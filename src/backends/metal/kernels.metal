#include <metal_stdlib>
#include "metal_kernel_types.h"

using namespace metal;

/* Reproduces broadcast_offset() from the CPU reference kernel: walk the input's
 * own dims right-aligned against the output index, treating size-1 dims as
 * broadcast (coord 0). Input is contiguous, so strides are implied by sizes. */
static int gd_broadcast_offset(thread const int *out_index,
                               int out_ndim,
                               constant int *in_sizes,
                               int in_ndim)
{
    int stride = 1;
    int offset = 0;
    for (int i = in_ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_ndim - i);
        int coord = (in_sizes[i] == 1) ? 0 : out_index[out_pos];
        offset += coord * stride;
        stride *= in_sizes[i];
    }
    return offset;
}

/* Shared body for the broadcasting binary ops; OP is 0=add, 1=mul. */
static inline void gd_binary(device const float *a,
                             device const float *b,
                             device float *out,
                             constant gd_metal_ew_params &p,
                             uint gid,
                             int op)
{
    if ((int)gid >= p.numel) {
        return;
    }
    int index[GD_METAL_MAX_DIMS];
    int lin = (int)gid;
    for (int i = p.ndim - 1; i >= 0; --i) {
        index[i] = lin % p.out_sizes[i];
        lin /= p.out_sizes[i];
    }
    int ao = gd_broadcast_offset(index, p.ndim, p.a_sizes, p.a_ndim);
    int bo = gd_broadcast_offset(index, p.ndim, p.b_sizes, p.b_ndim);
    out[gid] = (op == 0) ? (a[ao] + b[bo]) : (a[ao] * b[bo]);
}

kernel void gd_add(device const float *a          [[buffer(0)]],
                   device const float *b          [[buffer(1)]],
                   device float *out              [[buffer(2)]],
                   constant gd_metal_ew_params &p  [[buffer(3)]],
                   uint gid                        [[thread_position_in_grid]])
{
    gd_binary(a, b, out, p, gid, 0);
}

kernel void gd_mul(device const float *a          [[buffer(0)]],
                   device const float *b          [[buffer(1)]],
                   device float *out              [[buffer(2)]],
                   constant gd_metal_ew_params &p  [[buffer(3)]],
                   uint gid                        [[thread_position_in_grid]])
{
    gd_binary(a, b, out, p, gid, 1);
}

/* ---- Unary elementwise (contiguous) -------------------------------------- */

kernel void gd_scale(device const float *x            [[buffer(0)]],
                     device float *out                [[buffer(1)]],
                     constant gd_metal_unary_params &p [[buffer(2)]],
                     uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    out[gid] = x[gid] * p.scale;
}

kernel void gd_relu(device const float *x            [[buffer(0)]],
                    device float *out                [[buffer(1)]],
                    constant gd_metal_unary_params &p [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = x[gid];
    out[gid] = v > 0.0f ? v : 0.0f;
}

kernel void gd_silu(device const float *x            [[buffer(0)]],
                    device float *out                [[buffer(1)]],
                    constant gd_metal_unary_params &p [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = x[gid];
    out[gid] = v / (1.0f + exp(-v));
}

/* Identity/reshape copy of contiguous 4-byte elements (F32/I32); bit-exact. */
kernel void gd_copy(device const uint *x             [[buffer(0)]],
                    device uint *out                 [[buffer(1)]],
                    constant gd_metal_unary_params &p [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    out[gid] = x[gid];
}

/* dtype conversion over raw 4-byte words; matches the CPU reference's C casts
 * (truncation toward zero for F32->I32). */
kernel void gd_cast(device const uint *x              [[buffer(0)]],
                    device uint *out                  [[buffer(1)]],
                    constant gd_metal_cast_params &p   [[buffer(2)]],
                    uint gid                           [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    uint raw = x[gid];
    if (p.src_dtype == p.dst_dtype) {
        out[gid] = raw;
    } else if (p.src_dtype == GD_METAL_DT_F32 && p.dst_dtype == GD_METAL_DT_I32) {
        int v = (int)as_type<float>(raw);
        out[gid] = as_type<uint>(v);
    } else { /* I32 -> F32 */
        float f = (float)as_type<int>(raw);
        out[gid] = as_type<uint>(f);
    }
}

/* ---- Matmul / linear ----------------------------------------------------- */

/* Batched matmul; one thread per output element. Reproduces the CPU reference's
 * batch broadcasting and transpose addressing. Accumulates in float (v1). */
kernel void gd_matmul(device const float *a              [[buffer(0)]],
                      device const float *b              [[buffer(1)]],
                      device float *out                  [[buffer(2)]],
                      constant gd_metal_matmul_params &p  [[buffer(3)]],
                      uint2 gid                          [[thread_position_in_grid]])
{
    int batch_total = 1;
    for (int i = 0; i < p.batch_ndim; ++i) {
        batch_total *= p.out_batch_sizes[i];
    }
    int col = (int)gid.x;
    int grow = (int)gid.y; /* batch_lin * m + row */
    if (col >= p.n || grow >= batch_total * p.m) {
        return;
    }
    int batch_lin = grow / p.m;
    int row = grow % p.m;

    int bidx[GD_METAL_MAX_DIMS];
    int tmp = batch_lin;
    for (int i = p.batch_ndim - 1; i >= 0; --i) {
        bidx[i] = tmp % p.out_batch_sizes[i];
        tmp /= p.out_batch_sizes[i];
    }
    int a_base = 0, a_bstride = 1;
    for (int i = p.a_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.a_batch_ndim - i);
        int coord = (p.a_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        a_base += coord * a_bstride * p.a_mat;
        a_bstride *= p.a_batch_sizes[i];
    }
    int b_base = 0, b_bstride = 1;
    for (int i = p.b_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.b_batch_ndim - i);
        int coord = (p.b_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        b_base += coord * b_bstride * p.b_mat;
        b_bstride *= p.b_batch_sizes[i];
    }

    float acc = 0.0f;
    for (int kk = 0; kk < p.k; ++kk) {
        int a_off = p.trans_a ? (kk * p.a_cols + row) : (row * p.a_cols + kk);
        int b_off = p.trans_b ? (col * p.b_cols + kk) : (kk * p.b_cols + col);
        acc += a[a_base + a_off] * b[b_base + b_off];
    }
    out[batch_lin * p.out_mat + row * p.n + col] = acc;
}

/* out[r,o] = bias[o] + sum_k x[r,k] * w[...]. bias is always bound; read only
 * when has_bias. One thread per (out_feature, row). */
kernel void gd_linear(device const float *x               [[buffer(0)]],
                      device const float *w               [[buffer(1)]],
                      device const float *bias            [[buffer(2)]],
                      device float *out                   [[buffer(3)]],
                      constant gd_metal_linear_params &p   [[buffer(4)]],
                      uint2 gid                           [[thread_position_in_grid]])
{
    int o = (int)gid.x;
    int r = (int)gid.y;
    if (o >= p.out_features || r >= p.rows) {
        return;
    }
    float acc = p.has_bias ? bias[o] : 0.0f;
    for (int kk = 0; kk < p.in_features; ++kk) {
        int w_off = p.trans_w ? (o * p.in_features + kk) : (kk * p.out_features + o);
        acc += x[r * p.in_features + kk] * w[w_off];
    }
    out[r * p.out_features + o] = acc;
}

/* Tiled batched matmul. One threadgroup computes a GD_METAL_GEMM_TILE square
 * output block for a single batch (grid z = batch index). K is streamed in tiles
 * through threadgroup memory, so each A/B element is read from device memory
 * once per tile instead of once per output element. Reproduces the naive
 * kernel's batch-broadcast and transpose addressing, so it is a drop-in for all
 * shapes the reference kernel accepted. */
kernel void gd_matmul_tiled(device const float *a              [[buffer(0)]],
                            device const float *b              [[buffer(1)]],
                            device float *out                  [[buffer(2)]],
                            constant gd_metal_matmul_params &p  [[buffer(3)]],
                            uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
{
    threadgroup float As[GD_METAL_GEMM_TILE][GD_METAL_GEMM_TILE];
    threadgroup float Bs[GD_METAL_GEMM_TILE][GD_METAL_GEMM_TILE];

    int col = (int)(tgpos.x * GD_METAL_GEMM_TILE + lid.x);
    int row = (int)(tgpos.y * GD_METAL_GEMM_TILE + lid.y);
    int batch_lin = (int)tgpos.z;

    int bidx[GD_METAL_MAX_DIMS];
    int tmp = batch_lin;
    for (int i = p.batch_ndim - 1; i >= 0; --i) {
        bidx[i] = tmp % p.out_batch_sizes[i];
        tmp /= p.out_batch_sizes[i];
    }
    int a_base = 0, a_bstride = 1;
    for (int i = p.a_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.a_batch_ndim - i);
        int coord = (p.a_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        a_base += coord * a_bstride * p.a_mat;
        a_bstride *= p.a_batch_sizes[i];
    }
    int b_base = 0, b_bstride = 1;
    for (int i = p.b_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.b_batch_ndim - i);
        int coord = (p.b_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        b_base += coord * b_bstride * p.b_mat;
        b_bstride *= p.b_batch_sizes[i];
    }

    float acc = 0.0f;
    int n_tiles = (p.k + GD_METAL_GEMM_TILE - 1) / GD_METAL_GEMM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        int a_k = t * GD_METAL_GEMM_TILE + (int)lid.x;
        int b_k = t * GD_METAL_GEMM_TILE + (int)lid.y;

        if (row < p.m && a_k < p.k) {
            int a_off = p.trans_a ? (a_k * p.a_cols + row) : (row * p.a_cols + a_k);
            As[lid.y][lid.x] = a[a_base + a_off];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }
        if (col < p.n && b_k < p.k) {
            int b_off = p.trans_b ? (col * p.b_cols + b_k) : (b_k * p.b_cols + col);
            Bs[lid.y][lid.x] = b[b_base + b_off];
        } else {
            Bs[lid.y][lid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < GD_METAL_GEMM_TILE; ++i) {
            acc += As[lid.y][i] * Bs[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.m && col < p.n) {
        out[batch_lin * p.out_mat + row * p.n + col] = acc;
    }
}

/* Tiled linear: out[r,o] = bias[o] + sum_k x[r,k] * w[...]. Same tiling as the
 * matmul kernel (M=rows, N=out_features, K=in_features), with the weight layout
 * selected by trans_w and an optional bias add on store. */
kernel void gd_linear_tiled(device const float *x               [[buffer(0)]],
                            device const float *w               [[buffer(1)]],
                            device const float *bias            [[buffer(2)]],
                            device float *out                   [[buffer(3)]],
                            constant gd_metal_linear_params &p   [[buffer(4)]],
                            uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
{
    threadgroup float Xs[GD_METAL_GEMM_TILE][GD_METAL_GEMM_TILE];
    threadgroup float Ws[GD_METAL_GEMM_TILE][GD_METAL_GEMM_TILE];

    int o = (int)(tgpos.x * GD_METAL_GEMM_TILE + lid.x);
    int r = (int)(tgpos.y * GD_METAL_GEMM_TILE + lid.y);

    float acc = 0.0f;
    int n_tiles = (p.in_features + GD_METAL_GEMM_TILE - 1) / GD_METAL_GEMM_TILE;
    for (int t = 0; t < n_tiles; ++t) {
        int x_k = t * GD_METAL_GEMM_TILE + (int)lid.x;
        int w_k = t * GD_METAL_GEMM_TILE + (int)lid.y;

        if (r < p.rows && x_k < p.in_features) {
            Xs[lid.y][lid.x] = x[r * p.in_features + x_k];
        } else {
            Xs[lid.y][lid.x] = 0.0f;
        }
        if (o < p.out_features && w_k < p.in_features) {
            int w_off = p.trans_w ? (o * p.in_features + w_k) : (w_k * p.out_features + o);
            Ws[lid.y][lid.x] = w[w_off];
        } else {
            Ws[lid.y][lid.x] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = 0; i < GD_METAL_GEMM_TILE; ++i) {
            acc += Xs[lid.y][i] * Ws[i][lid.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r < p.rows && o < p.out_features) {
        out[r * p.out_features + o] = (p.has_bias ? bias[o] : 0.0f) + acc;
    }
}

/* ---- Reductions ---------------------------------------------------------- */

/* sum/mean over one dim; one thread per output position reduces the d axis. */
kernel void gd_reduce(device const float *x               [[buffer(0)]],
                      device float *out                   [[buffer(1)]],
                      constant gd_metal_reduce_params &p   [[buffer(2)]],
                      uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float acc = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        acc += x[(o * p.d + c) * p.inner + in];
    }
    if (p.mean) {
        acc /= (float)p.d;
    }
    out[o * p.inner + in] = acc;
}

/* Stable softmax over one dim; one thread per (outer,inner) position. */
kernel void gd_softmax(device const float *x               [[buffer(0)]],
                       device float *out                   [[buffer(1)]],
                       constant gd_metal_softmax_params &p  [[buffer(2)]],
                       uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float maxv = -INFINITY;
    for (int c = 0; c < p.d; ++c) {
        float v = x[(o * p.d + c) * p.inner + in];
        if (v > maxv) {
            maxv = v;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        float e = exp(x[(o * p.d + c) * p.inner + in] - maxv);
        out[(o * p.d + c) * p.inner + in] = e;
        sum += e;
    }
    for (int c = 0; c < p.d; ++c) {
        int idx = (o * p.d + c) * p.inner + in;
        out[idx] = out[idx] / sum;
    }
}

/* Threadgroup size for the RMSNorm-family reductions. Power of two so the
 * tree reduction is exact; sized to fit the shared partial arrays. */
#define GD_RMS_TG 256

/* RMSNorm over the last dim; one threadgroup per row, threads cooperatively
 * reduce sum(x^2) then write the normalized row. */
kernel void gd_rms_norm(device const float *x               [[buffer(0)]],
                        device const float *weight          [[buffer(1)]],
                        device float *out                   [[buffer(2)]],
                        constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                        uint tgid  [[threadgroup_position_in_grid]],
                        uint tid    [[thread_index_in_threadgroup]],
                        uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows) {
        return;
    }
    threadgroup float part[GD_RMS_TG];
    int base = r * p.last;
    float local = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = x[base + c];
        local += v * v;
    }
    part[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            part[tid] += part[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(part[0] / (float)p.last + p.eps);
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        out[base + c] = x[base + c] * inv * weight[c];
    }
}

/* RMSNorm backward dx; one threadgroup per row. Threads cooperatively reduce
 * both sum(x^2) (for the rms) and A = sum_c go*weight*x in a single pass, then
 * write dx. */
kernel void gd_rms_norm_bwd(device const float *x               [[buffer(0)]],
                            device const float *weight          [[buffer(1)]],
                            device const float *go              [[buffer(2)]],
                            device float *dx                    [[buffer(3)]],
                            constant gd_metal_rmsnorm_params &p  [[buffer(4)]],
                            uint tgid  [[threadgroup_position_in_grid]],
                            uint tid    [[thread_index_in_threadgroup]],
                            uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows) {
        return;
    }
    threadgroup float pss[GD_RMS_TG];
    threadgroup float pa[GD_RMS_TG];
    int base = r * p.last;
    float lss = 0.0f;
    float la = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float v = x[base + c];
        lss += v * v;
        la += go[base + c] * weight[c] * v;
    }
    pss[tid] = lss;
    pa[tid] = la;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            pss[tid] += pss[tid + stride];
            pa[tid] += pa[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = 1.0f / sqrt(pss[0] / (float)p.last + p.eps);
    float inv3 = inv * inv * inv;
    float A = pa[0];
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        dx[base + c] = inv * go[base + c] * weight[c] - x[base + c] * inv3 * A / (float)p.last;
    }
}

/* RMSNorm backward dweight = sum_r go[r,c] * x[r,c] * rms_inv[r]. One threadgroup
 * per channel-tile; the per-row rms_inv is computed cooperatively once per row
 * tile and cached in threadgroup memory, instead of being recomputed for every
 * channel (the previous one-thread-per-channel kernel recomputed it `last`
 * times, which dominated GPT's backward). */
kernel void gd_rms_norm_wbwd(device const float *x               [[buffer(0)]],
                             device const float *go              [[buffer(1)]],
                             device float *dweight               [[buffer(2)]],
                             constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                             uint tgid  [[threadgroup_position_in_grid]],
                             uint tid    [[thread_index_in_threadgroup]],
                             uint tgsz   [[threads_per_threadgroup]])
{
    threadgroup float inv_sh[GD_RMS_TG];
    int c = (int)(tgid * tgsz + tid);
    float acc = 0.0f;

    for (int rb = 0; rb < p.rows; rb += GD_RMS_TG) {
        int tile = p.rows - rb;
        if (tile > GD_RMS_TG) {
            tile = GD_RMS_TG;
        }
        /* Cooperatively compute rms_inv for the rows in this tile (one row per
         * thread for tid < tile). */
        if ((int)tid < tile) {
            int rbase = (rb + (int)tid) * p.last;
            float ss = 0.0f;
            for (int cc = 0; cc < p.last; ++cc) {
                float v = x[rbase + cc];
                ss += v * v;
            }
            inv_sh[tid] = 1.0f / sqrt(ss / (float)p.last + p.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c < p.last) {
            for (int i = 0; i < tile; ++i) {
                int rbase = (rb + i) * p.last;
                acc += go[rbase + c] * x[rbase + c] * inv_sh[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (c < p.last) {
        dweight[c] = acc;
    }
}

/* Mean cross-entropy to a single scalar. One threadgroup: each thread sums the
 * per-position negative log-likelihood for a strided subset of positions, then a
 * threadgroup reduction produces the mean. Targets are int32; class index is
 * assumed valid (matches a validated graph). */
#define GD_CE_TG 256
kernel void gd_cross_entropy(device const float *logits      [[buffer(0)]],
                             device const int *targets       [[buffer(1)]],
                             device float *out               [[buffer(2)]],
                             constant gd_metal_ce_params &p   [[buffer(3)]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float partial[GD_CE_TG];
    float local = 0.0f;
    int total = p.outer * p.inner;

    for (int pos = (int)tid; pos < total; pos += (int)tgsz) {
        int o = pos / p.inner;
        int in = pos % p.inner;
        int target = targets[pos];
        float maxv = -INFINITY;
        for (int c = 0; c < p.classes; ++c) {
            float v = logits[(o * p.classes + c) * p.inner + in];
            if (v > maxv) {
                maxv = v;
            }
        }
        float sum = 0.0f;
        for (int c = 0; c < p.classes; ++c) {
            sum += exp(logits[(o * p.classes + c) * p.inner + in] - maxv);
        }
        float logit_t = logits[(o * p.classes + target) * p.inner + in];
        local += -(logit_t - maxv - log(sum));
    }

    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[0] = partial[0] / (float)p.positions;
    }
}

/* ---- Backward kernels ---------------------------------------------------- */

kernel void gd_relu_bwd(device const float *x             [[buffer(0)]],
                        device const float *go            [[buffer(1)]],
                        device float *dx                  [[buffer(2)]],
                        constant gd_metal_unary_params &p [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    dx[gid] = x[gid] > 0.0f ? go[gid] : 0.0f;
}

kernel void gd_silu_bwd(device const float *x             [[buffer(0)]],
                        device const float *go            [[buffer(1)]],
                        device float *dx                  [[buffer(2)]],
                        constant gd_metal_unary_params &p [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = x[gid];
    float s = 1.0f / (1.0f + exp(-xv));
    float grad = s * (1.0f + xv * (1.0f - s));
    dx[gid] = go[gid] * grad;
}

/* dx = y * (go - sum_c go*y) along `dim`; one thread per (outer,inner). */
kernel void gd_softmax_bwd(device const float *y               [[buffer(0)]],
                           device const float *go              [[buffer(1)]],
                           device float *dx                    [[buffer(2)]],
                           constant gd_metal_softmax_params &p  [[buffer(3)]],
                           uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float dot = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        int idx = (o * p.d + c) * p.inner + in;
        dot += go[idx] * y[idx];
    }
    for (int c = 0; c < p.d; ++c) {
        int idx = (o * p.d + c) * p.inner + in;
        dx[idx] = y[idx] * (go[idx] - dot);
    }
}

/* Broadcast the reduced gradient back over `dim`; one thread per (outer,inner). */
kernel void gd_sum_bwd(device const float *go              [[buffer(0)]],
                       device float *dx                    [[buffer(1)]],
                       constant gd_metal_reduce_params &p   [[buffer(2)]],
                       uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float scale = p.mean ? (1.0f / (float)p.d) : 1.0f;
    float g = go[o * p.inner + in] * scale;
    for (int c = 0; c < p.d; ++c) {
        dx[(o * p.d + c) * p.inner + in] = g;
    }
}

/* dlogits = (go_scalar/positions) * (softmax - onehot); one thread per position. */
kernel void gd_cross_entropy_bwd(device const float *logits     [[buffer(0)]],
                                 device const int *targets      [[buffer(1)]],
                                 device const float *go_scalar  [[buffer(2)]],
                                 device float *dlogits          [[buffer(3)]],
                                 constant gd_metal_ce_params &p  [[buffer(4)]],
                                 uint gid                       [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    int pos = o * p.inner + in;
    int target = targets[pos];
    float scale = go_scalar[0] / (float)p.positions;
    float maxv = -INFINITY;
    for (int c = 0; c < p.classes; ++c) {
        float v = logits[(o * p.classes + c) * p.inner + in];
        if (v > maxv) {
            maxv = v;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < p.classes; ++c) {
        sum += exp(logits[(o * p.classes + c) * p.inner + in] - maxv);
    }
    for (int c = 0; c < p.classes; ++c) {
        float pc = exp(logits[(o * p.classes + c) * p.inner + in] - maxv) / sum;
        float onehot = (c == target) ? 1.0f : 0.0f;
        dlogits[(o * p.classes + c) * p.inner + in] = scale * (pc - onehot);
    }
}

/* Sum `go` down into `out` (target shape) following right-aligned broadcast
 * rules. One thread owns one target element and sums only the `go` positions
 * that broadcast to it (iterating the reduced dims), so the total work is
 * O(go_numel) split across target_numel threads instead of a single serial
 * scan. Both tensors are contiguous. */
kernel void gd_reduce_to(device const float *go                [[buffer(0)]],
                         device float *out                     [[buffer(1)]],
                         constant gd_metal_reduce_to_params &p  [[buffer(2)]],
                         uint gid                              [[thread_position_in_grid]])
{
    int t = (int)gid;
    if (t >= p.target_numel) {
        return;
    }

    /* Target coordinates (contiguous target). */
    int tcoord[GD_METAL_MAX_DIMS];
    int lin = t;
    for (int k = p.target_ndim - 1; k >= 0; --k) {
        tcoord[k] = (p.target_sizes[k] > 0) ? (lin % p.target_sizes[k]) : 0;
        if (p.target_sizes[k] > 0) {
            lin /= p.target_sizes[k];
        }
    }

    /* Contiguous strides for go. */
    int go_stride[GD_METAL_MAX_DIMS];
    int s = 1;
    for (int i = p.go_ndim - 1; i >= 0; --i) {
        go_stride[i] = s;
        s *= p.go_sizes[i];
    }

    /* Partition go dims into fixed (matched to a target coord) and reduced
     * (extra leading dim, or target dim of size 1 that go broadcasts over). */
    int reduce_dims[GD_METAL_MAX_DIMS];
    int n_reduce = 0;
    int reduce_count = 1;
    int base = 0;
    for (int i = 0; i < p.go_ndim; ++i) {
        int out_pos = p.target_ndim - (p.go_ndim - i);
        bool reduced = (out_pos < 0) ||
                       (p.target_sizes[out_pos] == 1 && p.go_sizes[i] > 1);
        if (reduced) {
            reduce_dims[n_reduce++] = i;
            reduce_count *= p.go_sizes[i];
        } else {
            int coord = (out_pos < 0 || p.go_sizes[i] == 1) ? 0 : tcoord[out_pos];
            base += coord * go_stride[i];
        }
    }

    float acc = 0.0f;
    for (int r = 0; r < reduce_count; ++r) {
        int rem = r;
        int goff = base;
        for (int j = n_reduce - 1; j >= 0; --j) {
            int dim = reduce_dims[j];
            goff += (rem % p.go_sizes[dim]) * go_stride[dim];
            rem /= p.go_sizes[dim];
        }
        acc += go[goff];
    }
    out[t] = acc;
}

/* ---- Optimizer ----------------------------------------------------------- */

kernel void gd_step_inc(device float *step [[buffer(0)]],
                        uint gid           [[thread_position_in_grid]])
{
    if ((int)gid != 0) {
        return;
    }
    step[0] += 1.0f;
}

kernel void gd_adamw(device float *param                 [[buffer(0)]],
                     device const float *grad            [[buffer(1)]],
                     device float *m                     [[buffer(2)]],
                     device float *v                     [[buffer(3)]],
                     device const float *step            [[buffer(4)]],
                     constant gd_metal_adamw_params &p    [[buffer(5)]],
                     uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float t = step[0];
    float bc1 = 1.0f - pow(p.beta1, t);
    float bc2 = 1.0f - pow(p.beta2, t);
    float g = grad[gid];
    float mi = p.beta1 * m[gid] + (1.0f - p.beta1) * g;
    float vi = p.beta2 * v[gid] + (1.0f - p.beta2) * g * g;
    float mhat = mi / bc1;
    float vhat = vi / bc2;
    float pp = param[gid];

    m[gid] = mi;
    v[gid] = vi;
    pp -= p.lr * p.weight_decay * pp;            /* decoupled weight decay */
    pp -= p.lr * mhat / (sqrt(vhat) + p.eps);
    param[gid] = pp;
}

/* ---- GPT primitives (G0) ------------------------------------------------- */

/* MSL lacks erf(); Abramowitz & Stegun 7.1.26 (max abs error ~1.5e-7), which is
 * far below the 1e-4 GELU parity tolerance against the CPU's libm erf. */
static inline float gd_erff(float x)
{
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                       - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

kernel void gd_gelu(device const float *x             [[buffer(0)]],
                    device float *out                 [[buffer(1)]],
                    constant gd_metal_gelu_params &p   [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = x[gid];
    if (p.tanh_approx) {
        const float c = 0.7978845608028654f; /* sqrt(2/pi) */
        float inner = c * (xv + 0.044715f * xv * xv * xv);
        out[gid] = 0.5f * xv * (1.0f + tanh(inner));
    } else {
        const float inv_sqrt2 = 0.7071067811865476f;
        out[gid] = 0.5f * xv * (1.0f + gd_erff(xv * inv_sqrt2));
    }
}

kernel void gd_gelu_bwd(device const float *x             [[buffer(0)]],
                        device const float *go            [[buffer(1)]],
                        device float *dx                  [[buffer(2)]],
                        constant gd_metal_gelu_params &p   [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = x[gid];
    if (p.tanh_approx) {
        const float c = 0.7978845608028654f;
        float u = c * (xv + 0.044715f * xv * xv * xv);
        float t = tanh(u);
        float du = c * (1.0f + 3.0f * 0.044715f * xv * xv);
        float g = 0.5f * (1.0f + t) + 0.5f * xv * (1.0f - t * t) * du;
        dx[gid] = go[gid] * g;
    } else {
        const float inv_sqrt2 = 0.7071067811865476f;
        const float inv_sqrt2pi = 0.3989422804014327f;
        float cdf = 0.5f * (1.0f + gd_erff(xv * inv_sqrt2));
        float pdf = inv_sqrt2pi * exp(-0.5f * xv * xv);
        dx[gid] = go[gid] * (cdf + xv * pdf);
    }
}

/* Physical axis permutation; one thread per (4-byte) output element. */
kernel void gd_transpose(device const uint *in                [[buffer(0)]],
                         device uint *out                     [[buffer(1)]],
                         constant gd_metal_transpose_params &p [[buffer(2)]],
                         uint gid                             [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    int out_index[GD_METAL_MAX_DIMS];
    int lin = (int)gid;
    for (int k = p.ndim - 1; k >= 0; --k) {
        out_index[k] = lin % p.out_sizes[k];
        lin /= p.out_sizes[k];
    }
    int in_off = 0;
    for (int k = 0; k < p.ndim; ++k) {
        in_off += out_index[k] * p.in_strides[p.perm[k]];
    }
    out[gid] = in[in_off];
}

/* Row gather: out[p, c] = table[ids[p], c]; one thread per output element. */
kernel void gd_embedding(device const float *table           [[buffer(0)]],
                         device const int *ids               [[buffer(1)]],
                         device float *out                   [[buffer(2)]],
                         constant gd_metal_embedding_params &p [[buffer(3)]],
                         uint gid                            [[thread_position_in_grid]])
{
    int total = p.n * p.dim;
    if ((int)gid >= total) {
        return;
    }
    int row = (int)gid / p.dim;
    int c = (int)gid % p.dim;
    int id = ids[row];
    out[gid] = table[id * p.dim + c];
}

/* ---- Scaled dot-product attention (reference, dense + causal, GQA) -------- */

static inline bool gd_sdpa_allowed(int i, int j, int Tq, int Tk, int causal, int window)
{
    int qpos = i + (Tk - Tq);
    if (causal && j > qpos) {
        return false;
    }
    if (window > 0 && (qpos - j) >= window) {
        return false;
    }
    return true;
}

static inline float gd_sdpa_dot(device const float *a, device const float *b, int n)
{
    float s = 0.0f;
    for (int c = 0; c < n; ++c) {
        s += a[c] * b[c];
    }
    return s;
}

/* Additive attention bias broadcast over [B, Hq, Tq, Tk]. */
static inline float gd_sdpa_bias_at(device const float *bias,
                                    constant gd_metal_sdpa_params &p,
                                    int b, int hq, int i, int j)
{
    if (!p.has_bias) {
        return 0.0f;
    }
    int bb = (p.Bb == 1) ? 0 : b;
    int hb = (p.Hb == 1) ? 0 : hq;
    int ib = (p.Tqb == 1) ? 0 : i;
    int jb = (p.Tkb == 1) ? 0 : j;
    return bias[((bb * p.Hb + hb) * p.Tqb + ib) * p.Tkb + jb];
}

/* Forward: one thread per (b, hq, i); two-pass stable softmax, no score matrix. */
kernel void gd_sdpa(device const float *q              [[buffer(0)]],
                    device const float *k              [[buffer(1)]],
                    device const float *v              [[buffer(2)]],
                    device const float *bias           [[buffer(3)]],
                    device float *out                  [[buffer(4)]],
                    constant gd_metal_sdpa_params &p    [[buffer(5)]],
                    uint gid                           [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq;
    if ((int)gid >= total) {
        return;
    }
    int i = (int)gid % p.Tq;
    int r = (int)gid / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    int obase = qbase;

    float m = -INFINITY;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            if (s > m) {
                m = s;
            }
        }
    }
    for (int c = 0; c < p.Dh; ++c) {
        out[obase + c] = 0.0f;
    }
    float sum = 0.0f;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            int vbase = kbase;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            float e = exp(s - m);
            sum += e;
            for (int c = 0; c < p.Dh; ++c) {
                out[obase + c] += e * v[vbase + c];
            }
        }
    }
    if (sum > 0.0f) {
        for (int c = 0; c < p.Dh; ++c) {
            out[obase + c] /= sum;
        }
    }
}

/* Backward dq: one thread per (b, hq, i); recompute softmax stats then dq. */
kernel void gd_sdpa_bwd_dq(device const float *go            [[buffer(0)]],
                           device const float *q             [[buffer(1)]],
                           device const float *k             [[buffer(2)]],
                           device const float *v             [[buffer(3)]],
                           device const float *bias          [[buffer(4)]],
                           device float *dq                  [[buffer(5)]],
                           constant gd_metal_sdpa_params &p   [[buffer(6)]],
                           uint gid                          [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq;
    if ((int)gid >= total) {
        return;
    }
    int i = (int)gid % p.Tq;
    int r = (int)gid / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    int gobase = qbase;

    float m = -INFINITY;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            if (s > m) {
                m = s;
            }
        }
    }
    float sum = 0.0f;
    float dsum = 0.0f;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            sum += exp(s - m);
        }
    }
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            float pj = exp(s - m) / sum;
            dsum += pj * gd_sdpa_dot(go + gobase, v + kbase, p.Dh);
        }
    }
    for (int c = 0; c < p.Dh; ++c) {
        dq[qbase + c] = 0.0f;
    }
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            float pj = exp(s - m) / sum;
            float dp = gd_sdpa_dot(go + gobase, v + kbase, p.Dh);
            float ds = pj * (dp - dsum);
            for (int c = 0; c < p.Dh; ++c) {
                dq[qbase + c] += p.scale * ds * k[kbase + c];
            }
        }
    }
}

/* Backward dk/dv: one thread per (b, hkv, j); loop the query-head group and all
 * query positions, accumulating into this kv slot (no cross-thread conflict). */
kernel void gd_sdpa_bwd_dkv(device const float *go            [[buffer(0)]],
                            device const float *q             [[buffer(1)]],
                            device const float *k             [[buffer(2)]],
                            device const float *v             [[buffer(3)]],
                            device const float *bias          [[buffer(4)]],
                            device float *dk                  [[buffer(5)]],
                            device float *dv                  [[buffer(6)]],
                            constant gd_metal_sdpa_params &p   [[buffer(7)]],
                            uint gid                          [[thread_position_in_grid]])
{
    int total = p.B * p.Hkv * p.Tk;
    if ((int)gid >= total) {
        return;
    }
    int j = (int)gid % p.Tk;
    int r = (int)gid / p.Tk;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;

    for (int c = 0; c < p.Dh; ++c) {
        dk[kbase + c] = 0.0f;
        dv[kbase + c] = 0.0f;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int i = 0; i < p.Tq; ++i) {
            if (!gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                continue;
            }
            int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
            int gobase = qbase;
            float m = -INFINITY;
            for (int jj = 0; jj < p.Tk; ++jj) {
                if (gd_sdpa_allowed(i, jj, p.Tq, p.Tk, p.causal, p.window)) {
                    int kb = ((b * p.Tk + jj) * p.Hkv + hkv) * p.Dh;
                    float s = p.scale * gd_sdpa_dot(q + qbase, k + kb, p.Dh)
                              + gd_sdpa_bias_at(bias, p, b, hq, i, jj);
                    if (s > m) {
                        m = s;
                    }
                }
            }
            float sum = 0.0f;
            float dsum = 0.0f;
            for (int jj = 0; jj < p.Tk; ++jj) {
                if (gd_sdpa_allowed(i, jj, p.Tq, p.Tk, p.causal, p.window)) {
                    int kb = ((b * p.Tk + jj) * p.Hkv + hkv) * p.Dh;
                    float s = p.scale * gd_sdpa_dot(q + qbase, k + kb, p.Dh)
                              + gd_sdpa_bias_at(bias, p, b, hq, i, jj);
                    sum += exp(s - m);
                }
            }
            for (int jj = 0; jj < p.Tk; ++jj) {
                if (gd_sdpa_allowed(i, jj, p.Tq, p.Tk, p.causal, p.window)) {
                    int kb = ((b * p.Tk + jj) * p.Hkv + hkv) * p.Dh;
                    float s = p.scale * gd_sdpa_dot(q + qbase, k + kb, p.Dh)
                              + gd_sdpa_bias_at(bias, p, b, hq, i, jj);
                    float pjj = exp(s - m) / sum;
                    dsum += pjj * gd_sdpa_dot(go + gobase, v + kb, p.Dh);
                }
            }
            float sj = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                       + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            float pj = exp(sj - m) / sum;
            float dp = gd_sdpa_dot(go + gobase, v + kbase, p.Dh);
            float ds = pj * (dp - dsum);
            for (int c = 0; c < p.Dh; ++c) {
                dv[kbase + c] += pj * go[gobase + c];
                dk[kbase + c] += p.scale * ds * q[qbase + c];
            }
        }
    }
}

/* Rotary position embedding; one thread per (.., head) row. Serves forward and
 * the transpose (backward) rotation via sin_sign. */
kernel void gd_rope(device const float *x                 [[buffer(0)]],
                    device const int *pos                 [[buffer(1)]],
                    device float *out                     [[buffer(2)]],
                    constant gd_metal_rope_params &p       [[buffer(3)]],
                    uint gid                              [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) {
        return;
    }
    int pos_idx = (int)gid / p.heads;
    int pp = pos[pos_idx];
    int base = (int)gid * p.head_dim;
    int rd_half = p.n_dims / 2;
    for (int d = p.n_dims; d < p.head_dim; ++d) {
        out[base + d] = x[base + d];
    }
    for (int i = 0; i < rd_half; ++i) {
        float inv = pow(p.theta, -((float)(2 * i)) / (float)p.n_dims);
        float angle = (float)pp * inv;
        float c = cos(angle);
        float s = sin(angle) * p.sin_sign;
        int a = p.interleaved ? (2 * i) : i;
        int bb = p.interleaved ? (2 * i + 1) : (i + rd_half);
        float x1 = x[base + a];
        float x2 = x[base + bb];
        out[base + a] = x1 * c - x2 * s;
        out[base + bb] = x1 * s + x2 * c;
    }
}

/* Scatter-add backward. One thread owns one dtable element (v, c) and sums the
 * gradient rows whose id == v, avoiding atomics and the serial scan. Work per
 * thread is O(n_ids); total O(vocab*dim*n_ids) spread across vocab*dim threads. */
kernel void gd_embedding_bwd(device const float *go               [[buffer(0)]],
                             device const int *ids                [[buffer(1)]],
                             device float *dtable                 [[buffer(2)]],
                             constant gd_metal_embedding_params &p [[buffer(3)]],
                             uint gid                            [[thread_position_in_grid]])
{
    int idx = (int)gid;
    if (idx >= p.vocab * p.dim) {
        return;
    }
    int v = idx / p.dim;
    int c = idx % p.dim;
    float acc = 0.0f;
    for (int row = 0; row < p.n; ++row) {
        if (ids[row] == v) {
            acc += go[row * p.dim + c];
        }
    }
    dtable[idx] = acc;
}
