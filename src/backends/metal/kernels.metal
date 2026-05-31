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

/* Causal block-skip bounds (B1). These are threadgroup-uniform: they depend only
 * on the block index, not the per-thread row, so every thread in the group runs
 * the same number of staging iterations and barriers stay aligned. The
 * per-element gd_sdpa_allowed() predicate still runs inside visited blocks, so
 * partial (diagonal) blocks and the window mask remain exact. */

/* For a query block covering [q0, q0+bq), the largest causal key index any query
 * in the block may attend is min(q0+bq-1, Tq-1) + (Tk-Tq). Key blocks past that
 * are fully masked; returns the exclusive key bound to cap the kb loop. */
static inline int gd_sdpa_kb_end(int q0, int bq, int Tq, int Tk, int causal)
{
    if (!causal) {
        return Tk;
    }
    int qmax = q0 + bq - 1;
    if (qmax > Tq - 1) {
        qmax = Tq - 1;
    }
    int lim = qmax + (Tk - Tq) + 1; /* keys j < lim may be attended */
    if (lim < 0) {
        lim = 0;
    }
    if (lim > Tk) {
        lim = Tk;
    }
    return lim;
}

/* Symmetric bound for the dk/dv kernel, which iterates query blocks per key
 * block. Keys [k0, ..) can only be attended by queries i >= k0 - (Tk-Tq); query
 * blocks fully below that are masked. Returns the qblk-aligned first query block
 * to start the qb loop from. */
static inline int gd_sdpa_qb_start(int k0, int qblk, int Tk, int Tq, int causal)
{
    if (!causal) {
        return 0;
    }
    int qmin = k0 - (Tk - Tq);
    if (qmin <= 0) {
        return 0;
    }
    return (qmin / qblk) * qblk;
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

/* FlashAttention-style tiled forward (G3). One threadgroup per (b, hq,
 * query-block of GD_SDPA_BQ queries); K/V key tiles are staged into threadgroup
 * memory once per block and reused by every query in the block, and softmax is
 * computed online in a single streaming pass (running max/denom/accumulator with
 * rescale). Capped to head_dim <= GD_SDPA_DHT; the host falls back to the
 * reference kernel for larger Dh. */
#define GD_SDPA_BQ GD_METAL_SDPA_BQ
#define GD_SDPA_BK GD_METAL_SDPA_BK
#define GD_SDPA_DHT GD_METAL_SDPA_DHT
kernel void gd_sdpa_tiled(device const float *q              [[buffer(0)]],
                          device const float *k              [[buffer(1)]],
                          device const float *v              [[buffer(2)]],
                          device const float *bias           [[buffer(3)]],
                          device float *out                  [[buffer(4)]],
                          constant gd_metal_sdpa_params &p    [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb; /* b*Hq + hq */
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        /* Cooperatively stage this key tile (all threads participate). */
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                    float s = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (s > m) ? s : m;
                    float corr = exp(m - mnew);
                    float e = exp(s - mnew);
                    l = l * corr + e;
                    for (int c = 0; c < p.Dh; ++c) {
                        acc[c] = acc[c] * corr + e * vsh[jj * p.Dh + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        if (l > 0.0f) {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = acc[c] / l;
            }
        } else {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = 0.0f;
            }
        }
    }
}

/* ---- Split-K (flash-decoding) forward SDPA ------------------------------- *
 * Same tiled online-softmax body as gd_sdpa_tiled, but each threadgroup scans
 * only key range [s*split_len, (s+1)*split_len) for split s, writing the partial
 * state (acc[Dh], m, l) to scratch. This shortens the critical path of the
 * heaviest query block from Tk to ~Tk/n_splits. gd_sdpa_combine merges the
 * splits. Scratch layout per (b, hq, i, s): Dh acc floats, then m, then l;
 * stride (Dh+2), index ((((b*Hq+hq)*Tq+i)*n_splits)+s)*(Dh+2). */
kernel void gd_sdpa_splitk(device const float *q              [[buffer(0)]],
                           device const float *k              [[buffer(1)]],
                           device const float *v              [[buffer(2)]],
                           device const float *bias           [[buffer(3)]],
                           device float *partials             [[buffer(4)]],
                           constant gd_metal_sdpa_params &p    [[buffer(5)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                    float ss = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        ss += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    for (int c = 0; c < p.Dh; ++c) {
                        acc[c] = acc[c] * corr + e * vsh[jj * p.Dh + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        for (int c = 0; c < p.Dh; ++c) {
            partials[base + c] = acc[c];
        }
        partials[base + p.Dh] = m;
        partials[base + p.Dh + 1] = l;
    }
}

/* Combine split-K partials into the final attention output. One thread per
 * (b, hq, i): online-merge the n_splits partial (acc, m, l) triples. */
kernel void gd_sdpa_combine(device const float *partials       [[buffer(0)]],
                            device float *out                  [[buffer(1)]],
                            constant gd_metal_sdpa_params &p    [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq;
    if ((int)gid >= total) {
        return;
    }
    int i = (int)gid % p.Tq;
    int r = (int)gid / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;

    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        float ms = partials[base + p.Dh];
        float ls = partials[base + p.Dh + 1];
        if (ls <= 0.0f) {
            continue;
        }
        float mnew = (ms > m) ? ms : m;
        float corr_o = exp(m - mnew);
        float corr_s = exp(ms - mnew);
        l = l * corr_o + ls * corr_s;
        for (int c = 0; c < p.Dh; ++c) {
            acc[c] = acc[c] * corr_o + partials[base + c] * corr_s;
        }
        m = mnew;
    }
    if (l > 0.0f) {
        for (int c = 0; c < p.Dh; ++c) {
            out[qbase + c] = acc[c] / l;
        }
    } else {
        for (int c = 0; c < p.Dh; ++c) {
            out[qbase + c] = 0.0f;
        }
    }
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

/* FlashAttention-2 style backward (G3). A stats pre-pass computes, per query,
 * the softmax max `m`, denominator `l`, and `D = sum_j p_ij (dO_i . v_j)` once;
 * dq and dk/dv then visit each (query,key) pair a single time using those
 * stats, so the total work is O(B*Hq*Tq*Tk*Dh) instead of the reference
 * O(B*Hkv*Tk * Tq*Tk) recompute. Stats layout: 3 floats per (b,hq,i):
 * [m, l, D] at index ((b*Hq + hq)*Tq + i)*3. */
kernel void gd_sdpa_bwd_stats(device const float *go            [[buffer(0)]],
                              device const float *q             [[buffer(1)]],
                              device const float *k             [[buffer(2)]],
                              device const float *v             [[buffer(3)]],
                              device const float *bias          [[buffer(4)]],
                              device float *stats               [[buffer(5)]],
                              constant gd_metal_sdpa_params &p   [[buffer(6)]],
                              uint tgid [[threadgroup_position_in_grid]],
                              uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        goreg[c] = active ? go[qbase + c] : 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (s > m) ? s : m;
                    float corr = exp(m - mnew);
                    float e = exp(s - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int sbase = ((b * p.Hq + hq) * p.Tq + i) * 3;
        stats[sbase + 0] = m;
        stats[sbase + 1] = l;
        stats[sbase + 2] = (l > 0.0f) ? (raw / l) : 0.0f;
    }
}

/* Backward dq: one threadgroup per (b, hq, query-block); K/V tiles staged and
 * reused across the block; single pass over keys using stats. */
kernel void gd_sdpa_bwd_dq(device const float *go            [[buffer(0)]],
                           device const float *q             [[buffer(1)]],
                           device const float *k             [[buffer(2)]],
                           device const float *v             [[buffer(3)]],
                           device const float *bias          [[buffer(4)]],
                           device float *dq                  [[buffer(5)]],
                           constant gd_metal_sdpa_params &p   [[buffer(6)]],
                           device const float *stats         [[buffer(7)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;
    int sbase = active ? (((b * p.Hq + hq) * p.Tq + i) * 3) : 0;
    float m = active ? stats[sbase + 0] : 0.0f;
    float l = active ? stats[sbase + 1] : 0.0f;
    float D = active ? stats[sbase + 2] : 0.0f;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    float dqacc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        goreg[c] = active ? go[qbase + c] : 0.0f;
        dqacc[c] = 0.0f;
    }
    bool run = active && l > 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (run) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float pj = exp(s - m) / l;
                    float ds = pj * (dp - D);
                    for (int c = 0; c < p.Dh; ++c) {
                        dqacc[c] += p.scale * ds * ksh[jj * p.Dh + c];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (int c = 0; c < p.Dh; ++c) {
            dq[qbase + c] = dqacc[c];
        }
    }
}

/* Backward dk/dv: one threadgroup per (b, hkv, key-block); Q/dO and per-query
 * stats are staged into threadgroup memory and reused across the key block.
 * Each thread owns one kv slot, so dk/dv accumulate conflict-free in registers. */
kernel void gd_sdpa_bwd_dkv(device const float *go            [[buffer(0)]],
                            device const float *q             [[buffer(1)]],
                            device const float *k             [[buffer(2)]],
                            device const float *v             [[buffer(3)]],
                            device const float *bias          [[buffer(4)]],
                            device float *dk                  [[buffer(5)]],
                            device float *dv                  [[buffer(6)]],
                            constant gd_metal_sdpa_params &p   [[buffer(7)]],
                            device const float *stats         [[buffer(8)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.Tk + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int kblk = (int)tgid % n_kb;
    int r = (int)tgid / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int j = kblk * GD_SDPA_BQ + (int)tid;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DHT];
    float vreg[GD_SDPA_DHT];
    float dkacc[GD_SDPA_DHT];
    float dvacc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        kreg[c] = active ? k[kbase + c] : 0.0f;
        vreg[c] = active ? v[kbase + c] : 0.0f;
        dkacc[c] = 0.0f;
        dvacc[c] = 0.0f;
    }

    int qb_start = gd_sdpa_qb_start(kblk * GD_SDPA_BQ, GD_SDPA_BK, p.Tk, p.Tq, p.causal);
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = qb_start; qb < p.Tq; qb += GD_SDPA_BK) {
            int tile = p.Tq - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
                int ii = idx / p.Dh;
                int c = idx % p.Dh;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = q[qb2 + c];
                gsh[ii * p.Dh + c] = go[qb2 + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_BQ) {
                int sb = ((b * p.Hq + hq) * p.Tq + (qb + ii)) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    if (!gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window)) {
                        continue;
                    }
                    float l = lsh[ii];
                    if (l <= 0.0f) {
                        continue;
                    }
                    float mm = msh[ii];
                    float D = dsh[ii];
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qsh[ii * p.Dh + c] * kreg[c];
                        dp += gsh[ii * p.Dh + c] * vreg[c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float pj = exp(s - mm) / l;
                    float ds = pj * (dp - D);
                    for (int c = 0; c < p.Dh; ++c) {
                        dvacc[c] += pj * gsh[ii * p.Dh + c];
                        dkacc[c] += p.scale * ds * qsh[ii * p.Dh + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        for (int c = 0; c < p.Dh; ++c) {
            dk[kbase + c] = dkacc[c];
            dv[kbase + c] = dvacc[c];
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
