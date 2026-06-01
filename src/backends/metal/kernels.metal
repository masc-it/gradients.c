#include <metal_stdlib>
#include "metal_kernel_types.h"

using namespace metal;

#define GD_CLIP_NORM_TG 256

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
    if (p.same_shape) {
        out[gid] = (op == 0) ? (a[gid] + b[gid]) : (a[gid] * b[gid]);
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

static inline float gd_sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    float e = exp(x);
    return e / (1.0f + e);
}

static inline float gd_powlu_gate(float z, float m)
{
    float s = gd_sigmoid_stable(z);
    if (z <= 0.0f) {
        return z * s;
    }
    float r = sqrt(max(z, 0.0f));
    float a = m / (r + 1.0f);
    return pow(z, a) * s;
}

static inline float gd_powlu_gate_grad(float z, float m)
{
    float s = gd_sigmoid_stable(z);
    if (z <= 0.0f) {
        return s * (1.0f + z * (1.0f - s));
    }
    float r = sqrt(max(z, 0.0f));
    float rp1 = r + 1.0f;
    float a = m / rp1;
    float g = pow(z, a);
    float da = -m / (2.0f * r * rp1 * rp1);
    float lz = log(max(z, 0x1p-126f));
    return g * s * (a / z + da * lz + (1.0f - s));
}

kernel void gd_powlu(device const float *x1            [[buffer(0)]],
                     device const float *x2            [[buffer(1)]],
                     device float *out                 [[buffer(2)]],
                     constant gd_metal_powlu_params &p  [[buffer(3)]],
                     uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    out[gid] = x1[gid] * gd_powlu_gate(x2[gid], p.m);
}

kernel void gd_powlu_bwd(device const float *x1            [[buffer(0)]],
                         device const float *x2            [[buffer(1)]],
                         device const float *go            [[buffer(2)]],
                         device float *dx1                 [[buffer(3)]],
                         device float *dx2                 [[buffer(4)]],
                         constant gd_metal_powlu_params &p  [[buffer(5)]],
                         uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float gate = gd_powlu_gate(x2[gid], p.m);
    float grad = gd_powlu_gate_grad(x2[gid], p.m);
    dx1[gid] = go[gid] * gate;
    dx2[gid] = go[gid] * x1[gid] * grad;
}

/* Fused SwiGLU activation: hh = silu(gate) * up. Produces `act = silu(gate)` as
 * a second output so the unfused backward (mul_bwd reads act, silu_bwd reads
 * gate) is unchanged. Bit-identical to gd_silu followed by gd_mul on equal
 * shapes. See docs/metal_gpu_fuse.md F1. */
kernel void gd_silu_mul(device const float *gate         [[buffer(0)]],
                        device const float *up           [[buffer(1)]],
                        device float *hh                 [[buffer(2)]],
                        device float *act                [[buffer(3)]],
                        constant gd_metal_unary_params &p [[buffer(4)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = gate[gid];
    float a = v / (1.0f + exp(-v));
    act[gid] = a;
    hh[gid] = a * up[gid];
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
    threadgroup float As[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup float Bs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    const int nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN)
                       * (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);

    int tx = (int)lid.x; /* 0 .. BN/TN */
    int ty = (int)lid.y; /* 0 .. BM/TM */
    int tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    int m0 = (int)tgpos.y * GD_METAL_GEMM_BM; /* first output row of block */
    int n0 = (int)tgpos.x * GD_METAL_GEMM_BN; /* first output col of block */
    int row0 = m0 + ty * GD_METAL_GEMM_TM;
    int col0 = n0 + tx * GD_METAL_GEMM_TN;
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

    float4 acc[GD_METAL_GEMM_TM];
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }
    int n_tiles = (p.k + GD_METAL_GEMM_BK - 1) / GD_METAL_GEMM_BK;
    for (int t = 0; t < n_tiles; ++t) {
        int kbase = t * GD_METAL_GEMM_BK;
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BM;
            int mr = e % GD_METAL_GEMM_BM;
            int gr = m0 + mr;
            int gk = kbase + kr;
            float val = 0.0f;
            if (gr < p.m && gk < p.k) {
                int a_off = p.trans_a ? (gk * p.a_cols + gr) : (gr * p.a_cols + gk);
                val = a[a_base + a_off];
            }
            As[kr][mr] = val;
        }
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BN;
            int nc = e % GD_METAL_GEMM_BN;
            int gk = kbase + kr;
            int gc = n0 + nc;
            float val = 0.0f;
            if (gc < p.n && gk < p.k) {
                int b_off = p.trans_b ? (gc * p.b_cols + gk) : (gk * p.b_cols + gc);
                val = b[b_base + b_off];
            }
            Bs[kr][nc] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            float4 breg = *(threadgroup float4 *)(&Bs[kk][tx * GD_METAL_GEMM_TN]);
            float4 areg = *(threadgroup float4 *)(&As[kk][ty * GD_METAL_GEMM_TM]);
            acc[0] += areg.x * breg;
            acc[1] += areg.y * breg;
            acc[2] += areg.z * breg;
            acc[3] += areg.w * breg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        int gr = row0 + tm;
        if (gr >= p.m) {
            continue;
        }
        float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (int tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            int gc = col0 + tn;
            if (gc < p.n) {
                out[batch_lin * p.out_mat + gr * p.n + gc] = vals[tn];
            }
        }
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
    threadgroup float Xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup float Ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    const int nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN)
                       * (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);

    int tx = (int)lid.x;
    int ty = (int)lid.y;
    int tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    int r0 = (int)tgpos.y * GD_METAL_GEMM_BM; /* first row of block */
    int o0 = (int)tgpos.x * GD_METAL_GEMM_BN; /* first out-feature of block */
    int row0 = r0 + ty * GD_METAL_GEMM_TM;
    int col0 = o0 + tx * GD_METAL_GEMM_TN;

    float4 acc[GD_METAL_GEMM_TM];
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }
    int n_tiles = (p.in_features + GD_METAL_GEMM_BK - 1) / GD_METAL_GEMM_BK;
    for (int t = 0; t < n_tiles; ++t) {
        int kbase = t * GD_METAL_GEMM_BK;
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BM;
            int mr = e % GD_METAL_GEMM_BM;
            int gr = r0 + mr;
            int gk = kbase + kr;
            Xs[kr][mr] = (gr < p.rows && gk < p.in_features)
                             ? x[gr * p.in_features + gk] : 0.0f;
        }
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BN;
            int nc = e % GD_METAL_GEMM_BN;
            int gk = kbase + kr;
            int gc = o0 + nc;
            float val = 0.0f;
            if (gc < p.out_features && gk < p.in_features) {
                int w_off = p.trans_w ? (gc * p.in_features + gk) : (gk * p.out_features + gc);
                val = w[w_off];
            }
            Ws[kr][nc] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            float4 wreg = *(threadgroup float4 *)(&Ws[kk][tx * GD_METAL_GEMM_TN]);
            float4 xreg = *(threadgroup float4 *)(&Xs[kk][ty * GD_METAL_GEMM_TM]);
            acc[0] += xreg.x * wreg;
            acc[1] += xreg.y * wreg;
            acc[2] += xreg.z * wreg;
            acc[3] += xreg.w * wreg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        int gr = row0 + tm;
        if (gr >= p.rows) {
            continue;
        }
        float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (int tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            int gc = col0 + tn;
            if (gc < p.out_features) {
                out[gr * p.out_features + gc] = (p.has_bias ? bias[gc] : 0.0f) + vals[tn];
            }
        }
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

/* F4 fused residual add + RMSNorm forward. Writes both boundary values: sum
 * (the residual stream, because later forward/backward nodes may read it) and
 * normalized output. sum is kept in threadgroup memory, so RMSNorm does not
 * reload the just-written add output from device memory. */
kernel void gd_add_rms_norm(device const float *a               [[buffer(0)]],
                            device const float *b               [[buffer(1)]],
                            device const float *weight          [[buffer(2)]],
                            device float *sum_out               [[buffer(3)]],
                            device float *norm_out              [[buffer(4)]],
                            constant gd_metal_rmsnorm_params &p  [[buffer(5)]],
                            uint tgid  [[threadgroup_position_in_grid]],
                            uint tid    [[thread_index_in_threadgroup]],
                            uint tgsz   [[threads_per_threadgroup]])
{
    int r = (int)tgid;
    if (r >= p.rows || p.last > GD_METAL_FUSED_RMS_MAX) {
        return;
    }
    threadgroup float part[GD_RMS_TG];
    threadgroup float sum_sh[GD_METAL_FUSED_RMS_MAX];
    int base = r * p.last;
    float local = 0.0f;
    for (int c = (int)tid; c < p.last; c += (int)tgsz) {
        float s = a[base + c] + b[base + c];
        sum_sh[c] = s;
        sum_out[base + c] = s;
        local += s * s;
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
        norm_out[base + c] = sum_sh[c] * inv * weight[c];
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

/* F4 fused RMSNorm backward + residual-gradient add. Materializes the raw
 * RMSNorm dx (for graph parity / any outside consumer) and the accumulated
 * residual gradient dsum = dx + dS_out for add backward. */
kernel void gd_rms_norm_bwd_add(device const float *x               [[buffer(0)]],
                                device const float *weight          [[buffer(1)]],
                                device const float *go              [[buffer(2)]],
                                device const float *ds_out          [[buffer(3)]],
                                device float *dx                    [[buffer(4)]],
                                device float *dsum                  [[buffer(5)]],
                                constant gd_metal_rmsnorm_params &p  [[buffer(6)]],
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
        float d = inv * go[base + c] * weight[c] - x[base + c] * inv3 * A / (float)p.last;
        dx[base + c] = d;
        dsum[base + c] = d + ds_out[base + c];
    }
}

/* RMSNorm backward dweight = sum_r go[r,c] * x[r,c] * rms_inv[r]. Pass 1 writes
 * one partial per (row-block, channel); pass 2 reduces row-block partials. This
 * parallelizes the row dimension for long sequences instead of one channel
 * thread serially scanning every row. */
kernel void gd_rms_norm_wbwd(device const float *x               [[buffer(0)]],
                             device const float *go              [[buffer(1)]],
                             device float *partial               [[buffer(2)]],
                             constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                             uint tgid  [[threadgroup_position_in_grid]],
                             uint tid    [[thread_index_in_threadgroup]],
                             uint tgsz   [[threads_per_threadgroup]])
{
    threadgroup float inv_sh[GD_RMS_TG];
    int row_blocks = (p.rows + (int)tgsz - 1) / (int)tgsz;
    int rb = (int)tgid % row_blocks;
    int cb = (int)tgid / row_blocks;
    int c = cb * (int)tgsz + (int)tid;
    int row0 = rb * (int)tgsz;
    int tile = p.rows - row0;
    float acc = 0.0f;
    if (tile > (int)tgsz) {
        tile = (int)tgsz;
    }

    if ((int)tid < tile) {
        int rbase = (row0 + (int)tid) * p.last;
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
            int rbase = (row0 + i) * p.last;
            acc += go[rbase + c] * x[rbase + c] * inv_sh[i];
        }
        partial[rb * p.last + c] = acc;
    }
}

kernel void gd_rms_norm_wbwd_reduce(device const float *partial      [[buffer(0)]],
                                    device float *dweight            [[buffer(1)]],
                                    constant gd_metal_rmsnorm_params &p [[buffer(2)]],
                                    uint gid                         [[thread_position_in_grid]])
{
    int c = (int)gid;
    if (c >= p.last) {
        return;
    }
    int row_blocks = (p.rows + GD_RMS_TG - 1) / GD_RMS_TG;
    float acc = 0.0f;
    for (int rb = 0; rb < row_blocks; ++rb) {
        acc += partial[rb * p.last + c];
    }
    dweight[c] = acc;
}

/* Mean cross-entropy to a single scalar. One threadgroup: each thread sums the
 * per-position negative log-likelihood for a strided subset of positions, then a
 * threadgroup reduction produces the mean. Targets are int32; class index is
 * assumed valid (matches a validated graph). */
#define GD_CE_TG 256
/* Pass 1: one threadgroup per position, threads cooperate over classes and
 * write an unnormalized per-position loss. Pass 2 reduces losses to the scalar
 * mean (gd_cross_entropy_reduce). */
kernel void gd_cross_entropy(device const float *logits      [[buffer(0)]],
                             device const int *targets       [[buffer(1)]],
                             device float *losses            [[buffer(2)]],
                             constant gd_metal_ce_params &p   [[buffer(3)]],
                             uint gid  [[threadgroup_position_in_grid]],
                             uint tid  [[thread_index_in_threadgroup]],
                             uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float red[GD_CE_TG];
    int total = p.outer * p.inner;
    int pos = (int)gid;
    if (pos >= total) {
        return;
    }
    int o = pos / p.inner;
    int in = pos % p.inner;
    int target = targets[pos];

    float lmax = -INFINITY;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lmax = max(lmax, logits[(o * p.classes + c) * p.inner + in]);
    }
    red[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lsum = 0.0f;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lsum += exp(logits[(o * p.classes + c) * p.inner + in] - maxv);
    }
    red[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float logit_t = logits[(o * p.classes + target) * p.inner + in];
        losses[pos] = -(logit_t - maxv - log(red[0]));
    }
}

kernel void gd_cross_entropy_reduce(device const float *losses           [[buffer(0)]],
                                    device float *out                    [[buffer(1)]],
                                    constant gd_metal_ce_params &p        [[buffer(2)]],
                                    uint tid  [[thread_index_in_threadgroup]],
                                    uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float partial[GD_CE_TG];
    float local = 0.0f;
    for (int pos = (int)tid; pos < p.positions; pos += (int)tgsz) {
        local += losses[pos];
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

/* Fused LM-head CE helper kernels. GEMMs are done by MPS into logits_chunk
 * [rows, chunk_size]. These kernels maintain row-wise softmax stats across
 * chunks and convert chunk logits to dlogits in-place during backward. */
kernel void gd_lmce_fwd_chunk(device const float *logits             [[buffer(0)]],
                              device const int *targets             [[buffer(1)]],
                              device float *m_out                   [[buffer(2)]],
                              device float *l_out                   [[buffer(3)]],
                              device float *target_logit            [[buffer(4)]],
                              constant gd_metal_lmce_params &p       [[buffer(5)]],
                              uint gid                              [[thread_position_in_grid]])
{
    int row = (int)gid;
    if (row >= p.rows) {
        return;
    }
    float old_m = p.first_chunk ? -INFINITY : m_out[row];
    float old_l = p.first_chunk ? 0.0f : l_out[row];
    float cm = -INFINITY;
    int base = row * p.chunk_size;
    for (int c = 0; c < p.chunk_size; ++c) {
        cm = max(cm, logits[base + c]);
    }
    float mnew = max(old_m, cm);
    float lnew = old_l * exp(old_m - mnew);
    for (int c = 0; c < p.chunk_size; ++c) {
        lnew += exp(logits[base + c] - mnew);
    }
    int t = targets[row];
    if (t >= p.chunk_start && t < p.chunk_start + p.chunk_size) {
        target_logit[row] = logits[base + (t - p.chunk_start)];
    }
    m_out[row] = mnew;
    l_out[row] = lnew;
}

kernel void gd_lmce_loss_rows(device const float *m                  [[buffer(0)]],
                              device const float *l                  [[buffer(1)]],
                              device const float *target_logit       [[buffer(2)]],
                              device float *losses                   [[buffer(3)]],
                              constant gd_metal_lmce_params &p       [[buffer(4)]],
                              uint gid                              [[thread_position_in_grid]])
{
    int row = (int)gid;
    if (row >= p.rows) {
        return;
    }
    losses[row] = -(target_logit[row] - m[row] - log(l[row]));
}

kernel void gd_lmce_dlogits_chunk(device float *logits               [[buffer(0)]],
                                  device const int *targets          [[buffer(1)]],
                                  device const float *go_scalar      [[buffer(2)]],
                                  device const float *m              [[buffer(3)]],
                                  device const float *l              [[buffer(4)]],
                                  constant gd_metal_lmce_params &p    [[buffer(5)]],
                                  uint gid                           [[thread_position_in_grid]])
{
    int idx = (int)gid;
    int total = p.rows * p.chunk_size;
    if (idx >= total) {
        return;
    }
    int row = idx / p.chunk_size;
    int c = idx - row * p.chunk_size;
    int cls = p.chunk_start + c;
    float prob = exp(logits[idx] - m[row]) / l[row];
    float onehot = (cls == targets[row]) ? 1.0f : 0.0f;
    logits[idx] = (go_scalar[0] / (float)p.rows) * (prob - onehot);
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

/* dlogits = (go_scalar/positions) * (softmax - onehot).
 * One threadgroup per position; the GD_CE_TG threads cooperate over the class
 * dimension (two threadgroup reductions: row max, then sum-exp), then each
 * thread writes its slice of `dlogits`. Replaces the prior one-thread-per-token
 * scan, which serialized the V-length softmax on a single thread and left the
 * GPU badly under-occupied. Math is identical (fp32). */
kernel void gd_cross_entropy_bwd(device const float *logits     [[buffer(0)]],
                                 device const int *targets      [[buffer(1)]],
                                 device const float *go_scalar  [[buffer(2)]],
                                 device float *dlogits          [[buffer(3)]],
                                 constant gd_metal_ce_params &p  [[buffer(4)]],
                                 uint gid  [[threadgroup_position_in_grid]],
                                 uint tid  [[thread_index_in_threadgroup]],
                                 uint tgsz [[threads_per_threadgroup]])
{
    threadgroup float red[GD_CE_TG];
    int total = p.outer * p.inner;
    int pos = (int)gid;
    if (pos >= total) {
        return;
    }
    int o = pos / p.inner;
    int in = pos % p.inner;
    int target = targets[pos];
    float scale = go_scalar[0] / (float)p.positions;

    float lmax = -INFINITY;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lmax = max(lmax, logits[(o * p.classes + c) * p.inner + in]);
    }
    red[tid] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maxv = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lsum = 0.0f;
    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
        lsum += exp(logits[(o * p.classes + c) * p.inner + in] - maxv);
    }
    red[tid] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgsz / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum = red[0];

    for (int c = (int)tid; c < p.classes; c += (int)tgsz) {
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

    /* Fast path for the dominant GPT case: reduce only extra leading dims,
     * e.g. [B, M, N] -> [M, N]. Both tensors are contiguous, so each output
     * element is a short strided sum over leading slices. Avoids the generic
     * coordinate/divmod path for millions of elements with reduce_count=4/8. */
    if (p.go_ndim == p.target_ndim + 1) {
        bool leading_only = true;
        for (int i = 0; i < p.target_ndim; ++i) {
            if (p.go_sizes[i + 1] != p.target_sizes[i]) {
                leading_only = false;
            }
        }
        if (leading_only) {
            float acc = 0.0f;
            for (int r = 0; r < p.go_sizes[0]; ++r) {
                acc += go[r * p.target_numel + t];
            }
            out[t] = acc;
            return;
        }
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

/* ---- Gradient clipping / optimizer -------------------------------------- */

kernel void gd_clip_norm_partial(device const float *grad                 [[buffer(0)]],
                                 device float *scratch                   [[buffer(1)]],
                                 constant gd_metal_clip_norm_params &p   [[buffer(2)]],
                                 uint gid                                [[thread_position_in_grid]],
                                 uint tid                                [[thread_index_in_threadgroup]],
                                 uint tg                                 [[threadgroup_position_in_grid]])
{
    threadgroup float partial[GD_CLIP_NORM_TG];
    float v = 0.0f;
    if ((int)gid < p.numel) {
        float g = grad[gid];
        v = g * g;
    }
    partial[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GD_CLIP_NORM_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        scratch[p.scratch_offset + (int)tg] = partial[0];
    }
}

kernel void gd_clip_norm_finalize(device float *scratch                  [[buffer(0)]],
                                  device float *norm_out                 [[buffer(1)]],
                                  constant gd_metal_clip_norm_params &p  [[buffer(2)]],
                                  uint gid                               [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float sumsq = 0.0f;
    for (int i = 0; i < p.total_groups; ++i) {
        sumsq += scratch[i];
    }
    float norm = sqrt(sumsq);
    float scale = 1.0f;
    if (norm > p.max_norm) {
        scale = p.max_norm / (norm + p.eps);
    }
    scratch[p.scale_index] = scale;
    norm_out[0] = norm;
}

kernel void gd_clip_norm_scale(device float *grad                       [[buffer(0)]],
                               device const float *scratch              [[buffer(1)]],
                               constant gd_metal_clip_norm_params &p    [[buffer(2)]],
                               uint gid                                 [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    grad[gid] *= scratch[p.scale_index];
}

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
                     device const float *lr_tensor       [[buffer(6)]],
                     uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float lr = (p.use_lr_tensor != 0 ? lr_tensor[0] : p.lr) * p.lr_scale;
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
    pp -= lr * p.weight_decay * pp;            /* decoupled weight decay */
    pp -= lr * mhat / (sqrt(vhat) + p.eps);
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

static inline bool gd_sdpa_allowed(int i, int j, int Tq, int Tk,
                                   int causal, int window, int prefix_len)
{
    int qpos = i + (Tk - Tq);

    if (causal) {
        if (prefix_len > 0) {
            if (qpos < prefix_len) {
                if (j >= prefix_len) {
                    return false;
                }
            } else if (j > qpos) {
                return false;
            }
        } else if (j > qpos) {
            return false;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (qpos >= prefix_len && j >= prefix_len && (qpos - j) >= window) {
                return false;
            }
        } else if ((qpos - j) >= window) {
            return false;
        }
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
static inline int gd_sdpa_kb_end(int q0, int bq, int Tq, int Tk,
                                  int causal, int prefix_len)
{
    if (!causal) {
        return Tk;
    }
    int qmax = q0 + bq - 1;
    if (qmax > Tq - 1) {
        qmax = Tq - 1;
    }
    int qmaxpos = qmax + (Tk - Tq);
    int lim = qmaxpos + 1; /* keys j < lim may be attended */
    if (prefix_len > 0 && qmaxpos < prefix_len) {
        lim = prefix_len;
    }
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
static inline int gd_sdpa_qb_start(int k0, int qblk, int Tk, int Tq,
                                    int causal, int prefix_len)
{
    if (!causal) {
        return 0;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return 0;
    }
    int qmin = k0 - (Tk - Tq);
    if (qmin <= 0) {
        return 0;
    }
    if (qmin > Tq) {
        return Tq;
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
#define GD_SDPA_DKV_KEYS GD_METAL_SDPA_DKV_KEYS
#define GD_SDPA_DKV_LANES (GD_SDPA_BQ / GD_SDPA_DKV_KEYS)
#define GD_SDPA_DKV_CMAX ((GD_SDPA_DHT + GD_SDPA_DKV_LANES - 1) / GD_SDPA_DKV_LANES)
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

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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

    int qb_start = gd_sdpa_qb_start(kblk * GD_SDPA_BQ, GD_SDPA_BK, p.Tk, p.Tq, p.causal, p.prefix_len);
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
                    if (!gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
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

/* ---- Split-K backward SDPA (long context) ------------------------------- *
 * The three backward passes are critical-path bound exactly like the forward
 * (the heaviest query/key block scans the full opposite range). Split-K shortens
 * that scan: stats/dq split the KEY range per query block, dkv splits the QUERY
 * range per key block. stats merges online (m,l,raw); dq and dkv are plain
 * partial sums (the softmax stats are already known), reduced by an add pass.
 * Key/query slice length is derived in-kernel from n_splits so host and kernel
 * always agree. Partial layouts (float offsets within one scratch buffer, bound
 * at byte offsets by the host):
 *   stats : [m, l, raw] per (b,hq,i,s), stride 3
 *   dq    : [Dh] per (b,hq,i,s), stride Dh
 *   dkv   : [dk[Dh], dv[Dh]] per (b,hkv,j,s), stride 2*Dh */
static inline int gd_sdpa_split_len(int T, int n)
{
    int len = (T + n - 1) / n;
    len = ((len + GD_SDPA_BK - 1) / GD_SDPA_BK) * GD_SDPA_BK;
    return (len < GD_SDPA_BK) ? GD_SDPA_BK : len;
}

/* Fused stats+dq split pass. For each query row and key split, stream keys once
 * and accumulate the scalar softmax stats plus two vector partials:
 *   raw = sum e * dp
 *   acc = sum e * dp * k
 *   ksum = sum e * k
 * where e is exp(score - m_split) under the split-local online max and
 * dp = dot(dO_i, V_j). The combine pass merges split-local sums with the same
 * max correction as FlashAttention and writes both final stats (for dkv) and dq:
 *   D  = raw / l
 *   dq = scale * (acc/l - D * ksum/l)
 * This removes the old second full key scan in gd_sdpa_bwd_dq_split. */
kernel void gd_sdpa_bwd_stats_dq_split(device const float *go          [[buffer(0)]],
                                       device const float *q           [[buffer(1)]],
                                       device const float *k           [[buffer(2)]],
                                       device const float *v           [[buffer(3)]],
                                       device const float *bias        [[buffer(4)]],
                                       device float *stats_part        [[buffer(5)]],
                                       device float *dq_part           [[buffer(6)]],
                                       constant gd_metal_sdpa_params &p [[buffer(7)]],
                                       uint tgid [[threadgroup_position_in_grid]],
                                       uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int sp = (int)tgid % p.n_splits;
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
    float goreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    float ksum[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        goreg[c] = active ? go[qbase + c] : 0.0f;
        acc[c] = 0.0f;
        ksum[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
    int slen = gd_sdpa_split_len(p.Tk, p.n_splits);
    int k_lo = sp * slen;
    int k_hi = k_lo + slen;
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        ss += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int c = 0; c < p.Dh; ++c) {
                        float kc = ksh[jj * p.Dh + c];
                        acc[c] = acc[c] * corr + e * dp * kc;
                        ksum[c] = ksum[c] * corr + e * kc;
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int sbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 3;
        int vbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 2 * p.Dh;
        stats_part[sbase + 0] = m;
        stats_part[sbase + 1] = l;
        stats_part[sbase + 2] = raw;
        for (int c = 0; c < p.Dh; ++c) {
            dq_part[vbase + c] = acc[c];
            dq_part[vbase + p.Dh + c] = ksum[c];
        }
    }
}

kernel void gd_sdpa_bwd_stats_dq_combine(device const float *stats_part [[buffer(0)]],
                                         device const float *dq_part    [[buffer(1)]],
                                         device float *stats            [[buffer(2)]],
                                         device float *dq               [[buffer(3)]],
                                         constant gd_metal_sdpa_params &p [[buffer(4)]],
                                         uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int i = row % p.Tq;
    int r = row / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;

    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;
    float acc = 0.0f;
    float ksum = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        int sbase = (row * p.n_splits + sp) * 3;
        float ls = stats_part[sbase + 1];
        if (ls <= 0.0f) {
            continue;
        }
        float ms = stats_part[sbase + 0];
        float mnew = (ms > m) ? ms : m;
        float corr_o = exp(m - mnew);
        float corr_s = exp(ms - mnew);
        int vbase = (row * p.n_splits + sp) * 2 * p.Dh;
        l = l * corr_o + ls * corr_s;
        raw = raw * corr_o + stats_part[sbase + 2] * corr_s;
        acc = acc * corr_o + dq_part[vbase + c] * corr_s;
        ksum = ksum * corr_o + dq_part[vbase + p.Dh + c] * corr_s;
        m = mnew;
    }
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float D = raw * inv_l;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    dq[qbase + c] = p.scale * (acc * inv_l - D * ksum * inv_l);
    if (c == 0) {
        int sb = ((b * p.Hq + hq) * p.Tq + i) * 3;
        stats[sb + 0] = m;
        stats[sb + 1] = l;
        stats[sb + 2] = D;
    }
}

kernel void gd_sdpa_bwd_dkv_split(device const float *go            [[buffer(0)]],
                                  device const float *q             [[buffer(1)]],
                                  device const float *k             [[buffer(2)]],
                                  device const float *v             [[buffer(3)]],
                                  device const float *bias          [[buffer(4)]],
                                  device float *part                [[buffer(5)]],
                                  constant gd_metal_sdpa_params &p   [[buffer(6)]],
                                  device const float *stats         [[buffer(7)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];
    threadgroup float ss_part[GD_SDPA_BK * GD_SDPA_DKV_KEYS * GD_SDPA_DKV_LANES];
    threadgroup float dp_part[GD_SDPA_BK * GD_SDPA_DKV_KEYS * GD_SDPA_DKV_LANES];
    threadgroup float pjsh[GD_SDPA_BK * GD_SDPA_DKV_KEYS];
    threadgroup float dssh[GD_SDPA_BK * GD_SDPA_DKV_KEYS];

    int n_kb = (p.Tk + GD_SDPA_DKV_KEYS - 1) / GD_SDPA_DKV_KEYS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_LANES;
    int j = kblk * GD_SDPA_DKV_KEYS + local_key;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DKV_CMAX];
    float vreg[GD_SDPA_DKV_CMAX];
    float dkacc[GD_SDPA_DKV_CMAX];
    float dvacc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        kreg[nchan] = active ? k[kbase + c] : 0.0f;
        vreg[nchan] = active ? v[kbase + c] : 0.0f;
        dkacc[nchan] = 0.0f;
        dvacc[nchan] = 0.0f;
        nchan++;
    }

    int qb_start = gd_sdpa_qb_start(kblk * GD_SDPA_DKV_KEYS, GD_SDPA_BK,
                                    p.Tk, p.Tq, p.causal, p.prefix_len);
    int slen = gd_sdpa_split_len(p.Tq, p.n_splits);
    int q_lo = s * slen;
    int q_hi = q_lo + slen;
    if (q_lo < qb_start) {
        q_lo = qb_start;
    }
    if (q_hi > p.Tq) {
        q_hi = p.Tq;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = q_lo; qb < q_hi; qb += GD_SDPA_BK) {
            int tile = q_hi - qb;
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

            for (int ii = 0; ii < tile; ++ii) {
                float ss = 0.0f;
                float dp = 0.0f;
                for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                    ss += qsh[ii * p.Dh + c] * kreg[x];
                    dp += gsh[ii * p.Dh + c] * vreg[x];
                }
                int rb = (ii * GD_SDPA_DKV_KEYS + local_key) * GD_SDPA_DKV_LANES + lane;
                ss_part[rb] = ss;
                dp_part[rb] = dp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    int ob = ii * GD_SDPA_DKV_KEYS + local_key;
                    float pj = 0.0f;
                    float ds = 0.0f;
                    if (active && gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                        float l = lsh[ii];
                        if (l > 0.0f) {
                            int rb = ob * GD_SDPA_DKV_LANES;
                            float ss = 0.0f;
                            float dp = 0.0f;
                            for (int ln = 0; ln < GD_SDPA_DKV_LANES; ++ln) {
                                ss += ss_part[rb + ln];
                                dp += dp_part[rb + ln];
                            }
                            ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                            pj = exp(ss - msh[ii]) / l;
                            ds = pj * (dp - dsh[ii]);
                        }
                    }
                    pjsh[ob] = pj;
                    dssh[ob] = ds;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int ob = ii * GD_SDPA_DKV_KEYS + local_key;
                    float pj = pjsh[ob];
                    float ds = dssh[ob];
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        dvacc[x] += pj * gsh[ii * p.Dh + c];
                        dkacc[x] += p.scale * ds * qsh[ii * p.Dh + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int base = (((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits + s) * 2 * p.Dh;
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            part[base + c] = dkacc[x];
            part[base + p.Dh + c] = dvacc[x];
        }
    }
}

/* Sum per-split dk/dv partials. One thread owns one (row, channel), writing both
 * dk and dv for that channel. */
kernel void gd_sdpa_bwd_dkv_reduce(device const float *part         [[buffer(0)]],
                                   device float *dk                 [[buffer(1)]],
                                   device float *dv                 [[buffer(2)]],
                                   constant gd_metal_sdpa_params &p  [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hkv * p.Tk * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int j = row % p.Tk;
    int r = row / p.Tk;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
    int pbase = ((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits * 2 * p.Dh;
    float adk = 0.0f;
    float adv = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        adk += part[pbase + s * 2 * p.Dh + c];
        adv += part[pbase + s * 2 * p.Dh + p.Dh + c];
    }
    dk[kbase + c] = adk;
    dv[kbase + c] = adv;
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

/* Embedding backward: zero the dense table gradient, then scatter-add one
 * gradient element per (token,row-channel) with atomic float adds. This changes
 * the old deterministic gather-by-vocab O(vocab*dim*n_ids) kernel into
 * O(vocab*dim + n_ids*dim). Atomic order can differ for duplicate token ids, but
 * remains within the GPU parity tolerances used for training. */
kernel void gd_embedding_bwd(device float *dtable                 [[buffer(0)]],
                             constant gd_metal_embedding_params &p [[buffer(1)]],
                             uint gid                            [[thread_position_in_grid]])
{
    int idx = (int)gid;
    if (idx < p.vocab * p.dim) {
        dtable[idx] = 0.0f;
    }
}

kernel void gd_embedding_bwd_scatter(device const float *go               [[buffer(0)]],
                                     device const int *ids                [[buffer(1)]],
                                     device atomic_float *dtable          [[buffer(2)]],
                                     constant gd_metal_embedding_params &p [[buffer(3)]],
                                     uint gid                            [[thread_position_in_grid]])
{
    int idx = (int)gid;
    int total = p.n * p.dim;
    if (idx >= total) {
        return;
    }
    int row = idx / p.dim;
    int c = idx % p.dim;
    int v = ids[row];
    if (v >= 0 && v < p.vocab) {
        atomic_fetch_add_explicit(&dtable[v * p.dim + c], go[idx], memory_order_relaxed);
    }
}
