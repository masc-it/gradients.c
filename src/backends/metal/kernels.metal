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

/* RMSNorm over the last dim; one thread per row. */
kernel void gd_rms_norm(device const float *x               [[buffer(0)]],
                        device const float *weight          [[buffer(1)]],
                        device float *out                   [[buffer(2)]],
                        constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                        uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) {
        return;
    }
    int r = (int)gid;
    float sumsq = 0.0f;
    for (int c = 0; c < p.last; ++c) {
        float v = x[r * p.last + c];
        sumsq += v * v;
    }
    float inv = 1.0f / sqrt(sumsq / (float)p.last + p.eps);
    for (int c = 0; c < p.last; ++c) {
        out[r * p.last + c] = x[r * p.last + c] * inv * weight[c];
    }
}

/* Mean cross-entropy to a single scalar; naive single-thread reduction (v1).
 * Targets are int32; class index assumed valid (matches a validated graph). */
kernel void gd_cross_entropy(device const float *logits      [[buffer(0)]],
                             device const int *targets       [[buffer(1)]],
                             device float *out               [[buffer(2)]],
                             constant gd_metal_ce_params &p   [[buffer(3)]],
                             uint gid                        [[thread_position_in_grid]])
{
    if ((int)gid != 0) {
        return;
    }
    float loss = 0.0f;
    for (int o = 0; o < p.outer; ++o) {
        for (int in = 0; in < p.inner; ++in) {
            int pos = o * p.inner + in;
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
            loss += -(logit_t - maxv - log(sum));
        }
    }
    out[0] = loss / (float)p.positions;
}
