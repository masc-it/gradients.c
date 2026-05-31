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

/* RMSNorm backward dx; one thread per row. */
kernel void gd_rms_norm_bwd(device const float *x               [[buffer(0)]],
                            device const float *weight          [[buffer(1)]],
                            device const float *go              [[buffer(2)]],
                            device float *dx                    [[buffer(3)]],
                            constant gd_metal_rmsnorm_params &p  [[buffer(4)]],
                            uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) {
        return;
    }
    int r = (int)gid;
    int base = r * p.last;
    float sumsq = 0.0f;
    for (int c = 0; c < p.last; ++c) {
        float v = x[base + c];
        sumsq += v * v;
    }
    float inv = 1.0f / sqrt(sumsq / (float)p.last + p.eps);
    float inv3 = inv * inv * inv;
    float A = 0.0f;
    for (int c = 0; c < p.last; ++c) {
        A += go[base + c] * weight[c] * x[base + c];
    }
    for (int c = 0; c < p.last; ++c) {
        dx[base + c] = inv * go[base + c] * weight[c] - x[base + c] * inv3 * A / (float)p.last;
    }
}

/* RMSNorm backward dweight; one thread per channel, reducing over rows. */
kernel void gd_rms_norm_wbwd(device const float *x               [[buffer(0)]],
                             device const float *go              [[buffer(1)]],
                             device float *dweight               [[buffer(2)]],
                             constant gd_metal_rmsnorm_params &p  [[buffer(3)]],
                             uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid >= p.last) {
        return;
    }
    int c = (int)gid;
    float acc = 0.0f;
    for (int r = 0; r < p.rows; ++r) {
        int base = r * p.last;
        float sumsq = 0.0f;
        for (int cc = 0; cc < p.last; ++cc) {
            float v = x[base + cc];
            sumsq += v * v;
        }
        float inv = 1.0f / sqrt(sumsq / (float)p.last + p.eps);
        acc += go[base + c] * x[base + c] * inv;
    }
    dweight[c] = acc;
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

/* Sum `go` down into `out` (target shape) following broadcast rules. Naive
 * single-thread scatter-add (matches the CPU accumulation order). */
kernel void gd_reduce_to(device const float *go                [[buffer(0)]],
                         device float *out                     [[buffer(1)]],
                         constant gd_metal_reduce_to_params &p  [[buffer(2)]],
                         uint gid                              [[thread_position_in_grid]])
{
    if ((int)gid != 0) {
        return;
    }
    for (int i = 0; i < p.target_numel; ++i) {
        out[i] = 0.0f;
    }
    for (int i = 0; i < p.go_numel; ++i) {
        int idx[GD_METAL_MAX_DIMS];
        int lin = i;
        for (int k = p.go_ndim - 1; k >= 0; --k) {
            idx[k] = lin % p.go_sizes[k];
            lin /= p.go_sizes[k];
        }
        int off = gd_broadcast_offset(idx, p.go_ndim, p.target_sizes, p.target_ndim);
        out[off] += go[i];
    }
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

/* Scatter-add backward; naive single-thread (zero then accumulate by id). */
kernel void gd_embedding_bwd(device const float *go               [[buffer(0)]],
                             device const int *ids                [[buffer(1)]],
                             device float *dtable                 [[buffer(2)]],
                             constant gd_metal_embedding_params &p [[buffer(3)]],
                             uint gid                            [[thread_position_in_grid]])
{
    if ((int)gid != 0) {
        return;
    }
    for (int i = 0; i < p.vocab * p.dim; ++i) {
        dtable[i] = 0.0f;
    }
    for (int row = 0; row < p.n; ++row) {
        int id = ids[row];
        for (int c = 0; c < p.dim; ++c) {
            dtable[id * p.dim + c] += go[row * p.dim + c];
        }
    }
}
