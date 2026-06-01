#include "cpu_backend.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "../../core/internal.h"

static int64_t desc_numel(const gd_tensor_desc *desc)
{
    int64_t numel = 1;
    int i = 0;

    for (i = 0; i < desc->ndim; ++i) {
        numel *= desc->sizes[i];
    }
    return numel;
}

static void unravel(int64_t linear, const gd_tensor_desc *desc, int64_t *index)
{
    int i = 0;

    for (i = desc->ndim - 1; i >= 0; --i) {
        index[i] = linear % desc->sizes[i];
        linear /= desc->sizes[i];
    }
}

/* Linear offset into a contiguous input, broadcasting against an output index. */
static int64_t broadcast_offset(const int64_t *out_index,
                                int out_ndim,
                                const gd_tensor_desc *in_desc)
{
    int64_t stride = 1;
    int64_t offset = 0;
    int i = 0;

    for (i = in_desc->ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_desc->ndim - i);
        int64_t coord = in_desc->sizes[i] == 1 ? 0 : out_index[out_pos];

        offset += coord * stride;
        stride *= in_desc->sizes[i];
    }
    return offset;
}

gd_status _gd_cpu_k_elementwise(_gd_op_kind op,
                                const gd_tensor_desc *out_desc,
                                float *out,
                                const gd_tensor_desc *a_desc,
                                const float *a,
                                const gd_tensor_desc *b_desc,
                                const float *b)
{
    int64_t total = desc_numel(out_desc);
    int64_t lin = 0;
    int64_t index[GD_MAX_DIMS];

    for (lin = 0; lin < total; ++lin) {
        int64_t ao = 0;
        int64_t bo = 0;

        unravel(lin, out_desc, index);
        ao = broadcast_offset(index, out_desc->ndim, a_desc);
        bo = broadcast_offset(index, out_desc->ndim, b_desc);
        if (op == _GD_OP_ADD) {
            out[lin] = a[ao] + b[bo];
        } else {
            out[lin] = a[ao] * b[bo];
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_scale(const gd_tensor_desc *desc, float *out, const float *x, float scale)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        out[i] = x[i] * scale;
    }
    return GD_OK;
}

gd_status _gd_cpu_k_relu(const gd_tensor_desc *desc, float *out, const float *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        out[i] = x[i] > 0.0F ? x[i] : 0.0F;
    }
    return GD_OK;
}

gd_status _gd_cpu_k_silu(const gd_tensor_desc *desc, float *out, const float *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        out[i] = x[i] / (1.0F + expf(-x[i]));
    }
    return GD_OK;
}

static float sigmoidf_stable(float x)
{
    if (x >= 0.0F) {
        float e = expf(-x);
        return 1.0F / (1.0F + e);
    }
    {
        float e = expf(x);
        return e / (1.0F + e);
    }
}

static float powlu_gate(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0F) {
        return z * s;
    }
    {
        float r = sqrtf(z > 0.0F ? z : 0.0F);
        float a = m / (r + 1.0F);
        return powf(z, a) * s;
    }
}

static float powlu_gate_grad(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0F) {
        return s * (1.0F + z * (1.0F - s));
    }
    {
        float r = sqrtf(z > 0.0F ? z : 0.0F);
        float rp1 = r + 1.0F;
        float a = m / rp1;
        float g = powf(z, a);
        float da = -m / (2.0F * r * rp1 * rp1);
        float lz = logf(z > FLT_MIN ? z : FLT_MIN);
        return g * s * (a / z + da * lz + (1.0F - s));
    }
}

gd_status _gd_cpu_k_powlu(const gd_tensor_desc *desc, float *out,
                          const float *x1, const float *x2, float m)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        out[i] = x1[i] * powlu_gate(x2[i], m);
    }
    return GD_OK;
}

gd_status _gd_cpu_k_powlu_bwd(const gd_tensor_desc *desc, float *dx1, float *dx2,
                              const float *x1, const float *x2, const float *go,
                              float m)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float gate = powlu_gate(x2[i], m);
        float grad = powlu_gate_grad(x2[i], m);
        dx1[i] = go[i] * gate;
        dx2[i] = go[i] * x1[i] * grad;
    }
    return GD_OK;
}

gd_status _gd_cpu_k_matmul(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *a_desc,
                           const float *a,
                           bool trans_a,
                           const gd_tensor_desc *b_desc,
                           const float *b,
                           bool trans_b)
{
    int batch_ndim = out_desc->ndim - 2;
    int64_t m = out_desc->sizes[out_desc->ndim - 2];
    int64_t n = out_desc->sizes[out_desc->ndim - 1];
    int64_t k = trans_a ? a_desc->sizes[a_desc->ndim - 2] : a_desc->sizes[a_desc->ndim - 1];
    int64_t a_rows = a_desc->sizes[a_desc->ndim - 2];
    int64_t a_cols = a_desc->sizes[a_desc->ndim - 1];
    int64_t b_rows = b_desc->sizes[b_desc->ndim - 2];
    int64_t b_cols = b_desc->sizes[b_desc->ndim - 1];
    int64_t a_mat = a_rows * a_cols;
    int64_t b_mat = b_rows * b_cols;
    int64_t out_mat = m * n;
    int64_t batch_total = 1;
    int64_t batch_lin = 0;
    int64_t bidx[GD_MAX_DIMS];
    int i = 0;

    for (i = 0; i < batch_ndim; ++i) {
        batch_total *= out_desc->sizes[i];
    }

    for (batch_lin = 0; batch_lin < batch_total; ++batch_lin) {
        int64_t a_base = 0;
        int64_t b_base = 0;
        int64_t a_bstride = 1;
        int64_t b_bstride = 1;
        int64_t row = 0;

        if (batch_ndim > 0) {
            int64_t tmp = batch_lin;
            for (i = batch_ndim - 1; i >= 0; --i) {
                bidx[i] = tmp % out_desc->sizes[i];
                tmp /= out_desc->sizes[i];
            }
            for (i = a_desc->ndim - 3; i >= 0; --i) {
                int out_pos = batch_ndim - (a_desc->ndim - 2 - i);
                int64_t coord = a_desc->sizes[i] == 1 ? 0 : bidx[out_pos];
                a_base += coord * a_bstride * a_mat;
                a_bstride *= a_desc->sizes[i];
            }
            for (i = b_desc->ndim - 3; i >= 0; --i) {
                int out_pos = batch_ndim - (b_desc->ndim - 2 - i);
                int64_t coord = b_desc->sizes[i] == 1 ? 0 : bidx[out_pos];
                b_base += coord * b_bstride * b_mat;
                b_bstride *= b_desc->sizes[i];
            }
        }

        for (row = 0; row < m; ++row) {
            int64_t col = 0;
            for (col = 0; col < n; ++col) {
                double acc = 0.0;
                int64_t kk = 0;
                for (kk = 0; kk < k; ++kk) {
                    int64_t a_off = trans_a ? (kk * a_cols + row) : (row * a_cols + kk);
                    int64_t b_off = trans_b ? (col * b_cols + kk) : (kk * b_cols + col);
                    acc += (double)a[a_base + a_off] * (double)b[b_base + b_off];
                }
                out[batch_lin * out_mat + row * n + col] = (float)acc;
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_linear(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           const gd_tensor_desc *w_desc,
                           const float *w,
                           bool trans_w,
                           const float *bias)
{
    int64_t in_features = x_desc->sizes[x_desc->ndim - 1];
    int64_t out_features = out_desc->sizes[out_desc->ndim - 1];
    int64_t rows = desc_numel(x_desc) / in_features;
    int64_t r = 0;

    (void)w_desc;
    for (r = 0; r < rows; ++r) {
        int64_t o = 0;
        for (o = 0; o < out_features; ++o) {
            double acc = bias != NULL ? (double)bias[o] : 0.0;
            int64_t kk = 0;
            for (kk = 0; kk < in_features; ++kk) {
                int64_t w_off = trans_w ? (o * in_features + kk) : (kk * out_features + o);
                acc += (double)x[r * in_features + kk] * (double)w[w_off];
            }
            out[r * out_features + o] = (float)acc;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_reduce(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           int dim,
                           bool mean)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = x_desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    (void)out_desc;
    for (i = 0; i < dim; ++i) {
        outer *= x_desc->sizes[i];
    }
    for (i = dim + 1; i < x_desc->ndim; ++i) {
        inner *= x_desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double acc = 0.0;
            int64_t c = 0;
            for (c = 0; c < d; ++c) {
                acc += (double)x[(o * d + c) * inner + in];
            }
            if (mean) {
                acc /= (double)d;
            }
            out[o * inner + in] = (float)acc;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_softmax(const gd_tensor_desc *desc, float *out, const float *x, int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        inner *= desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;

            for (c = 0; c < d; ++c) {
                double v = (double)x[(o * d + c) * inner + in];
                if (v > max_val) {
                    max_val = v;
                }
            }
            for (c = 0; c < d; ++c) {
                double e = exp((double)x[(o * d + c) * inner + in] - max_val);
                out[(o * d + c) * inner + in] = (float)e;
                sum += e;
            }
            for (c = 0; c < d; ++c) {
                out[(o * d + c) * inner + in] = (float)((double)out[(o * d + c) * inner + in] / sum);
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rms_norm(const gd_tensor_desc *desc,
                             float *out,
                             const float *x,
                             const float *weight,
                             float eps)
{
    int64_t last = desc->sizes[desc->ndim - 1];
    int64_t rows = desc_numel(desc) / last;
    int64_t r = 0;

    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;
        int64_t c = 0;

        for (c = 0; c < last; ++c) {
            double v = (double)x[r * last + c];
            sumsq += v * v;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        for (c = 0; c < last; ++c) {
            out[r * last + c] = (float)((double)x[r * last + c] * inv * (double)weight[c]);
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rms_norm_bwd(const gd_tensor_desc *desc, float *dx,
                                 const float *x, const float *weight,
                                 const float *go, float eps)
{
    int64_t last = desc->sizes[desc->ndim - 1];
    int64_t rows = desc_numel(desc) / last;
    int64_t r = 0;

    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;
        double inv3 = 0.0;
        double A = 0.0; /* sum_c go_c * w_c * x_c */
        int64_t c = 0;

        for (c = 0; c < last; ++c) {
            double v = (double)x[r * last + c];
            sumsq += v * v;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        inv3 = inv * inv * inv;
        for (c = 0; c < last; ++c) {
            A += (double)go[r * last + c] * (double)weight[c] * (double)x[r * last + c];
        }
        for (c = 0; c < last; ++c) {
            double gi = (double)go[r * last + c];
            double xi = (double)x[r * last + c];
            double wi = (double)weight[c];
            dx[r * last + c] = (float)(inv * gi * wi - xi * inv3 * A / (double)last);
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rms_norm_wbwd(const gd_tensor_desc *x_desc, float *dweight,
                                  const float *x, const float *go, float eps)
{
    int64_t last = x_desc->sizes[x_desc->ndim - 1];
    int64_t rows = desc_numel(x_desc) / last;
    int64_t r = 0;
    int64_t c = 0;

    for (c = 0; c < last; ++c) {
        dweight[c] = 0.0F;
    }
    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;

        for (c = 0; c < last; ++c) {
            double v = (double)x[r * last + c];
            sumsq += v * v;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        for (c = 0; c < last; ++c) {
            dweight[c] += (float)((double)go[r * last + c] * (double)x[r * last + c] * inv);
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_cross_entropy(float *out,
                                  const gd_tensor_desc *logits_desc,
                                  const float *logits,
                                  const gd_tensor_desc *targets_desc,
                                  const void *targets,
                                  int class_dim)
{
    int64_t classes = logits_desc->sizes[class_dim];
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t positions = 0;
    int64_t o = 0;
    double loss = 0.0;
    int is_i64 = targets_desc->dtype == GD_DTYPE_I64;
    int i = 0;

    for (i = 0; i < class_dim; ++i) {
        outer *= logits_desc->sizes[i];
    }
    for (i = class_dim + 1; i < logits_desc->ndim; ++i) {
        inner *= logits_desc->sizes[i];
    }
    positions = outer * inner;

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;
            int64_t pos = o * inner + in;
            int64_t target = is_i64 ? (int64_t)((const int64_t *)targets)[pos]
                                    : (int64_t)((const int32_t *)targets)[pos];

            if (target < 0 || target >= classes) {
                return _gd_error(GD_ERR_SHAPE, "cross_entropy target out of range");
            }
            for (c = 0; c < classes; ++c) {
                double v = (double)logits[(o * classes + c) * inner + in];
                if (v > max_val) {
                    max_val = v;
                }
            }
            for (c = 0; c < classes; ++c) {
                sum += exp((double)logits[(o * classes + c) * inner + in] - max_val);
            }
            {
                double logit_t = (double)logits[(o * classes + target) * inner + in];
                loss += -(logit_t - max_val - log(sum));
            }
        }
    }

    out[0] = (float)(loss / (double)positions);
    return GD_OK;
}

gd_status _gd_cpu_k_lm_cross_entropy(float *out,
                                     float *row_max,
                                     float *row_sum,
                                     const gd_tensor_desc *hidden_desc,
                                     const float *hidden,
                                     const gd_tensor_desc *weight_desc,
                                     const float *weight,
                                     const gd_tensor_desc *targets_desc,
                                     const void *targets)
{
    int64_t D = hidden_desc->sizes[hidden_desc->ndim - 1];
    int64_t V = weight_desc->sizes[0];
    int64_t N = desc_numel(hidden_desc) / D;
    int is_i64 = targets_desc->dtype == GD_DTYPE_I64;
    double loss = 0.0;
    int64_t n = 0;

    for (n = 0; n < N; ++n) {
        int64_t target = is_i64 ? ((const int64_t *)targets)[n] : (int64_t)((const int32_t *)targets)[n];
        double max_val = -HUGE_VAL;
        double sum = 0.0;
        double target_logit = 0.0;
        int64_t v = 0;
        if (target < 0 || target >= V) {
            return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy target out of range");
        }
        for (v = 0; v < V; ++v) {
            double s = 0.0;
            int64_t d = 0;
            for (d = 0; d < D; ++d) {
                s += (double)hidden[n * D + d] * (double)weight[v * D + d];
            }
            if (v == target) {
                target_logit = s;
            }
            if (s > max_val) {
                max_val = s;
            }
        }
        for (v = 0; v < V; ++v) {
            double s = 0.0;
            int64_t d = 0;
            for (d = 0; d < D; ++d) {
                s += (double)hidden[n * D + d] * (double)weight[v * D + d];
            }
            sum += exp(s - max_val);
        }
        row_max[n] = (float)max_val;
        row_sum[n] = (float)sum;
        loss += -(target_logit - max_val - log(sum));
    }
    out[0] = (float)(loss / (double)N);
    return GD_OK;
}

gd_status _gd_cpu_k_cast(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *x_desc,
                         const void *x)
{
    int64_t total = desc_numel(out_desc);
    int64_t i = 0;
    gd_dtype src = x_desc->dtype;
    gd_dtype dst = out_desc->dtype;

    if (src == GD_DTYPE_F32 && dst == GD_DTYPE_F32) {
        for (i = 0; i < total; ++i) {
            ((float *)out)[i] = ((const float *)x)[i];
        }
    } else if (src == GD_DTYPE_F32 && dst == GD_DTYPE_I32) {
        for (i = 0; i < total; ++i) {
            ((int32_t *)out)[i] = (int32_t)((const float *)x)[i];
        }
    } else if (src == GD_DTYPE_I32 && dst == GD_DTYPE_F32) {
        for (i = 0; i < total; ++i) {
            ((float *)out)[i] = (float)((const int32_t *)x)[i];
        }
    } else if (src == GD_DTYPE_I32 && dst == GD_DTYPE_I32) {
        for (i = 0; i < total; ++i) {
            ((int32_t *)out)[i] = ((const int32_t *)x)[i];
        }
    } else {
        return _gd_error(GD_ERR_UNSUPPORTED, "cast dtype pair is not implemented in CPU_REF");
    }
    return GD_OK;
}

gd_status _gd_cpu_k_copy(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *in_desc,
                         const void *in)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = desc_numel(out_desc);

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "copy requires a fixed-size dtype");
    }
    if (desc_numel(in_desc) != total) {
        return _gd_error(GD_ERR_SHAPE, "copy requires equal element counts");
    }
    memcpy(out, in, (size_t)total * elem);
    return GD_OK;
}

gd_status _gd_cpu_k_gelu(const gd_tensor_desc *desc, float *out, const float *x, int tanh_approx)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    if (tanh_approx) {
        const double c = 0.7978845608028654; /* sqrt(2/pi) */
        for (i = 0; i < total; ++i) {
            double xv = (double)x[i];
            double inner = c * (xv + 0.044715 * xv * xv * xv);
            out[i] = (float)(0.5 * xv * (1.0 + tanh(inner)));
        }
    } else {
        const double inv_sqrt2 = 0.7071067811865476;
        for (i = 0; i < total; ++i) {
            double xv = (double)x[i];
            out[i] = (float)(0.5 * xv * (1.0 + erf(xv * inv_sqrt2)));
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_gelu_bwd(const gd_tensor_desc *desc, float *dx, const float *x,
                             const float *go, int tanh_approx)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    if (tanh_approx) {
        const double c = 0.7978845608028654; /* sqrt(2/pi) */
        for (i = 0; i < total; ++i) {
            double xv = (double)x[i];
            double u = c * (xv + 0.044715 * xv * xv * xv);
            double t = tanh(u);
            double du = c * (1.0 + 3.0 * 0.044715 * xv * xv);
            double g = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * du;
            dx[i] = (float)((double)go[i] * g);
        }
    } else {
        const double inv_sqrt2 = 0.7071067811865476;
        const double inv_sqrt2pi = 0.3989422804014327;
        for (i = 0; i < total; ++i) {
            double xv = (double)x[i];
            double cdf = 0.5 * (1.0 + erf(xv * inv_sqrt2));
            double pdf = inv_sqrt2pi * exp(-0.5 * xv * xv);
            dx[i] = (float)((double)go[i] * (cdf + xv * pdf));
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_transpose(const gd_tensor_desc *out_desc, void *out,
                              const gd_tensor_desc *in_desc, const void *in,
                              const int *perm)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = desc_numel(out_desc);
    int64_t in_strides[GD_MAX_DIMS];
    int64_t out_index[GD_MAX_DIMS];
    int ndim = out_desc->ndim;
    int64_t stride = 1;
    int64_t lin = 0;
    int k = 0;

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "transpose requires a fixed-size dtype");
    }
    if (in_desc->ndim != ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose rank mismatch");
    }
    for (k = ndim - 1; k >= 0; --k) {
        in_strides[k] = stride;
        stride *= in_desc->sizes[k];
    }
    for (lin = 0; lin < total; ++lin) {
        int64_t in_off = 0;
        unravel(lin, out_desc, out_index);
        for (k = 0; k < ndim; ++k) {
            in_off += out_index[k] * in_strides[perm[k]];
        }
        memcpy((unsigned char *)out + (size_t)lin * elem,
               (const unsigned char *)in + (size_t)in_off * elem, elem);
    }
    return GD_OK;
}

static int64_t embedding_id(const gd_tensor_desc *ids_desc, const void *ids, int64_t p)
{
    if (ids_desc->dtype == GD_DTYPE_I64) {
        return ((const int64_t *)ids)[p];
    }
    return (int64_t)((const int32_t *)ids)[p];
}

gd_status _gd_cpu_k_embedding(const gd_tensor_desc *out_desc, float *out,
                              const gd_tensor_desc *table_desc, const float *table,
                              const gd_tensor_desc *ids_desc, const void *ids)
{
    int64_t vocab = table_desc->sizes[0];
    int64_t dim = table_desc->sizes[1];
    int64_t n = desc_numel(ids_desc);
    int64_t p = 0;

    (void)out_desc;
    for (p = 0; p < n; ++p) {
        int64_t id = embedding_id(ids_desc, ids, p);
        if (id < 0 || id >= vocab) {
            return _gd_error(GD_ERR_SHAPE, "embedding id out of range");
        }
        memcpy(out + p * dim, table + id * dim, (size_t)dim * sizeof(float));
    }
    return GD_OK;
}

gd_status _gd_cpu_k_embedding_bwd(const gd_tensor_desc *table_desc, float *dtable,
                                  const gd_tensor_desc *go_desc, const float *go,
                                  const gd_tensor_desc *ids_desc, const void *ids)
{
    int64_t vocab = table_desc->sizes[0];
    int64_t dim = table_desc->sizes[1];
    int64_t n = desc_numel(ids_desc);
    int64_t p = 0;
    int64_t c = 0;

    (void)go_desc;
    for (p = 0; p < vocab * dim; ++p) {
        dtable[p] = 0.0F;
    }
    for (p = 0; p < n; ++p) {
        int64_t id = embedding_id(ids_desc, ids, p);
        if (id < 0 || id >= vocab) {
            return _gd_error(GD_ERR_SHAPE, "embedding id out of range");
        }
        for (c = 0; c < dim; ++c) {
            dtable[id * dim + c] += go[p * dim + c];
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rope(const gd_tensor_desc *desc, float *out, const float *x,
                         const gd_tensor_desc *pos_desc, const void *pos,
                         float theta, int n_dims, int interleaved, float sin_sign)
{
    int ndim = desc->ndim;
    int64_t head_dim = desc->sizes[ndim - 1];
    int64_t heads = desc->sizes[ndim - 2];
    int64_t rows = desc_numel(desc) / head_dim;
    int half = n_dims / 2;
    int64_t r = 0;

    for (r = 0; r < rows; ++r) {
        int64_t pos_idx = r / heads;
        int64_t p = embedding_id(pos_desc, pos, pos_idx);
        int64_t base = r * head_dim;
        int i = 0;
        int64_t d = 0;

        for (d = n_dims; d < head_dim; ++d) {
            out[base + d] = x[base + d];
        }
        for (i = 0; i < half; ++i) {
            double inv = pow((double)theta, -(2.0 * (double)i) / (double)n_dims);
            double angle = (double)p * inv;
            double c = cos(angle);
            double s = sin(angle) * (double)sin_sign;
            int a = interleaved ? (2 * i) : i;
            int bb = interleaved ? (2 * i + 1) : (i + half);
            double x1 = (double)x[base + a];
            double x2 = (double)x[base + bb];
            out[base + a] = (float)(x1 * c - x2 * s);
            out[base + bb] = (float)(x1 * s + x2 * c);
        }
    }
    return GD_OK;
}

#define GD_SDPA_MAX_HEAD_DIM 256

static int sdpa_allowed(int64_t i, int64_t j, int64_t Tq, int64_t Tk,
                        int causal, int window, int prefix_len)
{
    int64_t qpos = i + (Tk - Tq);

    if (causal) {
        if (prefix_len > 0) {
            if (qpos < prefix_len) {
                if (j >= prefix_len) {
                    return 0;
                }
            } else if (j > qpos) {
                return 0;
            }
        } else if (j > qpos) {
            return 0;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (qpos >= prefix_len && j >= prefix_len && (qpos - j) >= window) {
                return 0;
            }
        } else if ((qpos - j) >= window) {
            return 0;
        }
    }
    return 1;
}

static double sdpa_dot(const float *a, const float *b, int64_t n)
{
    double acc = 0.0;
    int64_t c = 0;
    for (c = 0; c < n; ++c) {
        acc += (double)a[c] * (double)b[c];
    }
    return acc;
}

/* Additive attention bias, broadcast over [B, Hq, Tq, Tk]. */
static double sdpa_bias_at(const gd_tensor_desc *bd, const float *bias,
                           int64_t b, int64_t hq, int64_t i, int64_t j)
{
    int64_t Bb = 0, Hb = 0, Tqb = 0, Tkb = 0, bb = 0, hb = 0, ib = 0, jb = 0;
    if (bias == NULL) {
        return 0.0;
    }
    Bb = bd->sizes[0];
    Hb = bd->sizes[1];
    Tqb = bd->sizes[2];
    Tkb = bd->sizes[3];
    bb = (Bb == 1) ? 0 : b;
    hb = (Hb == 1) ? 0 : hq;
    ib = (Tqb == 1) ? 0 : i;
    jb = (Tkb == 1) ? 0 : j;
    return (double)bias[((bb * Hb + hb) * Tqb + ib) * Tkb + jb];
}

gd_status _gd_cpu_k_sdpa(const gd_tensor_desc *o_desc, float *o,
                         const gd_tensor_desc *q_desc, const float *q,
                         const gd_tensor_desc *k_desc, const float *k,
                         const gd_tensor_desc *v_desc, const float *v,
                         const gd_tensor_desc *bias_desc, const float *bias,
                         float scale, int causal, int window, int prefix_len)
{
    int64_t B = q_desc->sizes[0];
    int64_t Tq = q_desc->sizes[1];
    int64_t Hq = q_desc->sizes[2];
    int64_t Dh = q_desc->sizes[3];
    int64_t Tk = k_desc->sizes[1];
    int64_t Hkv = k_desc->sizes[2];
    int64_t group = Hkv > 0 ? Hq / Hkv : 0;
    int64_t b = 0, hq = 0, i = 0, j = 0, c = 0;

    (void)o_desc;
    (void)v_desc;
    if (Dh > GD_SDPA_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa head_dim exceeds reference limit");
    }
    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                const float *qr = q + (((b * Tq + i) * Hq + hq) * Dh);
                float *orow = o + (((b * Tq + i) * Hq + hq) * Dh);
                double acc[GD_SDPA_MAX_HEAD_DIM];
                double m = -HUGE_VAL;
                double sum = 0.0;

                for (c = 0; c < Dh; ++c) {
                    acc[c] = 0.0;
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double e = exp(s - m);
                        sum += e;
                        for (c = 0; c < Dh; ++c) {
                            acc[c] += e * (double)vr[c];
                        }
                    }
                }
                for (c = 0; c < Dh; ++c) {
                    orow[c] = sum > 0.0 ? (float)(acc[c] / sum) : 0.0F;
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_sdpa_bwd(const gd_tensor_desc *q_desc, const float *q,
                             const gd_tensor_desc *k_desc, const float *k,
                             const gd_tensor_desc *v_desc, const float *v,
                             const gd_tensor_desc *bias_desc, const float *bias,
                             const float *go,
                             float *dq, float *dk, float *dv,
                             float scale, int causal, int window, int prefix_len)
{
    int64_t B = q_desc->sizes[0];
    int64_t Tq = q_desc->sizes[1];
    int64_t Hq = q_desc->sizes[2];
    int64_t Dh = q_desc->sizes[3];
    int64_t Tk = k_desc->sizes[1];
    int64_t Hkv = k_desc->sizes[2];
    int64_t group = Hkv > 0 ? Hq / Hkv : 0;
    int64_t b = 0, hq = 0, i = 0, j = 0, c = 0;

    (void)v_desc;
    if (Dh > GD_SDPA_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa head_dim exceeds reference limit");
    }
    for (j = 0; j < B * Tk * Hkv * Dh; ++j) {
        dk[j] = 0.0F;
        dv[j] = 0.0F;
    }
    for (j = 0; j < B * Tq * Hq * Dh; ++j) {
        dq[j] = 0.0F;
    }

    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                const float *qr = q + (((b * Tq + i) * Hq + hq) * Dh);
                const float *gor = go + (((b * Tq + i) * Hq + hq) * Dh);
                float *dqr = dq + (((b * Tq + i) * Hq + hq) * Dh);
                double m = -HUGE_VAL;
                double sum = 0.0;
                double dsum = 0.0; /* D = sum_j p_j * dp_j */

                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        sum += exp(s - m);
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double p = exp(s - m) / sum;
                        dsum += p * sdpa_dot(gor, vr, Dh);
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        float *dkr = dk + (((b * Tk + j) * Hkv + hkv) * Dh);
                        float *dvr = dv + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double p = exp(s - m) / sum;
                        double dp = sdpa_dot(gor, vr, Dh);
                        double ds = p * (dp - dsum);
                        for (c = 0; c < Dh; ++c) {
                            dvr[c] += (float)(p * (double)gor[c]);
                            dqr[c] += (float)((double)scale * ds * (double)kr[c]);
                            dkr[c] += (float)((double)scale * ds * (double)qr[c]);
                        }
                    }
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_relu_bwd(const gd_tensor_desc *desc,
                             float *dx,
                             const float *x,
                             const float *go)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        dx[i] = x[i] > 0.0F ? go[i] : 0.0F;
    }
    return GD_OK;
}

gd_status _gd_cpu_k_silu_bwd(const gd_tensor_desc *desc,
                             float *dx,
                             const float *x,
                             const float *go)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        double s = 1.0 / (1.0 + exp(-(double)x[i]));
        double grad = s * (1.0 + (double)x[i] * (1.0 - s));
        dx[i] = (float)((double)go[i] * grad);
    }
    return GD_OK;
}

gd_status _gd_cpu_k_softmax_bwd(const gd_tensor_desc *desc,
                                float *dx,
                                const float *y,
                                const float *go,
                                int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        inner *= desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double dot = 0.0;
            int64_t c = 0;

            for (c = 0; c < d; ++c) {
                int64_t idx = (o * d + c) * inner + in;
                dot += (double)go[idx] * (double)y[idx];
            }
            for (c = 0; c < d; ++c) {
                int64_t idx = (o * d + c) * inner + in;
                dx[idx] = (float)((double)y[idx] * ((double)go[idx] - dot));
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_sum_bwd(const gd_tensor_desc *x_desc,
                            float *dx,
                            const float *go,
                            int dim,
                            bool mean)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = x_desc->sizes[dim];
    int64_t o = 0;
    double scale = 1.0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= x_desc->sizes[i];
    }
    for (i = dim + 1; i < x_desc->ndim; ++i) {
        inner *= x_desc->sizes[i];
    }
    if (mean) {
        scale = 1.0 / (double)d;
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double g = (double)go[o * inner + in] * scale;
            int64_t c = 0;
            for (c = 0; c < d; ++c) {
                dx[(o * d + c) * inner + in] = (float)g;
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_cross_entropy_bwd(const gd_tensor_desc *logits_desc,
                                      float *dlogits,
                                      const float *logits,
                                      const gd_tensor_desc *targets_desc,
                                      const void *targets,
                                      const float *go_scalar,
                                      int class_dim)
{
    int64_t classes = logits_desc->sizes[class_dim];
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t positions = 0;
    int64_t o = 0;
    int is_i64 = targets_desc->dtype == GD_DTYPE_I64;
    double scale = 0.0;
    int i = 0;

    for (i = 0; i < class_dim; ++i) {
        outer *= logits_desc->sizes[i];
    }
    for (i = class_dim + 1; i < logits_desc->ndim; ++i) {
        inner *= logits_desc->sizes[i];
    }
    positions = outer * inner;
    scale = (double)go_scalar[0] / (double)positions;

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;
            int64_t pos = o * inner + in;
            int64_t target = is_i64 ? (int64_t)((const int64_t *)targets)[pos]
                                    : (int64_t)((const int32_t *)targets)[pos];

            if (target < 0 || target >= classes) {
                return _gd_error(GD_ERR_SHAPE, "cross_entropy target out of range");
            }
            for (c = 0; c < classes; ++c) {
                double v = (double)logits[(o * classes + c) * inner + in];
                if (v > max_val) {
                    max_val = v;
                }
            }
            for (c = 0; c < classes; ++c) {
                sum += exp((double)logits[(o * classes + c) * inner + in] - max_val);
            }
            for (c = 0; c < classes; ++c) {
                double p = exp((double)logits[(o * classes + c) * inner + in] - max_val) / sum;
                double onehot = (c == target) ? 1.0 : 0.0;
                dlogits[(o * classes + c) * inner + in] = (float)(scale * (p - onehot));
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_lm_cross_entropy_bwd(const gd_tensor_desc *hidden_desc,
                                         float *dhidden,
                                         const float *hidden,
                                         const gd_tensor_desc *weight_desc,
                                         float *dweight,
                                         const float *weight,
                                         const gd_tensor_desc *targets_desc,
                                         const void *targets,
                                         const float *go_scalar,
                                         const float *row_max,
                                         const float *row_sum)
{
    int64_t D = hidden_desc->sizes[hidden_desc->ndim - 1];
    int64_t V = weight_desc->sizes[0];
    int64_t N = desc_numel(hidden_desc) / D;
    int is_i64 = targets_desc->dtype == GD_DTYPE_I64;
    double scale = (double)go_scalar[0] / (double)N;
    int64_t i = 0;

    for (i = 0; i < N * D; ++i) {
        dhidden[i] = 0.0F;
    }
    for (i = 0; i < V * D; ++i) {
        dweight[i] = 0.0F;
    }

    for (int64_t n = 0; n < N; ++n) {
        int64_t target = is_i64 ? ((const int64_t *)targets)[n] : (int64_t)((const int32_t *)targets)[n];
        double max_val = (double)row_max[n];
        double sum = (double)row_sum[n];
        if (target < 0 || target >= V) {
            return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy target out of range");
        }
        for (int64_t v = 0; v < V; ++v) {
            double s = 0.0;
            double dl = 0.0;
            for (int64_t d = 0; d < D; ++d) {
                s += (double)hidden[n * D + d] * (double)weight[v * D + d];
            }
            dl = scale * (exp(s - max_val) / sum - (v == target ? 1.0 : 0.0));
            for (int64_t d = 0; d < D; ++d) {
                dhidden[n * D + d] += (float)(dl * (double)weight[v * D + d]);
                dweight[v * D + d] += (float)(dl * (double)hidden[n * D + d]);
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_assert_finite(const gd_tensor_desc *desc, const float *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        if (!isfinite(x[i])) {
            return _gd_error(GD_ERR_INVALID_STATE, "assert_finite: non-finite value");
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_assert_close(const gd_tensor_desc *a_desc,
                                 const float *a,
                                 const float *b,
                                 float atol,
                                 float rtol)
{
    int64_t total = desc_numel(a_desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        double diff = fabs((double)a[i] - (double)b[i]);
        double tol = (double)atol + (double)rtol * fabs((double)b[i]);
        if (diff > tol) {
            return _gd_error(GD_ERR_INVALID_STATE, "assert_close: tensors differ");
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_reduce_to(const gd_tensor_desc *target_desc,
                              float *out,
                              const gd_tensor_desc *go_desc,
                              const float *go)
{
    int64_t target_total = desc_numel(target_desc);
    int64_t go_total = desc_numel(go_desc);
    int64_t i = 0;
    int64_t index[GD_MAX_DIMS];

    for (i = 0; i < target_total; ++i) {
        out[i] = 0.0F;
    }
    for (i = 0; i < go_total; ++i) {
        int64_t off = 0;
        unravel(i, go_desc, index);
        off = broadcast_offset(index, go_desc->ndim, target_desc);
        out[off] += go[i];
    }
    return GD_OK;
}

gd_status _gd_cpu_k_step_inc(float *step)
{
    step[0] += 1.0F;
    return GD_OK;
}

gd_status _gd_cpu_k_clip_grad_norm(const gd_tensor_desc * const *grad_descs,
                                   float **grads,
                                   int n_grads,
                                   float max_norm,
                                   float eps,
                                   float *norm_out)
{
    double sumsq = 0.0;
    double norm;
    double scale = 1.0;
    int g = 0;

    if (grad_descs == NULL || grads == NULL || n_grads <= 0 || norm_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid clip_grad_norm arguments");
    }
    if (!isfinite(max_norm) || max_norm <= 0.0F || !isfinite(eps) || eps < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid clip_grad_norm config");
    }
    for (g = 0; g < n_grads; ++g) {
        int64_t total;
        int64_t i;
        if (grad_descs[g] == NULL || grads[g] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "null clip_grad_norm input");
        }
        total = desc_numel(grad_descs[g]);
        for (i = 0; i < total; ++i) {
            double v = (double)grads[g][i];
            sumsq += v * v;
        }
    }
    norm = sqrt(sumsq);
    if (norm > (double)max_norm) {
        scale = (double)max_norm / (norm + (double)eps);
    }
    for (g = 0; g < n_grads; ++g) {
        int64_t total = desc_numel(grad_descs[g]);
        int64_t i;
        for (i = 0; i < total; ++i) {
            grads[g][i] = (float)((double)grads[g][i] * scale);
        }
    }
    norm_out[0] = (float)norm;
    return GD_OK;
}

gd_status _gd_cpu_k_adamw(const gd_tensor_desc *param_desc,
                          float *param,
                          const float *grad,
                          float *m,
                          float *v,
                          const float *step,
                          const float *lr_tensor,
                          float lr,
                          float lr_scale,
                          float beta1,
                          float beta2,
                          float eps,
                          float weight_decay)
{
    int64_t total = desc_numel(param_desc);
    int64_t i = 0;
    double t = (double)step[0];
    double bc1 = 1.0 - pow((double)beta1, t);
    double bc2 = 1.0 - pow((double)beta2, t);
    double step_lr = (lr_tensor != NULL ? (double)lr_tensor[0] : (double)lr) *
                     (double)lr_scale;

    if (t < 1.0) {
        return _gd_error(GD_ERR_INVALID_STATE, "adamw step counter must be >= 1");
    }
    if (!isfinite(step_lr) || step_lr < 0.0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw lr must be finite and nonnegative");
    }
    for (i = 0; i < total; ++i) {
        double g = (double)grad[i];
        double mi = (double)beta1 * (double)m[i] + (1.0 - (double)beta1) * g;
        double vi = (double)beta2 * (double)v[i] + (1.0 - (double)beta2) * g * g;
        double mhat = mi / bc1;
        double vhat = vi / bc2;
        double p = (double)param[i];

        m[i] = (float)mi;
        v[i] = (float)vi;
        p -= step_lr * (double)weight_decay * p;        /* decoupled weight decay */
        p -= step_lr * mhat / (sqrt(vhat) + (double)eps);
        param[i] = (float)p;
    }
    return GD_OK;
}
