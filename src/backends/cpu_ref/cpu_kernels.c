#include "cpu_backend.h"

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

gd_status _gd_cpu_k_adamw(const gd_tensor_desc *param_desc,
                          float *param,
                          const float *grad,
                          float *m,
                          float *v,
                          const float *step,
                          float lr,
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

    if (t < 1.0) {
        return _gd_error(GD_ERR_INVALID_STATE, "adamw step counter must be >= 1");
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
        p -= (double)lr * (double)weight_decay * p;        /* decoupled weight decay */
        p -= (double)lr * mhat / (sqrt(vhat) + (double)eps);
        param[i] = (float)p;
    }
    return GD_OK;
}
