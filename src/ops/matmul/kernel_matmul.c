#include "../../backends/cpu_ref/cpu_backend.h"

#include "../../core/internal.h"

#include <stdint.h>

static int64_t desc_numel(const gd_tensor_desc *desc)
{
    int64_t numel = 1;
    int i = 0;

    for (i = 0; i < desc->ndim; ++i) {
        numel *= desc->sizes[i];
    }
    return numel;
}

static gd_status load_float(const gd_tensor_desc *desc, const void *data,
                            int64_t i, float *out)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        *out = ((const float *)data)[i];
        return GD_OK;
    case GD_DTYPE_F16:
        *out = _gd_f16_bits_to_f32(((const uint16_t *)data)[i]);
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF matmul supports F32/F16 only");
    }
}

static gd_status store_float(const gd_tensor_desc *desc, void *data,
                             int64_t i, float value)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        ((float *)data)[i] = value;
        return GD_OK;
    case GD_DTYPE_F16:
        ((uint16_t *)data)[i] = _gd_f32_to_f16_bits(value);
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF matmul supports F32/F16 only");
    }
}

gd_status _gd_cpu_k_matmul(const gd_tensor_desc *out_desc,
                           void *out,
                           const gd_tensor_desc *a_desc,
                           const void *a,
                           bool trans_a,
                           const gd_tensor_desc *b_desc,
                           const void *b,
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
                    float av = 0.0F;
                    float bv = 0.0F;
                    gd_status status = load_float(a_desc, a, a_base + a_off, &av);
                    if (status != GD_OK) {
                        return status;
                    }
                    status = load_float(b_desc, b, b_base + b_off, &bv);
                    if (status != GD_OK) {
                        return status;
                    }
                    acc += (double)av * (double)bv;
                }
                {
                    int64_t out_off = batch_lin * out_mat + row * n + col;
                    gd_status status = store_float(out_desc, out, out_off, (float)acc);
                    if (status != GD_OK) {
                        return status;
                    }
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_linear(const gd_tensor_desc *out_desc,
                           void *out,
                           const gd_tensor_desc *x_desc,
                           const void *x,
                           const gd_tensor_desc *w_desc,
                           const void *w,
                           bool trans_w,
                           const void *bias)
{
    int64_t in_features = x_desc->sizes[x_desc->ndim - 1];
    int64_t out_features = out_desc->sizes[out_desc->ndim - 1];
    int64_t rows = desc_numel(x_desc) / in_features;
    int64_t r = 0;

    (void)w_desc;
    for (r = 0; r < rows; ++r) {
        int64_t o = 0;
        for (o = 0; o < out_features; ++o) {
            double acc = 0.0;
            int64_t kk = 0;
            if (bias != NULL) {
                float bv = 0.0F;
                gd_status status = load_float(out_desc, bias, o, &bv);
                if (status != GD_OK) {
                    return status;
                }
                acc = (double)bv;
            }
            for (kk = 0; kk < in_features; ++kk) {
                int64_t w_off = trans_w ? (o * in_features + kk) : (kk * out_features + o);
                float xv = 0.0F;
                float wv = 0.0F;
                gd_status status = load_float(x_desc, x, r * in_features + kk, &xv);
                if (status != GD_OK) {
                    return status;
                }
                status = load_float(w_desc, w, w_off, &wv);
                if (status != GD_OK) {
                    return status;
                }
                acc += (double)xv * (double)wv;
            }
            {
                gd_status status = store_float(out_desc, out, r * out_features + o, (float)acc);
                if (status != GD_OK) {
                    return status;
                }
            }
        }
    }
    return GD_OK;
}
