#include "../../backends/cpu_ref/cpu_backend.h"

#include <math.h>
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

static int64_t embedding_id(const gd_tensor_desc *ids_desc, const void *ids, int64_t p)
{
    if (ids_desc->dtype == GD_DTYPE_I64) {
        return ((const int64_t *)ids)[p];
    }
    return (int64_t)((const int32_t *)ids)[p];
}

gd_status _gd_cpu_k_rope(const gd_tensor_desc *desc, void *out, const void *x,
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
            float value = 0.0F;
            gd_status status = _gd_cpu_load_float(desc, x, base + d, &value);
            if (status != GD_OK) {
                return status;
            }
            status = _gd_cpu_store_float(desc, out, base + d, value);
            if (status != GD_OK) {
                return status;
            }
        }
        for (i = 0; i < half; ++i) {
            double inv = pow((double)theta, -(2.0 * (double)i) / (double)n_dims);
            double angle = (double)p * inv;
            double c = cos(angle);
            double s = sin(angle) * (double)sin_sign;
            int a = interleaved ? (2 * i) : i;
            int bb = interleaved ? (2 * i + 1) : (i + half);
            float f1 = 0.0F;
            float f2 = 0.0F;
            double x1 = 0.0;
            double x2 = 0.0;
            gd_status status = _gd_cpu_load_float(desc, x, base + a, &f1);
            if (status != GD_OK) {
                return status;
            }
            status = _gd_cpu_load_float(desc, x, base + bb, &f2);
            if (status != GD_OK) {
                return status;
            }
            x1 = (double)f1;
            x2 = (double)f2;
            status = _gd_cpu_store_float(desc, out, base + a, (float)(x1 * c - x2 * s));
            if (status != GD_OK) {
                return status;
            }
            status = _gd_cpu_store_float(desc, out, base + bb, (float)(x1 * s + x2 * c));
            if (status != GD_OK) {
                return status;
            }
        }
    }
    return GD_OK;
}
