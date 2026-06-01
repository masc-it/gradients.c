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
