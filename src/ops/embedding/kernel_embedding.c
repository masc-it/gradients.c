#include "../../backends/cpu_ref/cpu_backend.h"

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
