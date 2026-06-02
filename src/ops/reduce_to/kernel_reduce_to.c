#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <stdlib.h>

gd_status _gd_cpu_k_reduce_to(const gd_tensor_desc *target_desc,
                              void *out,
                              const gd_tensor_desc *go_desc,
                              const void *go)
{
    int64_t target_total = _gd_cpu_desc_numel(target_desc);
    int64_t go_total = _gd_cpu_desc_numel(go_desc);
    int64_t i = 0;
    int64_t index[GD_MAX_DIMS];
    float *acc = NULL;

    if (target_total <= 0) {
        return GD_OK;
    }
    acc = (float *)calloc((size_t)target_total, sizeof(float));
    if (acc == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "cpu reduce_to allocation failed");
    }
    for (i = 0; i < go_total; ++i) {
        int64_t off = 0;
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(go_desc, go, i, &v);
        if (status != GD_OK) {
            free(acc);
            return status;
        }
        _gd_cpu_unravel(i, go_desc, index);
        off = _gd_cpu_broadcast_offset(index, go_desc->ndim, target_desc);
        acc[off] += v;
    }
    for (i = 0; i < target_total; ++i) {
        gd_status status = _gd_cpu_store_float(target_desc, out, i, acc[i]);
        if (status != GD_OK) {
            free(acc);
            return status;
        }
    }
    free(acc);
    return GD_OK;
}
