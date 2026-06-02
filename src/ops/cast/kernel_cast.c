#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <string.h>

gd_status _gd_cpu_k_cast(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *x_desc,
                         const void *x)
{
    int64_t total = _gd_cpu_desc_numel(out_desc);
    int64_t i = 0;

    if (_gd_cpu_desc_numel(x_desc) != total) {
        return _gd_error(GD_ERR_SHAPE, "cast requires equal element counts");
    }
    if (x_desc->dtype == out_desc->dtype) {
        size_t elem = gd_dtype_sizeof(out_desc->dtype);
        if (elem == 0U) {
            return _gd_error(GD_ERR_DTYPE, "cast requires a fixed-size dtype");
        }
        memcpy(out, x, (size_t)total * elem);
        return GD_OK;
    }
    for (i = 0; i < total; ++i) {
        float value = 0.0F;
        gd_status status = _gd_cpu_load_float(x_desc, x, i, &value);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(out_desc, out, i, value);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}
