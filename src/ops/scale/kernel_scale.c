#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

gd_status _gd_cpu_k_scale(const gd_tensor_desc *desc, void *out, const void *x, float scale)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v * scale);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}
