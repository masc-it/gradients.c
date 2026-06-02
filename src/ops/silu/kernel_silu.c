#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_silu(const gd_tensor_desc *desc, void *out, const void *x)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v / (1.0F + expf(-v)));
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_silu_bwd(const gd_tensor_desc *desc,
                             void *dx,
                             const void *x,
                             const void *go)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float xf = 0.0F;
        float gf = 0.0F;
        double s = 0.0;
        double grad = 0.0;
        gd_status status = _gd_cpu_load_float(desc, x, i, &xf);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_load_float(desc, go, i, &gf);
        if (status != GD_OK) { return status; }
        s = 1.0 / (1.0 + exp(-(double)xf));
        grad = s * (1.0 + (double)xf * (1.0 - s));
        status = _gd_cpu_store_float(desc, dx, i, (float)((double)gf * grad));
        if (status != GD_OK) { return status; }
    }
    return GD_OK;
}
