#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_assert_finite(const gd_tensor_desc *desc, const float *x)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        if (!isfinite(x[i])) {
            return _gd_error(GD_ERR_INVALID_STATE, "assert_finite: non-finite value");
        }
    }
    return GD_OK;
}
