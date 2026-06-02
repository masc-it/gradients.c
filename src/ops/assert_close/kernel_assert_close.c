#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_assert_close(const gd_tensor_desc *a_desc,
                                 const float *a,
                                 const float *b,
                                 float atol,
                                 float rtol)
{
    int64_t total = _gd_cpu_desc_numel(a_desc);
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
