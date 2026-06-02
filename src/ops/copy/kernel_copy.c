#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <string.h>

gd_status _gd_cpu_k_copy(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *in_desc,
                         const void *in)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = _gd_cpu_desc_numel(out_desc);

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "copy requires a fixed-size dtype");
    }
    if (out_desc->dtype != in_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "copy requires matching dtypes");
    }
    if (_gd_cpu_desc_numel(in_desc) != total) {
        return _gd_error(GD_ERR_SHAPE, "copy requires equal element counts");
    }
    memcpy(out, in, (size_t)total * elem);
    return GD_OK;
}
