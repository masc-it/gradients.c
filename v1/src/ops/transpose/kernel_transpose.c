#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <string.h>

gd_status _gd_cpu_k_transpose(const gd_tensor_desc *out_desc, void *out,
                              const gd_tensor_desc *in_desc, const void *in,
                              const int *perm)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = _gd_cpu_desc_numel(out_desc);
    int64_t in_strides[GD_MAX_DIMS];
    int64_t out_index[GD_MAX_DIMS];
    int ndim = out_desc->ndim;
    int64_t stride = 1;
    int64_t lin = 0;
    int k = 0;

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "transpose requires a fixed-size dtype");
    }
    if (in_desc->ndim != ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose rank mismatch");
    }
    for (k = ndim - 1; k >= 0; --k) {
        in_strides[k] = stride;
        stride *= in_desc->sizes[k];
    }
    for (lin = 0; lin < total; ++lin) {
        int64_t in_off = 0;
        _gd_cpu_unravel(lin, out_desc, out_index);
        for (k = 0; k < ndim; ++k) {
            in_off += out_index[k] * in_strides[perm[k]];
        }
        memcpy((unsigned char *)out + (size_t)lin * elem,
               (const unsigned char *)in + (size_t)in_off * elem, elem);
    }
    return GD_OK;
}
