#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

gd_status _gd_cpu_k_add(const gd_tensor_desc *out_desc,
                                void *out,
                                const gd_tensor_desc *a_desc,
                                const void *a,
                                const gd_tensor_desc *b_desc,
                                const void *b)
{
    int64_t total = _gd_cpu_desc_numel(out_desc);
    int64_t lin = 0;
    int64_t index[GD_MAX_DIMS];

    for (lin = 0; lin < total; ++lin) {
        int64_t ao = 0;
        int64_t bo = 0;
        float av = 0.0F;
        float bv = 0.0F;
        float y = 0.0F;
        gd_status status = GD_OK;

        _gd_cpu_unravel(lin, out_desc, index);
        ao = _gd_cpu_broadcast_offset(index, out_desc->ndim, a_desc);
        bo = _gd_cpu_broadcast_offset(index, out_desc->ndim, b_desc);
        status = _gd_cpu_load_float(a_desc, a, ao, &av);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_load_float(b_desc, b, bo, &bv);
        if (status != GD_OK) {
            return status;
        }
        y = av + bv;
        status = _gd_cpu_store_float(out_desc, out, lin, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}
