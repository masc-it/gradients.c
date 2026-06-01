#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static int64_t refresh_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status amp_refresh_param_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    void *param = NULL;
    float *master = NULL;
    int32_t *found_inf = NULL;
    const gd_tensor_desc *param_desc = NULL;
    const gd_tensor_desc *master_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    gd_status status = GD_OK;
    int64_t n = 0;
    int64_t i = 0;
    int dim = 0;

    status = _gd_cpu_exec_input(exec, node, 0, &param, &param_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&master, &master_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    if ((param_desc->dtype != GD_DTYPE_F32 && param_desc->dtype != GD_DTYPE_F16) ||
        master_desc->dtype != GD_DTYPE_F32 || found_desc->dtype != GD_DTYPE_I32 ||
        found_desc->ndim != 0 || param_desc->ndim != master_desc->ndim) {
        return _gd_error(GD_ERR_DTYPE, "amp_refresh_param dtype mismatch");
    }
    for (dim = 0; dim < param_desc->ndim; ++dim) {
        if (param_desc->sizes[dim] != master_desc->sizes[dim]) {
            return _gd_error(GD_ERR_SHAPE, "amp_refresh_param shape mismatch");
        }
    }
    if (found_inf[0] != 0) {
        return GD_OK;
    }
    n = refresh_numel(param_desc);
    for (i = 0; i < n; ++i) {
        status = _gd_cpu_store_float(param_desc, param, i, master[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_amp_refresh_param = {
    .kind = _GD_OP_AMP_REFRESH_PARAM,
    .name = "amp_refresh_param",
    .support = _gd_cpu_support_default,
    .run = amp_refresh_param_run,
};
