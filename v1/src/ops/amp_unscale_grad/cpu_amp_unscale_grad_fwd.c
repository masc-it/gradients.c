#include <math.h>
#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static int64_t amp_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status amp_unscale_grad_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    float *grad = NULL;
    float *scale = NULL;
    int32_t *found_inf = NULL;
    const gd_tensor_desc *grad_desc = NULL;
    const gd_tensor_desc *scale_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    gd_status status = GD_OK;
    int64_t n = 0;
    int64_t i = 0;

    status = _gd_cpu_exec_input(exec, node, 0, (void **)&grad, &grad_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&scale, &scale_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    if (grad_desc->dtype != GD_DTYPE_F32 || scale_desc->dtype != GD_DTYPE_F32 ||
        found_desc->dtype != GD_DTYPE_I32 || scale_desc->ndim != 0 || found_desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE, "amp_unscale_grad expects F32 grad/scale and I32 found_inf");
    }
    if (!isfinite(scale[0]) || scale[0] <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "AMP loss scale must be finite and positive");
    }
    n = amp_numel(grad_desc);
    for (i = 0; i < n; ++i) {
        float g = grad[i];
        if (!isfinite(g)) {
            found_inf[0] = 1;
        } else {
            grad[i] = g / scale[0];
        }
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_amp_unscale_grad = {
    .kind = _GD_OP_AMP_UNSCALE_GRAD,
    .name = "amp_unscale_grad",
    .support = _gd_cpu_support_default,
    .run = amp_unscale_grad_run,
};
