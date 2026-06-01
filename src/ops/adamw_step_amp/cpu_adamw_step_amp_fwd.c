#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static gd_status adamw_step_amp_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    float *param = NULL;
    float *grad = NULL;
    float *m = NULL;
    float *v = NULL;
    float *step = NULL;
    int32_t *found_inf = NULL;
    float *lr = NULL;
    const gd_tensor_desc *param_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    const gd_tensor_desc *lr_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, (void **)&param, &param_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&grad, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, (void **)&m, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, (void **)&v, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, (void **)&step, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 5, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_require_f32(param_desc);
    if (status != GD_OK) { return status; }
    if (found_desc->dtype != GD_DTYPE_I32 || found_desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE, "adamw_step_amp expects I32 found_inf");
    }
    if (node->n_inputs == 7) {
        status = _gd_cpu_exec_input(exec, node, 6, (void **)&lr, &lr_desc);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_require_f32(lr_desc);
        if (status != GD_OK) { return status; }
        if (lr_desc->ndim != 0) {
            return _gd_error(GD_ERR_SHAPE, "adamw lr tensor must be scalar");
        }
    }
    if (found_inf[0] != 0) {
        return GD_OK;
    }
    return _gd_cpu_k_adamw(param_desc, param, grad, m, v, step, lr, node->attrs.lr,
                           node->attrs.scale, node->attrs.beta1, node->attrs.beta2,
                           node->attrs.eps, node->attrs.weight_decay);
}

const _gd_cpu_op _gd_cpu_op_adamw_step_amp = {
    .kind = _GD_OP_ADAMW_STEP_AMP,
    .name = "adamw_step_amp",
    .support = _gd_cpu_support_default,
    .run = adamw_step_amp_run,
};
