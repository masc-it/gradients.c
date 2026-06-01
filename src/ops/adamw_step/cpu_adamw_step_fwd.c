#include "../../backends/cpu_ref/cpu_op.h"

static gd_status adamw_step_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *param_data = NULL;
    void *grad_data = NULL;
    void *m_data = NULL;
    void *v_data = NULL;
    void *step_data = NULL;
    void *lr_data = NULL;
    const gd_tensor_desc *param_desc = NULL;
    const gd_tensor_desc *lr_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &param_data, &param_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &grad_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 2, &m_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 3, &v_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 4, &step_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(param_desc);
    if (status != GD_OK) {
        return status;
    }
    if (node->n_inputs != 5 && node->n_inputs != 6) {
        return _gd_error(GD_ERR_INTERNAL, "adamw_step expects 5 or 6 inputs");
    }
    if (node->n_inputs == 6) {
        status = _gd_cpu_exec_input(exec, node, 5, &lr_data, &lr_desc);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_require_f32(lr_desc);
        if (status != GD_OK) {
            return status;
        }
        if (lr_desc->ndim != 0) {
            return _gd_error(GD_ERR_SHAPE, "adamw lr tensor must be scalar");
        }
    }
    return _gd_cpu_k_adamw(param_desc, param_data, grad_data, m_data, v_data, step_data,
                           lr_data, node->attrs.lr, node->attrs.scale, node->attrs.beta1,
                           node->attrs.beta2, node->attrs.eps, node->attrs.weight_decay);
}

const _gd_cpu_op _gd_cpu_op_adamw_step = {
    .kind = _GD_OP_ADAMW_STEP,
    .name = "adamw_step",
    .support = _gd_cpu_support_default,
    .run = adamw_step_run,
};
