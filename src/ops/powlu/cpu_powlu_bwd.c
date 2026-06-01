#include "../../backends/cpu_ref/cpu_op.h"

static gd_status powlu_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x1_data = NULL;
    void *x2_data = NULL;
    void *go_data = NULL;
    void *dx1_data = NULL;
    void *dx2_data = NULL;
    const gd_tensor_desc *dx1_desc = NULL;

    if (node->n_outputs != 2) {
        return _gd_error(GD_ERR_INTERNAL, "powlu_bwd expects 2 outputs");
    }
    status = _gd_cpu_exec_input(exec, node, 0, &x1_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &x2_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 2, &go_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &dx1_data, &dx1_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 1, &dx2_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(dx1_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_powlu_bwd(dx1_desc, dx1_data, dx2_data, x1_data, x2_data, go_data,
                               node->attrs.powlu_m);
}

const _gd_cpu_op _gd_cpu_op_powlu_bwd = {
    .kind = _GD_OP_POWLU_BWD,
    .name = "powlu_bwd",
    .support = _gd_cpu_support_default,
    .run = powlu_bwd_run,
};
