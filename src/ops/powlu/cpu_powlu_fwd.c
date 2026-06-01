#include "../../backends/cpu_ref/cpu_op.h"

static gd_status powlu_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x1_data = NULL;
    void *x2_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &x1_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &x2_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_powlu(out_desc, out_data, x1_data, x2_data, node->attrs.powlu_m);
}

const _gd_cpu_op _gd_cpu_op_powlu = {
    .kind = _GD_OP_POWLU,
    .name = "powlu",
    .support = _gd_cpu_support_default,
    .run = powlu_run,
};
