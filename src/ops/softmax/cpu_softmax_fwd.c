#include "../../backends/cpu_ref/cpu_op.h"

static gd_status softmax_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &x_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_softmax(out_desc, out_data, x_data, node->attrs.dim);
}

const _gd_cpu_op _gd_cpu_op_softmax = {
    .kind = _GD_OP_SOFTMAX,
    .name = "softmax",
    .support = _gd_cpu_support_default,
    .run = softmax_run,
};
