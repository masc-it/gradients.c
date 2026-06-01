#include "../../backends/cpu_ref/cpu_op.h"

static gd_status softmax_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *y_data = NULL;
    void *go_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &y_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &go_data, NULL);
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
    return _gd_cpu_k_softmax_bwd(out_desc, out_data, y_data, go_data, node->attrs.dim);
}

const _gd_cpu_op _gd_cpu_op_softmax_bwd = {
    .kind = _GD_OP_SOFTMAX_BWD,
    .name = "softmax_bwd",
    .support = _gd_cpu_support_default,
    .run = softmax_bwd_run,
};
