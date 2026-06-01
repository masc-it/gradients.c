#include "../../backends/cpu_ref/cpu_op.h"

static gd_status linear_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x_data = NULL;
    void *w_data = NULL;
    void *bias_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *x_desc = NULL;
    const gd_tensor_desc *w_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &x_data, &x_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &w_data, &w_desc);
    if (status != GD_OK) {
        return status;
    }
    if (node->attrs.has_bias) {
        status = _gd_cpu_exec_input(exec, node, 2, &bias_data, NULL);
        if (status != GD_OK) {
            return status;
        }
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_linear(out_desc, out_data, x_desc, x_data, w_desc, w_data,
                            node->attrs.trans_b, bias_data);
}

const _gd_cpu_op _gd_cpu_op_linear = {
    .kind = _GD_OP_LINEAR,
    .name = "linear",
    .support = _gd_cpu_support_default,
    .run = linear_run,
};
