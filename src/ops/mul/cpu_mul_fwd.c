#include "../../backends/cpu_ref/cpu_op.h"

static gd_status mul_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *a_data = NULL;
    void *b_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *a_desc = NULL;
    const gd_tensor_desc *b_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &a_data, &a_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &b_data, &b_desc);
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
    return _gd_cpu_k_elementwise(_GD_OP_MUL, out_desc, out_data, a_desc, a_data, b_desc, b_data);
}

const _gd_cpu_op _gd_cpu_op_mul = {
    .kind = _GD_OP_MUL,
    .name = "mul",
    .support = _gd_cpu_support_default,
    .run = mul_run,
};
