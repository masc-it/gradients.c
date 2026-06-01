#include "../../backends/cpu_ref/cpu_op.h"

static gd_status reduce_to_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *go_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *go_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &go_data, &go_desc);
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
    return _gd_cpu_k_reduce_to(out_desc, out_data, go_desc, go_data);
}

const _gd_cpu_op _gd_cpu_op_reduce_to = {
    .kind = _GD_OP_REDUCE_TO,
    .name = "reduce_to",
    .support = _gd_cpu_support_default,
    .run = reduce_to_run,
};
