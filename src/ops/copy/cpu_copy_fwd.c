#include "../../backends/cpu_ref/cpu_op.h"

static gd_status copy_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *in_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *in_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &in_data, &in_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_copy(out_desc, out_data, in_desc, in_data);
}

const _gd_cpu_op _gd_cpu_op_copy = {
    .kind = _GD_OP_COPY,
    .name = "copy",
    .support = _gd_cpu_support_default,
    .run = copy_run,
};
