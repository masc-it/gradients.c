#include "../../backends/cpu_ref/cpu_op.h"

static gd_status dropout_run(_gd_cpu_exec *exec, const _gd_node *node)
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
    if (out_desc->dtype != GD_DTYPE_F32 && out_desc->dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_DTYPE, "dropout output must be F32 or F16");
    }
    return _gd_cpu_k_dropout(out_desc, out_data, x_data, node->attrs.scale,
                             node->attrs.dropout_seed, _gd_cpu_exec_run_id(exec));
}

const _gd_cpu_op _gd_cpu_op_dropout = {
    .kind = _GD_OP_DROPOUT,
    .name = "dropout",
    .support = _gd_cpu_support_default,
    .run = dropout_run,
};
