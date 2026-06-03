#include "../../backends/cpu_ref/cpu_op.h"

static gd_status rope_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x_data = NULL;
    void *pos_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *pos_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &x_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &pos_data, &pos_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    if (out_desc->dtype != GD_DTYPE_F32 && out_desc->dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_DTYPE, "rope_bwd output must be F32 or F16");
    }
    return _gd_cpu_k_rope(out_desc, out_data, x_data, pos_desc, pos_data,
                          node->attrs.rope_theta, node->attrs.rope_n_dims,
                          node->attrs.rope_interleaved, -1.0F);
}

const _gd_cpu_op _gd_cpu_op_rope_bwd = {
    .kind = _GD_OP_ROPE_BWD,
    .name = "rope_bwd",
    .support = _gd_cpu_support_default,
    .run = rope_bwd_run,
};
