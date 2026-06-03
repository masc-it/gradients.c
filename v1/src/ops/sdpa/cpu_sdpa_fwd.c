#include "../../backends/cpu_ref/cpu_op.h"

static gd_status sdpa_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *q_data = NULL;
    void *k_data = NULL;
    void *v_data = NULL;
    void *bias_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *q_desc = NULL;
    const gd_tensor_desc *k_desc = NULL;
    const gd_tensor_desc *v_desc = NULL;
    const gd_tensor_desc *bias_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &q_data, &q_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &k_data, &k_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 2, &v_data, &v_desc);
    if (status != GD_OK) {
        return status;
    }
    if (node->attrs.has_bias) {
        status = _gd_cpu_exec_input(exec, node, 3, &bias_data, &bias_desc);
        if (status != GD_OK) {
            return status;
        }
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_sdpa(out_desc, out_data, q_desc, q_data, k_desc, k_data, v_desc, v_data,
                          bias_desc, bias_data, node->attrs.attn_scale, node->attrs.causal,
                          node->attrs.sliding_window, node->attrs.prefix_len);
}

const _gd_cpu_op _gd_cpu_op_sdpa = {
    .kind = _GD_OP_SDPA,
    .name = "sdpa",
    .support = _gd_cpu_support_default,
    .run = sdpa_run,
};
