#include "../../backends/cpu_ref/cpu_op.h"

#include "sdpa_varlen_kernel.h"

static gd_status sdpa_varlen_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *go_data = NULL;
    void *q_data = NULL;
    void *k_data = NULL;
    void *v_data = NULL;
    void *cu_data = NULL;
    void *dq_data = NULL;
    void *dk_data = NULL;
    void *dv_data = NULL;
    const gd_tensor_desc *go_desc = NULL;
    const gd_tensor_desc *q_desc = NULL;
    const gd_tensor_desc *k_desc = NULL;
    const gd_tensor_desc *v_desc = NULL;
    const gd_tensor_desc *cu_desc = NULL;
    const gd_tensor_desc *dq_desc = NULL;
    const gd_tensor_desc *dk_desc = NULL;
    const gd_tensor_desc *dv_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &go_data, &go_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, &q_data, &q_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, &k_data, &k_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, &v_data, &v_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, &cu_data, &cu_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 0, &dq_data, &dq_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 1, &dk_data, &dk_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 2, &dv_data, &dv_desc);
    if (status != GD_OK) { return status; }

    return _gd_cpu_k_sdpa_varlen_bwd(dq_desc, dq_data,
                                     dk_desc, dk_data,
                                     dv_desc, dv_data,
                                     go_desc, go_data,
                                     q_desc, q_data,
                                     k_desc, k_data,
                                     v_desc, v_data,
                                     cu_desc, cu_data,
                                     node->attrs.attn_scale,
                                     node->attrs.causal,
                                     node->attrs.sliding_window,
                                     node->attrs.prefix_len,
                                     node->attrs.max_seqlen);
}

const _gd_cpu_op _gd_cpu_op_sdpa_varlen_bwd = {
    .kind = _GD_OP_SDPA_VARLEN_BWD,
    .name = "sdpa_varlen_bwd",
    .support = _gd_cpu_support_default,
    .run = sdpa_varlen_bwd_run,
};
