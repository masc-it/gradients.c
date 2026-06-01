#include "../../backends/cpu_ref/cpu_op.h"

static gd_status lm_cross_entropy_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *hidden_data = NULL;
    void *weight_data = NULL;
    void *targets_data = NULL;
    void *out_data = NULL;
    void *row_max_data = NULL;
    void *row_sum_data = NULL;
    const gd_tensor_desc *hidden_desc = NULL;
    const gd_tensor_desc *weight_desc = NULL;
    const gd_tensor_desc *targets_desc = NULL;

    if (node->n_outputs != 3) {
        return _gd_error(GD_ERR_INTERNAL, "lm_cross_entropy expects 3 outputs");
    }
    status = _gd_cpu_exec_input(exec, node, 0, &hidden_data, &hidden_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &weight_data, &weight_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 2, &targets_data, &targets_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 1, &row_max_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 2, &row_sum_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(hidden_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_lm_cross_entropy(out_data, row_max_data, row_sum_data,
                                      hidden_desc, hidden_data, weight_desc, weight_data,
                                      targets_desc, targets_data);
}

const _gd_cpu_op _gd_cpu_op_lm_cross_entropy = {
    .kind = _GD_OP_LM_CROSS_ENTROPY,
    .name = "lm_cross_entropy",
    .support = _gd_cpu_support_default,
    .run = lm_cross_entropy_run,
};
