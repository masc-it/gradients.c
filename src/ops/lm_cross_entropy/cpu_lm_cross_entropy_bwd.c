#include "../../backends/cpu_ref/cpu_op.h"

static gd_status lm_cross_entropy_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *hidden_data = NULL;
    void *weight_data = NULL;
    void *targets_data = NULL;
    void *go_data = NULL;
    void *row_max_data = NULL;
    void *row_sum_data = NULL;
    void *dhidden_data = NULL;
    void *dweight_data = NULL;
    const gd_tensor_desc *hidden_desc = NULL;
    const gd_tensor_desc *weight_desc = NULL;
    const gd_tensor_desc *targets_desc = NULL;
    const gd_tensor_desc *dhidden_desc = NULL;
    const gd_tensor_desc *dweight_desc = NULL;

    if (node->n_inputs != 6 || node->n_outputs != 2) {
        return _gd_error(GD_ERR_INTERNAL, "lm_cross_entropy_bwd expects stats inputs");
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
    status = _gd_cpu_exec_input(exec, node, 3, &go_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 4, &row_max_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 5, &row_sum_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &dhidden_data, &dhidden_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 1, &dweight_data, &dweight_desc);
    if (status != GD_OK) {
        return status;
    }
    if ((dhidden_desc->dtype != GD_DTYPE_F32 && dhidden_desc->dtype != GD_DTYPE_F16) ||
        (dweight_desc->dtype != GD_DTYPE_F32 && dweight_desc->dtype != GD_DTYPE_F16)) {
        return _gd_error(GD_ERR_DTYPE, "lm_cross_entropy_bwd outputs must be F32 or F16");
    }
    return _gd_cpu_k_lm_cross_entropy_bwd(hidden_desc, dhidden_desc, dhidden_data, hidden_data,
                                          weight_desc, dweight_desc, dweight_data, weight_data,
                                          targets_desc, targets_data, go_data,
                                          row_max_data, row_sum_data);
}

const _gd_cpu_op _gd_cpu_op_lm_cross_entropy_bwd = {
    .kind = _GD_OP_LM_CROSS_ENTROPY_BWD,
    .name = "lm_cross_entropy_bwd",
    .support = _gd_cpu_support_default,
    .run = lm_cross_entropy_bwd_run,
};
