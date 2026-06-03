#include "../../backends/cpu_ref/cpu_op.h"

static gd_status cross_entropy_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *logits_data = NULL;
    void *targets_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *logits_desc = NULL;
    const gd_tensor_desc *targets_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &logits_data, &logits_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &targets_data, &targets_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_cross_entropy(out_data, logits_desc, logits_data,
                                   targets_desc, targets_data, node->attrs.dim,
                                   node->attrs.has_ignore_index, node->attrs.ignore_index);
}

const _gd_cpu_op _gd_cpu_op_cross_entropy = {
    .kind = _GD_OP_CROSS_ENTROPY,
    .name = "cross_entropy",
    .support = _gd_cpu_support_default,
    .run = cross_entropy_run,
};
