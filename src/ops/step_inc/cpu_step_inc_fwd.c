#include "../../backends/cpu_ref/cpu_op.h"

static gd_status step_inc_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *step_data = NULL;
    const gd_tensor_desc *step_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &step_data, &step_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(step_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_step_inc(step_data);
}

const _gd_cpu_op _gd_cpu_op_step_inc = {
    .kind = _GD_OP_STEP_INC,
    .name = "step_inc",
    .support = _gd_cpu_support_default,
    .run = step_inc_run,
};
