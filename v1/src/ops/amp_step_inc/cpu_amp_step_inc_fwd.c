#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static gd_status amp_step_inc_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    float *step = NULL;
    int32_t *found_inf = NULL;
    const gd_tensor_desc *step_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    gd_status status = GD_OK;

    status = _gd_cpu_exec_input(exec, node, 0, (void **)&step, &step_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    if (step_desc->dtype != GD_DTYPE_F32 || found_desc->dtype != GD_DTYPE_I32 ||
        step_desc->ndim != 0 || found_desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE, "amp_step_inc expects F32 step and I32 found_inf");
    }
    if (found_inf[0] == 0) {
        step[0] += 1.0F;
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_amp_step_inc = {
    .kind = _GD_OP_AMP_STEP_INC,
    .name = "amp_step_inc",
    .support = _gd_cpu_support_default,
    .run = amp_step_inc_run,
};
