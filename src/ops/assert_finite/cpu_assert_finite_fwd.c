#include "../../backends/cpu_ref/cpu_op.h"

static gd_status assert_finite_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *x_data = NULL;
    const gd_tensor_desc *x_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &x_data, &x_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(x_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_assert_finite(x_desc, x_data);
}

const _gd_cpu_op _gd_cpu_op_assert_finite = {
    .kind = _GD_OP_ASSERT_FINITE,
    .name = "assert_finite",
    .support = _gd_cpu_support_default,
    .run = assert_finite_run,
};
