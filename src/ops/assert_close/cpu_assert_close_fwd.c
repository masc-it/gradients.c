#include "../../backends/cpu_ref/cpu_op.h"

static gd_status assert_close_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *a_data = NULL;
    void *b_data = NULL;
    const gd_tensor_desc *a_desc = NULL;

    status = _gd_cpu_exec_input(exec, node, 0, &a_data, &a_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &b_data, NULL);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(a_desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_cpu_k_assert_close(a_desc, a_data, b_data, node->attrs.atol, node->attrs.rtol);
}

const _gd_cpu_op _gd_cpu_op_assert_close = {
    .kind = _GD_OP_ASSERT_CLOSE,
    .name = "assert_close",
    .support = _gd_cpu_support_default,
    .run = assert_close_run,
};
