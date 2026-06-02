#include <stdbool.h>
#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static int64_t adamw_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;

    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status validate_refresh(const gd_tensor_desc *param_desc,
                                  const gd_tensor_desc *refresh_desc)
{
    int d = 0;

    if ((refresh_desc->dtype != GD_DTYPE_F16 && refresh_desc->dtype != GD_DTYPE_F32) ||
        refresh_desc->ndim != param_desc->ndim) {
        return _gd_error(GD_ERR_DTYPE, "adamw_step refresh dtype unsupported");
    }
    for (d = 0; d < refresh_desc->ndim; ++d) {
        if (refresh_desc->sizes[d] != param_desc->sizes[d]) {
            return _gd_error(GD_ERR_SHAPE, "adamw_step refresh shape mismatch");
        }
    }
    return GD_OK;
}

static gd_status adamw_step_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *param_data = NULL;
    void *grad_data = NULL;
    void *m_data = NULL;
    void *v_data = NULL;
    void *step_data = NULL;
    void *lr_data = NULL;
    void *refresh_data = NULL;
    int32_t *found_inf = NULL;
    const gd_tensor_desc *param_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    const gd_tensor_desc *lr_desc = NULL;
    const gd_tensor_desc *refresh_desc = NULL;
    bool has_found_inf = node->attrs.adamw_has_found_inf;
    bool has_refresh = node->attrs.adamw_has_refresh;
    bool has_lr = node->attrs.adamw_has_lr ||
                  (!has_found_inf && !has_refresh && node->n_inputs == 6);
    int input = 5;

    status = _gd_cpu_exec_input(exec, node, 0, &param_data, &param_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, &grad_data, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, &m_data, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, &v_data, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, &step_data, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_require_f32(param_desc);
    if (status != GD_OK) { return status; }
    if (has_found_inf) {
        status = _gd_cpu_exec_input(exec, node, input++, (void **)&found_inf, &found_desc);
        if (status != GD_OK) { return status; }
        if (found_desc->dtype != GD_DTYPE_I32 || found_desc->ndim != 0) {
            return _gd_error(GD_ERR_DTYPE, "adamw_step found_inf must be I32 scalar");
        }
    }
    if (has_lr) {
        status = _gd_cpu_exec_input(exec, node, input++, &lr_data, &lr_desc);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_require_f32(lr_desc);
        if (status != GD_OK) { return status; }
        if (lr_desc->ndim != 0) {
            return _gd_error(GD_ERR_SHAPE, "adamw lr tensor must be scalar");
        }
    }
    if (has_refresh) {
        status = _gd_cpu_exec_input(exec, node, input++, &refresh_data, &refresh_desc);
        if (status != GD_OK) { return status; }
        status = validate_refresh(param_desc, refresh_desc);
        if (status != GD_OK) { return status; }
    }
    if (input != node->n_inputs) {
        return _gd_error(GD_ERR_INTERNAL, "adamw_step input/flag mismatch");
    }
    if (has_found_inf && found_inf[0] != 0) {
        return GD_OK;
    }
    status = _gd_cpu_k_adamw(param_desc, param_data, grad_data, m_data, v_data, step_data,
                             lr_data, node->attrs.lr, node->attrs.scale, node->attrs.beta1,
                             node->attrs.beta2, node->attrs.eps, node->attrs.weight_decay);
    if (status == GD_OK && has_refresh) {
        int64_t n = adamw_numel(param_desc);
        int64_t i = 0;

        for (i = 0; i < n; ++i) {
            status = _gd_cpu_store_float(refresh_desc, refresh_data, i, ((float *)param_data)[i]);
            if (status != GD_OK) { return status; }
        }
    }
    return status;
}

const _gd_cpu_op _gd_cpu_op_adamw_step = {
    .kind = _GD_OP_ADAMW_STEP,
    .name = "adamw_step",
    .support = _gd_cpu_support_default,
    .run = adamw_step_run,
};
