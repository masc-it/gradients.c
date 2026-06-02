#include <stdbool.h>
#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static int64_t adamw_amp_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status adamw_step_amp_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    float *param = NULL;
    float *grad = NULL;
    float *m = NULL;
    float *v = NULL;
    float *step = NULL;
    int32_t *found_inf = NULL;
    float *lr = NULL;
    void *refresh = NULL;
    const gd_tensor_desc *param_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    const gd_tensor_desc *lr_desc = NULL;
    const gd_tensor_desc *refresh_desc = NULL;
    bool has_refresh = false;

    status = _gd_cpu_exec_input(exec, node, 0, (void **)&param, &param_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&grad, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, (void **)&m, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, (void **)&v, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, (void **)&step, NULL);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 5, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_require_f32(param_desc);
    if (status != GD_OK) { return status; }
    if (found_desc->dtype != GD_DTYPE_I32 || found_desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE, "adamw_step_amp expects I32 found_inf");
    }
    if (node->n_inputs == 7 || node->n_inputs == 8) {
        int refresh_index = 6;
        if (node->n_inputs == 8) {
            status = _gd_cpu_exec_input(exec, node, 6, (void **)&lr, &lr_desc);
            if (status != GD_OK) { return status; }
            if (lr_desc->dtype != GD_DTYPE_F32 || lr_desc->ndim != 0) {
                return _gd_error(GD_ERR_DTYPE, "adamw lr tensor must be F32 scalar");
            }
            refresh_index = 7;
        }
        status = _gd_cpu_exec_input(exec, node, refresh_index, &refresh, &refresh_desc);
        if (status != GD_OK) { return status; }
        if (node->n_inputs == 7 && refresh_desc->dtype == GD_DTYPE_F32 &&
            refresh_desc->ndim == 0) {
            lr = (float *)refresh;
            lr_desc = refresh_desc;
        } else {
            int d = 0;
            has_refresh = true;
            if ((refresh_desc->dtype != GD_DTYPE_F16 && refresh_desc->dtype != GD_DTYPE_F32) ||
                refresh_desc->ndim != param_desc->ndim) {
                return _gd_error(GD_ERR_DTYPE,
                                 "adamw_step_amp extra input must be lr or refresh param");
            }
            for (d = 0; d < refresh_desc->ndim; ++d) {
                if (refresh_desc->sizes[d] != param_desc->sizes[d]) {
                    return _gd_error(GD_ERR_SHAPE, "adamw_step_amp refresh shape mismatch");
                }
            }
        }
        if (lr_desc != NULL && lr_desc->ndim != 0) {
            return _gd_error(GD_ERR_SHAPE, "adamw lr tensor must be scalar");
        }
    }
    if (found_inf[0] != 0) {
        return GD_OK;
    }
    status = _gd_cpu_k_adamw(param_desc, param, grad, m, v, step, lr, node->attrs.lr,
                             node->attrs.scale, node->attrs.beta1, node->attrs.beta2,
                             node->attrs.eps, node->attrs.weight_decay);
    if (status == GD_OK && has_refresh) {
        int64_t n = adamw_amp_numel(param_desc);
        int64_t i = 0;
        for (i = 0; i < n; ++i) {
            status = _gd_cpu_store_float(refresh_desc, refresh, i, param[i]);
            if (status != GD_OK) { return status; }
        }
    }
    return status;
}

const _gd_cpu_op _gd_cpu_op_adamw_step_amp = {
    .kind = _GD_OP_ADAMW_STEP_AMP,
    .name = "adamw_step_amp",
    .support = _gd_cpu_support_default,
    .run = adamw_step_amp_run,
};
