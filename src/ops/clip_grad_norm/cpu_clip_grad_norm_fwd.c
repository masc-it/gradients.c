#include "../../backends/cpu_ref/cpu_op.h"

static gd_status clip_grad_norm_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *out_data = NULL;
    float *grad_data[_GD_OP_MAX_INPUTS];
    const gd_tensor_desc *grad_descs[_GD_OP_MAX_INPUTS];
    const gd_tensor_desc *out_desc = NULL;
    int i = 0;

    if (node->n_inputs <= 0 || node->n_outputs != 1) {
        return _gd_error(GD_ERR_INTERNAL, "clip_grad_norm expects inputs and one output");
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_require_f32(out_desc);
    if (status != GD_OK) {
        return status;
    }
    if (out_desc->ndim != 0) {
        return _gd_error(GD_ERR_SHAPE, "clip_grad_norm output must be scalar");
    }
    for (i = 0; i < node->n_inputs; ++i) {
        void *grad = NULL;
        status = _gd_cpu_exec_input(exec, node, i, &grad, &grad_descs[i]);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_require_f32(grad_descs[i]);
        if (status != GD_OK) {
            return status;
        }
        grad_data[i] = grad;
    }
    return _gd_cpu_k_clip_grad_norm(grad_descs, grad_data, node->n_inputs,
                                    node->attrs.scale, node->attrs.eps, out_data);
}

const _gd_cpu_op _gd_cpu_op_clip_grad_norm = {
    .kind = _GD_OP_CLIP_GRAD_NORM,
    .name = "clip_grad_norm",
    .support = _gd_cpu_support_default,
    .run = clip_grad_norm_run,
};
