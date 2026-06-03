#include <float.h>
#include <math.h>
#include <stdint.h>

#include "../../backends/cpu_ref/cpu_op.h"

static int64_t amp_clip_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status amp_clip_grad_norm_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    float *scale = NULL;
    int32_t *found_inf = NULL;
    float *out = NULL;
    const gd_tensor_desc *scale_desc = NULL;
    const gd_tensor_desc *found_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;
    float *grads[_GD_OP_MAX_INPUTS];
    const gd_tensor_desc *grad_descs[_GD_OP_MAX_INPUTS];
    double sumsq = 0.0;
    double norm = 0.0;
    double clip = 1.0;
    int found = 0;
    int i = 0;

    status = _gd_cpu_exec_input(exec, node, 0, (void **)&scale, &scale_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, (void **)&found_inf, &found_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 0, (void **)&out, &out_desc);
    if (status != GD_OK) { return status; }
    if (scale_desc->dtype != GD_DTYPE_F32 || scale_desc->ndim != 0 ||
        found_desc->dtype != GD_DTYPE_I32 || found_desc->ndim != 0 ||
        out_desc->dtype != GD_DTYPE_F32 || out_desc->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE,
                         "amp_clip_grad_norm expects F32 scale/output and I32 found_inf");
    }
    if (!isfinite(scale[0]) || scale[0] <= 0.0F) {
        found_inf[0] = 1;
        out[0] = FLT_MAX;
        return GD_OK;
    }
    found = found_inf[0] != 0;
    for (i = 2; i < node->n_inputs; ++i) {
        int64_t total = 0;
        int64_t j = 0;
        status = _gd_cpu_exec_input(exec, node, i, (void **)&grads[i], &grad_descs[i]);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_require_f32(grad_descs[i]);
        if (status != GD_OK) { return status; }
        total = amp_clip_numel(grad_descs[i]);
        for (j = 0; j < total; ++j) {
            float g = grads[i][j];
            if (!isfinite(g)) {
                found = 1;
                found_inf[0] = 1;
            } else {
                float u = g / scale[0];
                grads[i][j] = u;
                sumsq += (double)u * (double)u;
            }
        }
    }
    if (found != 0) {
        out[0] = FLT_MAX;
        return GD_OK;
    }
    norm = sqrt(sumsq);
    if (norm > (double)node->attrs.scale) {
        clip = (double)node->attrs.scale / (norm + (double)node->attrs.eps);
    }
    for (i = 2; i < node->n_inputs; ++i) {
        int64_t total = amp_clip_numel(grad_descs[i]);
        int64_t j = 0;
        if (clip == 1.0) {
            continue;
        }
        for (j = 0; j < total; ++j) {
            grads[i][j] = (float)((double)grads[i][j] * clip);
        }
    }
    out[0] = (float)norm;
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_amp_clip_grad_norm = {
    .kind = _GD_OP_AMP_CLIP_GRAD_NORM,
    .name = "amp_clip_grad_norm",
    .support = _gd_cpu_support_default,
    .run = amp_clip_grad_norm_run,
};
