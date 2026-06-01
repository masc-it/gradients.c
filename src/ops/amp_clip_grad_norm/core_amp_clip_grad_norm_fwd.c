#include "../op_impl.h"
#include "../meta_common.h"

#include <math.h>

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status amp_clip_grad_norm_meta(const gd_tensor_desc *const *inputs,
                                         int n_inputs,
                                         _gd_op_attrs *attrs,
                                         gd_tensor_desc *outputs,
                                         int *n_outputs)
{
    gd_status status = GD_OK;
    gd_device device;
    int i = 0;

    if (attrs == NULL || !isfinite(attrs->scale) || attrs->scale <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "amp_clip_grad_norm max_norm must be finite and positive");
    }
    if (n_inputs < 3) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "amp_clip_grad_norm needs scale, found_inf, and gradients");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (inputs[0]->dtype != GD_DTYPE_F32 || inputs[0]->ndim != 0 ||
        inputs[1]->dtype != GD_DTYPE_I32 || inputs[1]->ndim != 0) {
        return _gd_error(GD_ERR_DTYPE,
                         "amp_clip_grad_norm expects F32 scale and I32 found_inf");
    }
    device = inputs[0]->device;
    if (!gd_device_equal(device, inputs[1]->device)) {
        return _gd_error(GD_ERR_DEVICE,
                         "amp_clip_grad_norm inputs must share device");
    }
    for (i = 2; i < n_inputs; ++i) {
        if (inputs[i]->dtype != GD_DTYPE_F32) {
            return _gd_error(GD_ERR_DTYPE,
                             "amp_clip_grad_norm requires F32 gradients");
        }
        if (!gd_device_equal(device, inputs[i]->device)) {
            return _gd_error(GD_ERR_DEVICE,
                             "amp_clip_grad_norm inputs must share device");
        }
    }
    return gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 0, NULL, &outputs[0]);
}

const _gd_op_def _gd_opdef_amp_clip_grad_norm = {
    .kind = _GD_OP_AMP_CLIP_GRAD_NORM,
    .name = "amp_clip_grad_norm",
    .min_inputs = 3,
    .max_inputs = _GD_OP_MAX_INPUTS,
    .n_outputs = 1,
    .flags = GD_OPF_MUTATES,
    .meta = amp_clip_grad_norm_meta,
};
