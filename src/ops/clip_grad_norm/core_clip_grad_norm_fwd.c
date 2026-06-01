#include "../op_impl.h"
#include "../meta_common.h"

#include <math.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../core/tensor_internal.h"
#include "../../graph/graph_internal.h"

static gd_status clip_grad_norm_meta(const gd_tensor_desc *const *inputs,
                                     int n_inputs,
                                     _gd_op_attrs *attrs,
                                     gd_tensor_desc *outputs,
                                     int *n_outputs)
{
    gd_status status = GD_OK;
    gd_device device = inputs[0]->device;
    int i = 0;

    if (attrs == NULL || !isfinite(attrs->scale) || attrs->scale <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "clip_grad_norm max_norm must be finite and positive");
    }
    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i]->dtype != GD_DTYPE_F32) {
            return _gd_error(GD_ERR_DTYPE, "clip_grad_norm requires F32 gradients");
        }
        if (!gd_device_equal(device, inputs[i]->device)) {
            return _gd_error(GD_ERR_DEVICE, "clip_grad_norm gradients must share device");
        }
    }
    return gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 0, NULL, &outputs[0]);
}

const _gd_op_def _gd_opdef_clip_grad_norm = {
    .kind = _GD_OP_CLIP_GRAD_NORM,
    .name = "clip_grad_norm",
    .min_inputs = 1,
    .max_inputs = 256,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_MUTATES,
    .meta = clip_grad_norm_meta,
};

gd_status gd_clip_grad_norm(gd_context *ctx,
                            gd_tensor **params,
                            int n_params,
                            float max_norm,
                            gd_tensor **norm_out)
{
    gd_status status = GD_OK;
    gd_tensor *grads[_GD_OP_MAX_INPUTS];
    _gd_op_attrs attrs = {0};
    gd_tensor *norm = NULL;
    int i = 0;

    if (ctx == NULL || params == NULL || n_params <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_clip_grad_norm argument is invalid");
    }
    if (n_params > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_clip_grad_norm too many parameters");
    }
    if (!isfinite(max_norm) || max_norm <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "clip_grad_norm max_norm must be finite and positive");
    }
    if (norm_out != NULL) {
        *norm_out = NULL;
    }
    for (i = 0; i < n_params; ++i) {
        gd_tensor *grad = NULL;
        int missing_grad = 0;

        if (params[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "clip_grad_norm parameter is NULL");
        }
        if (!gd_tensor_requires_grad(params[i])) {
            return _gd_error(GD_ERR_INVALID_STATE,
                             "clip_grad_norm parameter must require grad");
        }
        status = gd_tensor_grad(params[i], &grad);
        if (status != GD_OK) {
            return status;
        }
        if (grad == NULL) {
            missing_grad = 1;
            status = _gd_tensor_ensure_grad(ctx, params[i], &grad);
            if (status != GD_OK) {
                return status;
            }
        }
        if (missing_grad != 0) {
            status = _gd_tensor_zero(grad);
            if (status != GD_OK) {
                return status;
            }
        }
        grads[i] = grad;
    }

    attrs.scale = max_norm;
    attrs.eps = 1e-6F;
    status = _gd_emit_checked(ctx, _GD_OP_CLIP_GRAD_NORM, grads, n_params,
                              &attrs, &norm, 1);
    if (status != GD_OK) {
        return status;
    }
    if (norm_out != NULL) {
        *norm_out = norm;
    } else {
        gd_tensor_release(norm);
    }
    return GD_OK;
}
