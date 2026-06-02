#include "../op_impl.h"
#include "../meta_common.h"
#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#define GD_RMS_NORM_QKV_N_INPUTS 5
#define GD_RMS_NORM_QKV_N_OUTPUTS 4

static gd_status require_same_dtype_device(const gd_tensor_desc *x,
                                           const gd_tensor_desc *y,
                                           const char *name)
{
    gd_status status = GD_OK;

    if (y == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv input desc is NULL");
    }
    if (x->dtype != y->dtype) {
        return _gd_error(GD_ERR_DTYPE, name);
    }
    status = _gd_meta_require_same_device(x, y, "rms_norm_qkv inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (x->layout != GD_LAYOUT_CONTIGUOUS || y->layout != GD_LAYOUT_CONTIGUOUS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "rms_norm_qkv requires contiguous tensors");
    }
    return GD_OK;
}

static gd_status set_projected_desc(const gd_tensor_desc *x,
                                    int64_t out_features,
                                    gd_tensor_desc *out)
{
    int64_t sizes[GD_MAX_DIMS];
    int i = 0;

    for (i = 0; i < x->ndim - 1; ++i) {
        sizes[i] = x->sizes[i];
    }
    sizes[x->ndim - 1] = out_features;
    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, sizes, out);
}

static gd_status rms_norm_qkv_meta(const gd_tensor_desc *const *inputs,
                                   int n_inputs,
                                   _gd_op_attrs *attrs,
                                   gd_tensor_desc *outputs,
                                   int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *x = NULL;
    const gd_tensor_desc *weight = NULL;
    const gd_tensor_desc *wq = NULL;
    const gd_tensor_desc *wk = NULL;
    const gd_tensor_desc *wv = NULL;
    int64_t d = 0;

    if (inputs == NULL || attrs == NULL || outputs == NULL || n_outputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv meta arguments are NULL");
    }
    if (n_inputs != GD_RMS_NORM_QKV_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv input count mismatch");
    }
    x = inputs[0];
    weight = inputs[1];
    wq = inputs[2];
    wk = inputs[3];
    wv = inputs[4];
    if (x == NULL || weight == NULL || wq == NULL || wk == NULL || wv == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv input desc is NULL");
    }
    if (attrs->eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv eps must be positive");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "rms_norm_qkv requires floating-point inputs");
    }
    if (x->ndim < 1) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm_qkv input must have rank >= 1");
    }
    if (x->layout != GD_LAYOUT_CONTIGUOUS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "rms_norm_qkv requires contiguous tensors");
    }
    d = x->sizes[x->ndim - 1];
    status = require_same_dtype_device(x, weight,
                                       "rms_norm_qkv norm weight dtype must match input");
    if (status != GD_OK) { return status; }
    status = require_same_dtype_device(x, wq, "rms_norm_qkv q weight dtype must match input");
    if (status != GD_OK) { return status; }
    status = require_same_dtype_device(x, wk, "rms_norm_qkv k weight dtype must match input");
    if (status != GD_OK) { return status; }
    status = require_same_dtype_device(x, wv, "rms_norm_qkv v weight dtype must match input");
    if (status != GD_OK) { return status; }
    if (weight->ndim != 1 || weight->sizes[0] != d) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm_qkv norm weight must be [last_dim]");
    }
    if (wq->ndim != 2 || wk->ndim != 2 || wv->ndim != 2 ||
        wq->sizes[0] != d || wk->sizes[0] != d || wv->sizes[0] != d) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm_qkv projection weights must be [last_dim,out]");
    }
    status = _gd_meta_set_output_count(GD_RMS_NORM_QKV_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    outputs[0] = *x; /* normalized hidden, kept for backward dW without recompute */
    status = set_projected_desc(x, wq->sizes[1], &outputs[1]);
    if (status != GD_OK) { return status; }
    status = set_projected_desc(x, wk->sizes[1], &outputs[2]);
    if (status != GD_OK) { return status; }
    return set_projected_desc(x, wv->sizes[1], &outputs[3]);
}

const _gd_op_def _gd_opdef_rms_norm_qkv = {
    .kind = _GD_OP_RMS_NORM_QKV,
    .name = "rms_norm_qkv",
    .min_inputs = GD_RMS_NORM_QKV_N_INPUTS,
    .max_inputs = GD_RMS_NORM_QKV_N_INPUTS,
    .n_outputs = GD_RMS_NORM_QKV_N_OUTPUTS,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = rms_norm_qkv_meta,
};

gd_status gd_rms_norm_qkv(gd_context *ctx,
                          gd_tensor *x,
                          gd_tensor *weight,
                          gd_tensor *wq,
                          gd_tensor *wk,
                          gd_tensor *wv,
                          float eps,
                          gd_tensor **norm,
                          gd_tensor **q,
                          gd_tensor **k,
                          gd_tensor **v)
{
    gd_tensor *inputs[GD_RMS_NORM_QKV_N_INPUTS] = {x, weight, wq, wk, wv};
    gd_tensor *outputs[GD_RMS_NORM_QKV_N_OUTPUTS] = {0};
    _gd_op_attrs attrs = {0};
    gd_status status = GD_OK;

    if (ctx == NULL || x == NULL || weight == NULL || wq == NULL || wk == NULL ||
        wv == NULL || norm == NULL || q == NULL || k == NULL || v == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_rms_norm_qkv argument is NULL");
    }
    if (eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm_qkv eps must be positive");
    }
    *norm = NULL;
    *q = NULL;
    *k = NULL;
    *v = NULL;
    attrs.eps = eps;
    attrs.compute = gd_context_compute_policy(ctx);
    status = _gd_emit_checked(ctx, _GD_OP_RMS_NORM_QKV, inputs, GD_RMS_NORM_QKV_N_INPUTS,
                              &attrs, outputs, GD_RMS_NORM_QKV_N_OUTPUTS);
    if (status != GD_OK) {
        return status;
    }
    *norm = outputs[0];
    *q = outputs[1];
    *k = outputs[2];
    *v = outputs[3];
    return GD_OK;
}
