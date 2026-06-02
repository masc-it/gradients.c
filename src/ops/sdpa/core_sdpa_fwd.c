#include "../op_impl.h"
#include "../meta_common.h"

#include <limits.h>
#include <math.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

static gd_status sdpa_check_bias_desc(const gd_tensor_desc *bias,
                                      const gd_tensor_desc *out_desc,
                                      const gd_tensor_desc *k_desc)
{
    int64_t want[4];
    int i = 0;

    if (bias->dtype != out_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa bias must share dtype with q");
    }
    if (!gd_device_equal(bias->device, out_desc->device)) {
        return _gd_error(GD_ERR_DEVICE, "sdpa bias must share a device with q");
    }
    if (bias->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE, "sdpa bias must be 4D [B,Hq,Tq,Tk] (broadcastable)");
    }
    want[0] = out_desc->sizes[0];
    want[1] = out_desc->sizes[2];
    want[2] = out_desc->sizes[1];
    want[3] = k_desc->sizes[1];
    for (i = 0; i < 4; ++i) {
        if (bias->sizes[i] != 1 && bias->sizes[i] != want[i]) {
            return _gd_error(GD_ERR_SHAPE, "sdpa bias is not broadcastable to [B,Hq,Tq,Tk]");
        }
    }
    return GD_OK;
}

static gd_status sdpa_meta(const gd_tensor_desc *const *inputs,
                           int n_inputs,
                           _gd_op_attrs *attrs,
                           gd_tensor_desc *outputs,
                           int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *q = inputs[0];
    const gd_tensor_desc *k = inputs[1];
    int head_dim = 0;

    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_sdpa(inputs[0], inputs[1], inputs[2], &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    if (n_inputs == 4) {
        status = sdpa_check_bias_desc(inputs[3], &outputs[0], k);
        if (status != GD_OK) {
            return status;
        }
    }
    if (q->sizes[3] > (int64_t)INT_MAX || q->sizes[2] > (int64_t)INT_MAX ||
        k->sizes[2] > (int64_t)INT_MAX) {
        return _gd_error(GD_ERR_SHAPE, "sdpa dimension exceeds int range");
    }
    head_dim = (int)q->sizes[3];
    if (attrs->prefix_len < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa prefix_len must be non-negative");
    }
    if ((int64_t)attrs->prefix_len > k->sizes[1]) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa prefix_len exceeds key sequence length");
    }
    if (attrs->sliding_window < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa sliding_window must be non-negative");
    }
    if (attrs->prefix_len > 0 && !attrs->causal) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa prefix_len requires causal=true");
    }
    attrs->head_dim = head_dim;
    attrs->n_q_heads = (int)q->sizes[2];
    attrs->n_kv_heads = (int)k->sizes[2];
    if (attrs->attn_scale <= 0.0F) {
        attrs->attn_scale = (float)(1.0 / sqrt((double)head_dim));
    }
    attrs->has_bias = n_inputs == 4;
    return GD_OK;
}

const _gd_op_def _gd_opdef_sdpa = {
    .kind = _GD_OP_SDPA,
    .name = "sdpa",
    .min_inputs = 3,
    .max_inputs = 4,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = sdpa_meta,
};

gd_status gd_sdpa(gd_context *ctx,
                  gd_tensor *q,
                  gd_tensor *k,
                  gd_tensor *v,
                  gd_tensor *bias,
                  const gd_sdpa_config *config,
                  gd_tensor **out)
{
    gd_tensor *inputs[4];
    _gd_op_attrs attrs = {0};
    int n_inputs = 3;

    if (q == NULL || k == NULL || v == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_sdpa argument is NULL");
    }
    *out = NULL;
    attrs.attn_scale = (config != NULL) ? config->scale : 0.0F;
    attrs.causal = (config != NULL && config->causal) ? 1 : 0;
    attrs.sliding_window = (config != NULL) ? config->sliding_window : 0;
    attrs.prefix_len = (config != NULL) ? config->prefix_len : 0;
    inputs[0] = q;
    inputs[1] = k;
    inputs[2] = v;
    if (bias != NULL) {
        inputs[3] = bias;
        n_inputs = 4;
    }
    return _gd_emit_checked(ctx, _GD_OP_SDPA, inputs, n_inputs, &attrs, out, 1);
}
