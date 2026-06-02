#include "../op_impl.h"
#include "../meta_common.h"

#include <limits.h>
#include <math.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#define GD_SDPA_VARLEN_N_INPUTS 4
#define GD_SDPA_VARLEN_N_OUTPUTS 1

static gd_status sdpa_varlen_meta(const gd_tensor_desc *const *inputs,
                                  int n_inputs,
                                  _gd_op_attrs *attrs,
                                  gd_tensor_desc *outputs,
                                  int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *q = NULL;
    const gd_tensor_desc *k = NULL;
    const gd_tensor_desc *v = NULL;
    const gd_tensor_desc *cu = NULL;
    int64_t sizes[3];
    int head_dim = 0;

    if (inputs == NULL || outputs == NULL || n_outputs == NULL || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen meta arguments are NULL");
    }
    if (n_inputs != GD_SDPA_VARLEN_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen input count mismatch");
    }
    q = inputs[0];
    k = inputs[1];
    v = inputs[2];
    cu = inputs[3];
    if (q == NULL || k == NULL || v == NULL || cu == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen input desc is NULL");
    }
    status = _gd_meta_set_output_count(GD_SDPA_VARLEN_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (!_gd_dtype_is_float(q->dtype) || q->dtype == GD_DTYPE_BF16) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_varlen supports F32/F16 q/k/v only");
    }
    if (k->dtype != q->dtype || v->dtype != q->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_varlen requires matching q/k/v dtype");
    }
    if (!gd_device_equal(q->device, k->device) || !gd_device_equal(q->device, v->device) ||
        !gd_device_equal(q->device, cu->device)) {
        return _gd_error(GD_ERR_DEVICE, "sdpa_varlen inputs must share a device");
    }
    if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen q/k/v must be 3D [N,H,Dh]");
    }
    if (cu->ndim != 1 || cu->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen cu_seqlens must be I32 [B+1]");
    }
    if (cu->sizes[0] < 2) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen cu_seqlens must contain at least two offsets");
    }
    if (q->sizes[0] != k->sizes[0] || q->sizes[0] != v->sizes[0]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen q/k/v must share packed token count");
    }
    if (k->sizes[2] != q->sizes[2] || v->sizes[2] != q->sizes[2]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen q/k/v head_dim mismatch");
    }
    if (v->sizes[1] != k->sizes[1]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen k/v head count mismatch");
    }
    if (q->sizes[1] <= 0 || k->sizes[1] <= 0 || q->sizes[2] <= 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen head dimensions must be positive");
    }
    if ((q->sizes[1] % k->sizes[1]) != 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen Hq must be a multiple of Hkv");
    }
    if (q->sizes[0] > (int64_t)INT_MAX || q->sizes[1] > (int64_t)INT_MAX ||
        k->sizes[1] > (int64_t)INT_MAX || q->sizes[2] > (int64_t)INT_MAX ||
        cu->sizes[0] - 1 > (int64_t)INT_MAX) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen dimension exceeds int range");
    }
    if (attrs->prefix_len < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen prefix_len must be non-negative");
    }
    if (attrs->sliding_window < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen sliding_window must be non-negative");
    }
    if (attrs->prefix_len > 0 && !attrs->causal) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen prefix_len requires causal=true");
    }
    if (attrs->max_seqlen < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen max_seqlen must be non-negative");
    }
    if (attrs->max_seqlen == 0) {
        attrs->max_seqlen = (int)q->sizes[0];
    }
    if ((int64_t)attrs->prefix_len > (int64_t)attrs->max_seqlen) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen prefix_len exceeds max_seqlen");
    }
    head_dim = (int)q->sizes[2];
    attrs->head_dim = head_dim;
    attrs->n_q_heads = (int)q->sizes[1];
    attrs->n_kv_heads = (int)k->sizes[1];
    if (attrs->attn_scale <= 0.0F) {
        attrs->attn_scale = (float)(1.0 / sqrt((double)head_dim));
    }
    sizes[0] = q->sizes[0];
    sizes[1] = q->sizes[1];
    sizes[2] = q->sizes[2];
    return gd_tensor_desc_contiguous(q->dtype, q->device, 3, sizes, &outputs[0]);
}

const _gd_op_def _gd_opdef_sdpa_varlen = {
    .kind = _GD_OP_SDPA_VARLEN,
    .name = "sdpa_varlen",
    .min_inputs = GD_SDPA_VARLEN_N_INPUTS,
    .max_inputs = GD_SDPA_VARLEN_N_INPUTS,
    .n_outputs = GD_SDPA_VARLEN_N_OUTPUTS,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = sdpa_varlen_meta,
};

gd_status gd_sdpa_varlen(gd_context *ctx,
                         gd_tensor *q,
                         gd_tensor *k,
                         gd_tensor *v,
                         gd_tensor *cu_seqlens,
                         const gd_sdpa_varlen_config *config,
                         gd_tensor **out)
{
    gd_tensor *inputs[GD_SDPA_VARLEN_N_INPUTS] = {q, k, v, cu_seqlens};
    _gd_op_attrs attrs = {0};

    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_sdpa_varlen argument is NULL");
    }
    *out = NULL;
    attrs.attn_scale = (config != NULL) ? config->scale : 0.0F;
    attrs.causal = (config != NULL && config->causal) ? 1 : 0;
    attrs.sliding_window = (config != NULL) ? config->sliding_window : 0;
    attrs.prefix_len = (config != NULL) ? config->prefix_len : 0;
    attrs.max_seqlen = (config != NULL) ? config->max_seqlen : 0;
    return _gd_emit_checked(ctx, _GD_OP_SDPA_VARLEN, inputs, GD_SDPA_VARLEN_N_INPUTS,
                            &attrs, out, GD_SDPA_VARLEN_N_OUTPUTS);
}
