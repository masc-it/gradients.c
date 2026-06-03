#include "../op_impl.h"
#include "../meta_common.h"

#include <limits.h>
#include <math.h>

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#define GD_SDPA_DECODE_N_INPUTS 4

static gd_status sdpa_decode_meta(const gd_tensor_desc *const *inputs,
                                  int n_inputs,
                                  _gd_op_attrs *attrs,
                                  gd_tensor_desc *outputs,
                                  int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *q = NULL;
    const gd_tensor_desc *k = NULL;
    const gd_tensor_desc *v = NULL;
    const gd_tensor_desc *pos = NULL;
    int head_dim = 0;
    int i = 0;

    status = _gd_meta_set_output_count(1, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (inputs == NULL || n_inputs != GD_SDPA_DECODE_N_INPUTS || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode meta arguments are NULL");
    }
    q = inputs[0];
    k = inputs[1];
    v = inputs[2];
    pos = inputs[3];
    if (q == NULL || k == NULL || v == NULL || pos == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode input desc is NULL");
    }
    if (q->dtype != GD_DTYPE_F32 && q->dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_decode supports F32/F16 q/k/v only");
    }
    if (q->dtype != k->dtype || q->dtype != v->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_decode q/k/v must share dtype");
    }
    if (pos->dtype != GD_DTYPE_I32 && pos->dtype != GD_DTYPE_I64) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_decode cache_pos must be I32 or I64");
    }
    if (pos->ndim != 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode cache_pos must be scalar");
    }
    status = _gd_meta_require_same_device(q, k, "sdpa_decode inputs must share a device");
    if (status != GD_OK) { return status; }
    status = _gd_meta_require_same_device(q, v, "sdpa_decode inputs must share a device");
    if (status != GD_OK) { return status; }
    status = _gd_meta_require_same_device(q, pos, "sdpa_decode inputs must share a device");
    if (status != GD_OK) { return status; }
    if (q->ndim != 4 || k->ndim != 4 || v->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode q/cache tensors must be 4D [B,T,H,Dh]");
    }
    for (i = 0; i < 4; ++i) {
        if (k->sizes[i] != v->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "sdpa_decode k/v cache shapes must match");
        }
    }
    if (q->sizes[0] != k->sizes[0]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode batch mismatch");
    }
    if (q->sizes[1] <= 0 || k->sizes[1] <= 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode sequence dimensions must be positive");
    }
    if (q->sizes[3] != k->sizes[3]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode head_dim mismatch");
    }
    if (k->sizes[2] == 0 || q->sizes[2] % k->sizes[2] != 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode query heads must be a multiple of kv heads");
    }
    if (q->sizes[3] > (int64_t)INT_MAX || q->sizes[2] > (int64_t)INT_MAX ||
        k->sizes[2] > (int64_t)INT_MAX || q->sizes[1] > (int64_t)INT_MAX ||
        k->sizes[1] > (int64_t)INT_MAX) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_decode dimension exceeds int range");
    }
    if (attrs->prefix_len < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode prefix_len must be non-negative");
    }
    if ((int64_t)attrs->prefix_len > k->sizes[1]) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode prefix_len exceeds cache length");
    }
    if (attrs->sliding_window < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode sliding_window must be non-negative");
    }
    head_dim = (int)q->sizes[3];
    attrs->head_dim = head_dim;
    attrs->n_q_heads = (int)q->sizes[2];
    attrs->n_kv_heads = (int)k->sizes[2];
    attrs->causal = 1;
    if (attrs->attn_scale <= 0.0F) {
        attrs->attn_scale = (float)(1.0 / sqrt((double)head_dim));
    }
    return gd_tensor_desc_contiguous(q->dtype, q->device, 4, q->sizes, &outputs[0]);
}

const _gd_op_def _gd_opdef_sdpa_decode = {
    .kind = _GD_OP_SDPA_DECODE,
    .name = "sdpa_decode",
    .min_inputs = GD_SDPA_DECODE_N_INPUTS,
    .max_inputs = GD_SDPA_DECODE_N_INPUTS,
    .n_outputs = 1,
    .flags = GD_OPF_PUBLIC,
    .meta = sdpa_decode_meta,
};

gd_status gd_sdpa_decode(gd_context *ctx,
                         gd_tensor *q,
                         gd_tensor *k_cache,
                         gd_tensor *v_cache,
                         gd_tensor *cache_pos,
                         const gd_sdpa_decode_config *config,
                         gd_tensor **out)
{
    gd_tensor *inputs[GD_SDPA_DECODE_N_INPUTS] = {q, k_cache, v_cache, cache_pos};
    _gd_op_attrs attrs = {0};

    if (ctx == NULL || q == NULL || k_cache == NULL || v_cache == NULL ||
        cache_pos == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_sdpa_decode argument is NULL");
    }
    *out = NULL;
    attrs.attn_scale = config != NULL ? config->scale : 0.0F;
    attrs.sliding_window = config != NULL ? config->sliding_window : 0;
    attrs.prefix_len = config != NULL ? config->prefix_len : 0;
    attrs.causal = 1;
    return _gd_emit_checked(ctx, _GD_OP_SDPA_DECODE, inputs, GD_SDPA_DECODE_N_INPUTS,
                            &attrs, out, 1);
}
