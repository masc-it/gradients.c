#include "../op_impl.h"
#include "../meta_common.h"

#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#define GD_KV_CACHE_APPEND_N_INPUTS 5

static gd_status kv_cache_append_meta(const gd_tensor_desc *const *inputs,
                                      int n_inputs,
                                      _gd_op_attrs *attrs,
                                      gd_tensor_desc *outputs,
                                      int *n_outputs)
{
    const gd_tensor_desc *k_cache = NULL;
    const gd_tensor_desc *v_cache = NULL;
    const gd_tensor_desc *pos = NULL;
    const gd_tensor_desc *k_new = NULL;
    const gd_tensor_desc *v_new = NULL;
    gd_status status = GD_OK;
    int i = 0;

    (void)attrs;
    (void)outputs;
    status = _gd_meta_set_output_count(0, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (inputs == NULL || n_inputs != GD_KV_CACHE_APPEND_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "kv_cache_append input count mismatch");
    }
    k_cache = inputs[0];
    v_cache = inputs[1];
    pos = inputs[2];
    k_new = inputs[3];
    v_new = inputs[4];
    if (k_cache == NULL || v_cache == NULL || pos == NULL || k_new == NULL || v_new == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "kv_cache_append input desc is NULL");
    }
    if (k_cache->dtype != GD_DTYPE_F32 && k_cache->dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_DTYPE, "kv_cache_append supports F32/F16 cache tensors only");
    }
    if (k_cache->dtype != v_cache->dtype || k_cache->dtype != k_new->dtype ||
        k_cache->dtype != v_new->dtype) {
        return _gd_error(GD_ERR_DTYPE, "kv_cache_append k/v dtypes must match");
    }
    if (pos->dtype != GD_DTYPE_I32 && pos->dtype != GD_DTYPE_I64) {
        return _gd_error(GD_ERR_DTYPE, "kv_cache_append cache_pos must be I32 or I64");
    }
    if (pos->ndim != 0) {
        return _gd_error(GD_ERR_SHAPE, "kv_cache_append cache_pos must be scalar");
    }
    status = _gd_meta_require_same_device(k_cache, v_cache,
                                          "kv_cache_append inputs must share a device");
    if (status != GD_OK) { return status; }
    status = _gd_meta_require_same_device(k_cache, pos,
                                          "kv_cache_append inputs must share a device");
    if (status != GD_OK) { return status; }
    status = _gd_meta_require_same_device(k_cache, k_new,
                                          "kv_cache_append inputs must share a device");
    if (status != GD_OK) { return status; }
    status = _gd_meta_require_same_device(k_cache, v_new,
                                          "kv_cache_append inputs must share a device");
    if (status != GD_OK) { return status; }
    if (k_cache->ndim != 4 || v_cache->ndim != 4 || k_new->ndim != 4 || v_new->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE,
                         "kv_cache_append tensors must be 4D [B,T,Hkv,Dh]");
    }
    if (k_cache->layout != GD_LAYOUT_CONTIGUOUS || v_cache->layout != GD_LAYOUT_CONTIGUOUS ||
        k_new->layout != GD_LAYOUT_CONTIGUOUS || v_new->layout != GD_LAYOUT_CONTIGUOUS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "kv_cache_append requires contiguous tensors");
    }
    for (i = 0; i < 4; ++i) {
        if (k_cache->sizes[i] != v_cache->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "kv_cache_append cache k/v shapes must match");
        }
        if (k_new->sizes[i] != v_new->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "kv_cache_append new k/v shapes must match");
        }
    }
    if (k_cache->sizes[0] != k_new->sizes[0] ||
        k_cache->sizes[2] != k_new->sizes[2] ||
        k_cache->sizes[3] != k_new->sizes[3]) {
        return _gd_error(GD_ERR_SHAPE,
                         "kv_cache_append cache/new shapes must match on B,Hkv,Dh");
    }
    if (k_new->sizes[1] <= 0 || k_cache->sizes[1] <= 0) {
        return _gd_error(GD_ERR_SHAPE, "kv_cache_append sequence dimensions must be positive");
    }
    return GD_OK;
}

const _gd_op_def _gd_opdef_kv_cache_append = {
    .kind = _GD_OP_KV_CACHE_APPEND,
    .name = "kv_cache_append",
    .min_inputs = GD_KV_CACHE_APPEND_N_INPUTS,
    .max_inputs = GD_KV_CACHE_APPEND_N_INPUTS,
    .n_outputs = 0,
    .flags = GD_OPF_PUBLIC | GD_OPF_MUTATES,
    .meta = kv_cache_append_meta,
};

gd_status gd_kv_cache_append(gd_context *ctx,
                             gd_tensor *k_cache,
                             gd_tensor *v_cache,
                             gd_tensor *cache_pos,
                             gd_tensor *k_new,
                             gd_tensor *v_new)
{
    gd_tensor *inputs[GD_KV_CACHE_APPEND_N_INPUTS] = {k_cache, v_cache, cache_pos, k_new, v_new};

    if (ctx == NULL || k_cache == NULL || v_cache == NULL || cache_pos == NULL ||
        k_new == NULL || v_new == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_kv_cache_append argument is NULL");
    }
    return _gd_emit_checked(ctx, _GD_OP_KV_CACHE_APPEND, inputs,
                            GD_KV_CACHE_APPEND_N_INPUTS, NULL, NULL, 0);
}
