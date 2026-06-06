#include <gradients/ops.h>

#include "../op_common.h"

#include <limits.h>
#include <math.h>
#include <string.h>

#define GD_SDPA_DECODE_MAX_HEAD_DIM 64

static gd_status gd_sdpa_decode_validate(gd_context *ctx,
                                         const gd_tensor *q,
                                         const gd_tensor *k_cache,
                                         const gd_tensor *v_cache,
                                         const gd_tensor *cache_pos,
                                         const gd_sdpa_decode_config *config,
                                         gd_backend_sdpa_decode_args *args)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || q == NULL || k_cache == NULL || v_cache == NULL || cache_pos == NULL ||
        args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, q);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, k_cache);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, v_cache);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, cache_pos);
    if (st != GD_OK) {
        return st;
    }
    if ((q->dtype != GD_DTYPE_F16 && q->dtype != GD_DTYPE_F32) ||
        k_cache->dtype != q->dtype || v_cache->dtype != q->dtype) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sdpa_decode requires matching f16/f32 q/k/v");
    }
    if (cache_pos->dtype != GD_DTYPE_I32 || cache_pos->rank != 0U) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sdpa_decode cache_pos must be an i32 scalar");
    }
    if (q->rank != 4U || k_cache->rank != 4U || v_cache->rank != 4U) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_decode expects q/cache tensors [B,T,H,Dh]");
    }
    for (i = 0U; i < 4U; ++i) {
        if (k_cache->shape[i] != v_cache->shape[i]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "sdpa_decode k/v cache shapes must match");
        }
    }
    if (q->shape[0] != k_cache->shape[0] || q->shape[1] <= 0 || k_cache->shape[1] <= 0 ||
        q->shape[2] <= 0 || k_cache->shape[2] <= 0 || q->shape[3] <= 0 ||
        q->shape[3] != k_cache->shape[3] || (q->shape[2] % k_cache->shape[2]) != 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_decode shape mismatch");
    }
    if (q->shape[0] > (int64_t)INT_MAX || q->shape[1] > (int64_t)INT_MAX ||
        k_cache->shape[1] > (int64_t)INT_MAX || q->shape[2] > (int64_t)INT_MAX ||
        k_cache->shape[2] > (int64_t)INT_MAX || q->shape[3] > GD_SDPA_DECODE_MAX_HEAD_DIM) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sdpa_decode dimensions exceed Metal kernel limits");
    }
    if (!gd_tensor_is_contiguous(q) || !gd_tensor_is_contiguous(k_cache) ||
        !gd_tensor_is_contiguous(v_cache) || !gd_tensor_is_contiguous(cache_pos)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "sdpa_decode requires contiguous tensors");
    }
    memset(args, 0, sizeof(*args));
    args->scale = config != NULL ? config->scale : 0.0f;
    args->sliding_window = (uint32_t)(config != NULL ? config->sliding_window : 0);
    args->prefix_len = (uint32_t)(config != NULL ? config->prefix_len : 0);
    if ((config != NULL && (config->sliding_window < 0 || config->prefix_len < 0)) ||
        (config != NULL && config->prefix_len > k_cache->shape[1])) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_decode window/prefix must be non-negative and in range");
    }
    if (args->scale <= 0.0f) {
        args->scale = (float)(1.0 / sqrt((double)q->shape[3]));
    } else if (!isfinite(args->scale)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_decode scale must be finite");
    }
    return GD_OK;
}

gd_status gd_sdpa_decode(gd_context *ctx,
                         const gd_tensor *q,
                         const gd_tensor *k_cache,
                         const gd_tensor *v_cache,
                         const gd_tensor *cache_pos,
                         const gd_sdpa_decode_config *config,
                         gd_tensor *out)
{
    gd_status st;
    gd_backend_sdpa_decode_args args;
    gd_tensor y;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view vv;
    gd_backend_tensor_view pv;
    gd_backend_tensor_view yv;
    int64_t shape[4];
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || q == NULL || k_cache == NULL || v_cache == NULL || cache_pos == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sdpa_decode_validate(ctx, q, k_cache, v_cache, cache_pos, config, &args);
    if (st != GD_OK) {
        return st;
    }
    shape[0] = q->shape[0];
    shape[1] = q->shape[1];
    shape[2] = q->shape[2];
    shape[3] = q->shape[3];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, q->dtype, gd_shape_make(4U, shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(q, &qv) || !gd_op_tensor_view_from_tensor(k_cache, &kv) ||
        !gd_op_tensor_view_from_tensor(v_cache, &vv) || !gd_op_tensor_view_from_tensor(cache_pos, &pv) ||
        !gd_op_tensor_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_decode invalid tensor view");
    }
    st = gd_backend_sdpa_decode(gd_context_backend(ctx), &qv, &kv, &vv, &pv, &yv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sdpa_decode failed");
    }
    *out = y;
    return GD_OK;
}
