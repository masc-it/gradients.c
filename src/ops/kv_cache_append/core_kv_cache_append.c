#include <gradients/ops.h>

#include "../op_common.h"

#include <limits.h>
#include <string.h>

static bool gd_kv_cache_append_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static gd_status gd_kv_cache_append_validate(gd_context *ctx,
                                             const gd_tensor *k_cache,
                                             const gd_tensor *v_cache,
                                             int32_t cache_pos,
                                             const gd_tensor *k_new,
                                             const gd_tensor *v_new,
                                             gd_backend_kv_cache_append_args *args)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || k_cache == NULL || v_cache == NULL || k_new == NULL || v_new == NULL ||
        args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, k_cache);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, v_cache);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, k_new);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, v_new);
    if (st != GD_OK) {
        return st;
    }
    if (gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN || k_cache->requires_grad ||
        v_cache->requires_grad || k_new->requires_grad || v_new->requires_grad) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_at is inference/eval only");
    }
    if (!gd_kv_cache_append_dtype_supported(k_cache->dtype) || v_cache->dtype != k_cache->dtype ||
        k_new->dtype != k_cache->dtype || v_new->dtype != k_cache->dtype) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_at requires matching f16/f32 tensors");
    }
    if (k_cache->rank != 4U || v_cache->rank != 4U || k_new->rank != 4U || v_new->rank != 4U) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_at expects [B,T,Hkv,Dh] tensors");
    }
    for (i = 0U; i < 4U; ++i) {
        if (k_cache->shape[i] != v_cache->shape[i]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append_at cache K/V shapes must match");
        }
        if (k_new->shape[i] != v_new->shape[i]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append_at new K/V shapes must match");
        }
    }
    if (k_cache->shape[0] != k_new->shape[0] || k_cache->shape[2] != k_new->shape[2] ||
        k_cache->shape[3] != k_new->shape[3] || k_cache->shape[1] <= 0 || k_new->shape[1] <= 0) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_at cache/new shape mismatch");
    }
    if (cache_pos < 0 || (int64_t)cache_pos > k_cache->shape[1] ||
        k_new->shape[1] > k_cache->shape[1] - (int64_t)cache_pos) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_at cache_pos/Tnew exceed cache length");
    }
    if (k_cache->shape[0] > (int64_t)INT_MAX || k_cache->shape[1] > (int64_t)INT_MAX ||
        k_new->shape[1] > (int64_t)INT_MAX || k_cache->shape[2] > (int64_t)INT_MAX ||
        k_cache->shape[3] > (int64_t)INT_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_at dimensions exceed kernel limits");
    }
    if (!gd_tensor_is_contiguous(k_cache) || !gd_tensor_is_contiguous(v_cache) ||
        !gd_tensor_is_contiguous(k_new) || !gd_tensor_is_contiguous(v_new)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_at requires contiguous tensors");
    }
    memset(args, 0, sizeof(*args));
    args->cache_pos = cache_pos;
    return GD_OK;
}

gd_status gd_kv_cache_append_at(gd_context *ctx,
                                gd_tensor *k_cache,
                                gd_tensor *v_cache,
                                int32_t cache_pos,
                                const gd_tensor *k_new,
                                const gd_tensor *v_new)
{
    gd_status st;
    gd_backend_kv_cache_append_args args;
    gd_backend_tensor_view kcv;
    gd_backend_tensor_view vcv;
    gd_backend_tensor_view knv;
    gd_backend_tensor_view vnv;
    if (ctx == NULL || k_cache == NULL || v_cache == NULL || k_new == NULL || v_new == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_kv_cache_append_validate(ctx, k_cache, v_cache, cache_pos, k_new, v_new, &args);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(k_cache, &kcv) ||
        !gd_op_tensor_view_from_tensor(v_cache, &vcv) ||
        !gd_op_tensor_view_from_tensor(k_new, &knv) ||
        !gd_op_tensor_view_from_tensor(v_new, &vnv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_at invalid tensor view");
    }
    st = gd_backend_kv_cache_append_at(gd_context_backend(ctx), &kcv, &vcv, &knv, &vnv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend kv_cache_append_at failed");
    }
    k_cache->version += 1U;
    if (k_cache->version == 0U) {
        k_cache->version = 1U;
    }
    v_cache->version += 1U;
    if (v_cache->version == 0U) {
        v_cache->version = 1U;
    }
    return GD_OK;
}
