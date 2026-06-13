#include <gradients/ops.h>

#include "../op_common.h"

#include <limits.h>
#include <string.h>

static bool gd_kv_cache_append_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static gd_status gd_kv_cache_append_validate_common(gd_context *ctx,
                                                    const gd_tensor *k_cache,
                                                    const gd_tensor *v_cache,
                                                    const gd_tensor *k_new,
                                                    const gd_tensor *v_new,
                                                    bool packed)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || k_cache == NULL || v_cache == NULL || k_new == NULL || v_new == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, k_cache);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, v_cache);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, k_new);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, v_new);
    if (st != GD_OK) { return st; }
    if (gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN || k_cache->requires_grad ||
        v_cache->requires_grad || k_new->requires_grad || v_new->requires_grad) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append is inference/eval only");
    }
    if (!gd_kv_cache_append_dtype_supported(k_cache->dtype) || v_cache->dtype != k_cache->dtype ||
        k_new->dtype != k_cache->dtype || v_new->dtype != k_cache->dtype) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append requires matching f16/f32 tensors");
    }
    if (k_cache->rank != 4U || v_cache->rank != 4U ||
        k_new->rank != (packed ? 3U : 4U) || v_new->rank != (packed ? 3U : 4U)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    packed ? "kv_cache_append_packed expects cache [B,T,H,D], new [N,H,D]" :
                                             "kv_cache_append expects [B,T,H,D] tensors");
    }
    for (i = 0U; i < 4U; ++i) {
        if (k_cache->shape[i] != v_cache->shape[i]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append cache K/V shapes must match");
        }
    }
    if (packed) {
        for (i = 0U; i < 3U; ++i) {
            if (k_new->shape[i] != v_new->shape[i]) {
                return gd_context_set_error(ctx,
                                            GD_ERR_INVALID_ARGUMENT,
                                            "kv_cache_append_packed new K/V shapes must match");
            }
        }
        if (k_new->shape[0] <= 0 || k_cache->shape[0] <= 0 || k_cache->shape[1] <= 0 ||
            k_cache->shape[2] != k_new->shape[1] || k_cache->shape[3] != k_new->shape[2]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append_packed cache/new shape mismatch");
        }
    } else {
        for (i = 0U; i < 4U; ++i) {
            if (k_new->shape[i] != v_new->shape[i]) {
                return gd_context_set_error(ctx,
                                            GD_ERR_INVALID_ARGUMENT,
                                            "kv_cache_append new K/V shapes must match");
            }
        }
        if (k_cache->shape[0] != k_new->shape[0] || k_cache->shape[2] != k_new->shape[2] ||
            k_cache->shape[3] != k_new->shape[3] || k_cache->shape[1] <= 0 || k_new->shape[1] <= 0) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append cache/new shape mismatch");
        }
        if (k_new->shape[1] > k_cache->shape[1]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "kv_cache_append Tnew exceeds cache length");
        }
    }
    if (k_cache->shape[0] > (int64_t)INT_MAX || k_cache->shape[1] > (int64_t)INT_MAX ||
        k_cache->shape[2] > (int64_t)INT_MAX || k_cache->shape[3] > (int64_t)INT_MAX ||
        (packed ? k_new->shape[0] : k_new->shape[1]) > (int64_t)INT_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append dimensions exceed kernel limits");
    }
    if (!gd_tensor_is_contiguous(k_cache) || !gd_tensor_is_contiguous(v_cache) ||
        !gd_tensor_is_contiguous(k_new) || !gd_tensor_is_contiguous(v_new)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append requires contiguous tensors");
    }
    return GD_OK;
}

static gd_status gd_kv_cache_append_validate_positions_tensor(gd_context *ctx,
                                                              const gd_tensor *k_cache,
                                                              const gd_tensor *cache_pos,
                                                              const char *op_name)
{
    gd_status st;
    if (ctx == NULL || k_cache == NULL || cache_pos == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, cache_pos);
    if (st != GD_OK) {
        return st;
    }
    if (cache_pos->dtype != GD_DTYPE_I32 || cache_pos->rank != 1U ||
        cache_pos->shape[0] != k_cache->shape[0]) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    op_name != NULL ? op_name : "kv cache positions must be i32 [B]");
    }
    if (!gd_tensor_is_contiguous(cache_pos)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append requires contiguous cache positions");
    }
    return GD_OK;
}

static gd_status gd_kv_cache_append_validate_at(gd_context *ctx,
                                                const gd_tensor *k_cache,
                                                const gd_tensor *v_cache,
                                                int32_t cache_pos,
                                                const gd_tensor *k_new,
                                                const gd_tensor *v_new,
                                                gd_backend_kv_cache_append_args *args)
{
    gd_status st = gd_kv_cache_append_validate_common(ctx, k_cache, v_cache, k_new, v_new, false);
    if (st != GD_OK) {
        return st;
    }
    if (cache_pos < 0 || (int64_t)cache_pos > k_cache->shape[1] ||
        k_new->shape[1] > k_cache->shape[1] - (int64_t)cache_pos) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_at cache_pos/Tnew exceed cache length");
    }
    memset(args, 0, sizeof(*args));
    args->cache_pos = cache_pos;
    return GD_OK;
}

static gd_status gd_kv_cache_append_validate_positions(gd_context *ctx,
                                                       const gd_tensor *k_cache,
                                                       const gd_tensor *v_cache,
                                                       const gd_tensor *cache_pos,
                                                       const gd_tensor *k_new,
                                                       const gd_tensor *v_new,
                                                       gd_backend_kv_cache_append_args *args)
{
    gd_status st = gd_kv_cache_append_validate_common(ctx, k_cache, v_cache, k_new, v_new, false);
    if (st != GD_OK) {
        return st;
    }
    st = gd_kv_cache_append_validate_positions_tensor(ctx,
                                                      k_cache,
                                                      cache_pos,
                                                      "kv_cache_append_positions cache_pos must be i32 [B]");
    if (st != GD_OK) {
        return st;
    }
    memset(args, 0, sizeof(*args));
    args->cache_pos = -1;
    return GD_OK;
}

static gd_status gd_kv_cache_append_validate_packed(gd_context *ctx,
                                                    const gd_tensor *k_cache,
                                                    const gd_tensor *v_cache,
                                                    const gd_tensor *cache_pos,
                                                    const gd_tensor *cu_seqlens,
                                                    const gd_tensor *k_new,
                                                    const gd_tensor *v_new,
                                                    gd_backend_kv_cache_append_args *args)
{
    gd_status st = gd_kv_cache_append_validate_common(ctx, k_cache, v_cache, k_new, v_new, true);
    if (st != GD_OK) {
        return st;
    }
    st = gd_kv_cache_append_validate_positions_tensor(ctx,
                                                      k_cache,
                                                      cache_pos,
                                                      "kv_cache_append_packed cache_pos must be i32 [B]");
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, cu_seqlens);
    if (st != GD_OK) {
        return st;
    }
    if (cu_seqlens->dtype != GD_DTYPE_I32 || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] != k_cache->shape[0] + 1) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_packed cu_seqlens must be i32 [B+1]");
    }
    if (!gd_tensor_is_contiguous(cu_seqlens)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "kv_cache_append_packed requires contiguous cu_seqlens");
    }
    memset(args, 0, sizeof(*args));
    args->cache_pos = -1;
    return GD_OK;
}

static void gd_kv_cache_bump_versions(gd_tensor *k_cache, gd_tensor *v_cache)
{
    k_cache->version += 1U;
    if (k_cache->version == 0U) {
        k_cache->version = 1U;
    }
    v_cache->version += 1U;
    if (v_cache->version == 0U) {
        v_cache->version = 1U;
    }
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
    st = gd_kv_cache_append_validate_at(ctx, k_cache, v_cache, cache_pos, k_new, v_new, &args);
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
    gd_kv_cache_bump_versions(k_cache, v_cache);
    return GD_OK;
}

gd_status gd_kv_cache_append_positions(gd_context *ctx,
                                       gd_tensor *k_cache,
                                       gd_tensor *v_cache,
                                       const gd_tensor *cache_pos,
                                       const gd_tensor *k_new,
                                       const gd_tensor *v_new)
{
    gd_status st;
    gd_backend_kv_cache_append_args args;
    gd_backend_tensor_view kcv;
    gd_backend_tensor_view vcv;
    gd_backend_tensor_view pv;
    gd_backend_tensor_view knv;
    gd_backend_tensor_view vnv;
    if (ctx == NULL || k_cache == NULL || v_cache == NULL || cache_pos == NULL ||
        k_new == NULL || v_new == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_kv_cache_append_validate_positions(ctx, k_cache, v_cache, cache_pos, k_new, v_new, &args);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(k_cache, &kcv) ||
        !gd_op_tensor_view_from_tensor(v_cache, &vcv) ||
        !gd_op_tensor_view_from_tensor(cache_pos, &pv) ||
        !gd_op_tensor_view_from_tensor(k_new, &knv) ||
        !gd_op_tensor_view_from_tensor(v_new, &vnv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_positions invalid tensor view");
    }
    st = gd_backend_kv_cache_append_positions(gd_context_backend(ctx), &kcv, &vcv, &pv, &knv, &vnv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend kv_cache_append_positions failed");
    }
    gd_kv_cache_bump_versions(k_cache, v_cache);
    return GD_OK;
}

gd_status gd_kv_cache_append_packed(gd_context *ctx,
                                    gd_tensor *k_cache,
                                    gd_tensor *v_cache,
                                    const gd_tensor *cache_pos,
                                    const gd_tensor *cu_seqlens,
                                    const gd_tensor *k_new,
                                    const gd_tensor *v_new)
{
    gd_status st;
    gd_backend_kv_cache_append_args args;
    gd_backend_tensor_view kcv;
    gd_backend_tensor_view vcv;
    gd_backend_tensor_view pv;
    gd_backend_tensor_view cuv;
    gd_backend_tensor_view knv;
    gd_backend_tensor_view vnv;
    if (ctx == NULL || k_cache == NULL || v_cache == NULL || cache_pos == NULL ||
        cu_seqlens == NULL || k_new == NULL || v_new == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_kv_cache_append_validate_packed(ctx, k_cache, v_cache, cache_pos, cu_seqlens,
                                            k_new, v_new, &args);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(k_cache, &kcv) ||
        !gd_op_tensor_view_from_tensor(v_cache, &vcv) ||
        !gd_op_tensor_view_from_tensor(cache_pos, &pv) ||
        !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) ||
        !gd_op_tensor_view_from_tensor(k_new, &knv) ||
        !gd_op_tensor_view_from_tensor(v_new, &vnv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "kv_cache_append_packed invalid tensor view");
    }
    st = gd_backend_kv_cache_append_packed(gd_context_backend(ctx), &kcv, &vcv, &pv, &cuv, &knv, &vnv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend kv_cache_append_packed failed");
    }
    gd_kv_cache_bump_versions(k_cache, v_cache);
    return GD_OK;
}
