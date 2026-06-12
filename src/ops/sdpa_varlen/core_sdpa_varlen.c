#include "sdpa_varlen_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static gd_status gd_sdpa_varlen_config_attrs(gd_context *ctx,
                                             const gd_tensor *q,
                                             const gd_sdpa_varlen_config *config,
                                             gd_sdpa_varlen_attrs *attrs)
{
    double scale;
    if (ctx == NULL || q == NULL || attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(attrs, 0, sizeof(*attrs));
    attrs->scale = config != NULL ? config->scale : 0.0f;
    attrs->causal = (config != NULL && config->causal) ? 1 : 0;
    attrs->sliding_window = config != NULL ? config->sliding_window : 0;
    attrs->prefix_len = config != NULL ? config->prefix_len : 0;
    attrs->max_seqlen = config != NULL ? config->max_seqlen : 0;
    if (attrs->sliding_window < 0 || attrs->prefix_len < 0 || attrs->max_seqlen < 0) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen window/prefix/max_seqlen must be non-negative");
    }
    if (attrs->prefix_len > 0 && attrs->causal == 0) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen prefix_len requires causal=true");
    }
    if (attrs->max_seqlen == 0) {
        if (q->shape[0] > (int64_t)INT_MAX) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_varlen N exceeds int range");
        }
        attrs->max_seqlen = (int32_t)q->shape[0];
    }
    if (attrs->prefix_len > attrs->max_seqlen) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen prefix_len exceeds max_seqlen");
    }
    if (attrs->scale <= 0.0f) {
        scale = 1.0 / sqrt((double)q->shape[2]);
        attrs->scale = (float)scale;
    } else if (!isfinite(attrs->scale)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_varlen scale must be finite");
    }
    return GD_OK;
}

static gd_status gd_sdpa_varlen_validate(gd_context *ctx,
                                         const gd_tensor *q,
                                         const gd_tensor *k,
                                         const gd_tensor *v,
                                         const gd_tensor *cu_seqlens,
                                         const gd_sdpa_varlen_config *config,
                                         gd_sdpa_varlen_attrs *attrs)
{
    gd_status st;
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || attrs == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, q);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, k);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, v);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, cu_seqlens);
    if (st != GD_OK) {
        return st;
    }
    if ((q->dtype != GD_DTYPE_F16 && q->dtype != GD_DTYPE_F32) || k->dtype != q->dtype ||
        v->dtype != q->dtype) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sdpa_varlen requires matching f16/f32 q/k/v");
    }
    if (cu_seqlens->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "sdpa_varlen cu_seqlens must be i32");
    }
    if (q->rank != 3U || k->rank != 3U || v->rank != 3U || cu_seqlens->rank != 1U) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen expects q/k/v [N,H,Dh] and cu_seqlens [B+1]");
    }
    if (cu_seqlens->shape[0] < 2 || q->shape[0] <= 0 || q->shape[1] <= 0 ||
        q->shape[2] <= 0 || k->shape[1] <= 0 || k->shape[0] != q->shape[0] ||
        v->shape[0] != q->shape[0] || k->shape[2] != q->shape[2] ||
        v->shape[1] != k->shape[1] || v->shape[2] != q->shape[2]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_varlen shape mismatch");
    }
    if ((q->shape[1] % k->shape[1]) != 0) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen Hq must be a multiple of Hkv");
    }
    if (q->shape[0] > (int64_t)INT_MAX || q->shape[1] > (int64_t)INT_MAX ||
        k->shape[1] > (int64_t)INT_MAX || q->shape[2] > GD_SDPA_VARLEN_MAX_HEAD_DIM ||
        cu_seqlens->shape[0] - 1 > (int64_t)INT_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "sdpa_varlen dimensions exceed Metal kernel limits");
    }
    if (!gd_tensor_is_contiguous(q) || !gd_tensor_is_contiguous(k) ||
        !gd_tensor_is_contiguous(v) || !gd_tensor_is_contiguous(cu_seqlens)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "sdpa_varlen requires contiguous tensors");
    }
    return gd_sdpa_varlen_config_attrs(ctx, q, config, attrs);
}

static gd_backend_sdpa_varlen_args gd_sdpa_varlen_backend_args(const gd_sdpa_varlen_attrs *attrs)
{
    gd_backend_sdpa_varlen_args args;
    memset(&args, 0, sizeof(args));
    if (attrs != NULL) {
        args.scale = attrs->scale;
        args.causal = attrs->causal != 0 ? 1U : 0U;
        args.sliding_window = (uint32_t)attrs->sliding_window;
        args.prefix_len = (uint32_t)attrs->prefix_len;
        args.max_seqlen = (uint32_t)attrs->max_seqlen;
    }
    return args;
}

static bool gd_sdpa_varlen_can_save_forward_stats(const gd_tensor *q,
                                                  const gd_sdpa_varlen_attrs *attrs)
{
    return q != NULL && attrs != NULL && q->dtype == GD_DTYPE_F16 && q->shape[2] == 64 &&
           attrs->causal != 0 && attrs->sliding_window > 0;
}

static gd_status gd_sdpa_varlen_validate_forward_stats(gd_context *ctx,
                                                       const gd_tensor *q,
                                                       const gd_tensor *stats)
{
    gd_status st;
    if (ctx == NULL || q == NULL || stats == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, stats);
    if (st != GD_OK) {
        return st;
    }
    if (stats->dtype != GD_DTYPE_F32 || stats->rank != 3U ||
        stats->shape[0] != q->shape[0] || stats->shape[1] != q->shape[1] ||
        stats->shape[2] != 3 || !gd_tensor_is_contiguous(stats)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen forward stats must be contiguous f32 [N,H,3]");
    }
    return GD_OK;
}

gd_status gd_sdpa_varlen(gd_context *ctx,
                         const gd_tensor *q,
                         const gd_tensor *k,
                         const gd_tensor *v,
                         const gd_tensor *cu_seqlens,
                         const gd_sdpa_varlen_config *config,
                         gd_tensor *out)
{
    gd_status st;
    gd_sdpa_varlen_attrs attrs;
    gd_backend_sdpa_varlen_args backend_args;
    gd_tensor y;
    gd_tensor stats;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view vv;
    gd_backend_tensor_view cuv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view statsv;
    int64_t shape[3];
    int64_t stats_shape[3];
    bool save_stats;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&stats, 0, sizeof(stats));
    memset(&statsv, 0, sizeof(statsv));
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sdpa_varlen_validate(ctx, q, k, v, cu_seqlens, config, &attrs);
    if (st != GD_OK) {
        return st;
    }
    shape[0] = q->shape[0];
    shape[1] = q->shape[1];
    shape[2] = q->shape[2];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, q->dtype, gd_shape_make(3U, shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    save_stats = (q->requires_grad || k->requires_grad || v->requires_grad) &&
                 gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN &&
                 gd_sdpa_varlen_can_save_forward_stats(q, &attrs);
    if (save_stats) {
        stats_shape[0] = q->shape[0];
        stats_shape[1] = q->shape[1];
        stats_shape[2] = 3;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F32,
                             gd_shape_make(3U, stats_shape),
                             256U,
                             &stats);
        if (st != GD_OK) {
            return st;
        }
        stats.is_leaf = false;
    }
    if (!gd_op_tensor_view_from_tensor(q, &qv) || !gd_op_tensor_view_from_tensor(k, &kv) ||
        !gd_op_tensor_view_from_tensor(v, &vv) || !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) ||
        !gd_op_tensor_view_from_tensor(&y, &yv) ||
        (save_stats && !gd_op_tensor_view_from_tensor(&stats, &statsv))) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_varlen invalid tensor view");
    }
    backend_args = gd_sdpa_varlen_backend_args(&attrs);
    st = gd_backend_sdpa_varlen(gd_context_backend(ctx),
                                &qv,
                                &kv,
                                &vv,
                                &cuv,
                                &yv,
                                save_stats ? &statsv : NULL,
                                &backend_args);
    if (st == GD_ERR_UNSUPPORTED && save_stats) {
        gd_context_clear_error(ctx);
        save_stats = false;
        st = gd_backend_sdpa_varlen(gd_context_backend(ctx), &qv, &kv, &vv, &cuv, &yv, NULL, &backend_args);
    }
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sdpa_varlen failed");
    }
    if (q->requires_grad || k->requires_grad || v->requires_grad) {
        const gd_tensor *inputs[4] = {q, k, v, cu_seqlens};
        gd_tensor *outputs[1] = {&y};
        const gd_tensor *saved[1] = {&stats};
        st = gd_autograd_record(ctx,
                                GD_OP_SDPA_VARLEN,
                                inputs,
                                4U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                save_stats ? saved : NULL,
                                save_stats ? 1U : 0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_sdpa_varlen_backward_with_stats(gd_context *ctx,
                                             const gd_tensor *q,
                                             const gd_tensor *k,
                                             const gd_tensor *v,
                                             const gd_tensor *cu_seqlens,
                                             const gd_tensor *forward_stats,
                                             const gd_tensor *grad_out,
                                             const gd_sdpa_varlen_config *config,
                                             gd_tensor *grad_q,
                                             gd_tensor *grad_k,
                                             gd_tensor *grad_v)
{
    gd_status st;
    gd_sdpa_varlen_attrs attrs;
    gd_backend_sdpa_varlen_args backend_args;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_tensor stats;
    int64_t q_shape[3];
    int64_t k_shape[3];
    int64_t stats_shape[3];
    gd_backend_tensor_view gov;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view vv;
    gd_backend_tensor_view cuv;
    gd_backend_tensor_view dqv;
    gd_backend_tensor_view dkv;
    gd_backend_tensor_view dvv;
    gd_backend_tensor_view statsv;
    bool need_grad_q = grad_q != NULL;
    bool need_grad_k = grad_k != NULL;
    bool need_grad_v = grad_v != NULL;
    bool using_forward_stats = forward_stats != NULL;
    if (grad_q != NULL) {
        memset(grad_q, 0, sizeof(*grad_q));
    }
    if (grad_k != NULL) {
        memset(grad_k, 0, sizeof(*grad_k));
    }
    if (grad_v != NULL) {
        memset(grad_v, 0, sizeof(*grad_v));
    }
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL ||
        grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_sdpa_varlen_validate(ctx, q, k, v, cu_seqlens, config, &attrs);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != q->dtype || grad_out->rank != 3U || grad_out->shape[0] != q->shape[0] ||
        grad_out->shape[1] != q->shape[1] || grad_out->shape[2] != q->shape[2] ||
        !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "sdpa_varlen backward grad_out must match output shape");
    }
    if (!need_grad_q && !need_grad_k && !need_grad_v) {
        return GD_OK;
    }
    q_shape[0] = q->shape[0];
    q_shape[1] = q->shape[1];
    q_shape[2] = q->shape[2];
    k_shape[0] = k->shape[0];
    k_shape[1] = k->shape[1];
    k_shape[2] = k->shape[2];
    stats_shape[0] = q->shape[0];
    stats_shape[1] = q->shape[1];
    stats_shape[2] = 3;
    if (using_forward_stats) {
        st = gd_sdpa_varlen_validate_forward_stats(ctx, q, forward_stats);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, q->dtype, gd_shape_make(3U, q_shape), 256U, &dq);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, k->dtype, gd_shape_make(3U, k_shape), 256U, &dk);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, v->dtype, gd_shape_make(3U, k_shape), 256U, &dv);
    if (st != GD_OK) {
        return st;
    }
    if (using_forward_stats) {
        stats = *forward_stats;
    } else {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(3U, stats_shape), 256U, &stats);
        if (st != GD_OK) {
            return st;
        }
        stats.is_leaf = false;
    }
    dq.is_leaf = false;
    dk.is_leaf = false;
    dv.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(grad_out, &gov) || !gd_op_tensor_view_from_tensor(q, &qv) ||
        !gd_op_tensor_view_from_tensor(k, &kv) || !gd_op_tensor_view_from_tensor(v, &vv) ||
        !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) || !gd_op_tensor_view_from_tensor(&dq, &dqv) ||
        !gd_op_tensor_view_from_tensor(&dk, &dkv) || !gd_op_tensor_view_from_tensor(&dv, &dvv) ||
        !gd_op_tensor_view_from_tensor(&stats, &statsv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "sdpa_varlen backward invalid tensor view");
    }
    backend_args = gd_sdpa_varlen_backend_args(&attrs);
    backend_args.use_forward_stats = using_forward_stats ? 1U : 0U;
    st = gd_backend_sdpa_varlen_backward(gd_context_backend(ctx),
                                         &gov,
                                         &qv,
                                         &kv,
                                         &vv,
                                         &cuv,
                                         &dqv,
                                         &dkv,
                                         &dvv,
                                         &statsv,
                                         &backend_args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend sdpa_varlen backward failed");
    }
    if (grad_q != NULL) {
        *grad_q = dq;
    }
    if (grad_k != NULL) {
        *grad_k = dk;
    }
    if (grad_v != NULL) {
        *grad_v = dv;
    }
    return GD_OK;
}

gd_status gd_sdpa_varlen_backward(gd_context *ctx,
                                  const gd_tensor *q,
                                  const gd_tensor *k,
                                  const gd_tensor *v,
                                  const gd_tensor *cu_seqlens,
                                  const gd_tensor *grad_out,
                                  const gd_sdpa_varlen_config *config,
                                  gd_tensor *grad_q,
                                  gd_tensor *grad_k,
                                  gd_tensor *grad_v)
{
    return gd_sdpa_varlen_backward_with_stats(ctx,
                                             q,
                                             k,
                                             v,
                                             cu_seqlens,
                                             NULL,
                                             grad_out,
                                             config,
                                             grad_q,
                                             grad_k,
                                             grad_v);
}
