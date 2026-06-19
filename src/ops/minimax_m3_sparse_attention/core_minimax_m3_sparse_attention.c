#include "minimax_m3_sparse_attention_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static gd_status gd_minimax_m3_sparse_resolve_config(gd_context *ctx,
                                                     const gd_tensor *q,
                                                     const gd_minimax_m3_sparse_config *config,
                                                     gd_minimax_m3_sparse_attention_attrs *attrs,
                                                     gd_backend_minimax_m3_sparse_args *args)
{
    double scale;
    int32_t block_size;
    int32_t topk;
    int32_t init_blocks;
    int32_t local_blocks;
    int32_t max_seqlen;
    if (ctx == NULL || q == NULL || attrs == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    block_size = config != NULL && config->block_size > 0 ? config->block_size :
                                                              GD_MINIMAX_M3_SPARSE_BLOCK_SIZE;
    topk = config != NULL && config->topk > 0 ? config->topk : 4;
    init_blocks = config != NULL ? config->init_blocks : 1;
    local_blocks = config != NULL ? config->local_blocks : 1;
    max_seqlen = config != NULL && config->max_seqlen > 0 ? config->max_seqlen : (int32_t)q->shape[0];
    if (block_size <= 0 || block_size > GD_MINIMAX_M3_SPARSE_MAX_BLOCK_SIZE || topk <= 0 ||
        topk > GD_MINIMAX_M3_SPARSE_MAX_TOPK || init_blocks < 0 || local_blocks < 0 ||
        max_seqlen <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 sparse config out of range");
    }
    if (config != NULL && config->scale > 0.0f) {
        scale = (double)config->scale;
    } else {
        scale = 1.0 / sqrt((double)q->shape[2]);
    }
    if (!isfinite(scale) || scale <= 0.0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 sparse scale must be finite");
    }
    memset(attrs, 0, sizeof(*attrs));
    attrs->scale = (float)scale;
    attrs->block_size = block_size;
    attrs->topk = topk;
    attrs->init_blocks = init_blocks;
    attrs->local_blocks = local_blocks;
    attrs->max_seqlen = max_seqlen;
    memset(args, 0, sizeof(*args));
    args->scale = attrs->scale;
    args->block_size = (uint32_t)block_size;
    args->topk = (uint32_t)topk;
    args->init_blocks = (uint32_t)init_blocks;
    args->local_blocks = (uint32_t)local_blocks;
    args->max_seqlen = (uint32_t)max_seqlen;
    return GD_OK;
}

static gd_status gd_minimax_m3_sparse_validate_qkv(gd_context *ctx,
                                                   const gd_tensor *q,
                                                   const gd_tensor *k,
                                                   const gd_tensor *v,
                                                   const gd_tensor *cu_seqlens,
                                                   const gd_tensor *topk_idx,
                                                   const gd_minimax_m3_sparse_config *config,
                                                   gd_minimax_m3_sparse_attention_attrs *attrs,
                                                   gd_backend_minimax_m3_sparse_args *args)
{
    gd_status st;
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || topk_idx == NULL ||
        attrs == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, q);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, k);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, v);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, cu_seqlens);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, topk_idx);
    if (st != GD_OK) { return st; }
    if (q->dtype != GD_DTYPE_F16 || k->dtype != q->dtype || v->dtype != q->dtype) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 sparse attention requires f16 q/k/v");
    }
    if (cu_seqlens->dtype != GD_DTYPE_I32 || topk_idx->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 sparse attention metadata must be i32");
    }
    if (q->rank != 3U || k->rank != 3U || v->rank != 3U || cu_seqlens->rank != 1U ||
        topk_idx->rank != 3U || cu_seqlens->shape[0] < 2 || q->shape[0] <= 0 ||
        q->shape[1] <= 0 || q->shape[2] <= 0 || k->shape[0] != q->shape[0] ||
        v->shape[0] != q->shape[0] || k->shape[1] <= 0 || v->shape[1] != k->shape[1] ||
        k->shape[2] != q->shape[2] || v->shape[2] != q->shape[2] ||
        (q->shape[1] % k->shape[1]) != 0 || topk_idx->shape[0] != k->shape[1] ||
        topk_idx->shape[1] != q->shape[0]) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "minimax_m3 sparse attention shape mismatch");
    }
    if (q->shape[2] > GD_MINIMAX_M3_SPARSE_MAX_HEAD_DIM || q->shape[0] > (int64_t)UINT32_MAX ||
        q->shape[1] > (int64_t)UINT32_MAX || k->shape[1] > (int64_t)UINT32_MAX ||
        cu_seqlens->shape[0] - 1 > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "minimax_m3 sparse attention dimensions exceed Metal limits");
    }
    if (!gd_tensor_is_contiguous(q) || !gd_tensor_is_contiguous(k) || !gd_tensor_is_contiguous(v) ||
        !gd_tensor_is_contiguous(cu_seqlens) || !gd_tensor_is_contiguous(topk_idx)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "minimax_m3 sparse attention requires contiguous tensors");
    }
    st = gd_minimax_m3_sparse_resolve_config(ctx, q, config, attrs, args);
    if (st != GD_OK) {
        return st;
    }
    if (topk_idx->shape[2] != (int64_t)attrs->topk) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 topk shape/config mismatch");
    }
    return GD_OK;
}

static gd_status gd_minimax_m3_sparse_validate_grad_out(gd_context *ctx,
                                                        const gd_tensor *q,
                                                        const gd_tensor *grad_out)
{
    gd_status st;
    if (ctx == NULL || q == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
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
                                    "minimax_m3 sparse grad_out must match output shape");
    }
    return GD_OK;
}

gd_status gd_minimax_m3_sparse_attention(gd_context *ctx,
                                         const gd_tensor *q,
                                         const gd_tensor *k,
                                         const gd_tensor *v,
                                         const gd_tensor *cu_seqlens,
                                         const gd_tensor *topk_idx,
                                         const gd_minimax_m3_sparse_config *config,
                                         gd_tensor *out)
{
    gd_status st;
    gd_minimax_m3_sparse_attention_attrs attrs;
    gd_backend_minimax_m3_sparse_args args;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view vv;
    gd_backend_tensor_view cuv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view ov;
    gd_tensor y;
    int64_t out_shape[3];
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || topk_idx == NULL ||
        out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_minimax_m3_sparse_validate_qkv(ctx, q, k, v, cu_seqlens, topk_idx, config, &attrs, &args);
    if (st != GD_OK) {
        return st;
    }
    out_shape[0] = q->shape[0];
    out_shape[1] = q->shape[1];
    out_shape[2] = q->shape[2];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, q->dtype, gd_shape_make(3U, out_shape), 256U, &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(q, &qv) || !gd_op_tensor_view_from_tensor(k, &kv) ||
        !gd_op_tensor_view_from_tensor(v, &vv) || !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) ||
        !gd_op_tensor_view_from_tensor(topk_idx, &tv) || !gd_op_tensor_view_from_tensor(&y, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 sparse invalid tensor view");
    }
    st = gd_backend_minimax_m3_sparse_attention(gd_context_backend(ctx), &qv, &kv, &vv, &cuv, &tv, &ov, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend minimax_m3 sparse attention failed");
    }
    if (q->requires_grad || k->requires_grad || v->requires_grad) {
        const gd_tensor *inputs[5] = {q, k, v, cu_seqlens, topk_idx};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_MINIMAX_M3_SPARSE_ATTENTION,
                                inputs,
                                5U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_minimax_m3_sparse_attention_backward_with_topk(
    gd_context *ctx,
    const gd_tensor *q,
    const gd_tensor *k,
    const gd_tensor *v,
    const gd_tensor *cu_seqlens,
    const gd_tensor *topk_idx,
    const gd_tensor *grad_out,
    const gd_minimax_m3_sparse_config *config,
    gd_tensor *grad_q,
    gd_tensor *grad_k,
    gd_tensor *grad_v)
{
    gd_status st;
    gd_minimax_m3_sparse_attention_attrs attrs;
    gd_backend_minimax_m3_sparse_args args;
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
    gd_backend_tensor_view tv;
    gd_backend_tensor_view dqv;
    gd_backend_tensor_view dkv;
    gd_backend_tensor_view dvv;
    gd_backend_tensor_view statsv;
    const bool need_q = grad_q != NULL;
    const bool need_k = grad_k != NULL;
    const bool need_v = grad_v != NULL;
    if (grad_q != NULL) { memset(grad_q, 0, sizeof(*grad_q)); }
    if (grad_k != NULL) { memset(grad_k, 0, sizeof(*grad_k)); }
    if (grad_v != NULL) { memset(grad_v, 0, sizeof(*grad_v)); }
    if (ctx == NULL || q == NULL || k == NULL || v == NULL || cu_seqlens == NULL || topk_idx == NULL ||
        grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_minimax_m3_sparse_validate_qkv(ctx, q, k, v, cu_seqlens, topk_idx, config, &attrs, &args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_minimax_m3_sparse_validate_grad_out(ctx, q, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (!need_q && !need_k && !need_v) {
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
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, q->dtype, gd_shape_make(3U, q_shape), 256U, &dq);
    if (st != GD_OK) { return st; }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, k->dtype, gd_shape_make(3U, k_shape), 256U, &dk);
    if (st != GD_OK) { return st; }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, v->dtype, gd_shape_make(3U, k_shape), 256U, &dv);
    if (st != GD_OK) { return st; }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(3U, stats_shape), 256U, &stats);
    if (st != GD_OK) { return st; }
    dq.is_leaf = false;
    dk.is_leaf = false;
    dv.is_leaf = false;
    stats.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(grad_out, &gov) || !gd_op_tensor_view_from_tensor(q, &qv) ||
        !gd_op_tensor_view_from_tensor(k, &kv) || !gd_op_tensor_view_from_tensor(v, &vv) ||
        !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) || !gd_op_tensor_view_from_tensor(topk_idx, &tv) ||
        !gd_op_tensor_view_from_tensor(&dq, &dqv) || !gd_op_tensor_view_from_tensor(&dk, &dkv) ||
        !gd_op_tensor_view_from_tensor(&dv, &dvv) || !gd_op_tensor_view_from_tensor(&stats, &statsv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 sparse backward invalid view");
    }
    st = gd_backend_minimax_m3_sparse_attention_backward(gd_context_backend(ctx),
                                                         &gov,
                                                         &qv,
                                                         &kv,
                                                         &vv,
                                                         &cuv,
                                                         &tv,
                                                         &dqv,
                                                         &dkv,
                                                         &dvv,
                                                         &statsv,
                                                         &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend minimax_m3 sparse backward failed");
    }
    if (need_q) { *grad_q = dq; }
    if (need_k) { *grad_k = dk; }
    if (need_v) { *grad_v = dv; }
    return GD_OK;
}

gd_status gd_minimax_m3_sparse_attention_backward(gd_context *ctx,
                                                  const gd_tensor *q,
                                                  const gd_tensor *k,
                                                  const gd_tensor *v,
                                                  const gd_tensor *cu_seqlens,
                                                  const gd_tensor *topk_idx,
                                                  const gd_tensor *grad_out,
                                                  const gd_minimax_m3_sparse_config *config,
                                                  gd_tensor *grad_q,
                                                  gd_tensor *grad_k,
                                                  gd_tensor *grad_v)
{
    return gd_minimax_m3_sparse_attention_backward_with_topk(ctx,
                                                             q,
                                                             k,
                                                             v,
                                                             cu_seqlens,
                                                             topk_idx,
                                                             grad_out,
                                                             config,
                                                             grad_q,
                                                             grad_k,
                                                             grad_v);
}

