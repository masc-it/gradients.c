#include <gradients/ops.h>

#include "../op_common.h"
#include "../minimax_m3_sparse_attention/minimax_m3_sparse_attention_impl.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static gd_status gd_minimax_m3_index_resolve_config(gd_context *ctx,
                                                    const gd_tensor *q,
                                                    const gd_minimax_m3_sparse_config *config,
                                                    gd_backend_minimax_m3_sparse_args *args)
{
    double scale;
    int32_t block_size;
    int32_t topk;
    int32_t init_blocks;
    int32_t local_blocks;
    int32_t max_seqlen;
    if (ctx == NULL || q == NULL || args == NULL) {
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
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 index config out of range");
    }
    if (config != NULL && config->scale > 0.0f) {
        scale = (double)config->scale;
    } else {
        scale = 1.0 / sqrt((double)q->shape[2]);
    }
    if (!isfinite(scale) || scale <= 0.0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 index scale must be finite");
    }
    memset(args, 0, sizeof(*args));
    args->scale = (float)scale;
    args->block_size = (uint32_t)block_size;
    args->topk = (uint32_t)topk;
    args->init_blocks = (uint32_t)init_blocks;
    args->local_blocks = (uint32_t)local_blocks;
    args->max_seqlen = (uint32_t)max_seqlen;
    return GD_OK;
}

static gd_status gd_minimax_m3_index_validate(gd_context *ctx,
                                              const gd_tensor *index_q,
                                              const gd_tensor *index_k,
                                              const gd_tensor *cu_seqlens,
                                              const gd_minimax_m3_sparse_config *config,
                                              gd_backend_minimax_m3_sparse_args *args)
{
    gd_status st;
    if (ctx == NULL || index_q == NULL || index_k == NULL || cu_seqlens == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, index_q);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, index_k);
    if (st != GD_OK) { return st; }
    st = gd_tensor_validate(ctx, cu_seqlens);
    if (st != GD_OK) { return st; }
    if (index_q->dtype != GD_DTYPE_F16 || index_k->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 indexer requires f16 q/k");
    }
    if (cu_seqlens->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 indexer cu_seqlens must be i32");
    }
    if (index_q->rank != 3U || index_k->rank != 3U || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] < 2 || index_q->shape[0] <= 0 || index_q->shape[1] <= 0 ||
        index_q->shape[2] <= 0 || index_k->shape[0] != index_q->shape[0] ||
        index_k->shape[1] != index_q->shape[1] || index_k->shape[2] != index_q->shape[2]) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "minimax_m3 indexer expects q/k [N,H,D] and cu [B+1]");
    }
    if (index_q->shape[2] > GD_MINIMAX_M3_SPARSE_MAX_HEAD_DIM ||
        index_q->shape[0] > (int64_t)UINT32_MAX || index_q->shape[1] > (int64_t)UINT32_MAX ||
        cu_seqlens->shape[0] - 1 > (int64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 indexer dimensions exceed Metal limits");
    }
    if (!gd_tensor_is_contiguous(index_q) || !gd_tensor_is_contiguous(index_k) ||
        !gd_tensor_is_contiguous(cu_seqlens)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "minimax_m3 indexer requires contiguous tensors");
    }
    return gd_minimax_m3_index_resolve_config(ctx, index_q, config, args);
}

gd_status gd_minimax_m3_index_topk(gd_context *ctx,
                                   const gd_tensor *index_q,
                                   const gd_tensor *index_k,
                                   const gd_tensor *cu_seqlens,
                                   const gd_minimax_m3_sparse_config *config,
                                   gd_tensor *topk_idx)
{
    gd_status st;
    gd_backend_minimax_m3_sparse_args args;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view cuv;
    gd_backend_tensor_view tv;
    gd_tensor out;
    int64_t topk_shape[3];
    if (topk_idx != NULL) {
        memset(topk_idx, 0, sizeof(*topk_idx));
    }
    if (ctx == NULL || index_q == NULL || index_k == NULL || cu_seqlens == NULL || topk_idx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_minimax_m3_index_validate(ctx, index_q, index_k, cu_seqlens, config, &args);
    if (st != GD_OK) {
        return st;
    }
    topk_shape[0] = index_k->shape[1];
    topk_shape[1] = index_q->shape[0];
    topk_shape[2] = (int64_t)args.topk;
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_I32, gd_shape_make(3U, topk_shape), 256U, &out);
    if (st != GD_OK) {
        return st;
    }
    out.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(index_q, &qv) || !gd_op_tensor_view_from_tensor(index_k, &kv) ||
        !gd_op_tensor_view_from_tensor(cu_seqlens, &cuv) || !gd_op_tensor_view_from_tensor(&out, &tv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "minimax_m3 indexer invalid tensor view");
    }
    st = gd_backend_minimax_m3_index_topk(gd_context_backend(ctx), &qv, &kv, &cuv, &tv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend minimax_m3 index_topk failed");
    }
    *topk_idx = out;
    return GD_OK;
}
