#ifndef GD_OP_MINIMAX_M3_SPARSE_ATTENTION_IMPL_H
#define GD_OP_MINIMAX_M3_SPARSE_ATTENTION_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

#define GD_MINIMAX_M3_SPARSE_BLOCK_SIZE 128
#define GD_MINIMAX_M3_SPARSE_MAX_BLOCK_SIZE 128
#define GD_MINIMAX_M3_SPARSE_MAX_HEAD_DIM 64
#define GD_MINIMAX_M3_SPARSE_MAX_TOPK 16

typedef struct gd_minimax_m3_sparse_attention_attrs {
    float scale;
    int32_t block_size;
    int32_t topk;
    int32_t init_blocks;
    int32_t local_blocks;
    int32_t max_seqlen;
} gd_minimax_m3_sparse_attention_attrs;

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
    gd_tensor *grad_v);

#endif /* GD_OP_MINIMAX_M3_SPARSE_ATTENTION_IMPL_H */
