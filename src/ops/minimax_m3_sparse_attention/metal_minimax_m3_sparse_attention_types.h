#ifndef GD_OP_MINIMAX_M3_SPARSE_ATTENTION_METAL_TYPES_H
#define GD_OP_MINIMAX_M3_SPARSE_ATTENTION_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_MINIMAX_M3_BLOCK_THREADS 128U
#define GD_METAL_MINIMAX_M3_ATTENTION_THREADS 64U
#define GD_METAL_MINIMAX_M3_QTILE 4U
#define GD_METAL_MINIMAX_M3_QTILE_THREADS (GD_METAL_MINIMAX_M3_ATTENTION_THREADS * GD_METAL_MINIMAX_M3_QTILE)
#define GD_METAL_MINIMAX_M3_DKV_QTILE 16U
#define GD_METAL_MINIMAX_M3_MAX_HEAD_DIM 64U
#define GD_METAL_MINIMAX_M3_MAX_TOPK 16U
#define GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE 128U

typedef struct gd_metal_minimax_m3_sparse_args {
    gd_metal_u64 q_offset;
    gd_metal_u64 k_offset;
    gd_metal_u64 v_offset;
    gd_metal_u64 cu_offset;
    gd_metal_u64 topk_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 grad_out_offset;
    gd_metal_u64 grad_q_offset;
    gd_metal_u64 grad_k_offset;
    gd_metal_u64 grad_v_offset;
    gd_metal_u64 stats_offset;
    gd_metal_u64 total_tokens;
    gd_metal_u32 batch;
    gd_metal_u32 hq;
    gd_metal_u32 hkv;
    gd_metal_u32 dh;
    gd_metal_u32 max_seqlen;
    gd_metal_u32 block_size;
    gd_metal_u32 topk;
    gd_metal_u32 init_blocks;
    gd_metal_u32 local_blocks;
    float scale;
} gd_metal_minimax_m3_sparse_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_minimax_m3_sparse_args) == 136U,
               "gd_metal_minimax_m3_sparse_args ABI mismatch");
#endif

#endif /* GD_OP_MINIMAX_M3_SPARSE_ATTENTION_METAL_TYPES_H */
