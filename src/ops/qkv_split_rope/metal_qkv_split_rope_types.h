#ifndef GD_OP_QKV_SPLIT_ROPE_METAL_TYPES_H
#define GD_OP_QKV_SPLIT_ROPE_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_QKV_SPLIT_ROPE_MAX_THREADS_PER_GROUP 256U
#define GD_METAL_QKV_SPLIT_ROPE_TABLE_POSITIONS 512U
#define GD_METAL_QKV_SPLIT_ROPE_TABLE_HEAD_DIM 64U
#define GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS (GD_METAL_QKV_SPLIT_ROPE_TABLE_HEAD_DIM / 2U)
#define GD_METAL_QKV_SPLIT_ROPE_TABLE_THETA 10000.0f

typedef struct gd_metal_qkv_split_rope_args {
    gd_metal_u64 qkv_offset;
    gd_metal_u64 pos_offset;
    gd_metal_u64 q_offset;
    gd_metal_u64 k_offset;
    gd_metal_u64 v_offset;
    gd_metal_u64 tokens;
    gd_metal_u32 heads;
    gd_metal_u32 head_dim;
    gd_metal_u32 n_dims;
    gd_metal_u32 interleaved;
    float freq_scale;
    float sin_sign;
} gd_metal_qkv_split_rope_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_qkv_split_rope_args) == 72U,
               "gd_metal_qkv_split_rope_args ABI mismatch");
#endif

#endif /* GD_OP_QKV_SPLIT_ROPE_METAL_TYPES_H */
