#ifndef GD_OP_SDPA_VARLEN_METAL_TYPES_H
#define GD_OP_SDPA_VARLEN_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_SDPA_BQ 64U
#define GD_METAL_SDPA_BK 16U
#define GD_METAL_SDPA_DHT 64U
#define GD_METAL_SDPA_DKV_LANES 8U
#define GD_METAL_SDPA_DKV_WIDE_KEYS 16U
#define GD_METAL_SDPA_DKV_WIDE_LANES 8U
#define GD_METAL_SDPA_DKV_WIDE_THREADS \
    (GD_METAL_SDPA_DKV_WIDE_KEYS * GD_METAL_SDPA_DKV_WIDE_LANES)
#define GD_METAL_SDPA_CAUSAL_QROWS 16U
#define GD_METAL_SDPA_CAUSAL_THREADS \
    (GD_METAL_SDPA_CAUSAL_QROWS * GD_METAL_SDPA_DKV_LANES)
#define GD_METAL_SDPA_SPLIT_MIN 512U
#define GD_METAL_SDPA_SPLIT_MAX 8U

#define GD_METAL_SDPA_DTYPE_F16 1U
#define GD_METAL_SDPA_DTYPE_F32 3U

typedef struct gd_metal_sdpa_varlen_args {
    gd_metal_u64 q_offset;
    gd_metal_u64 k_offset;
    gd_metal_u64 v_offset;
    gd_metal_u64 cu_offset;
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
    gd_metal_u32 n_qb_max;
    gd_metal_u32 causal;
    gd_metal_u32 window;
    gd_metal_u32 prefix_len;
    gd_metal_u32 dtype;
    float scale;
    gd_metal_u32 n_splits;
} gd_metal_sdpa_varlen_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_sdpa_varlen_args) == 136U,
               "gd_metal_sdpa_varlen_args ABI mismatch");
#endif

#endif /* GD_OP_SDPA_VARLEN_METAL_TYPES_H */
