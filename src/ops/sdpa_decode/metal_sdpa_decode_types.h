#ifndef GD_OP_SDPA_DECODE_METAL_TYPES_H
#define GD_OP_SDPA_DECODE_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_SDPA_DECODE_BQ 64U
#define GD_METAL_SDPA_DECODE_BK 16U
#define GD_METAL_SDPA_DECODE_DHT 64U

#define GD_METAL_SDPA_DECODE_DTYPE_F16 1U
#define GD_METAL_SDPA_DECODE_DTYPE_F32 3U

typedef struct gd_metal_sdpa_decode_args {
    gd_metal_u64 q_offset;
    gd_metal_u64 k_offset;
    gd_metal_u64 v_offset;
    gd_metal_u64 pos_offset;
    gd_metal_u64 out_offset;
    gd_metal_u32 batch;
    gd_metal_u32 tq;
    gd_metal_u32 tmax;
    gd_metal_u32 hq;
    gd_metal_u32 hkv;
    gd_metal_u32 dh;
    gd_metal_u32 window;
    gd_metal_u32 prefix_len;
    gd_metal_u32 dtype;
    float scale;
    gd_metal_u32 reserved0;
    gd_metal_u32 reserved1;
} gd_metal_sdpa_decode_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_sdpa_decode_args) == 88U,
               "gd_metal_sdpa_decode_args ABI mismatch");
#endif

#endif /* GD_OP_SDPA_DECODE_METAL_TYPES_H */
