#ifndef GD_OP_KV_CACHE_APPEND_METAL_TYPES_H
#define GD_OP_KV_CACHE_APPEND_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_KV_CACHE_APPEND_DTYPE_F16 1U
#define GD_METAL_KV_CACHE_APPEND_DTYPE_F32 3U

typedef struct gd_metal_kv_cache_append_args {
    gd_metal_u64 k_cache_offset;
    gd_metal_u64 v_cache_offset;
    gd_metal_u64 k_new_offset;
    gd_metal_u64 v_new_offset;
    gd_metal_u64 total_units;
    gd_metal_u32 tmax;
    gd_metal_u32 tnew;
    gd_metal_u32 row_bytes;
    gd_metal_u32 copy_unit;
    gd_metal_u32 cache_pos;
    gd_metal_u32 reserved0;
} gd_metal_kv_cache_append_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_kv_cache_append_args) == 64U,
               "gd_metal_kv_cache_append_args ABI mismatch");
#endif

#endif /* GD_OP_KV_CACHE_APPEND_METAL_TYPES_H */
