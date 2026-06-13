#ifndef GD_METAL_MEMORY_TYPES_H
#define GD_METAL_MEMORY_TYPES_H

#include "../../metal_abi.h"

typedef struct gd_metal_fill_args {
    gd_metal_u64 byte_offset;
    gd_metal_u64 count;
    gd_metal_u32 elem_size;
    gd_metal_u32 pattern;
} gd_metal_fill_args;

typedef struct gd_metal_accumulate_args {
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_accumulate_args;

typedef struct gd_metal_scale_args {
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
    float scale;
    float pad1;
} gd_metal_scale_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_fill_args) == 24U, "gd_metal_fill_args ABI mismatch");
_Static_assert(sizeof(gd_metal_accumulate_args) == 32U, "gd_metal_accumulate_args ABI mismatch");
_Static_assert(sizeof(gd_metal_scale_args) == 40U, "gd_metal_scale_args ABI mismatch");
_Static_assert(offsetof(gd_metal_scale_args, scale) == 32U,
               "gd_metal_scale_args scale offset mismatch");
#endif

#endif /* GD_METAL_MEMORY_TYPES_H */
