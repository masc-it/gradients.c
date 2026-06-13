#ifndef GD_OPS_SHARED_BINARY_METAL_TYPES_H
#define GD_OPS_SHARED_BINARY_METAL_TYPES_H

#include "../../../backends/metal/metal_abi.h"

#define GD_METAL_BINARY_REDUCE_SUFFIX_SIMDGROUPS 8U

typedef struct gd_metal_binary_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_binary_args;

typedef struct gd_metal_binary_bcast_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 rank;
    gd_metal_u64 out_shape[8];
    gd_metal_u64 x_strides[8];
    gd_metal_u64 y_strides[8];
} gd_metal_binary_bcast_args;

typedef struct gd_metal_binary_reduce_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_count;
    gd_metal_u64 dst_count;
    gd_metal_u32 dtype;
    gd_metal_u32 rank;
    float scale;
    gd_metal_u32 pad0;
    gd_metal_u64 src_shape[8];
    gd_metal_u64 src_strides[8];
    gd_metal_u64 dst_shape[8];
    gd_metal_u64 dst_strides[8];
} gd_metal_binary_reduce_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_binary_args) == 40U, "gd_metal_binary_args ABI mismatch");
_Static_assert(sizeof(gd_metal_binary_bcast_args) == 232U,
               "gd_metal_binary_bcast_args ABI mismatch");
_Static_assert(sizeof(gd_metal_binary_reduce_args) == 304U,
               "gd_metal_binary_reduce_args ABI mismatch");
#endif

#endif /* GD_OPS_SHARED_BINARY_METAL_TYPES_H */
