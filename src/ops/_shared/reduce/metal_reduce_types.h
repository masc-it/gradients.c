#ifndef GD_OPS_SHARED_REDUCE_METAL_TYPES_H
#define GD_OPS_SHARED_REDUCE_METAL_TYPES_H

#include "../../../backends/metal/metal_abi.h"

#define GD_METAL_REDUCE_MAX_SIMDGROUPS 8U
#define GD_METAL_REDUCE_CONTIGUOUS_SIMDGROUPS GD_METAL_REDUCE_MAX_SIMDGROUPS
#define GD_METAL_REDUCE_AXIS_SIMDGROUPS GD_METAL_REDUCE_MAX_SIMDGROUPS
#define GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH 4U

typedef struct gd_metal_reduce_contiguous_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_count;
    gd_metal_u64 dst_count;
    gd_metal_u32 dtype;
    gd_metal_u32 simdgroups_per_output;
    float scale;
    gd_metal_u32 pad0;
} gd_metal_reduce_contiguous_args;

typedef struct gd_metal_reduce_axis_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 outer_count;
    gd_metal_u64 reduce_count;
    gd_metal_u64 inner_count;
    gd_metal_u64 dst_count;
    gd_metal_u32 dtype;
    gd_metal_u32 simdgroups_per_output;
    float scale;
    gd_metal_u32 pad0;
} gd_metal_reduce_axis_args;

typedef struct gd_metal_broadcast_to_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 dst_count;
    gd_metal_u32 dtype;
    gd_metal_u32 rank;
    float scale;
    gd_metal_u32 pad0;
    gd_metal_u64 dst_shape[8];
    gd_metal_u64 src_strides[8];
} gd_metal_broadcast_to_args;

typedef struct gd_metal_broadcast_scalar_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 dst_count;
    float scale;
    gd_metal_u32 vector_width;
    gd_metal_u32 pad0;
    gd_metal_u32 pad1;
} gd_metal_broadcast_scalar_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_reduce_contiguous_args) == 48U,
               "gd_metal_reduce_contiguous_args ABI mismatch");
_Static_assert(sizeof(gd_metal_reduce_axis_args) == 64U,
               "gd_metal_reduce_axis_args ABI mismatch");
_Static_assert(sizeof(gd_metal_broadcast_to_args) == 168U,
               "gd_metal_broadcast_to_args ABI mismatch");
_Static_assert(sizeof(gd_metal_broadcast_scalar_args) == 40U,
               "gd_metal_broadcast_scalar_args ABI mismatch");
#endif

#endif /* GD_OPS_SHARED_REDUCE_METAL_TYPES_H */
