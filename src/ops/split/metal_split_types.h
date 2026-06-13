#ifndef GD_OP_SPLIT_METAL_TYPES_H
#define GD_OP_SPLIT_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_SPLIT_THREADS 256U

typedef struct gd_metal_split_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 count;
    gd_metal_u64 inner;
    gd_metal_u64 slice_axis;
    gd_metal_u64 full_axis;
    gd_metal_u64 axis_offset;
    gd_metal_u64 elem_size;
} gd_metal_split_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_split_args) == 64U,
               "gd_metal_split_args ABI mismatch");
#endif

#endif /* GD_OP_SPLIT_METAL_TYPES_H */
