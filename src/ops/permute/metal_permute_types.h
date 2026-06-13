#ifndef GD_OP_PERMUTE_METAL_TYPES_H
#define GD_OP_PERMUTE_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_PERMUTE_THREADS 256U
#define GD_METAL_PERMUTE_MAX_DIMS 8U

typedef struct gd_metal_permute_args {
    gd_metal_u64 src_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 count;
    gd_metal_u64 inner;
    gd_metal_u32 rank;
    gd_metal_u32 active_rank;
    gd_metal_u32 reserved0;
    gd_metal_u32 reserved1;
    gd_metal_u64 dst_shape[GD_METAL_PERMUTE_MAX_DIMS];
    gd_metal_u64 src_strides[GD_METAL_PERMUTE_MAX_DIMS];
    gd_metal_u32 axes[GD_METAL_PERMUTE_MAX_DIMS];
} gd_metal_permute_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_permute_args) == 208U,
               "gd_metal_permute_args ABI mismatch");
#endif

#endif /* GD_OP_PERMUTE_METAL_TYPES_H */
