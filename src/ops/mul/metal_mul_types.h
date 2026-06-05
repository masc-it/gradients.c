#ifndef GD_OP_MUL_METAL_TYPES_H
#define GD_OP_MUL_METAL_TYPES_H

#include "../_shared/binary/metal_binary_types.h"
#include "../../backends/metal/metal_abi.h"

#define GD_METAL_MUL_F16_VECTOR_WIDTH 4U
#define GD_METAL_MUL_REDUCE_SUFFIX_SIMDGROUPS 8U

typedef struct gd_metal_mul_backward_direct_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 grad_x_offset;
    gd_metal_u64 grad_y_offset;
    gd_metal_u64 count;
} gd_metal_mul_backward_direct_args;

typedef struct gd_metal_mul_reduce_suffix_args {
    gd_metal_u64 grad_offset;
    gd_metal_u64 other_offset;
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_count;
    gd_metal_u64 dst_count;
} gd_metal_mul_reduce_suffix_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_mul_backward_direct_args) == 48U,
               "gd_metal_mul_backward_direct_args ABI mismatch");
_Static_assert(sizeof(gd_metal_mul_reduce_suffix_args) == 40U,
               "gd_metal_mul_reduce_suffix_args ABI mismatch");
#endif

#endif /* GD_OP_MUL_METAL_TYPES_H */
