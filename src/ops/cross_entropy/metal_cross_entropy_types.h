#ifndef GD_OP_CROSS_ENTROPY_METAL_TYPES_H
#define GD_OP_CROSS_ENTROPY_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_CROSS_ENTROPY_MAX_SIMDGROUPS 8U

typedef struct gd_metal_cross_entropy_args {
    gd_metal_u64 logits_offset;
    gd_metal_u64 target_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 grad_out_offset;
    gd_metal_u64 row_max_offset;
    gd_metal_u64 row_inv_sum_offset;
    gd_metal_u64 rows;
    gd_metal_u64 classes;
    float scale;
    gd_metal_u32 simdgroups;
} gd_metal_cross_entropy_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_cross_entropy_args) == 72U,
               "gd_metal_cross_entropy_args ABI mismatch");
#endif

#endif /* GD_OP_CROSS_ENTROPY_METAL_TYPES_H */
