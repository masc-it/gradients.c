#ifndef GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H
#define GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_POWLU_SPLIT_LINEAR_MAX_THREADS_PER_GROUP 128U

typedef struct gd_metal_powlu_split_linear_bwd_x12_args {
    gd_metal_u64 x12_offset;
    gd_metal_u64 w_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 dx12_offset;
    gd_metal_u64 x12_row_bytes;
    gd_metal_u64 w_row_bytes;
    gd_metal_u64 grad_row_bytes;
    gd_metal_u64 dx12_row_bytes;
    gd_metal_u32 rows;
    gd_metal_u32 hidden;
    gd_metal_u32 out_cols;
    gd_metal_u32 pad0;
    float m;
    gd_metal_u32 pad1[3];
} gd_metal_powlu_split_linear_bwd_x12_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_powlu_split_linear_bwd_x12_args) == 96U,
               "gd_metal_powlu_split_linear_bwd_x12_args ABI mismatch");
_Static_assert(offsetof(gd_metal_powlu_split_linear_bwd_x12_args, rows) == 64U,
               "gd_metal_powlu_split_linear_bwd_x12_args rows offset mismatch");
_Static_assert(offsetof(gd_metal_powlu_split_linear_bwd_x12_args, m) == 80U,
               "gd_metal_powlu_split_linear_bwd_x12_args m offset mismatch");
#endif

#endif /* GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H */
