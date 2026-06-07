#ifndef GD_OP_POWLU_METAL_TYPES_H
#define GD_OP_POWLU_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_POWLU_ELEMENTS_PER_THREAD 1U
#define GD_METAL_POWLU_MAX_THREADS_PER_GROUP 256U

typedef struct gd_metal_powlu_fwd_args {
    gd_metal_u64 x1_offset;
    gd_metal_u64 x2_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 count;
    float m;
    gd_metal_u32 reserved0;
} gd_metal_powlu_fwd_args;

typedef struct gd_metal_powlu_bwd_args {
    gd_metal_u64 x1_offset;
    gd_metal_u64 x2_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 dx1_offset;
    gd_metal_u64 dx2_offset;
    gd_metal_u64 count;
    float m;
    gd_metal_u32 write_x1;
    gd_metal_u32 write_x2;
    gd_metal_u32 reserved0;
} gd_metal_powlu_bwd_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_powlu_fwd_args) == 40U,
               "gd_metal_powlu_fwd_args ABI mismatch");
_Static_assert(sizeof(gd_metal_powlu_bwd_args) == 64U,
               "gd_metal_powlu_bwd_args ABI mismatch");
#endif

#endif /* GD_OP_POWLU_METAL_TYPES_H */
