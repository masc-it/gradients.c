#ifndef GD_BACKENDS_METAL_PRIMITIVES_AMP_TYPES_H
#define GD_BACKENDS_METAL_PRIMITIVES_AMP_TYPES_H

#include "../../metal_abi.h"

typedef struct gd_metal_amp_state_args {
    gd_metal_u64 scale_offset;
    gd_metal_u64 flags_offset;
    float growth_factor;
    float backoff_factor;
    float min_scale;
    float max_scale;
    gd_metal_u32 growth_interval;
    gd_metal_u32 pad0;
} gd_metal_amp_state_args;

typedef struct gd_metal_amp_tensor_args {
    gd_metal_u64 dst_offset;
    gd_metal_u64 src_offset;
    gd_metal_u64 scale_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_amp_tensor_args;

typedef struct gd_metal_amp_unscale_args {
    gd_metal_u64 grad_offset;
    gd_metal_u64 inv_scale_offset;
    gd_metal_u64 found_inf_offset;
    gd_metal_u64 count;
    gd_metal_u32 grad_dtype;
    gd_metal_u32 pad0;
} gd_metal_amp_unscale_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_amp_state_args) == 40U, "gd_metal_amp_state_args ABI mismatch");
_Static_assert(sizeof(gd_metal_amp_tensor_args) == 40U, "gd_metal_amp_tensor_args ABI mismatch");
_Static_assert(sizeof(gd_metal_amp_unscale_args) == 40U, "gd_metal_amp_unscale_args ABI mismatch");
_Static_assert(offsetof(gd_metal_amp_unscale_args, count) == 24U,
               "gd_metal_amp_unscale_args count offset mismatch");
#endif

#endif /* GD_BACKENDS_METAL_PRIMITIVES_AMP_TYPES_H */
