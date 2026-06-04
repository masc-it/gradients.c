#ifndef GD_METAL_AMP_TYPES_H
#define GD_METAL_AMP_TYPES_H

#include "../../metal_abi.h"

typedef struct gd_metal_amp_unscale_args {
    gd_metal_u64 grad_offset;
    gd_metal_u64 found_inf_offset;
    gd_metal_u64 count;
    gd_metal_u32 grad_dtype;
    gd_metal_u32 pad0;
    float inv_scale;
    float pad1;
} gd_metal_amp_unscale_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_amp_unscale_args) == 40U, "gd_metal_amp_unscale_args ABI mismatch");
_Static_assert(offsetof(gd_metal_amp_unscale_args, inv_scale) == 32U,
               "gd_metal_amp_unscale_args inv_scale offset mismatch");
#endif

#endif /* GD_METAL_AMP_TYPES_H */
