#ifndef GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H
#define GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H

/* Op-local Metal ABI types for powlu_split_linear. Keep host/Metal layouts in sync.
 * See docs/guides/metal_tips.md before implementing Metal hot paths.
 * Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
 */
#include "../../backends/metal/metal_abi.h"

typedef struct gd_metal_powlu_split_linear_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_powlu_split_linear_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_powlu_split_linear_args) == 40U, "gd_metal_powlu_split_linear_args ABI mismatch");
#endif

#endif /* GD_OP_POWLU_SPLIT_LINEAR_METAL_TYPES_H */
