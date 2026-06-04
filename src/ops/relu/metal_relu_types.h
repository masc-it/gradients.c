#ifndef GD_OP_RELU_METAL_TYPES_H
#define GD_OP_RELU_METAL_TYPES_H

/* Op-local Metal ABI types for ReLU. Keep host/Metal layouts in sync. */

#include "../../backends/metal/metal_abi.h"

typedef struct gd_metal_relu_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 grad_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_relu_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_relu_args) == 40U, "gd_metal_relu_args ABI mismatch");
#endif

#endif /* GD_OP_RELU_METAL_TYPES_H */
