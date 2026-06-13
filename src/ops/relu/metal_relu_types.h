#ifndef GD_OP_RELU_METAL_TYPES_H
#define GD_OP_RELU_METAL_TYPES_H

/* Op-local FP16-only Metal ABI types for ReLU. Keep host/Metal layouts in sync. */

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_RELU_ELEMENTS_PER_THREAD 4U

typedef struct gd_metal_relu_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;    /* Forward output offset, or backward grad_x offset. */
    gd_metal_u64 grad_offset; /* Backward grad_out offset; zero for forward. */
    gd_metal_u64 count;
} gd_metal_relu_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_relu_args) == 32U, "gd_metal_relu_args ABI mismatch");
#endif

#endif /* GD_OP_RELU_METAL_TYPES_H */
