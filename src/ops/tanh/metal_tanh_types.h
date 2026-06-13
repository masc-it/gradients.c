#ifndef GD_OP_TANH_METAL_TYPES_H
#define GD_OP_TANH_METAL_TYPES_H

/* Op-local Metal ABI types for tanh. Keep host/Metal layouts in sync. */
#include "../../backends/metal/metal_abi.h"

#define GD_METAL_TANH_ELEMENTS_PER_THREAD 4U

typedef struct gd_metal_tanh_args {
    gd_metal_u64 x_offset;    /* Forward/direct-backward input, or saved tanh output. */
    gd_metal_u64 y_offset;    /* Forward output, or backward grad_x output. */
    gd_metal_u64 grad_offset; /* Backward grad_out offset; zero for forward. */
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_tanh_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_tanh_args) == 40U, "gd_metal_tanh_args ABI mismatch");
#endif

#endif /* GD_OP_TANH_METAL_TYPES_H */
