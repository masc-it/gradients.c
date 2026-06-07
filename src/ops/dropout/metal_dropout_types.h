#ifndef GD_OP_DROPOUT_METAL_TYPES_H
#define GD_OP_DROPOUT_METAL_TYPES_H

/* Op-local Metal ABI types for inverted dropout. */

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_DROPOUT_ELEMENTS_PER_THREAD 4U

typedef struct gd_metal_dropout_args {
    gd_metal_u64 x_offset;    /* Forward input or direct-backward grad_out offset. */
    gd_metal_u64 y_offset;    /* Forward output or backward grad_x offset. */
    gd_metal_u64 mask_offset; /* Forward/saved-backward mask offset; zero for recompute backward. */
    gd_metal_u64 count;
    gd_metal_u64 seed;
    float p;
    float scale;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_dropout_args;

typedef struct gd_metal_dropout_add_args {
    gd_metal_u64 residual_offset;
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 mask_offset;
    gd_metal_u64 count;
    gd_metal_u64 seed;
    float p;
    float scale;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
} gd_metal_dropout_add_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_dropout_args) == 56U, "gd_metal_dropout_args ABI mismatch");
_Static_assert(sizeof(gd_metal_dropout_add_args) == 64U, "gd_metal_dropout_add_args ABI mismatch");
#endif

#endif /* GD_OP_DROPOUT_METAL_TYPES_H */
