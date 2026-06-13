#ifndef GD_OP_ROPE_METAL_TYPES_H
#define GD_OP_ROPE_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

typedef struct gd_metal_rope_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 pos_offset;
    gd_metal_u64 out_offset;
    gd_metal_u32 rows;
    gd_metal_u32 head_dim;
    gd_metal_u32 heads;
    gd_metal_u32 n_dims;
    gd_metal_u32 lanes_per_row;
    gd_metal_u32 interleaved;
    float sin_sign;
    float freq_scale;
} gd_metal_rope_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_rope_args) == 56U, "gd_metal_rope_args ABI mismatch");
#endif

#endif /* GD_OP_ROPE_METAL_TYPES_H */
