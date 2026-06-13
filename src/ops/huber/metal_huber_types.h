#ifndef GD_OP_HUBER_METAL_TYPES_H
#define GD_OP_HUBER_METAL_TYPES_H

#include "../../backends/metal/metal_abi.h"

#define GD_METAL_HUBER_CHUNK_SIZE 4096U
#define GD_METAL_HUBER_MAX_SIMDGROUPS 8U

typedef struct gd_metal_huber_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 out_offset;      /* forward output or backward dx output */
    gd_metal_u64 grad_out_offset; /* backward scalar f32 grad_out */
    gd_metal_u64 dy_offset;       /* backward dy output */
    gd_metal_u64 count;
    gd_metal_u64 chunk_size;
    float scale;
    float delta;
    gd_metal_u32 write_x;
    gd_metal_u32 write_y;
    gd_metal_u32 simdgroups;
    gd_metal_u32 reserved0;
} gd_metal_huber_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_huber_args) == 80U,
               "gd_metal_huber_args ABI mismatch");
#endif

#endif /* GD_OP_HUBER_METAL_TYPES_H */
