#ifndef GD_METAL_RANDOM_TYPES_H
#define GD_METAL_RANDOM_TYPES_H

#include "../../metal_abi.h"

typedef struct gd_metal_rand_uniform_args {
    gd_metal_u64 byte_offset;
    gd_metal_u64 count;
    gd_metal_u32 dtype;
    gd_metal_u32 pad0;
    gd_metal_u64 seed;
    float low;
    float high;
} gd_metal_rand_uniform_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_rand_uniform_args) == 40U, "gd_metal_rand_uniform_args ABI mismatch");
#endif

#endif /* GD_METAL_RANDOM_TYPES_H */
