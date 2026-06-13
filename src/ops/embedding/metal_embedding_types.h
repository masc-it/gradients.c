#ifndef GD_METAL_EMBEDDING_TYPES_H
#define GD_METAL_EMBEDDING_TYPES_H

#include "../../backends/metal/metal_abi.h"

typedef struct gd_metal_embedding_args {
    gd_metal_u64 table_offset;
    gd_metal_u64 ids_offset;
    gd_metal_u64 out_offset;
    gd_metal_u64 grad_out_offset;
    gd_metal_u64 grad_table_offset;
    gd_metal_u64 scratch_offset;
    gd_metal_u64 ids_count;
    gd_metal_u64 vocab;
    gd_metal_u64 dim;
} gd_metal_embedding_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_embedding_args) == 72U,
               "gd_metal_embedding_args ABI size mismatch");
#endif

#endif /* GD_METAL_EMBEDDING_TYPES_H */
