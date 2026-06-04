#ifndef GD_OP_LINEAR_METAL_TYPES_H
#define GD_OP_LINEAR_METAL_TYPES_H

#include "../_shared/gemm/metal_gemm_types.h"

#define GD_METAL_REDUCE_ROWS_SIMDGROUPS 8U

typedef struct gd_metal_reduce_rows_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 x_row_bytes;
    gd_metal_u32 rows;
    gd_metal_u32 cols;
    gd_metal_u32 pad0;
} gd_metal_reduce_rows_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_reduce_rows_args) == 40U, "gd_metal_reduce_rows_args ABI mismatch");
#endif

#endif /* GD_OP_LINEAR_METAL_TYPES_H */
