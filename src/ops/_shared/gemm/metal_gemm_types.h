#ifndef GD_OPS_SHARED_GEMM_METAL_TYPES_H
#define GD_OPS_SHARED_GEMM_METAL_TYPES_H

#include "../../../backends/metal/metal_abi.h"

#define GD_METAL_GEMM_BM 64U
#define GD_METAL_GEMM_BN 64U
#define GD_METAL_GEMM_BK 8U
#define GD_METAL_GEMM_TM 4U
#define GD_METAL_GEMM_TN 4U
#define GD_METAL_GEMM_REG_NBLK 4U
#define GD_METAL_GEMM_REG_TILE (GD_METAL_GEMM_REG_NBLK * 8U)
#define GD_METAL_GEMM_REG_SIMDGROUPS 4U
#define GD_METAL_GEMM_TN_SPLIT8 8U
#define GD_METAL_GEMM_MAX_BATCH_DIMS 6U

typedef struct gd_metal_gemm_args {
    gd_metal_u64 x_offset;
    gd_metal_u64 w_offset;
    gd_metal_u64 bias_offset;
    gd_metal_u64 y_offset;
    gd_metal_u64 x_row_bytes;
    gd_metal_u64 w_row_bytes;
    gd_metal_u64 y_row_bytes;
    gd_metal_u32 rows;
    gd_metal_u32 cols;
    gd_metal_u32 inner;
    gd_metal_u32 has_bias;
    gd_metal_u32 batch_rank;
    gd_metal_u32 pad0;
    gd_metal_u64 x_batch_strides[GD_METAL_GEMM_MAX_BATCH_DIMS];
    gd_metal_u64 w_batch_strides[GD_METAL_GEMM_MAX_BATCH_DIMS];
    gd_metal_u64 y_batch_strides[GD_METAL_GEMM_MAX_BATCH_DIMS];
    gd_metal_u32 batch_shape[GD_METAL_GEMM_MAX_BATCH_DIMS];
    gd_metal_u32 pad1[2];
} gd_metal_gemm_args;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_gemm_args) == 256U, "gd_metal_gemm_args ABI mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, rows) == 56U, "gd_metal_gemm_args rows offset mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, has_bias) == 68U, "gd_metal_gemm_args has_bias offset mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, batch_rank) == 72U, "gd_metal_gemm_args batch_rank offset mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, x_batch_strides) == 80U, "gd_metal_gemm_args x batch offset mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, batch_shape) == 224U, "gd_metal_gemm_args batch shape offset mismatch");
#endif

#endif /* GD_OPS_SHARED_GEMM_METAL_TYPES_H */
