#ifndef GD_METAL_KERNEL_TYPES_H
#define GD_METAL_KERNEL_TYPES_H

#define GD_METAL_GEMM_BM 64U
#define GD_METAL_GEMM_BN 64U
#define GD_METAL_GEMM_BK 8U
#define GD_METAL_GEMM_TM 4U
#define GD_METAL_GEMM_TN 4U
#define GD_METAL_GEMM_REG_NBLK 4U
#define GD_METAL_GEMM_REG_TILE (GD_METAL_GEMM_REG_NBLK * 8U)
#define GD_METAL_GEMM_REG_SIMDGROUPS 4U
#define GD_METAL_REDUCE_ROWS_SIMDGROUPS 8U

#ifdef __METAL_VERSION__

typedef struct gd_metal_fill_args {
    ulong byte_offset;
    ulong count;
    uint elem_size;
    uint pattern;
} gd_metal_fill_args;

typedef struct gd_metal_rand_uniform_args {
    ulong byte_offset;
    ulong count;
    uint dtype;
    uint pad0;
    ulong seed;
    float low;
    float high;
} gd_metal_rand_uniform_args;

typedef struct gd_metal_gemm_args {
    ulong x_offset;
    ulong w_offset;
    ulong bias_offset;
    ulong y_offset;
    ulong x_row_bytes;
    ulong w_row_bytes;
    ulong y_row_bytes;
    uint rows;
    uint cols;
    uint inner;
    uint has_bias;
} gd_metal_gemm_args;

typedef struct gd_metal_reduce_rows_args {
    ulong x_offset;
    ulong y_offset;
    ulong x_row_bytes;
    uint rows;
    uint cols;
    uint pad0;
} gd_metal_reduce_rows_args;

typedef struct gd_metal_accumulate_args {
    ulong dst_offset;
    ulong src_offset;
    ulong count;
    uint dtype;
    uint pad0;
} gd_metal_accumulate_args;

#else

#include <stddef.h>
#include <stdint.h>

typedef struct gd_metal_fill_args {
    uint64_t byte_offset;
    uint64_t count;
    uint32_t elem_size;
    uint32_t pattern;
} gd_metal_fill_args;

typedef struct gd_metal_rand_uniform_args {
    uint64_t byte_offset;
    uint64_t count;
    uint32_t dtype;
    uint32_t pad0;
    uint64_t seed;
    float low;
    float high;
} gd_metal_rand_uniform_args;

typedef struct gd_metal_gemm_args {
    uint64_t x_offset;
    uint64_t w_offset;
    uint64_t bias_offset;
    uint64_t y_offset;
    uint64_t x_row_bytes;
    uint64_t w_row_bytes;
    uint64_t y_row_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t inner;
    uint32_t has_bias;
} gd_metal_gemm_args;

typedef struct gd_metal_reduce_rows_args {
    uint64_t x_offset;
    uint64_t y_offset;
    uint64_t x_row_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t pad0;
} gd_metal_reduce_rows_args;

typedef struct gd_metal_accumulate_args {
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t count;
    uint32_t dtype;
    uint32_t pad0;
} gd_metal_accumulate_args;

#endif /* __METAL_VERSION__ */

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_metal_fill_args) == 24U, "gd_metal_fill_args ABI mismatch");
_Static_assert(sizeof(gd_metal_rand_uniform_args) == 40U, "gd_metal_rand_uniform_args ABI mismatch");
_Static_assert(sizeof(gd_metal_gemm_args) == 72U, "gd_metal_gemm_args ABI mismatch");
_Static_assert(sizeof(gd_metal_accumulate_args) == 32U, "gd_metal_accumulate_args ABI mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, rows) == 56U, "gd_metal_gemm_args rows offset mismatch");
_Static_assert(offsetof(gd_metal_gemm_args, has_bias) == 68U, "gd_metal_gemm_args has_bias offset mismatch");
#endif

#endif /* GD_METAL_KERNEL_TYPES_H */
