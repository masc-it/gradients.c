#ifndef GRADIENTS_TENSOR_H
#define GRADIENTS_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/memory.h>
#include <gradients/status.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_MAX_DIMS 8U

typedef enum gd_dtype {
    GD_DTYPE_INVALID = 0,
    GD_DTYPE_F16 = 1,
    GD_DTYPE_BF16 = 2,
    GD_DTYPE_F32 = 3,
    GD_DTYPE_I32 = 4,
    GD_DTYPE_U8 = 5,
} gd_dtype;

typedef enum gd_device {
    GD_DEVICE_INVALID = 0,
    GD_DEVICE_GPU = 1,
} gd_device;

typedef enum gd_layout {
    GD_LAYOUT_STRIDED = 0,
} gd_layout;

/* Tensor descriptors are caller-owned metadata. Tensor bytes live in arena storage. */
typedef struct gd_tensor {
    gd_dtype dtype;
    gd_device device;
    gd_layout layout;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    int64_t strides[GD_MAX_DIMS];
    gd_span storage;
    size_t view_offset;
    bool is_view;
    bool requires_grad;
} gd_tensor;

size_t gd_dtype_size(gd_dtype dtype);
const char *gd_dtype_name(gd_dtype dtype);

gd_status gd_tensor_empty(gd_context *ctx,
                          gd_arena_kind arena,
                          gd_dtype dtype,
                          uint32_t rank,
                          const int64_t *shape,
                          size_t alignment,
                          gd_tensor *out);
gd_status gd_tensor_zeros(gd_context *ctx,
                          gd_arena_kind arena,
                          gd_dtype dtype,
                          uint32_t rank,
                          const int64_t *shape,
                          size_t alignment,
                          gd_tensor *out);
gd_status gd_tensor_ones(gd_context *ctx,
                         gd_arena_kind arena,
                         gd_dtype dtype,
                         uint32_t rank,
                         const int64_t *shape,
                         size_t alignment,
                         gd_tensor *out);
gd_status gd_tensor_rand(gd_context *ctx,
                         gd_arena_kind arena,
                         gd_dtype dtype,
                         uint32_t rank,
                         const int64_t *shape,
                         size_t alignment,
                         uint64_t seed,
                         gd_tensor *out);
gd_status gd_tensor_rand_uniform(gd_context *ctx,
                                 gd_arena_kind arena,
                                 gd_dtype dtype,
                                 uint32_t rank,
                                 const int64_t *shape,
                                 size_t alignment,
                                 uint64_t seed,
                                 float low,
                                 float high,
                                 gd_tensor *out);

gd_status gd_tensor_slice(gd_context *ctx,
                          const gd_tensor *base,
                          uint32_t dim,
                          int64_t start,
                          int64_t length,
                          gd_tensor *out);

/* Allocates packed output storage/descriptor. Data materialization is backend op work. */
gd_status gd_tensor_contiguous(gd_context *ctx,
                               gd_arena_kind arena,
                               const gd_tensor *src,
                               size_t alignment,
                               gd_tensor *out);

gd_status gd_tensor_zero_(gd_context *ctx, gd_tensor *tensor);
gd_status gd_tensor_one_(gd_context *ctx, gd_tensor *tensor);
gd_status gd_tensor_rand_(gd_context *ctx, gd_tensor *tensor, uint64_t seed);
gd_status gd_tensor_rand_uniform_(gd_context *ctx,
                                  gd_tensor *tensor,
                                  uint64_t seed,
                                  float low,
                                  float high);

bool gd_tensor_is_contiguous(const gd_tensor *tensor);
size_t gd_tensor_storage_offset(const gd_tensor *tensor);

gd_status gd_tensor_numel(const gd_tensor *tensor, int64_t *out_numel);
gd_status gd_tensor_logical_nbytes(const gd_tensor *tensor, size_t *out_nbytes);
gd_status gd_tensor_validate(gd_context *ctx, const gd_tensor *tensor);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TENSOR_H */
