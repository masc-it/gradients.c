#ifndef GD_CORE_BACKEND_H
#define GD_CORE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>

typedef struct gd_backend gd_backend;
typedef struct gd_backend_buffer gd_backend_buffer;

typedef struct gd_backend_fence {
    void *handle;
} gd_backend_fence;

typedef enum gd_backend_kind {
    GD_BACKEND_METAL = 1,
} gd_backend_kind;

typedef struct gd_backend_matrix_view {
    gd_backend_buffer *buffer;
    size_t offset;
    uint32_t rows;
    uint32_t cols;
    size_t row_bytes;
    uint32_t dtype;
} gd_backend_matrix_view;

typedef struct gd_backend_vector_view {
    gd_backend_buffer *buffer;
    size_t offset;
    uint32_t length;
    uint32_t dtype;
} gd_backend_vector_view;

typedef struct gd_backend_adamw_desc {
    gd_backend_buffer *param_buffer;
    size_t param_offset;
    gd_backend_buffer *grad_buffer;
    size_t grad_offset;
    gd_backend_buffer *m_buffer;
    size_t m_offset;
    gd_backend_buffer *v_buffer;
    size_t v_offset;
    size_t count;
    uint32_t param_dtype;
    uint32_t grad_dtype;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
} gd_backend_adamw_desc;

typedef struct gd_backend_tensor_view {
    gd_backend_buffer *buffer;
    size_t offset;
    size_t count;
    uint32_t dtype;
} gd_backend_tensor_view;

#include "backend_generated.h"

gd_status gd_backend_create_default(gd_backend **out_backend);
void gd_backend_destroy(gd_backend *backend);

gd_backend_kind gd_backend_kind_query(const gd_backend *backend);
const char *gd_backend_name(const gd_backend *backend);

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer);
void gd_backend_buffer_destroy(gd_backend_buffer *buffer);
size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer);
void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer);
bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer);

gd_status gd_backend_scope_begin(gd_backend *backend);
gd_status gd_backend_flush(gd_backend *backend);

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes);
gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes);

gd_status gd_backend_fill(gd_backend *backend,
                          gd_backend_buffer *buffer,
                          size_t offset,
                          size_t count,
                          size_t elem_size,
                          uint32_t pattern);
gd_status gd_backend_rand_uniform(gd_backend *backend,
                                  gd_backend_buffer *buffer,
                                  size_t offset,
                                  size_t count,
                                  uint32_t dtype,
                                  uint64_t seed,
                                  float low,
                                  float high);

gd_status gd_backend_matmul(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_matrix_view *y);
/* y = x * w^T, where w is stored row-major as (y.cols, x.cols). */
gd_status gd_backend_matmul_nt(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y);
/* y = x^T * w, where x is stored row-major as (inner, y.rows). */
gd_status gd_backend_matmul_tn(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y);
gd_status gd_backend_linear(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_vector_view *bias,
                            const gd_backend_matrix_view *y);
/* y[col] = sum_rows x[row, col]. */
gd_status gd_backend_reduce_rows(gd_backend *backend,
                                 const gd_backend_matrix_view *x,
                                 const gd_backend_vector_view *y);
/* dst[i] += src[i] for contiguous tensors, dtype values from gd_dtype. */
gd_status gd_backend_accumulate(gd_backend *backend,
                                gd_backend_buffer *dst_buffer,
                                size_t dst_offset,
                                gd_backend_buffer *src_buffer,
                                size_t src_offset,
                                size_t count,
                                uint32_t dtype);
gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc);

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence);
void gd_backend_fence_destroy(gd_backend_fence *fence);
bool gd_backend_fence_is_complete(gd_backend_fence *fence);
gd_status gd_backend_fence_wait(gd_backend_fence *fence);

#endif /* GD_CORE_BACKEND_H */
