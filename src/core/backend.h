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
    gd_backend_buffer *master_buffer; /* Optional FP32 master weights. */
    size_t master_offset;
    gd_backend_buffer *grad_buffer;
    size_t grad_offset;
    gd_backend_buffer *m_buffer;
    size_t m_offset;
    gd_backend_buffer *v_buffer;
    size_t v_offset;
    size_t count;
    uint32_t param_dtype;
    uint32_t grad_dtype;
    uint32_t has_master;
    uint32_t pad0;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
} gd_backend_adamw_desc;

typedef struct gd_backend_amp_unscale_desc {
    gd_backend_buffer *grad_buffer;
    size_t grad_offset;
    size_t count;
    uint32_t grad_dtype;
    float inv_scale;
    gd_backend_buffer *found_inf_buffer;
    size_t found_inf_offset;
} gd_backend_amp_unscale_desc;

typedef struct gd_backend_tensor_view {
    gd_backend_buffer *buffer;
    size_t offset;
    size_t count;
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[8];
    int64_t strides[8];
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
/* dst[i] = src[i] * scale for contiguous tensors, dtype values from gd_dtype. */
gd_status gd_backend_scale(gd_backend *backend,
                           gd_backend_buffer *dst_buffer,
                           size_t dst_offset,
                           gd_backend_buffer *src_buffer,
                           size_t src_offset,
                           size_t count,
                           uint32_t dtype,
                           float scale);
/* dst[j] = scale * sum(src chunk j) for contiguous all-reduce staging. */
gd_status gd_backend_reduce_contiguous(gd_backend *backend,
                                       const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst,
                                       float scale);
/* Reduce one axis of a contiguous tensor; dst is src with axis removed or kept as size 1. */
gd_status gd_backend_reduce_axis(gd_backend *backend,
                                 const gd_backend_tensor_view *src,
                                 const gd_backend_tensor_view *dst,
                                 uint32_t axis,
                                 float scale);
/* Broadcast a reduced-axis tensor back to dst = src expanded along axis, scaled. */
gd_status gd_backend_broadcast_axis(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    uint32_t axis,
                                    float scale);
/* dst = broadcast_to(src, dst.shape) * scale for contiguous tensors. */
gd_status gd_backend_broadcast_to(gd_backend *backend,
                                  const gd_backend_tensor_view *src,
                                  const gd_backend_tensor_view *dst,
                                  float scale);
/* Sigmoid autograd fast path: grad_x = grad_out * y * (1 - y), using saved forward y. */
gd_status gd_backend_sigmoid_backward_from_output(gd_backend *backend,
                                                  const gd_backend_tensor_view *sigmoid_out,
                                                  const gd_backend_tensor_view *grad_out,
                                                  const gd_backend_tensor_view *grad_x);
/* Inverted dropout forward: y = x * mask / (1 - p), writing a compact u8 mask. */
gd_status gd_backend_dropout_forward(gd_backend *backend,
                                     const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *y,
                                     const gd_backend_tensor_view *mask,
                                     float p,
                                     uint64_t seed);
/* Direct dropout backward recomputes the same stateless mask from p/seed. */
gd_status gd_backend_dropout_backward(gd_backend *backend,
                                      const gd_backend_tensor_view *grad_out,
                                      const gd_backend_tensor_view *grad_x,
                                      float p,
                                      uint64_t seed);
/* Autograd dropout backward consumes the saved compact u8 forward mask. */
gd_status gd_backend_dropout_backward_mask(gd_backend *backend,
                                           const gd_backend_tensor_view *mask,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_x,
                                           float scale);
/* dst[i] = src[0] * scale for contiguous tensors; supports same dtype and f32 scalar to f16 dst. */
gd_status gd_backend_broadcast_scalar(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale);
/* dst = sum_to_shape(src) * scale for contiguous broadcast-compatible tensors. */
gd_status gd_backend_reduce_broadcast(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale);
/* F16-only mul fast path: grad_x = grad_out * y and grad_y = grad_out * x for direct shapes. */
gd_status gd_backend_mul_backward_direct(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_x,
                                         const gd_backend_tensor_view *grad_y);
/* F16-only mul fast path: dst[j] = sum_r grad_out[r, j] * other[r, j] for suffix reductions. */
gd_status gd_backend_mul_reduce_suffix(gd_backend *backend,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *other,
                                       const gd_backend_tensor_view *dst);
/* F16-only: row_loss[n] = logsumexp(logits[n, :]) - logits[n, target[n]]. */
gd_status gd_backend_cross_entropy_loss(gd_backend *backend,
                                        const gd_backend_tensor_view *logits,
                                        const gd_backend_tensor_view *targets,
                                        const gd_backend_tensor_view *row_loss);
/* F16-only training fast path: also saves row max and reciprocal exp-sum. */
gd_status gd_backend_cross_entropy_loss_stats(gd_backend *backend,
                                              const gd_backend_tensor_view *logits,
                                              const gd_backend_tensor_view *targets,
                                              const gd_backend_tensor_view *row_loss,
                                              const gd_backend_tensor_view *row_max,
                                              const gd_backend_tensor_view *row_inv_sum);
/* F16-only direct backward: recomputes row stats. */
gd_status gd_backend_cross_entropy_backward(gd_backend *backend,
                                            const gd_backend_tensor_view *logits,
                                            const gd_backend_tensor_view *targets,
                                            const gd_backend_tensor_view *grad_loss,
                                            const gd_backend_tensor_view *grad_logits,
                                            float scale);
/* F16-only autograd fast path: consumes forward-saved row stats. */
gd_status gd_backend_cross_entropy_backward_stats(gd_backend *backend,
                                                  const gd_backend_tensor_view *logits,
                                                  const gd_backend_tensor_view *targets,
                                                  const gd_backend_tensor_view *row_max,
                                                  const gd_backend_tensor_view *row_inv_sum,
                                                  const gd_backend_tensor_view *grad_loss,
                                                  const gd_backend_tensor_view *grad_logits,
                                                  float scale);
/* MSE loss: out chunks contain scale * sum((x - y)^2) over each contiguous chunk. */
gd_status gd_backend_mse_forward(gd_backend *backend,
                                 const gd_backend_tensor_view *x,
                                 const gd_backend_tensor_view *y,
                                 const gd_backend_tensor_view *out,
                                 uint64_t chunk_size,
                                 float scale);
/* MSE backward: optional grad_x/grad_y receive +/- grad_out[0] * scale * (x - y). */
gd_status gd_backend_mse_backward(gd_backend *backend,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *y,
                                  const gd_backend_tensor_view *grad_out,
                                  const gd_backend_tensor_view *grad_x,
                                  const gd_backend_tensor_view *grad_y,
                                  float scale);
/* Huber loss: out chunks contain scale * sum(huber_delta(x - y)) over each contiguous chunk. */
gd_status gd_backend_huber_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x,
                                   const gd_backend_tensor_view *y,
                                   const gd_backend_tensor_view *out,
                                   uint64_t chunk_size,
                                   float scale,
                                   float delta);
/* Huber backward: optional grad_x/grad_y receive +/- grad_out[0] * scale * clamp(x - y, -delta, delta). */
gd_status gd_backend_huber_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x,
                                    const gd_backend_tensor_view *y,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x,
                                    const gd_backend_tensor_view *grad_y,
                                    float scale,
                                    float delta);

typedef struct gd_backend_concat_args {
    uint64_t count;       /* Number of elements in the slice tensor. */
    uint64_t inner;       /* Product of dimensions after the concat axis. */
    uint64_t slice_axis;  /* Axis length of the slice tensor. */
    uint64_t full_axis;   /* Axis length of the materialized concat tensor. */
    uint64_t axis_offset; /* Slice axis offset within the full tensor. */
} gd_backend_concat_args;

/* Copy contiguous src slice into its position in contiguous dst concat output. */
gd_status gd_backend_concat_to_full(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    const gd_backend_concat_args *args);
/* Copy one contiguous grad slice out of a contiguous full concat gradient. */
gd_status gd_backend_concat_from_full(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      const gd_backend_concat_args *args);

typedef struct gd_backend_sdpa_varlen_args {
    float scale;
    uint32_t causal;
    uint32_t sliding_window;
    uint32_t prefix_len;
    uint32_t max_seqlen;
} gd_backend_sdpa_varlen_args;

typedef struct gd_backend_sdpa_decode_args {
    float scale;
    uint32_t sliding_window;
    uint32_t prefix_len;
} gd_backend_sdpa_decode_args;

gd_status gd_backend_sdpa_varlen(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k,
                                 const gd_backend_tensor_view *v,
                                 const gd_backend_tensor_view *cu_seqlens,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_sdpa_varlen_args *args);
gd_status gd_backend_sdpa_varlen_backward(gd_backend *backend,
                                          const gd_backend_tensor_view *grad_out,
                                          const gd_backend_tensor_view *q,
                                          const gd_backend_tensor_view *k,
                                          const gd_backend_tensor_view *v,
                                          const gd_backend_tensor_view *cu_seqlens,
                                          const gd_backend_tensor_view *grad_q,
                                          const gd_backend_tensor_view *grad_k,
                                          const gd_backend_tensor_view *grad_v,
                                          const gd_backend_tensor_view *stats,
                                          const gd_backend_sdpa_varlen_args *args);
gd_status gd_backend_sdpa_decode(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k_cache,
                                 const gd_backend_tensor_view *v_cache,
                                 const gd_backend_tensor_view *cache_pos,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_sdpa_decode_args *args);

gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc);
gd_status gd_backend_amp_unscale(gd_backend *backend, const gd_backend_amp_unscale_desc *desc);

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence);
void gd_backend_fence_destroy(gd_backend_fence *fence);
bool gd_backend_fence_is_complete(gd_backend_fence *fence);
gd_status gd_backend_fence_wait(gd_backend_fence *fence);

#endif /* GD_CORE_BACKEND_H */
