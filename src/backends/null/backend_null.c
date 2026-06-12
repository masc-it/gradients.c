#include "../../core/backend.h"

struct gd_backend {
    int unused;
};

struct gd_backend_buffer {
    int unused;
};

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    if (out_backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_backend = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_destroy(gd_backend *backend)
{
    (void)backend;
}

gd_backend_kind gd_backend_kind_query(const gd_backend *backend)
{
    (void)backend;
    return 0;
}

const char *gd_backend_name(const gd_backend *backend)
{
    (void)backend;
    return "none";
}

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer)
{
    (void)backend;
    (void)nbytes;
    if (out_buffer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_buffer_destroy(gd_backend_buffer *buffer)
{
    (void)buffer;
}

size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer)
{
    (void)buffer;
    return 0U;
}

void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer)
{
    (void)buffer;
    return 0;
}

bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer)
{
    (void)buffer;
    return false;
}

gd_status gd_backend_scope_begin(gd_backend *backend)
{
    (void)backend;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_flush(gd_backend *backend)
{
    (void)backend;
    return GD_OK;
}

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)src;
    (void)nbytes;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)dst;
    (void)nbytes;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_fill(gd_backend *backend,
                          gd_backend_buffer *buffer,
                          size_t offset,
                          size_t count,
                          size_t elem_size,
                          uint32_t pattern)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)count;
    (void)elem_size;
    (void)pattern;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rand_uniform(gd_backend *backend,
                                  gd_backend_buffer *buffer,
                                  size_t offset,
                                  size_t count,
                                  uint32_t dtype,
                                  uint64_t seed,
                                  float low,
                                  float high)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)count;
    (void)dtype;
    (void)seed;
    (void)low;
    (void)high;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_matmul(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_matmul_nt(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_matmul_tn(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_batched_matmul(gd_backend *backend,
                                    const gd_backend_batched_matrix_view *x,
                                    const gd_backend_batched_matrix_view *w,
                                    const gd_backend_batched_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_batched_matmul_nt(gd_backend *backend,
                                       const gd_backend_batched_matrix_view *x,
                                       const gd_backend_batched_matrix_view *w,
                                       const gd_backend_batched_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_batched_matmul_tn(gd_backend *backend,
                                       const gd_backend_batched_matrix_view *x,
                                       const gd_backend_batched_matrix_view *w,
                                       const gd_backend_batched_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_linear(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_vector_view *bias,
                            const gd_backend_matrix_view *y)
{
    (void)backend;
    (void)x;
    (void)w;
    (void)bias;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_reduce_rows(gd_backend *backend,
                                 const gd_backend_matrix_view *x,
                                 const gd_backend_vector_view *y)
{
    (void)backend;
    (void)x;
    (void)y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_accumulate(gd_backend *backend,
                                gd_backend_buffer *dst_buffer,
                                size_t dst_offset,
                                gd_backend_buffer *src_buffer,
                                size_t src_offset,
                                size_t count,
                                uint32_t dtype)
{
    (void)backend;
    (void)dst_buffer;
    (void)dst_offset;
    (void)src_buffer;
    (void)src_offset;
    (void)count;
    (void)dtype;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_scale(gd_backend *backend,
                           gd_backend_buffer *dst_buffer,
                           size_t dst_offset,
                           gd_backend_buffer *src_buffer,
                           size_t src_offset,
                           size_t count,
                           uint32_t dtype,
                           float scale)
{
    (void)backend;
    (void)dst_buffer;
    (void)dst_offset;
    (void)src_buffer;
    (void)src_offset;
    (void)count;
    (void)dtype;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_reduce_contiguous(gd_backend *backend,
                                       const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst,
                                       float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_reduce_axis(gd_backend *backend,
                                 const gd_backend_tensor_view *src,
                                 const gd_backend_tensor_view *dst,
                                 uint32_t axis,
                                 float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)axis;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_broadcast_axis(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    uint32_t axis,
                                    float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)axis;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_broadcast_to(gd_backend *backend,
                                  const gd_backend_tensor_view *src,
                                  const gd_backend_tensor_view *dst,
                                  float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_sigmoid_backward_from_output(gd_backend *backend,
                                                  const gd_backend_tensor_view *sigmoid_out,
                                                  const gd_backend_tensor_view *grad_out,
                                                  const gd_backend_tensor_view *grad_x)
{
    (void)backend;
    (void)sigmoid_out;
    (void)grad_out;
    (void)grad_x;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_tanh_backward_from_output(gd_backend *backend,
                                               const gd_backend_tensor_view *tanh_out,
                                               const gd_backend_tensor_view *grad_out,
                                               const gd_backend_tensor_view *grad_x)
{
    (void)backend;
    (void)tanh_out;
    (void)grad_out;
    (void)grad_x;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_powlu_split_linear_backward_x12(gd_backend *backend,
                                                     const gd_backend_tensor_view *x12,
                                                     const gd_backend_matrix_view *w,
                                                     const gd_backend_matrix_view *grad_out,
                                                     const gd_backend_tensor_view *grad_x12,
                                                     float m)
{
    (void)backend;
    (void)x12;
    (void)w;
    (void)grad_out;
    (void)grad_x12;
    (void)m;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_dropout_forward(gd_backend *backend,
                                     const gd_backend_tensor_view *x,
                                     const gd_backend_tensor_view *y,
                                     const gd_backend_tensor_view *mask,
                                     float p,
                                     uint64_t seed)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)mask;
    (void)p;
    (void)seed;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_dropout_add_forward(gd_backend *backend,
                                         const gd_backend_tensor_view *residual,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *mask,
                                         float p,
                                         uint64_t seed)
{
    (void)backend;
    (void)residual;
    (void)x;
    (void)y;
    (void)mask;
    (void)p;
    (void)seed;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_dropout_backward(gd_backend *backend,
                                      const gd_backend_tensor_view *grad_out,
                                      const gd_backend_tensor_view *grad_x,
                                      float p,
                                      uint64_t seed)
{
    (void)backend;
    (void)grad_out;
    (void)grad_x;
    (void)p;
    (void)seed;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_dropout_backward_mask(gd_backend *backend,
                                           const gd_backend_tensor_view *mask,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_x,
                                           float scale)
{
    (void)backend;
    (void)mask;
    (void)grad_out;
    (void)grad_x;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_embedding_forward(gd_backend *backend,
                                       const gd_backend_tensor_view *table,
                                       const gd_backend_tensor_view *ids,
                                       const gd_backend_tensor_view *out,
                                       const gd_backend_embedding_args *args)
{
    (void)backend;
    (void)table;
    (void)ids;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_embedding_backward(gd_backend *backend,
                                        const gd_backend_tensor_view *grad_out,
                                        const gd_backend_tensor_view *ids,
                                        const gd_backend_tensor_view *grad_table,
                                        const gd_backend_tensor_view *scratch,
                                        const gd_backend_embedding_args *args)
{
    (void)backend;
    (void)grad_out;
    (void)ids;
    (void)grad_table;
    (void)scratch;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_broadcast_scalar(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_reduce_broadcast(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_mul_backward_direct(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_x,
                                         const gd_backend_tensor_view *grad_y)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)grad_out;
    (void)grad_x;
    (void)grad_y;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_mul_reduce_suffix(gd_backend *backend,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *other,
                                       const gd_backend_tensor_view *dst)
{
    (void)backend;
    (void)grad_out;
    (void)other;
    (void)dst;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_cross_entropy_loss(gd_backend *backend,
                                        const gd_backend_tensor_view *logits,
                                        const gd_backend_tensor_view *targets,
                                        const gd_backend_tensor_view *row_loss)
{
    (void)backend;
    (void)logits;
    (void)targets;
    (void)row_loss;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_cross_entropy_loss_stats(gd_backend *backend,
                                              const gd_backend_tensor_view *logits,
                                              const gd_backend_tensor_view *targets,
                                              const gd_backend_tensor_view *row_loss,
                                              const gd_backend_tensor_view *row_max,
                                              const gd_backend_tensor_view *row_inv_sum)
{
    (void)backend;
    (void)logits;
    (void)targets;
    (void)row_loss;
    (void)row_max;
    (void)row_inv_sum;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_cross_entropy_backward(gd_backend *backend,
                                            const gd_backend_tensor_view *logits,
                                            const gd_backend_tensor_view *targets,
                                            const gd_backend_tensor_view *grad_loss,
                                            const gd_backend_tensor_view *grad_logits,
                                            float scale)
{
    (void)backend;
    (void)logits;
    (void)targets;
    (void)grad_loss;
    (void)grad_logits;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_cross_entropy_backward_stats(gd_backend *backend,
                                                  const gd_backend_tensor_view *logits,
                                                  const gd_backend_tensor_view *targets,
                                                  const gd_backend_tensor_view *row_max,
                                                  const gd_backend_tensor_view *row_inv_sum,
                                                  const gd_backend_tensor_view *grad_loss,
                                                  const gd_backend_tensor_view *grad_logits,
                                                  float scale)
{
    (void)backend;
    (void)logits;
    (void)targets;
    (void)row_max;
    (void)row_inv_sum;
    (void)grad_loss;
    (void)grad_logits;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_lm_cross_entropy_online_update(gd_backend *backend,
                                                    const gd_backend_tensor_view *logits_chunk,
                                                    const gd_backend_tensor_view *targets,
                                                    const gd_backend_tensor_view *row_loss,
                                                    const gd_backend_tensor_view *row_max,
                                                    const gd_backend_tensor_view *row_inv_sum,
                                                    uint64_t class_start,
                                                    uint64_t total_classes,
                                                    float logits_softcap)
{
    (void)backend;
    (void)logits_chunk;
    (void)targets;
    (void)row_loss;
    (void)row_max;
    (void)row_inv_sum;
    (void)class_start;
    (void)total_classes;
    (void)logits_softcap;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_lm_cross_entropy_finalize(gd_backend *backend,
                                               const gd_backend_tensor_view *targets,
                                               const gd_backend_tensor_view *row_loss,
                                               const gd_backend_tensor_view *row_max,
                                               const gd_backend_tensor_view *row_inv_sum,
                                               const gd_backend_tensor_view *row_valid,
                                               uint64_t total_classes)
{
    (void)backend;
    (void)targets;
    (void)row_loss;
    (void)row_max;
    (void)row_inv_sum;
    (void)row_valid;
    (void)total_classes;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_lm_cross_entropy_reduce_normalize(gd_backend *backend,
                                                       const gd_backend_tensor_view *row_loss,
                                                       const gd_backend_tensor_view *row_valid,
                                                       const gd_backend_tensor_view *loss,
                                                       const gd_backend_tensor_view *inv_valid_count)
{
    (void)backend;
    (void)row_loss;
    (void)row_valid;
    (void)loss;
    (void)inv_valid_count;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_lm_cross_entropy_backward_chunk(gd_backend *backend,
                                                     const gd_backend_tensor_view *logits_chunk,
                                                     const gd_backend_tensor_view *targets,
                                                     const gd_backend_tensor_view *row_max,
                                                     const gd_backend_tensor_view *row_inv_sum,
                                                     const gd_backend_tensor_view *grad_loss,
                                                     const gd_backend_tensor_view *inv_valid_count,
                                                     const gd_backend_tensor_view *grad_logits_chunk,
                                                     uint64_t class_start,
                                                     uint64_t total_classes,
                                                     float scale,
                                                     float logits_softcap)
{
    (void)backend;
    (void)logits_chunk;
    (void)targets;
    (void)row_max;
    (void)row_inv_sum;
    (void)grad_loss;
    (void)inv_valid_count;
    (void)grad_logits_chunk;
    (void)class_start;
    (void)total_classes;
    (void)scale;
    (void)logits_softcap;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_mse_forward(gd_backend *backend,
                                 const gd_backend_tensor_view *x,
                                 const gd_backend_tensor_view *y,
                                 const gd_backend_tensor_view *out,
                                 uint64_t chunk_size,
                                 float scale)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)out;
    (void)chunk_size;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_mse_backward(gd_backend *backend,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *y,
                                  const gd_backend_tensor_view *grad_out,
                                  const gd_backend_tensor_view *grad_x,
                                  const gd_backend_tensor_view *grad_y,
                                  float scale)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)grad_out;
    (void)grad_x;
    (void)grad_y;
    (void)scale;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_huber_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x,
                                   const gd_backend_tensor_view *y,
                                   const gd_backend_tensor_view *out,
                                   uint64_t chunk_size,
                                   float scale,
                                   float delta)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)out;
    (void)chunk_size;
    (void)scale;
    (void)delta;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_huber_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x,
                                    const gd_backend_tensor_view *y,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x,
                                    const gd_backend_tensor_view *grad_y,
                                    float scale,
                                    float delta)
{
    (void)backend;
    (void)x;
    (void)y;
    (void)grad_out;
    (void)grad_x;
    (void)grad_y;
    (void)scale;
    (void)delta;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_powlu_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x1,
                                   const gd_backend_tensor_view *x2,
                                   const gd_backend_tensor_view *out,
                                   float m)
{
    (void)backend;
    (void)x1;
    (void)x2;
    (void)out;
    (void)m;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_powlu_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x1,
                                    const gd_backend_tensor_view *x2,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x1,
                                    const gd_backend_tensor_view *grad_x2,
                                    float m)
{
    (void)backend;
    (void)x1;
    (void)x2;
    (void)grad_out;
    (void)grad_x1;
    (void)grad_x2;
    (void)m;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_powlu_split_forward(gd_backend *backend,
                                         const gd_backend_tensor_view *x12,
                                         const gd_backend_tensor_view *out,
                                         float m)
{
    (void)backend;
    (void)x12;
    (void)out;
    (void)m;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_powlu_split_backward(gd_backend *backend,
                                          const gd_backend_tensor_view *x12,
                                          const gd_backend_tensor_view *grad_out,
                                          const gd_backend_tensor_view *grad_x12,
                                          float m)
{
    (void)backend;
    (void)x12;
    (void)grad_out;
    (void)grad_x12;
    (void)m;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_qkv_split_rope_forward(gd_backend *backend,
                                            const gd_backend_tensor_view *qkv,
                                            const gd_backend_tensor_view *pos_ids,
                                            const gd_backend_tensor_view *q,
                                            const gd_backend_tensor_view *k,
                                            const gd_backend_tensor_view *v,
                                            uint32_t n_heads,
                                            uint32_t head_dim,
                                            const gd_backend_rope_args *args)
{
    (void)backend;
    (void)qkv;
    (void)pos_ids;
    (void)q;
    (void)k;
    (void)v;
    (void)n_heads;
    (void)head_dim;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_qkv_split_rope_backward(gd_backend *backend,
                                             const gd_backend_tensor_view *grad_q,
                                             const gd_backend_tensor_view *grad_k,
                                             const gd_backend_tensor_view *grad_v,
                                             const gd_backend_tensor_view *pos_ids,
                                             const gd_backend_tensor_view *grad_qkv,
                                             uint32_t n_heads,
                                             uint32_t head_dim,
                                             const gd_backend_rope_args *args)
{
    (void)backend;
    (void)grad_q;
    (void)grad_k;
    (void)grad_v;
    (void)pos_ids;
    (void)grad_qkv;
    (void)n_heads;
    (void)head_dim;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_forward(gd_backend *backend,
                                      const gd_backend_tensor_view *x,
                                      const gd_backend_tensor_view *weight,
                                      const gd_backend_tensor_view *out,
                                      const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)weight;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_forward_stats(gd_backend *backend,
                                            const gd_backend_tensor_view *x,
                                            const gd_backend_tensor_view *weight,
                                            const gd_backend_tensor_view *out,
                                            const gd_backend_tensor_view *inv_rms,
                                            const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)weight;
    (void)out;
    (void)inv_rms;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_inv(gd_backend *backend,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *inv_rms,
                                  const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)inv_rms;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_backward(gd_backend *backend,
                                       const gd_backend_tensor_view *x,
                                       const gd_backend_tensor_view *weight,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *grad_x,
                                       const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)weight;
    (void)grad_out;
    (void)grad_x;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_backward_stats(gd_backend *backend,
                                             const gd_backend_tensor_view *x,
                                             const gd_backend_tensor_view *weight,
                                             const gd_backend_tensor_view *inv_rms,
                                             const gd_backend_tensor_view *grad_out,
                                             const gd_backend_tensor_view *grad_x,
                                             const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)weight;
    (void)inv_rms;
    (void)grad_out;
    (void)grad_x;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rms_norm_weight_backward_stats(gd_backend *backend,
                                                    const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *inv_rms,
                                                    const gd_backend_tensor_view *grad_out,
                                                    const gd_backend_tensor_view *grad_weight,
                                                    const gd_backend_tensor_view *partial,
                                                    const gd_backend_rms_norm_args *args)
{
    (void)backend;
    (void)x;
    (void)inv_rms;
    (void)grad_out;
    (void)grad_weight;
    (void)partial;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_concat_to_full(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    const gd_backend_concat_args *args)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_concat_from_full(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      const gd_backend_concat_args *args)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_split_from_full(gd_backend *backend,
                                     const gd_backend_tensor_view *full,
                                     const gd_backend_tensor_view *slice,
                                     const gd_backend_split_args *args)
{
    (void)backend;
    (void)full;
    (void)slice;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_split_to_full(gd_backend *backend,
                                   const gd_backend_tensor_view *slice,
                                   const gd_backend_tensor_view *full,
                                   const gd_backend_split_args *args)
{
    (void)backend;
    (void)slice;
    (void)full;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_permute(gd_backend *backend,
                             const gd_backend_tensor_view *src,
                             const gd_backend_tensor_view *dst,
                             const gd_backend_permute_args *args)
{
    (void)backend;
    (void)src;
    (void)dst;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_sdpa_varlen(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k,
                                 const gd_backend_tensor_view *v,
                                 const gd_backend_tensor_view *cu_seqlens,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_sdpa_varlen_args *args)
{
    (void)backend;
    (void)q;
    (void)k;
    (void)v;
    (void)cu_seqlens;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

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
                                          const gd_backend_sdpa_varlen_args *args)
{
    (void)backend;
    (void)grad_out;
    (void)q;
    (void)k;
    (void)v;
    (void)cu_seqlens;
    (void)grad_q;
    (void)grad_k;
    (void)grad_v;
    (void)stats;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_sdpa_decode(gd_backend *backend,
                                 const gd_backend_tensor_view *q,
                                 const gd_backend_tensor_view *k_cache,
                                 const gd_backend_tensor_view *v_cache,
                                 const gd_backend_tensor_view *cache_pos,
                                 const gd_backend_tensor_view *out,
                                 const gd_backend_sdpa_decode_args *args)
{
    (void)backend;
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)cache_pos;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_sdpa_decode_at(gd_backend *backend,
                                    const gd_backend_tensor_view *q,
                                    const gd_backend_tensor_view *k_cache,
                                    const gd_backend_tensor_view *v_cache,
                                    const gd_backend_tensor_view *out,
                                    const gd_backend_sdpa_decode_args *args)
{
    (void)backend;
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_sdpa_decode_positions(gd_backend *backend,
                                           const gd_backend_tensor_view *q,
                                           const gd_backend_tensor_view *k_cache,
                                           const gd_backend_tensor_view *v_cache,
                                           const gd_backend_tensor_view *cache_pos,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_sdpa_decode_args *args)
{
    (void)backend;
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)cache_pos;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_kv_cache_append_at(gd_backend *backend,
                                        const gd_backend_tensor_view *k_cache,
                                        const gd_backend_tensor_view *v_cache,
                                        const gd_backend_tensor_view *k_new,
                                        const gd_backend_tensor_view *v_new,
                                        const gd_backend_kv_cache_append_args *args)
{
    (void)backend;
    (void)k_cache;
    (void)v_cache;
    (void)k_new;
    (void)v_new;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_kv_cache_append_positions(gd_backend *backend,
                                               const gd_backend_tensor_view *k_cache,
                                               const gd_backend_tensor_view *v_cache,
                                               const gd_backend_tensor_view *cache_pos,
                                               const gd_backend_tensor_view *k_new,
                                               const gd_backend_tensor_view *v_new,
                                               const gd_backend_kv_cache_append_args *args)
{
    (void)backend;
    (void)k_cache;
    (void)v_cache;
    (void)cache_pos;
    (void)k_new;
    (void)v_new;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_kv_cache_append_packed(gd_backend *backend,
                                            const gd_backend_tensor_view *k_cache,
                                            const gd_backend_tensor_view *v_cache,
                                            const gd_backend_tensor_view *cache_pos,
                                            const gd_backend_tensor_view *cu_seqlens,
                                            const gd_backend_tensor_view *k_new,
                                            const gd_backend_tensor_view *v_new,
                                            const gd_backend_kv_cache_append_args *args)
{
    (void)backend;
    (void)k_cache;
    (void)v_cache;
    (void)cache_pos;
    (void)cu_seqlens;
    (void)k_new;
    (void)v_new;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_begin_step(gd_backend *backend,
                                    const gd_backend_amp_state_desc *desc)
{
    (void)backend;
    (void)desc;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_finish_step(gd_backend *backend,
                                     const gd_backend_amp_state_desc *desc)
{
    (void)backend;
    (void)desc;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_fill_scale(gd_backend *backend,
                                    gd_backend_buffer *dst_buffer,
                                    size_t dst_offset,
                                    size_t count,
                                    uint32_t dtype,
                                    gd_backend_buffer *scale_buffer,
                                    size_t scale_offset)
{
    (void)backend;
    (void)dst_buffer;
    (void)dst_offset;
    (void)count;
    (void)dtype;
    (void)scale_buffer;
    (void)scale_offset;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_scale(gd_backend *backend,
                               gd_backend_buffer *dst_buffer,
                               size_t dst_offset,
                               gd_backend_buffer *src_buffer,
                               size_t src_offset,
                               size_t count,
                               uint32_t dtype,
                               gd_backend_buffer *scale_buffer,
                               size_t scale_offset)
{
    (void)backend;
    (void)dst_buffer;
    (void)dst_offset;
    (void)src_buffer;
    (void)src_offset;
    (void)count;
    (void)dtype;
    (void)scale_buffer;
    (void)scale_offset;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc)
{
    (void)backend;
    (void)desc;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_adamw_batch(gd_backend *backend,
                                  const gd_backend_adamw_desc *descs,
                                  uint32_t desc_count)
{
    (void)backend;
    (void)descs;
    (void)desc_count;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_unscale(gd_backend *backend, const gd_backend_amp_unscale_desc *desc)
{
    (void)backend;
    (void)desc;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_unscale_batch(gd_backend *backend,
                                        const gd_backend_amp_unscale_desc *descs,
                                        uint32_t desc_count)
{
    (void)backend;
    (void)descs;
    (void)desc_count;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_grad_clip_scale(gd_backend *backend,
                                      const gd_backend_grad_norm_desc *descs,
                                      uint32_t desc_count,
                                      gd_backend_buffer *scale_buffer,
                                      size_t scale_offset,
                                      float max_norm,
                                      float eps)
{
    (void)backend;
    (void)descs;
    (void)desc_count;
    (void)scale_buffer;
    (void)scale_offset;
    (void)max_norm;
    (void)eps;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence)
{
    (void)backend;
    if (out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_fence_destroy(gd_backend_fence *fence)
{
    if (fence != NULL) {
        fence->handle = 0;
    }
}

bool gd_backend_fence_is_complete(gd_backend_fence *fence)
{
    (void)fence;
    return true;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    (void)fence;
    return GD_OK;
}
