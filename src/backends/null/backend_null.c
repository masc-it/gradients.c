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

gd_status gd_backend_adamw(gd_backend *backend, const gd_backend_adamw_desc *desc)
{
    (void)backend;
    (void)desc;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_amp_unscale(gd_backend *backend, const gd_backend_amp_unscale_desc *desc)
{
    (void)backend;
    (void)desc;
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
