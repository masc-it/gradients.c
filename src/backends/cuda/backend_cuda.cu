#include "cuda_backend_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

gd_status gd_cuda_status(cudaError_t err)
{
    if (err == cudaSuccess) {
        return GD_OK;
    }
    if (err == cudaErrorMemoryAllocation) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
        return GD_ERR_UNSUPPORTED;
    }
    return GD_ERR_INTERNAL;
}

bool gd_cuda_byte_range_valid(const gd_backend_buffer *buffer, size_t offset, size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

bool gd_cuda_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

gd_status gd_cuda_finish_if_immediate(gd_backend *backend)
{
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->scope_active) {
        return GD_OK;
    }
    return gd_cuda_status(cudaStreamSynchronize(backend->stream));
}

__global__ static void gd_cuda_fill_kernel(unsigned char *dst,
                                           size_t offset,
                                           size_t count,
                                           size_t elem_size,
                                           uint32_t pattern)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    unsigned char *p;
    if (i >= count) {
        return;
    }
    p = dst + offset + (i * elem_size);
    p[0] = (unsigned char)(pattern & 0xffU);
    if (elem_size >= 2U) {
        p[1] = (unsigned char)((pattern >> 8U) & 0xffU);
    }
    if (elem_size >= 4U) {
        p[2] = (unsigned char)((pattern >> 16U) & 0xffU);
        p[3] = (unsigned char)((pattern >> 24U) & 0xffU);
    }
}

static gd_cuda_memory_mode gd_cuda_memory_mode_from_env(void)
{
    const char *mode = getenv("GD_CUDA_MEMORY");
    if (mode == NULL || mode[0] == '\0' || strcmp(mode, "device") == 0) {
        return GD_CUDA_MEMORY_DEVICE;
    }
    if (strcmp(mode, "managed") == 0 || strcmp(mode, "unified") == 0) {
        return GD_CUDA_MEMORY_MANAGED;
    }
    return GD_CUDA_MEMORY_DEVICE;
}

static gd_status gd_cuda_activate(gd_backend *backend)
{
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_cuda_status(cudaSetDevice(backend->device));
}

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    gd_backend *backend;
    cudaError_t err;
    int device_count;
    if (out_backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    device_count = 0;
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        return err == cudaSuccess ? GD_ERR_UNSUPPORTED : gd_cuda_status(err);
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    backend = (gd_backend *)calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    backend->device = 0;
    err = cudaStreamCreateWithFlags(&backend->stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        free(backend);
        return gd_cuda_status(err);
    }
    err = cudaStreamCreateWithFlags(&backend->transfer_stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        (void)cudaStreamDestroy(backend->stream);
        free(backend);
        return gd_cuda_status(err);
    }
    backend->memory_mode = gd_cuda_memory_mode_from_env();
    backend->scope_active = false;
    *out_backend = backend;
    return GD_OK;
}

void gd_backend_destroy(gd_backend *backend)
{
    if (backend == NULL) {
        return;
    }
    (void)cudaSetDevice(backend->device);
    if (backend->stream != NULL) {
        (void)cudaStreamSynchronize(backend->stream);
    }
    if (backend->transfer_stream != NULL) {
        (void)cudaStreamSynchronize(backend->transfer_stream);
        (void)cudaStreamDestroy(backend->transfer_stream);
    }
    if (backend->stream != NULL) {
        (void)cudaStreamDestroy(backend->stream);
    }
    free(backend);
}

gd_backend_kind gd_backend_kind_query(const gd_backend *backend)
{
    if (backend == NULL) {
        return (gd_backend_kind)0;
    }
    return GD_BACKEND_CUDA;
}

const char *gd_backend_name(const gd_backend *backend)
{
    if (backend == NULL) {
        return "none";
    }
    return "cuda";
}

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer)
{
    gd_backend_buffer *buffer;
    cudaError_t err;
    void *ptr;
    if (backend == NULL || out_buffer == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = NULL;
    err = cudaSetDevice(backend->device);
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    buffer = (gd_backend_buffer *)calloc(1U, sizeof(*buffer));
    if (buffer == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    ptr = NULL;
    if (backend->memory_mode == GD_CUDA_MEMORY_MANAGED) {
        err = cudaMallocManaged(&ptr, nbytes, cudaMemAttachGlobal);
    } else {
        err = cudaMalloc(&ptr, nbytes);
    }
    if (err != cudaSuccess) {
        free(buffer);
        return gd_cuda_status(err);
    }
    buffer->ptr = ptr;
    buffer->host_ptr = backend->memory_mode == GD_CUDA_MEMORY_MANAGED ? ptr : NULL;
    buffer->nbytes = nbytes;
    buffer->host_visible = backend->memory_mode == GD_CUDA_MEMORY_MANAGED;
    *out_buffer = buffer;
    return GD_OK;
}

void gd_backend_buffer_destroy(gd_backend_buffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    if (buffer->ptr != NULL) {
        (void)cudaFree(buffer->ptr);
    }
    free(buffer);
}

size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->nbytes : 0U;
}

void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->host_ptr : NULL;
}

bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer)
{
    return buffer != NULL && buffer->host_visible;
}

gd_status gd_backend_scope_begin(gd_backend *backend)
{
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->scope_active) {
        return GD_ERR_BAD_STATE;
    }
    backend->scope_active = true;
    return GD_OK;
}

gd_status gd_backend_flush(gd_backend *backend)
{
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaStreamSynchronize(backend->stream));
    return st;
}

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes)
{
    gd_status st;
    if (backend == NULL || buffer == NULL || src == NULL || nbytes == 0U ||
        !gd_cuda_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaMemcpyAsync((unsigned char *)buffer->ptr + offset,
                                        src,
                                        nbytes,
                                        cudaMemcpyHostToDevice,
                                        backend->transfer_stream));
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_status(cudaStreamSynchronize(backend->transfer_stream));
}

gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes)
{
    gd_status st;
    if (backend == NULL || buffer == NULL || dst == NULL || nbytes == 0U ||
        !gd_cuda_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaMemcpyAsync(dst,
                                        (const unsigned char *)buffer->ptr + offset,
                                        nbytes,
                                        cudaMemcpyDeviceToHost,
                                        backend->transfer_stream));
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_status(cudaStreamSynchronize(backend->transfer_stream));
}

gd_status gd_backend_fill(gd_backend *backend,
                          gd_backend_buffer *buffer,
                          size_t offset,
                          size_t count,
                          size_t elem_size,
                          uint32_t pattern)
{
    size_t nbytes;
    unsigned int blocks;
    gd_status st;
    if (backend == NULL || buffer == NULL ||
        (elem_size != 1U && elem_size != 2U && elem_size != 4U) ||
        count > UINT32_MAX || !gd_cuda_count_bytes(count, elem_size, &nbytes) ||
        !gd_cuda_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    blocks = gd_cuda_blocks_for_count(count, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_fill_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U, backend->stream>>>(
        (unsigned char *)buffer->ptr,
        offset,
        count,
        elem_size,
        pattern);
    st = gd_cuda_status(cudaGetLastError());
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_finish_if_immediate(backend);
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

gd_status gd_backend_rope(gd_backend *backend,
                          const gd_backend_tensor_view *x,
                          const gd_backend_tensor_view *pos_ids,
                          const gd_backend_tensor_view *out,
                          const gd_backend_rope_args *args)
{
    (void)backend;
    (void)x;
    (void)pos_ids;
    (void)out;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_rope_backward(gd_backend *backend,
                                   const gd_backend_tensor_view *grad_out,
                                   const gd_backend_tensor_view *pos_ids,
                                   const gd_backend_tensor_view *grad_x,
                                   const gd_backend_rope_args *args)
{
    (void)backend;
    (void)grad_out;
    (void)pos_ids;
    (void)grad_x;
    (void)args;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_swiglu_forward(gd_backend *backend,
                                    const gd_backend_tensor_view *x1,
                                    const gd_backend_tensor_view *x2,
                                    const gd_backend_tensor_view *out)
{
    (void)backend;
    (void)x1;
    (void)x2;
    (void)out;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_swiglu_backward(gd_backend *backend,
                                     const gd_backend_tensor_view *x1,
                                     const gd_backend_tensor_view *x2,
                                     const gd_backend_tensor_view *grad_out,
                                     const gd_backend_tensor_view *grad_x1,
                                     const gd_backend_tensor_view *grad_x2)
{
    (void)backend;
    (void)x1;
    (void)x2;
    (void)grad_out;
    (void)grad_x1;
    (void)grad_x2;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_swiglu_split_forward(gd_backend *backend,
                                          const gd_backend_tensor_view *x12,
                                          const gd_backend_tensor_view *out)
{
    (void)backend;
    (void)x12;
    (void)out;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_swiglu_split_backward(gd_backend *backend,
                                           const gd_backend_tensor_view *x12,
                                           const gd_backend_tensor_view *grad_out,
                                           const gd_backend_tensor_view *grad_x12)
{
    (void)backend;
    (void)x12;
    (void)grad_out;
    (void)grad_x12;
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
                                 const gd_backend_tensor_view *stats,
                                 const gd_backend_sdpa_varlen_args *args)
{
    (void)backend;
    (void)q;
    (void)k;
    (void)v;
    (void)cu_seqlens;
    (void)out;
    (void)stats;
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
    cudaEvent_t event;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = NULL;
    event = NULL;
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    st = gd_cuda_status(cudaEventRecord(event, backend->stream));
    if (st != GD_OK) {
        (void)cudaEventDestroy(event);
        return st;
    }
    backend->scope_active = false;
    out_fence->handle = (void *)event;
    return GD_OK;
}

void gd_backend_fence_destroy(gd_backend_fence *fence)
{
    if (fence != NULL && fence->handle != NULL) {
        (void)cudaEventDestroy((cudaEvent_t)fence->handle);
        fence->handle = NULL;
    }
}

bool gd_backend_fence_is_complete(gd_backend_fence *fence)
{
    cudaError_t err;
    if (fence == NULL || fence->handle == NULL) {
        return true;
    }
    err = cudaEventQuery((cudaEvent_t)fence->handle);
    return err == cudaSuccess;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    if (fence == NULL || fence->handle == NULL) {
        return GD_OK;
    }
    return gd_cuda_status(cudaEventSynchronize((cudaEvent_t)fence->handle));
}
