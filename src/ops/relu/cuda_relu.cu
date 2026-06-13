#include "../../backends/cuda/cuda_backend_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>

#define GD_CUDA_RELU_ELEMENTS_PER_THREAD 4U

static __device__ __forceinline__ __half gd_cuda_relu_forward_value(__half v)
{
    const float vf = __half2float(v);
    /* Match Metal semantics: NaNs and signed zero pass through forward. */
    return vf < 0.0f ? __float2half(0.0f) : v;
}

static __device__ __forceinline__ __half gd_cuda_relu_backward_value(__half x, __half grad)
{
    const float xf = __half2float(x);
    /* Zero derivative at +/-0; NaN input passes grad_out through. */
    return xf <= 0.0f ? __float2half(0.0f) : grad;
}

static __global__ void gd_cuda_relu_kernel(const unsigned char *xbuf,
                                            unsigned char *ybuf,
                                            size_t x_offset,
                                            size_t y_offset,
                                            size_t count)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * GD_CUDA_RELU_ELEMENTS_PER_THREAD;
    const __half *x = (const __half *)(const void *)(xbuf + x_offset);
    __half *y = (__half *)(void *)(ybuf + y_offset);
    if (base >= count) {
        return;
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_RELU_ELEMENTS_PER_THREAD; ++lane) {
        const size_t i = base + (size_t)lane;
        if (i < count) {
            y[i] = gd_cuda_relu_forward_value(x[i]);
        }
    }
}

static __global__ void gd_cuda_relu_backward_kernel(const unsigned char *xbuf,
                                                     const unsigned char *gbuf,
                                                     unsigned char *dxbuf,
                                                     size_t x_offset,
                                                     size_t grad_offset,
                                                     size_t dx_offset,
                                                     size_t count)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * GD_CUDA_RELU_ELEMENTS_PER_THREAD;
    const __half *x = (const __half *)(const void *)(xbuf + x_offset);
    const __half *g = (const __half *)(const void *)(gbuf + grad_offset);
    __half *dx = (__half *)(void *)(dxbuf + dx_offset);
    if (base >= count) {
        return;
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_RELU_ELEMENTS_PER_THREAD; ++lane) {
        const size_t i = base + (size_t)lane;
        if (i < count) {
            dx[i] = gd_cuda_relu_backward_value(x[i], g[i]);
        }
    }
}

static gd_status gd_cuda_relu_validate_view(const gd_backend_tensor_view *view, size_t *out_nbytes)
{
    size_t nbytes;
    if (view == NULL || out_nbytes == NULL || view->buffer == NULL || view->count == 0U ||
        view->dtype != 1U || !gd_cuda_count_bytes(view->count, 2U, &nbytes) ||
        !gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_nbytes = nbytes;
    return GD_OK;
}

static gd_status gd_cuda_relu_validate_pair(const gd_backend_tensor_view *x,
                                             const gd_backend_tensor_view *y,
                                             size_t *out_nbytes)
{
    size_t nbytes;
    gd_status st = gd_cuda_relu_validate_view(x, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    if (y == NULL || y->buffer == NULL || y->count != x->count || y->dtype != x->dtype ||
        !gd_cuda_byte_range_valid(y->buffer, y->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_nbytes = nbytes;
    return GD_OK;
}

extern "C" gd_status gd_backend_relu(gd_backend *backend,
                                       const gd_backend_tensor_view *x,
                                       const gd_backend_tensor_view *y)
{
    size_t nbytes;
    size_t threads;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_relu_validate_pair(x, y, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    (void)nbytes;
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    threads = (x->count + GD_CUDA_RELU_ELEMENTS_PER_THREAD - 1U) /
              GD_CUDA_RELU_ELEMENTS_PER_THREAD;
    blocks = gd_cuda_blocks_for_count(threads, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_relu_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U, gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(y->buffer),
        x->offset,
        y->offset,
        x->count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

extern "C" gd_status gd_backend_relu_backward(gd_backend *backend,
                                                const gd_backend_tensor_view *x,
                                                const gd_backend_tensor_view *grad_out,
                                                const gd_backend_tensor_view *grad_x)
{
    size_t nbytes;
    size_t threads;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_relu_validate_pair(x, grad_out, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL || grad_x->buffer == NULL || grad_x->count != x->count ||
        grad_x->dtype != x->dtype || !gd_cuda_byte_range_valid(grad_x->buffer, grad_x->offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    threads = (x->count + GD_CUDA_RELU_ELEMENTS_PER_THREAD - 1U) /
              GD_CUDA_RELU_ELEMENTS_PER_THREAD;
    blocks = gd_cuda_blocks_for_count(threads, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_relu_backward_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                   gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
        (const unsigned char *)gd_cuda_buffer_const_ptr(grad_out->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(grad_x->buffer),
        x->offset,
        grad_out->offset,
        grad_x->offset,
        x->count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}
