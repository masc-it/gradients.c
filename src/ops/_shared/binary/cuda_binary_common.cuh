#ifndef GD_OPS_SHARED_BINARY_CUDA_COMMON_CUH
#define GD_OPS_SHARED_BINARY_CUDA_COMMON_CUH

#include "../../../backends/cuda/cuda_backend_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GD_CUDA_BINARY_MAX_DIMS 8U
#define GD_CUDA_BINARY_OP_ADD 1U
#define GD_CUDA_BINARY_OP_SUB 2U
#define GD_CUDA_BINARY_OP_MUL 3U

typedef struct gd_cuda_binary_args {
    size_t x_offset;
    size_t y_offset;
    size_t out_offset;
    size_t count;
    uint32_t dtype;
    uint32_t op;
} gd_cuda_binary_args;

typedef struct gd_cuda_binary_bcast_args {
    size_t x_offset;
    size_t y_offset;
    size_t out_offset;
    size_t count;
    uint32_t dtype;
    uint32_t op;
    uint32_t rank;
    uint32_t pad0;
    size_t out_shape[GD_CUDA_BINARY_MAX_DIMS];
    size_t x_strides[GD_CUDA_BINARY_MAX_DIMS];
    size_t y_strides[GD_CUDA_BINARY_MAX_DIMS];
} gd_cuda_binary_bcast_args;

static __device__ __forceinline__ __half gd_cuda_binary_apply_h(__half a,
                                                                 __half b,
                                                                 uint32_t op)
{
    switch (op) {
    case GD_CUDA_BINARY_OP_ADD:
        return __float2half(__half2float(a) + __half2float(b));
    case GD_CUDA_BINARY_OP_SUB:
        return __float2half(__half2float(a) - __half2float(b));
    case GD_CUDA_BINARY_OP_MUL:
    default:
        return __float2half(__half2float(a) * __half2float(b));
    }
}

static __device__ __forceinline__ float gd_cuda_binary_apply_f32(float a,
                                                                  float b,
                                                                  uint32_t op)
{
    switch (op) {
    case GD_CUDA_BINARY_OP_ADD:
        return a + b;
    case GD_CUDA_BINARY_OP_SUB:
        return a - b;
    case GD_CUDA_BINARY_OP_MUL:
    default:
        return a * b;
    }
}

static __global__ void gd_cuda_binary_direct_kernel(const unsigned char *xbuf,
                                                     const unsigned char *ybuf,
                                                     unsigned char *outbuf,
                                                     gd_cuda_binary_args args)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * 4U;
    if (base >= args.count) {
        return;
    }
    if (args.dtype == 1U) {
        const __half *x = (const __half *)(const void *)(xbuf + args.x_offset);
        const __half *y = (const __half *)(const void *)(ybuf + args.y_offset);
        __half *out = (__half *)(void *)(outbuf + args.out_offset);
#pragma unroll
        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const size_t i = base + (size_t)lane;
            if (i < args.count) {
                out[i] = gd_cuda_binary_apply_h(x[i], y[i], args.op);
            }
        }
    } else if (args.dtype == 3U) {
        const float *x = (const float *)(const void *)(xbuf + args.x_offset);
        const float *y = (const float *)(const void *)(ybuf + args.y_offset);
        float *out = (float *)(void *)(outbuf + args.out_offset);
#pragma unroll
        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const size_t i = base + (size_t)lane;
            if (i < args.count) {
                out[i] = gd_cuda_binary_apply_f32(x[i], y[i], args.op);
            }
        }
    }
}

static __device__ __forceinline__ size_t gd_cuda_binary_bcast_index(size_t linear,
                                                                     const size_t *shape,
                                                                     const size_t *strides,
                                                                     uint32_t rank)
{
    size_t rem = linear;
    size_t index = 0U;
    for (int dim = (int)rank - 1; dim >= 0; --dim) {
        const size_t extent = shape[dim];
        const size_t coord = extent > 1U ? rem % extent : 0U;
        if (extent > 1U) {
            rem /= extent;
        }
        index += coord * strides[dim];
    }
    return index;
}

static __global__ void gd_cuda_binary_bcast_kernel(const unsigned char *xbuf,
                                                    const unsigned char *ybuf,
                                                    unsigned char *outbuf,
                                                    gd_cuda_binary_bcast_args args)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    size_t xi;
    size_t yi;
    if (i >= args.count) {
        return;
    }
    xi = gd_cuda_binary_bcast_index(i, args.out_shape, args.x_strides, args.rank);
    yi = gd_cuda_binary_bcast_index(i, args.out_shape, args.y_strides, args.rank);
    if (args.dtype == 1U) {
        const __half *x = (const __half *)(const void *)(xbuf + args.x_offset);
        const __half *y = (const __half *)(const void *)(ybuf + args.y_offset);
        __half *out = (__half *)(void *)(outbuf + args.out_offset);
        out[i] = gd_cuda_binary_apply_h(x[xi], y[yi], args.op);
    } else if (args.dtype == 3U) {
        const float *x = (const float *)(const void *)(xbuf + args.x_offset);
        const float *y = (const float *)(const void *)(ybuf + args.y_offset);
        float *out = (float *)(void *)(outbuf + args.out_offset);
        out[i] = gd_cuda_binary_apply_f32(x[xi], y[yi], args.op);
    }
}

static __global__ void gd_cuda_binary_row_bcast_kernel(const unsigned char *xbuf,
                                                        const unsigned char *ybuf,
                                                        unsigned char *outbuf,
                                                        gd_cuda_binary_bcast_args args)
{
    const size_t col = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    const size_t row = (size_t)blockIdx.y;
    const size_t cols = args.out_shape[1];
    const size_t i = row * cols + col;
    const size_t xi = row * args.x_strides[0] + col * args.x_strides[1];
    const size_t yi = row * args.y_strides[0] + col * args.y_strides[1];
    if (col >= cols || i >= args.count) {
        return;
    }
    if (args.dtype == 1U) {
        const __half *x = (const __half *)(const void *)(xbuf + args.x_offset);
        const __half *y = (const __half *)(const void *)(ybuf + args.y_offset);
        __half *out = (__half *)(void *)(outbuf + args.out_offset);
        out[i] = gd_cuda_binary_apply_h(x[xi], y[yi], args.op);
    } else if (args.dtype == 3U) {
        const float *x = (const float *)(const void *)(xbuf + args.x_offset);
        const float *y = (const float *)(const void *)(ybuf + args.y_offset);
        float *out = (float *)(void *)(outbuf + args.out_offset);
        out[i] = gd_cuda_binary_apply_f32(x[xi], y[yi], args.op);
    }
}

static inline bool gd_cuda_binary_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == 1U) {
        *out_size = 2U;
        return true;
    }
    if (dtype == 3U) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static inline bool gd_cuda_binary_count_bytes(size_t count,
                                               uint32_t dtype,
                                               size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U || !gd_cuda_binary_dtype_size(dtype, &elem_size) ||
        count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static inline bool gd_cuda_binary_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_cuda_binary_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes);
}

static inline gd_status gd_cuda_binary_validate_direct_views(const gd_backend_tensor_view *x,
                                                              const gd_backend_tensor_view *y,
                                                              const gd_backend_tensor_view *out)
{
    if (!gd_cuda_binary_view_range_valid(x) || !gd_cuda_binary_view_range_valid(y) ||
        !gd_cuda_binary_view_range_valid(out) || x->count != y->count ||
        x->count != out->count || x->dtype != y->dtype || x->dtype != out->dtype ||
        out->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static inline bool gd_cuda_binary_broadcast_dims(const gd_backend_tensor_view *arg,
                                                  const gd_backend_tensor_view *out,
                                                  uint32_t dim,
                                                  size_t *out_stride)
{
    uint32_t prefix;
    int64_t arg_dim;
    int64_t out_dim;
    int64_t stride;
    if (arg == NULL || out == NULL || out_stride == NULL || dim >= out->rank ||
        out->rank > GD_CUDA_BINARY_MAX_DIMS || arg->rank > out->rank) {
        return false;
    }
    prefix = out->rank - arg->rank;
    out_dim = out->shape[dim];
    if (out_dim <= 0) {
        return false;
    }
    if (dim < prefix) {
        *out_stride = 0U;
        return true;
    }
    arg_dim = arg->shape[dim - prefix];
    stride = arg->strides[dim - prefix];
    if (arg_dim == out_dim && stride >= 0) {
        *out_stride = (size_t)stride;
        return true;
    }
    if (arg_dim == 1) {
        *out_stride = 0U;
        return true;
    }
    return false;
}

static inline gd_status gd_cuda_binary_validate_broadcast_views(const gd_backend_tensor_view *x,
                                                                 const gd_backend_tensor_view *y,
                                                                 const gd_backend_tensor_view *out)
{
    uint32_t dim;
    size_t stride;
    if (!gd_cuda_binary_view_range_valid(x) || !gd_cuda_binary_view_range_valid(y) ||
        !gd_cuda_binary_view_range_valid(out) || x->dtype != y->dtype ||
        x->dtype != out->dtype || out->rank > GD_CUDA_BINARY_MAX_DIMS ||
        x->rank > out->rank || y->rank > out->rank || out->count > UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (dim = 0U; dim < out->rank; ++dim) {
        const uint32_t x_prefix = out->rank - x->rank;
        const uint32_t y_prefix = out->rank - y->rank;
        const int64_t x_dim = dim < x_prefix ? 1 : x->shape[dim - x_prefix];
        const int64_t y_dim = dim < y_prefix ? 1 : y->shape[dim - y_prefix];
        const int64_t expected = x_dim > y_dim ? x_dim : y_dim;
        if (out->shape[dim] != expected ||
            !gd_cuda_binary_broadcast_dims(x, out, dim, &stride) ||
            !gd_cuda_binary_broadcast_dims(y, out, dim, &stride)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static inline void gd_cuda_binary_fill_bcast_args(gd_cuda_binary_bcast_args *args,
                                                   const gd_backend_tensor_view *x,
                                                   const gd_backend_tensor_view *y,
                                                   const gd_backend_tensor_view *out,
                                                   uint32_t op)
{
    uint32_t dim;
    memset(args, 0, sizeof(*args));
    args->x_offset = x->offset;
    args->y_offset = y->offset;
    args->out_offset = out->offset;
    args->count = out->count;
    args->dtype = out->dtype;
    args->op = op;
    args->rank = out->rank;
    for (dim = 0U; dim < out->rank; ++dim) {
        size_t stride = 0U;
        args->out_shape[dim] = (size_t)out->shape[dim];
        (void)gd_cuda_binary_broadcast_dims(x, out, dim, &stride);
        args->x_strides[dim] = stride;
        (void)gd_cuda_binary_broadcast_dims(y, out, dim, &stride);
        args->y_strides[dim] = stride;
    }
}

static inline gd_status gd_cuda_binary_dispatch(gd_backend *backend,
                                                 const gd_backend_tensor_view *x,
                                                 const gd_backend_tensor_view *y,
                                                 const gd_backend_tensor_view *out,
                                                 uint32_t op)
{
    gd_status st;
    cudaError_t err;
    bool direct;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_binary_validate_broadcast_views(x, y, out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    direct = x->count == out->count && y->count == out->count;
    if (direct) {
        gd_cuda_binary_args args;
        const size_t work_items = (out->count + 3U) / 4U;
        const unsigned int blocks = gd_cuda_blocks_for_count(work_items,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        st = gd_cuda_binary_validate_direct_views(x, y, out);
        if (st != GD_OK) {
            return st;
        }
        memset(&args, 0, sizeof(args));
        args.x_offset = x->offset;
        args.y_offset = y->offset;
        args.out_offset = out->offset;
        args.count = out->count;
        args.dtype = out->dtype;
        args.op = op;
        gd_cuda_binary_direct_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                       gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(y->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
            args);
    } else {
        gd_cuda_binary_bcast_args args;
        gd_cuda_binary_fill_bcast_args(&args, x, y, out, op);
        if (out->rank == 2U) {
            const unsigned int threads = GD_CUDA_DEFAULT_THREADS_PER_BLOCK;
            const unsigned int blocks_x = gd_cuda_blocks_for_count((size_t)out->shape[1], threads);
            const dim3 grid(blocks_x, (unsigned int)out->shape[0], 1U);
            gd_cuda_binary_row_bcast_kernel<<<grid, threads, 0U, gd_cuda_stream(backend)>>>(
                (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
                (const unsigned char *)gd_cuda_buffer_const_ptr(y->buffer),
                (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
                args);
        } else {
            const unsigned int blocks = gd_cuda_blocks_for_count(out->count, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
            gd_cuda_binary_bcast_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                          gd_cuda_stream(backend)>>>(
                (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
                (const unsigned char *)gd_cuda_buffer_const_ptr(y->buffer),
                (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
                args);
        }
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

#endif /* GD_OPS_SHARED_BINARY_CUDA_COMMON_CUH */
