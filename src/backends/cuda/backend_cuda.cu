#include "cuda_backend_internal.h"

#include <cuda_fp16.h>

#include <limits.h>
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

gd_status gd_cublas_status(cublasStatus_t status)
{
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return GD_OK;
    case CUBLAS_STATUS_ALLOC_FAILED:
        return GD_ERR_OUT_OF_MEMORY;
    case CUBLAS_STATUS_INVALID_VALUE:
        return GD_ERR_INVALID_ARGUMENT;
    case CUBLAS_STATUS_ARCH_MISMATCH:
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return GD_ERR_UNSUPPORTED;
    case CUBLAS_STATUS_NOT_INITIALIZED:
    case CUBLAS_STATUS_MAPPING_ERROR:
    case CUBLAS_STATUS_EXECUTION_FAILED:
    case CUBLAS_STATUS_INTERNAL_ERROR:
    default:
        return GD_ERR_INTERNAL;
    }
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

__device__ static __forceinline__ float gd_cuda_load_dtype_as_f32(const unsigned char *base,
                                                                  size_t offset,
                                                                  uint32_t dtype,
                                                                  size_t index)
{
    if (dtype == 1U) {
        const __half *ptr = (const __half *)(const void *)(base + offset);
        return __half2float(ptr[index]);
    }
    if (dtype == 3U) {
        const float *ptr = (const float *)(const void *)(base + offset);
        return ptr[index];
    }
    return 0.0f;
}

__device__ static __forceinline__ void gd_cuda_store_f32_as_dtype(unsigned char *base,
                                                                  size_t offset,
                                                                  uint32_t dtype,
                                                                  size_t index,
                                                                  float value)
{
    if (dtype == 1U) {
        __half *ptr = (__half *)(void *)(base + offset);
        ptr[index] = __float2half(value);
    } else if (dtype == 3U) {
        float *ptr = (float *)(void *)(base + offset);
        ptr[index] = value;
    }
}

__global__ static void gd_cuda_accumulate_kernel(unsigned char *dst,
                                                 size_t dst_offset,
                                                 const unsigned char *src,
                                                 size_t src_offset,
                                                 size_t count,
                                                 uint32_t dtype)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    if (i >= count) {
        return;
    }
    if (dtype == 1U) {
        __half *d = (__half *)(void *)(dst + dst_offset);
        const __half *s = (const __half *)(const void *)(src + src_offset);
        d[i] = __float2half(__half2float(d[i]) + __half2float(s[i]));
    } else if (dtype == 3U) {
        float *d = (float *)(void *)(dst + dst_offset);
        const float *s = (const float *)(const void *)(src + src_offset);
        d[i] += s[i];
    }
}

__global__ static void gd_cuda_scale_kernel(unsigned char *dst,
                                            size_t dst_offset,
                                            const unsigned char *src,
                                            size_t src_offset,
                                            size_t count,
                                            uint32_t dtype,
                                            float scale)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    if (i >= count) {
        return;
    }
    if (dtype == 1U) {
        __half *d = (__half *)(void *)(dst + dst_offset);
        const __half *s = (const __half *)(const void *)(src + src_offset);
        d[i] = scale == 1.0f ? s[i] : __float2half(__half2float(s[i]) * scale);
    } else if (dtype == 3U) {
        float *d = (float *)(void *)(dst + dst_offset);
        const float *s = (const float *)(const void *)(src + src_offset);
        d[i] = scale == 1.0f ? s[i] : s[i] * scale;
    }
}

#define GD_CUDA_REDUCE_MAX_DIMS 8U
#define GD_CUDA_REDUCE_BLOCK_THREADS 256U
#define GD_CUDA_REDUCE_ELEMS_PER_THREAD 4U

#define GD_CUDA_DTYPE_F16 1U
#define GD_CUDA_DTYPE_F32 3U

typedef struct gd_cuda_reduce_broadcast_args {
    size_t src_offset;
    size_t dst_offset;
    size_t src_count;
    size_t dst_count;
    uint32_t dtype;
    uint32_t rank;
    float scale;
    uint32_t pad0;
    size_t src_shape[GD_CUDA_REDUCE_MAX_DIMS];
    size_t src_strides[GD_CUDA_REDUCE_MAX_DIMS];
    size_t dst_shape[GD_CUDA_REDUCE_MAX_DIMS];
    size_t dst_strides[GD_CUDA_REDUCE_MAX_DIMS];
} gd_cuda_reduce_broadcast_args;

__device__ static size_t gd_cuda_reduce_base_offset(size_t dst_linear,
                                                    const gd_cuda_reduce_broadcast_args *args)
{
    size_t rem = dst_linear;
    size_t offset = 0U;
    for (int dim = (int)args->rank - 1; dim >= 0; --dim) {
        const size_t size = args->dst_shape[dim];
        const size_t coord = size > 1U ? rem % size : 0U;
        if (size > 1U) {
            rem /= size;
        }
        offset += coord * args->src_strides[dim];
    }
    return offset;
}

__device__ static size_t gd_cuda_reduce_repeat_count(const gd_cuda_reduce_broadcast_args *args)
{
    size_t count = 1U;
    for (uint32_t dim = 0U; dim < args->rank; ++dim) {
        if (args->dst_shape[dim] == 1U) {
            count *= args->src_shape[dim];
        }
    }
    return count;
}

__device__ static size_t gd_cuda_reduce_repeat_offset(size_t repeat_linear,
                                                      const gd_cuda_reduce_broadcast_args *args)
{
    size_t rem = repeat_linear;
    size_t offset = 0U;
    for (int dim = (int)args->rank - 1; dim >= 0; --dim) {
        if (args->dst_shape[dim] == 1U) {
            const size_t size = args->src_shape[dim];
            const size_t coord = size > 1U ? rem % size : 0U;
            if (size > 1U) {
                rem /= size;
            }
            offset += coord * args->src_strides[dim];
        }
    }
    return offset;
}

__global__ static void gd_cuda_reduce_broadcast_kernel(const unsigned char *src,
                                                       unsigned char *dst,
                                                       gd_cuda_reduce_broadcast_args args)
{
    const size_t dst_i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    float acc = 0.0f;
    size_t base;
    size_t repeats;
    if (dst_i >= args.dst_count) {
        return;
    }
    base = gd_cuda_reduce_base_offset(dst_i, &args);
    repeats = gd_cuda_reduce_repeat_count(&args);
    for (size_t r = 0U; r < repeats; ++r) {
        acc += gd_cuda_load_dtype_as_f32(src, args.src_offset, args.dtype,
                                         base + gd_cuda_reduce_repeat_offset(r, &args));
    }
    gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i, acc * args.scale);
}

__global__ static void gd_cuda_reduce_broadcast_suffix_kernel(const unsigned char *src,
                                                              unsigned char *dst,
                                                              gd_cuda_reduce_broadcast_args args)
{
    __shared__ float partial[GD_CUDA_REDUCE_BLOCK_THREADS];
    const size_t dst_i = (size_t)blockIdx.x;
    const size_t repeats = args.src_count / args.dst_count;
    float acc = 0.0f;
    if (dst_i >= args.dst_count) {
        return;
    }
    for (size_t r = (size_t)threadIdx.x; r < repeats; r += (size_t)blockDim.x) {
        acc += gd_cuda_load_dtype_as_f32(src, args.src_offset, args.dtype,
                                         r * args.dst_count + dst_i);
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i,
                                   partial[0] * args.scale);
    }
}

typedef struct gd_cuda_reduce_contiguous_args {
    size_t src_offset;
    size_t dst_offset;
    size_t src_count;
    size_t dst_count;
    uint32_t src_dtype;
    uint32_t dst_dtype;
    float scale;
    uint32_t pad0;
} gd_cuda_reduce_contiguous_args;

typedef struct gd_cuda_reduce_axis_args {
    size_t src_offset;
    size_t dst_offset;
    size_t outer_count;
    size_t reduce_count;
    size_t inner_count;
    size_t dst_count;
    uint32_t dtype;
    float scale;
} gd_cuda_reduce_axis_args;

typedef struct gd_cuda_broadcast_to_args {
    size_t src_offset;
    size_t dst_offset;
    size_t dst_count;
    uint32_t dtype;
    uint32_t rank;
    float scale;
    uint32_t pad0;
    size_t dst_shape[GD_CUDA_REDUCE_MAX_DIMS];
    size_t src_strides[GD_CUDA_REDUCE_MAX_DIMS];
} gd_cuda_broadcast_to_args;

typedef struct gd_cuda_broadcast_scalar_args {
    size_t src_offset;
    size_t dst_offset;
    size_t dst_count;
    uint32_t src_dtype;
    uint32_t dst_dtype;
    float scale;
    uint32_t pad0;
} gd_cuda_broadcast_scalar_args;

__device__ static __forceinline__ float gd_cuda_reduce_warp_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__device__ static __forceinline__ float gd_cuda_reduce_block_sum(float value)
{
    __shared__ float warp_sums[32];
    const uint32_t lane = (uint32_t)threadIdx.x & 31U;
    const uint32_t warp = (uint32_t)threadIdx.x >> 5U;
    const uint32_t warp_count = ((uint32_t)blockDim.x + 31U) >> 5U;
    value = gd_cuda_reduce_warp_sum(value);
    if (lane == 0U) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp == 0U) {
        value = gd_cuda_reduce_warp_sum(value);
    }
    return value;
}

__device__ static size_t gd_cuda_broadcast_to_offset(size_t linear,
                                                     const gd_cuda_broadcast_to_args *args)
{
    size_t rem = linear;
    size_t offset = 0U;
    for (int dim = (int)args->rank - 1; dim >= 0; --dim) {
        const size_t size = args->dst_shape[dim];
        const size_t coord = size > 1U ? rem % size : 0U;
        if (size > 1U) {
            rem /= size;
        }
        offset += coord * args->src_strides[dim];
    }
    return offset;
}

__global__ static void gd_cuda_reduce_contiguous_kernel(const unsigned char *src,
                                                        unsigned char *dst,
                                                        gd_cuda_reduce_contiguous_args args)
{
    const size_t dst_i = (size_t)blockIdx.x;
    const size_t chunk = (args.src_count + args.dst_count - 1U) / args.dst_count;
    const size_t begin = dst_i * chunk;
    size_t end = begin + chunk;
    float acc = 0.0f;
    if (end > args.src_count) {
        end = args.src_count;
    }
    for (size_t i = begin + (size_t)threadIdx.x; i < end; i += (size_t)blockDim.x) {
        acc += gd_cuda_load_dtype_as_f32(src, args.src_offset, args.src_dtype, i);
    }
    acc = gd_cuda_reduce_block_sum(acc);
    if (threadIdx.x == 0U) {
        gd_cuda_store_f32_as_dtype(dst,
                                   args.dst_offset,
                                   args.dst_dtype,
                                   dst_i,
                                   acc * args.scale);
    }
}

__global__ static void gd_cuda_reduce_axis_kernel(const unsigned char *src,
                                                  unsigned char *dst,
                                                  gd_cuda_reduce_axis_args args)
{
    const size_t dst_i = (size_t)blockIdx.x;
    const size_t inner_i = args.inner_count > 1U ? dst_i % args.inner_count : 0U;
    const size_t outer_i = args.inner_count > 1U ? dst_i / args.inner_count : dst_i;
    const size_t base = (outer_i * args.reduce_count * args.inner_count) + inner_i;
    float acc = 0.0f;
    for (size_t k = (size_t)threadIdx.x; k < args.reduce_count; k += (size_t)blockDim.x) {
        acc += gd_cuda_load_dtype_as_f32(src,
                                         args.src_offset,
                                         args.dtype,
                                         base + k * args.inner_count);
    }
    acc = gd_cuda_reduce_block_sum(acc);
    if (threadIdx.x == 0U) {
        gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i, acc * args.scale);
    }
}

__global__ static void gd_cuda_reduce_axis_last_kernel(const unsigned char *src,
                                                       unsigned char *dst,
                                                       gd_cuda_reduce_axis_args args)
{
    const size_t dst_i = (size_t)blockIdx.x;
    const size_t base = dst_i * args.reduce_count;
    float acc = 0.0f;
    for (size_t k = (size_t)threadIdx.x; k < args.reduce_count; k += (size_t)blockDim.x) {
        acc += gd_cuda_load_dtype_as_f32(src, args.src_offset, args.dtype, base + k);
    }
    acc = gd_cuda_reduce_block_sum(acc);
    if (threadIdx.x == 0U) {
        gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i, acc * args.scale);
    }
}

__global__ static void gd_cuda_broadcast_axis_kernel(const unsigned char *src,
                                                     unsigned char *dst,
                                                     gd_cuda_reduce_axis_args args)
{
    const size_t base_i = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                           (size_t)threadIdx.x) * GD_CUDA_REDUCE_ELEMS_PER_THREAD;
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_REDUCE_ELEMS_PER_THREAD; ++lane) {
        const size_t dst_i = base_i + (size_t)lane;
        if (dst_i < args.dst_count) {
            const size_t inner_i = args.inner_count > 1U ? dst_i % args.inner_count : 0U;
            const size_t quotient = args.inner_count > 1U ? dst_i / args.inner_count : dst_i;
            const size_t outer_i = quotient / args.reduce_count;
            const size_t src_i = outer_i * args.inner_count + inner_i;
            const float value = gd_cuda_load_dtype_as_f32(src,
                                                          args.src_offset,
                                                          args.dtype,
                                                          src_i) * args.scale;
            gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i, value);
        }
    }
}

__global__ static void gd_cuda_broadcast_to_kernel(const unsigned char *src,
                                                   unsigned char *dst,
                                                   gd_cuda_broadcast_to_args args)
{
    const size_t base_i = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                           (size_t)threadIdx.x) * GD_CUDA_REDUCE_ELEMS_PER_THREAD;
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_REDUCE_ELEMS_PER_THREAD; ++lane) {
        const size_t dst_i = base_i + (size_t)lane;
        if (dst_i < args.dst_count) {
            const size_t src_i = gd_cuda_broadcast_to_offset(dst_i, &args);
            const float value = gd_cuda_load_dtype_as_f32(src,
                                                          args.src_offset,
                                                          args.dtype,
                                                          src_i) * args.scale;
            gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dtype, dst_i, value);
        }
    }
}

__global__ static void gd_cuda_broadcast_scalar_kernel(const unsigned char *src,
                                                       unsigned char *dst,
                                                       gd_cuda_broadcast_scalar_args args)
{
    const float value = gd_cuda_load_dtype_as_f32(src, args.src_offset, args.src_dtype, 0U) *
                        args.scale;
    const size_t base_i = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                           (size_t)threadIdx.x) * GD_CUDA_REDUCE_ELEMS_PER_THREAD;
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_REDUCE_ELEMS_PER_THREAD; ++lane) {
        const size_t dst_i = base_i + (size_t)lane;
        if (dst_i < args.dst_count) {
            gd_cuda_store_f32_as_dtype(dst, args.dst_offset, args.dst_dtype, dst_i, value);
        }
    }
}

__global__ static void gd_cuda_mul_backward_direct_kernel(const unsigned char *xbuf,
                                                          const unsigned char *ybuf,
                                                          const unsigned char *gbuf,
                                                          unsigned char *dxbuf,
                                                          unsigned char *dybuf,
                                                          size_t x_offset,
                                                          size_t y_offset,
                                                          size_t grad_offset,
                                                          size_t dx_offset,
                                                          size_t dy_offset,
                                                          size_t count)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    if (i >= count) {
        return;
    }
    const __half *x = (const __half *)(const void *)(xbuf + x_offset);
    const __half *y = (const __half *)(const void *)(ybuf + y_offset);
    const __half *g = (const __half *)(const void *)(gbuf + grad_offset);
    __half *dx = (__half *)(void *)(dxbuf + dx_offset);
    __half *dy = (__half *)(void *)(dybuf + dy_offset);
    dx[i] = __float2half(__half2float(g[i]) * __half2float(y[i]));
    dy[i] = __float2half(__half2float(g[i]) * __half2float(x[i]));
}

__global__ static void gd_cuda_mul_reduce_suffix_kernel(const unsigned char *grad,
                                                        const unsigned char *other,
                                                        unsigned char *dst,
                                                        size_t grad_offset,
                                                        size_t other_offset,
                                                        size_t dst_offset,
                                                        size_t src_count,
                                                        size_t dst_count)
{
    __shared__ float partial[GD_CUDA_REDUCE_BLOCK_THREADS];
    const size_t dst_i = (size_t)blockIdx.x;
    const size_t repeats = src_count / dst_count;
    const __half *g = (const __half *)(const void *)(grad + grad_offset);
    const __half *o = (const __half *)(const void *)(other + other_offset);
    __half *d = (__half *)(void *)(dst + dst_offset);
    float acc = 0.0f;
    if (dst_i >= dst_count) {
        return;
    }
    for (size_t r = (size_t)threadIdx.x; r < repeats; r += (size_t)blockDim.x) {
        const size_t i = r * dst_count + dst_i;
        acc += __half2float(g[i]) * __half2float(o[i]);
    }
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        d[dst_i] = __float2half(partial[0]);
    }
}

#define GD_CUDA_REDUCE_ROWS_COLS 32U
#define GD_CUDA_REDUCE_ROWS_Y_THREADS 8U
#define GD_CUDA_REDUCE_ROWS_TILE_ROWS 256U

typedef struct gd_cuda_reduce_rows_args {
    size_t x_offset;
    size_t y_offset;
    size_t x_row_bytes;
    uint32_t rows;
    uint32_t cols;
    uint32_t tile_rows;
    uint32_t col_groups;
} gd_cuda_reduce_rows_args;

__device__ static __forceinline__ __half gd_cuda_matrix_load_f16(const unsigned char *base,
                                                                 size_t offset,
                                                                 size_t row_bytes,
                                                                 uint32_t row,
                                                                 uint32_t col)
{
    return *(const __half *)(const void *)(base + offset + ((size_t)row * row_bytes) +
                                          ((size_t)col * sizeof(__half)));
}

__global__ static void gd_cuda_reduce_rows_single_kernel(const unsigned char *xbuf,
                                                         unsigned char *ybuf,
                                                         gd_cuda_reduce_rows_args args)
{
    __shared__ float partial[GD_CUDA_REDUCE_ROWS_Y_THREADS][GD_CUDA_REDUCE_ROWS_COLS];
    const uint32_t col = ((uint32_t)blockIdx.x * GD_CUDA_REDUCE_ROWS_COLS) +
                         (uint32_t)threadIdx.x;
    float acc = 0.0f;
    if (col < args.cols) {
        for (uint32_t row = (uint32_t)threadIdx.y; row < args.rows;
             row += GD_CUDA_REDUCE_ROWS_Y_THREADS) {
            acc += __half2float(gd_cuda_matrix_load_f16(xbuf,
                                                        args.x_offset,
                                                        args.x_row_bytes,
                                                        row,
                                                        col));
        }
    }
    partial[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0U && col < args.cols) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t lane = 0U; lane < GD_CUDA_REDUCE_ROWS_Y_THREADS; ++lane) {
            sum += partial[lane][threadIdx.x];
        }
        __half *y = (__half *)(void *)(ybuf + args.y_offset);
        y[col] = __float2half(sum);
    }
}

__global__ static void gd_cuda_reduce_rows_partial_kernel(const unsigned char *xbuf,
                                                          float *partials,
                                                          gd_cuda_reduce_rows_args args)
{
    __shared__ float partial[GD_CUDA_REDUCE_ROWS_Y_THREADS][GD_CUDA_REDUCE_ROWS_COLS];
    const uint32_t block_linear = (uint32_t)blockIdx.x;
    const uint32_t row_tile = block_linear / args.col_groups;
    const uint32_t col_group = block_linear - row_tile * args.col_groups;
    const uint32_t col = (col_group * GD_CUDA_REDUCE_ROWS_COLS) + (uint32_t)threadIdx.x;
    const uint32_t row_begin = row_tile * args.tile_rows;
    uint32_t row_end = row_begin + args.tile_rows;
    float acc = 0.0f;
    if (row_end > args.rows) {
        row_end = args.rows;
    }
    if (col < args.cols) {
        for (uint32_t row = row_begin + (uint32_t)threadIdx.y; row < row_end;
             row += GD_CUDA_REDUCE_ROWS_Y_THREADS) {
            acc += __half2float(gd_cuda_matrix_load_f16(xbuf,
                                                        args.x_offset,
                                                        args.x_row_bytes,
                                                        row,
                                                        col));
        }
    }
    partial[threadIdx.y][threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.y == 0U && col < args.cols) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t lane = 0U; lane < GD_CUDA_REDUCE_ROWS_Y_THREADS; ++lane) {
            sum += partial[lane][threadIdx.x];
        }
        partials[(size_t)row_tile * (size_t)args.cols + (size_t)col] = sum;
    }
}

__global__ static void gd_cuda_reduce_rows_final_kernel(const float *partials,
                                                        unsigned char *ybuf,
                                                        uint32_t tiles,
                                                        uint32_t cols,
                                                        size_t y_offset)
{
    __shared__ float shared[GD_CUDA_REDUCE_BLOCK_THREADS];
    const uint32_t col = (uint32_t)blockIdx.x;
    float acc = 0.0f;
    if (col >= cols) {
        return;
    }
    for (uint32_t tile = (uint32_t)threadIdx.x; tile < tiles; tile += (uint32_t)blockDim.x) {
        acc += partials[(size_t)tile * (size_t)cols + (size_t)col];
    }
    shared[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        __half *y = (__half *)(void *)(ybuf + y_offset);
        y[col] = __float2half(shared[0]);
    }
}

__global__ static void gd_cuda_add_row_bias_f16_kernel(const unsigned char *biasbuf,
                                                       unsigned char *ybuf,
                                                       size_t bias_offset,
                                                       size_t y_offset,
                                                       size_t y_row_bytes,
                                                       uint32_t rows,
                                                       uint32_t cols)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * 4U;
#pragma unroll
    for (uint32_t lane = 0U; lane < 4U; ++lane) {
        const size_t linear = base + (size_t)lane;
        if (linear < (size_t)rows * (size_t)cols) {
            const uint32_t col = (uint32_t)(linear % (size_t)cols);
            const uint32_t row = (uint32_t)(linear / (size_t)cols);
            const __half *bias = (const __half *)(const void *)(biasbuf + bias_offset);
            __half *y = (__half *)(void *)(ybuf + y_offset + ((size_t)row * y_row_bytes));
            y[col] = __float2half(__half2float(y[col]) + __half2float(bias[col]));
        }
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

static bool gd_cuda_size_add(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool gd_cuda_matrix_bounds_ok(const gd_backend_matrix_view *view)
{
    const size_t elem_size = 2U;
    size_t last_row;
    size_t row_span;
    size_t matrix_bytes;
    if (view == NULL || view->buffer == NULL || view->rows == 0U || view->cols == 0U ||
        view->dtype != 1U || (view->offset % elem_size) != 0U ||
        (view->row_bytes % elem_size) != 0U ||
        (size_t)view->cols > SIZE_MAX / elem_size ||
        view->row_bytes < (size_t)view->cols * elem_size ||
        (size_t)(view->rows - 1U) > SIZE_MAX / view->row_bytes) {
        return false;
    }
    last_row = (size_t)(view->rows - 1U) * view->row_bytes;
    row_span = (size_t)view->cols * elem_size;
    if (!gd_cuda_size_add(last_row, row_span, &matrix_bytes)) {
        return false;
    }
    return gd_cuda_byte_range_valid(view->buffer, view->offset, matrix_bytes);
}

static bool gd_cuda_vector_bounds_ok(const gd_backend_vector_view *view)
{
    const size_t elem_size = 2U;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->length == 0U || view->dtype != 1U ||
        (view->offset % elem_size) != 0U || (size_t)view->length > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = (size_t)view->length * elem_size;
    return gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_cuda_matrix_ld(const gd_backend_matrix_view *view, int *out_ld)
{
    size_t ld;
    if (view == NULL || out_ld == NULL || (view->row_bytes % 2U) != 0U) {
        return false;
    }
    ld = view->row_bytes / 2U;
    if (ld == 0U || ld > (size_t)INT_MAX) {
        return false;
    }
    *out_ld = (int)ld;
    return true;
}

static bool gd_cuda_matrix_dims_fit_int(uint32_t a, uint32_t b, uint32_t c)
{
    return a <= (uint32_t)INT_MAX && b <= (uint32_t)INT_MAX && c <= (uint32_t)INT_MAX;
}

static gd_status gd_cuda_cublas_gemm(gd_backend *backend,
                                     cublasOperation_t op_a,
                                     cublasOperation_t op_b,
                                     const gd_backend_matrix_view *a,
                                     const gd_backend_matrix_view *b,
                                     const gd_backend_matrix_view *c,
                                     int m,
                                     int n,
                                     int k)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int lda;
    int ldb;
    int ldc;
    gd_status st;
    cublasStatus_t cb;
    if (backend == NULL || backend->cublas == NULL || a == NULL || b == NULL || c == NULL ||
        m <= 0 || n <= 0 || k <= 0 || !gd_cuda_matrix_ld(a, &lda) ||
        !gd_cuda_matrix_ld(b, &ldb) || !gd_cuda_matrix_ld(c, &ldc)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    cb = cublasGemmEx(backend->cublas,
                      op_a,
                      op_b,
                      m,
                      n,
                      k,
                      &alpha,
                      (const unsigned char *)a->buffer->ptr + a->offset,
                      CUDA_R_16F,
                      lda,
                      (const unsigned char *)b->buffer->ptr + b->offset,
                      CUDA_R_16F,
                      ldb,
                      &beta,
                      (unsigned char *)c->buffer->ptr + c->offset,
                      CUDA_R_16F,
                      ldc,
                      CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    st = gd_cublas_status(cb);
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_finish_if_immediate(backend);
}

static gd_status gd_cuda_cublaslt_linear_bias(gd_backend *backend,
                                              const gd_backend_matrix_view *x,
                                              const gd_backend_matrix_view *w,
                                              const gd_backend_vector_view *bias,
                                              const gd_backend_matrix_view *y)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int lda;
    int ldb;
    int ldc;
    int returned = 0;
    size_t workspace_limit = 8U * 1024U * 1024U;
    void *workspace = NULL;
    const void *bias_ptr;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDesc_t op_desc = NULL;
    cublasLtMatrixLayout_t a_desc = NULL;
    cublasLtMatrixLayout_t b_desc = NULL;
    cublasLtMatrixLayout_t c_desc = NULL;
    cublasLtMatrixLayout_t d_desc = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasStatus_t cb;
    cudaError_t err;
    gd_status st;
    memset(&heuristic, 0, sizeof(heuristic));
    if (backend == NULL || x == NULL || w == NULL || bias == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->cublaslt == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    if (!gd_cuda_matrix_ld(w, &lda) || !gd_cuda_matrix_ld(x, &ldb) ||
        !gd_cuda_matrix_ld(y, &ldc)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_cuda_matrix_dims_fit_int(y->cols, y->rows, x->cols)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    bias_ptr = (const unsigned char *)bias->buffer->ptr + bias->offset;
    cb = cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (cb != CUBLAS_STATUS_SUCCESS) {
        st = gd_cublas_status(cb);
        goto cleanup;
    }
    cb = cublasLtMatmulDescSetAttribute(op_desc,
                                        CUBLASLT_MATMUL_DESC_TRANSA,
                                        &transa,
                                        sizeof(transa));
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulDescSetAttribute(op_desc,
                                            CUBLASLT_MATMUL_DESC_TRANSB,
                                            &transb,
                                            sizeof(transb));
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulDescSetAttribute(op_desc,
                                            CUBLASLT_MATMUL_DESC_EPILOGUE,
                                            &epilogue,
                                            sizeof(epilogue));
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulDescSetAttribute(op_desc,
                                            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                            &bias_ptr,
                                            sizeof(bias_ptr));
    }
    if (cb != CUBLAS_STATUS_SUCCESS) {
        st = gd_cublas_status(cb);
        goto cleanup;
    }
    /* Reuse the row-major-to-column-major mapping from gd_backend_matmul:
     * C = Y^T[N, M] = W^T_storage[N, K] * X^T_storage[K, M].  cuBLASLt's
     * bias epilogue is row-wise in this column-major C, exactly matching the
     * output-channel bias vector of length N. */
    cb = cublasLtMatrixLayoutCreate(&a_desc,
                                    CUDA_R_16F,
                                    (uint64_t)y->cols,
                                    (uint64_t)x->cols,
                                    (int64_t)lda);
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatrixLayoutCreate(&b_desc,
                                        CUDA_R_16F,
                                        (uint64_t)x->cols,
                                        (uint64_t)y->rows,
                                        (int64_t)ldb);
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatrixLayoutCreate(&c_desc,
                                        CUDA_R_16F,
                                        (uint64_t)y->cols,
                                        (uint64_t)y->rows,
                                        (int64_t)ldc);
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatrixLayoutCreate(&d_desc,
                                        CUDA_R_16F,
                                        (uint64_t)y->cols,
                                        (uint64_t)y->rows,
                                        (int64_t)ldc);
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulPreferenceCreate(&pref);
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulPreferenceSetAttribute(pref,
                                                  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_limit,
                                                  sizeof(workspace_limit));
    }
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasLtMatmulAlgoGetHeuristic(backend->cublaslt,
                                            op_desc,
                                            a_desc,
                                            b_desc,
                                            c_desc,
                                            d_desc,
                                            pref,
                                            1,
                                            &heuristic,
                                            &returned);
    }
    if (cb != CUBLAS_STATUS_SUCCESS) {
        st = gd_cublas_status(cb);
        goto cleanup;
    }
    if (returned <= 0 || heuristic.workspaceSize > workspace_limit) {
        st = GD_ERR_UNSUPPORTED;
        goto cleanup;
    }
    if (heuristic.workspaceSize > 0U) {
        err = cudaMallocAsync(&workspace, heuristic.workspaceSize, backend->stream);
        if (err == cudaErrorMemoryAllocation || err == cudaErrorNotSupported) {
            st = GD_ERR_UNSUPPORTED;
            goto cleanup;
        }
        if (err != cudaSuccess) {
            st = gd_cuda_status(err);
            goto cleanup;
        }
    }
    cb = cublasLtMatmul(backend->cublaslt,
                        op_desc,
                        &alpha,
                        (const unsigned char *)w->buffer->ptr + w->offset,
                        a_desc,
                        (const unsigned char *)x->buffer->ptr + x->offset,
                        b_desc,
                        &beta,
                        (const unsigned char *)y->buffer->ptr + y->offset,
                        c_desc,
                        (unsigned char *)y->buffer->ptr + y->offset,
                        d_desc,
                        &heuristic.algo,
                        workspace,
                        heuristic.workspaceSize,
                        backend->stream);
    st = gd_cublas_status(cb);
    if (workspace != NULL) {
        err = cudaFreeAsync(workspace, backend->stream);
        if (st == GD_OK && err != cudaSuccess) {
            st = gd_cuda_status(err);
        }
        workspace = NULL;
    }

cleanup:
    if (pref != NULL) {
        (void)cublasLtMatmulPreferenceDestroy(pref);
    }
    if (d_desc != NULL) {
        (void)cublasLtMatrixLayoutDestroy(d_desc);
    }
    if (c_desc != NULL) {
        (void)cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != NULL) {
        (void)cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != NULL) {
        (void)cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (op_desc != NULL) {
        (void)cublasLtMatmulDescDestroy(op_desc);
    }
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_finish_if_immediate(backend);
}

static gd_status gd_cuda_add_row_bias_f16(gd_backend *backend,
                                          const gd_backend_vector_view *bias,
                                          const gd_backend_matrix_view *y)
{
    size_t total;
    size_t work_items;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || bias == NULL || y == NULL || bias->length != y->cols ||
        bias->dtype != 1U || y->dtype != 1U || !gd_cuda_vector_bounds_ok(bias) ||
        !gd_cuda_matrix_bounds_ok(y) ||
        ((size_t)y->rows != 0U && (size_t)y->cols > SIZE_MAX / (size_t)y->rows)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    total = (size_t)y->rows * (size_t)y->cols;
    work_items = (total + 3U) / 4U;
    if (work_items > (size_t)UINT_MAX * (size_t)GD_CUDA_DEFAULT_THREADS_PER_BLOCK) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    blocks = gd_cuda_blocks_for_count(work_items, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_add_row_bias_f16_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                      backend->stream>>>(
        (const unsigned char *)bias->buffer->ptr,
        (unsigned char *)y->buffer->ptr,
        bias->offset,
        y->offset,
        y->row_bytes,
        y->rows,
        y->cols);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

static gd_status gd_cuda_reduce_rows_f16(gd_backend *backend,
                                         const gd_backend_matrix_view *x,
                                         const gd_backend_vector_view *y)
{
    gd_cuda_reduce_rows_args args;
    uint32_t col_groups;
    uint32_t row_tiles;
    const dim3 block(GD_CUDA_REDUCE_ROWS_COLS, GD_CUDA_REDUCE_ROWS_Y_THREADS, 1U);
    gd_status st;
    cudaError_t err;
    if (backend == NULL || x == NULL || y == NULL || x->dtype != 1U || y->dtype != 1U ||
        x->cols != y->length || x->cols > (uint32_t)INT_MAX ||
        !gd_cuda_matrix_bounds_ok(x) || !gd_cuda_vector_bounds_ok(y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    col_groups = (x->cols + GD_CUDA_REDUCE_ROWS_COLS - 1U) / GD_CUDA_REDUCE_ROWS_COLS;
    row_tiles = (x->rows + GD_CUDA_REDUCE_ROWS_TILE_ROWS - 1U) /
                GD_CUDA_REDUCE_ROWS_TILE_ROWS;
    if (col_groups == 0U || row_tiles == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&args, 0, sizeof(args));
    args.x_offset = x->offset;
    args.y_offset = y->offset;
    args.x_row_bytes = x->row_bytes;
    args.rows = x->rows;
    args.cols = x->cols;
    args.tile_rows = GD_CUDA_REDUCE_ROWS_TILE_ROWS;
    args.col_groups = col_groups;
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    if (row_tiles == 1U) {
        gd_cuda_reduce_rows_single_kernel<<<col_groups, block, 0U, backend->stream>>>(
            (const unsigned char *)x->buffer->ptr,
            (unsigned char *)y->buffer->ptr,
            args);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return gd_cuda_status(err);
        }
        return gd_cuda_finish_if_immediate(backend);
    }
    {
        float *partials = NULL;
        size_t partial_count;
        size_t partial_bytes;
        size_t first_pass_blocks;
        if ((size_t)row_tiles > SIZE_MAX / (size_t)x->cols ||
            (size_t)row_tiles > SIZE_MAX / (size_t)col_groups) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        partial_count = (size_t)row_tiles * (size_t)x->cols;
        first_pass_blocks = (size_t)row_tiles * (size_t)col_groups;
        if (!gd_cuda_count_bytes(partial_count, sizeof(float), &partial_bytes) ||
            first_pass_blocks > (size_t)UINT_MAX) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        err = cudaMallocAsync((void **)&partials, partial_bytes, backend->stream);
        if (err == cudaErrorMemoryAllocation || err == cudaErrorNotSupported) {
            gd_cuda_reduce_rows_single_kernel<<<col_groups, block, 0U, backend->stream>>>(
                (const unsigned char *)x->buffer->ptr,
                (unsigned char *)y->buffer->ptr,
                args);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                return gd_cuda_status(err);
            }
            return gd_cuda_finish_if_immediate(backend);
        }
        if (err != cudaSuccess) {
            return gd_cuda_status(err);
        }
        gd_cuda_reduce_rows_partial_kernel<<<(unsigned int)first_pass_blocks,
                                             block,
                                             0U,
                                             backend->stream>>>(
            (const unsigned char *)x->buffer->ptr,
            partials,
            args);
        err = cudaGetLastError();
        if (err == cudaSuccess) {
            gd_cuda_reduce_rows_final_kernel<<<x->cols,
                                               GD_CUDA_REDUCE_BLOCK_THREADS,
                                               0U,
                                               backend->stream>>>(
                partials,
                (unsigned char *)y->buffer->ptr,
                row_tiles,
                x->cols,
                y->offset);
            err = cudaGetLastError();
        }
        {
            cudaError_t free_err = cudaFreeAsync(partials, backend->stream);
            if (err == cudaSuccess && free_err != cudaSuccess) {
                err = free_err;
            }
        }
        if (err != cudaSuccess) {
            return gd_cuda_status(err);
        }
    }
    return gd_cuda_finish_if_immediate(backend);
}

static bool gd_cuda_dtype_elem_size(uint32_t dtype, size_t *out_elem_size)
{
    if (out_elem_size == NULL) {
        return false;
    }
    if (dtype == 1U) {
        *out_elem_size = 2U;
        return true;
    }
    if (dtype == 3U) {
        *out_elem_size = 4U;
        return true;
    }
    return false;
}

static bool gd_cuda_tensor_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_cuda_dtype_elem_size(view->dtype, &elem_size) &&
           gd_cuda_count_bytes(view->count, elem_size, &nbytes) &&
           gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes);
}

static gd_status gd_cuda_validate_flat_same_dtype(const gd_backend_tensor_view *a,
                                                  const gd_backend_tensor_view *b)
{
    if (!gd_cuda_tensor_view_range_valid(a) || !gd_cuda_tensor_view_range_valid(b) ||
        a->dtype != b->dtype || a->count != b->count) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_cuda_tensor_shapes_equal(const gd_backend_tensor_view *x,
                                        const gd_backend_tensor_view *y)
{
    uint32_t dim;
    if (x == NULL || y == NULL || x->rank != y->rank) {
        return false;
    }
    for (dim = 0U; dim < x->rank; ++dim) {
        if (x->shape[dim] != y->shape[dim]) {
            return false;
        }
    }
    return true;
}

static bool gd_cuda_reduce_dtype_supported(uint32_t dtype)
{
    return dtype == GD_CUDA_DTYPE_F16 || dtype == GD_CUDA_DTYPE_F32;
}

static gd_status gd_cuda_reduce_contiguous_validate(const gd_backend_tensor_view *src,
                                                    const gd_backend_tensor_view *dst)
{
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        !gd_cuda_reduce_dtype_supported(src->dtype) ||
        !gd_cuda_reduce_dtype_supported(dst->dtype) || src->count < dst->count ||
        src->count > (size_t)UINT32_MAX || dst->count > (size_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_cuda_reduce_axis_counts(const gd_backend_tensor_view *full,
                                       uint32_t axis,
                                       size_t *out_outer,
                                       size_t *out_reduce,
                                       size_t *out_inner)
{
    size_t outer = 1U;
    size_t inner = 1U;
    uint32_t dim;
    if (full == NULL || out_outer == NULL || out_reduce == NULL || out_inner == NULL ||
        full->rank == 0U || full->rank > GD_CUDA_REDUCE_MAX_DIMS || axis >= full->rank ||
        full->shape[axis] <= 0) {
        return false;
    }
    for (dim = 0U; dim < axis; ++dim) {
        if (full->shape[dim] <= 0 || outer > SIZE_MAX / (size_t)full->shape[dim]) {
            return false;
        }
        outer *= (size_t)full->shape[dim];
    }
    for (dim = axis + 1U; dim < full->rank; ++dim) {
        if (full->shape[dim] <= 0 || inner > SIZE_MAX / (size_t)full->shape[dim]) {
            return false;
        }
        inner *= (size_t)full->shape[dim];
    }
    if (outer > SIZE_MAX / (size_t)full->shape[axis] ||
        outer * (size_t)full->shape[axis] > SIZE_MAX / inner ||
        outer * (size_t)full->shape[axis] * inner != full->count) {
        return false;
    }
    *out_outer = outer;
    *out_reduce = (size_t)full->shape[axis];
    *out_inner = inner;
    return true;
}

static bool gd_cuda_reduce_axis_reduced_shape_compatible(const gd_backend_tensor_view *full,
                                                         const gd_backend_tensor_view *reduced,
                                                         uint32_t axis)
{
    size_t outer;
    size_t reduce;
    size_t inner;
    uint32_t dim;
    if (full == NULL || reduced == NULL || axis >= full->rank || full->rank == 0U ||
        full->rank > GD_CUDA_REDUCE_MAX_DIMS || reduced->rank > full->rank ||
        full->dtype != reduced->dtype ||
        !gd_cuda_reduce_axis_counts(full, axis, &outer, &reduce, &inner) || reduce == 0U) {
        return false;
    }
    (void)outer;
    (void)inner;
    if (full->count % reduce != 0U || full->count / reduce != reduced->count) {
        return false;
    }
    if (reduced->rank == full->rank) {
        for (dim = 0U; dim < full->rank; ++dim) {
            const int64_t want = dim == axis ? 1 : full->shape[dim];
            if (reduced->shape[dim] != want) {
                return false;
            }
        }
        return true;
    }
    if (reduced->rank + 1U == full->rank) {
        for (dim = 0U; dim < full->rank; ++dim) {
            uint32_t reduced_dim;
            if (dim == axis) {
                continue;
            }
            reduced_dim = dim < axis ? dim : dim - 1U;
            if (reduced->shape[reduced_dim] != full->shape[dim]) {
                return false;
            }
        }
        return true;
    }
    return false;
}

static gd_status gd_cuda_reduce_axis_validate(const gd_backend_tensor_view *src,
                                              const gd_backend_tensor_view *dst,
                                              uint32_t axis)
{
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        !gd_cuda_reduce_dtype_supported(src->dtype) || src->count > (size_t)UINT32_MAX ||
        dst->count > (size_t)UINT32_MAX ||
        !gd_cuda_reduce_axis_reduced_shape_compatible(src, dst, axis)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cuda_broadcast_axis_validate(const gd_backend_tensor_view *src,
                                                 const gd_backend_tensor_view *dst,
                                                 uint32_t axis)
{
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        !gd_cuda_reduce_dtype_supported(src->dtype) || src->count > (size_t)UINT32_MAX ||
        dst->count > (size_t)UINT32_MAX ||
        !gd_cuda_reduce_axis_reduced_shape_compatible(dst, src, axis)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_cuda_reduce_axis_fill_args(gd_cuda_reduce_axis_args *args,
                                          const gd_backend_tensor_view *src,
                                          const gd_backend_tensor_view *dst,
                                          uint32_t axis,
                                          float scale)
{
    size_t outer;
    size_t reduce;
    size_t inner;
    memset(args, 0, sizeof(*args));
    (void)gd_cuda_reduce_axis_counts(src, axis, &outer, &reduce, &inner);
    args->src_offset = src->offset;
    args->dst_offset = dst->offset;
    args->outer_count = outer;
    args->reduce_count = reduce;
    args->inner_count = inner;
    args->dst_count = dst->count;
    args->dtype = src->dtype;
    args->scale = scale;
}

static void gd_cuda_broadcast_axis_fill_args(gd_cuda_reduce_axis_args *args,
                                             const gd_backend_tensor_view *src,
                                             const gd_backend_tensor_view *dst,
                                             uint32_t axis,
                                             float scale)
{
    size_t outer;
    size_t reduce;
    size_t inner;
    memset(args, 0, sizeof(*args));
    (void)gd_cuda_reduce_axis_counts(dst, axis, &outer, &reduce, &inner);
    args->src_offset = src->offset;
    args->dst_offset = dst->offset;
    args->outer_count = outer;
    args->reduce_count = reduce;
    args->inner_count = inner;
    args->dst_count = dst->count;
    args->dtype = dst->dtype;
    args->scale = scale;
}

static bool gd_cuda_broadcast_dim(const gd_backend_tensor_view *src,
                                  const gd_backend_tensor_view *dst,
                                  uint32_t dim,
                                  size_t *out_stride)
{
    uint32_t prefix;
    int64_t src_dim;
    int64_t dst_dim;
    int64_t stride;
    if (src == NULL || dst == NULL || out_stride == NULL || dim >= dst->rank ||
        dst->rank > GD_CUDA_REDUCE_MAX_DIMS || src->rank > dst->rank) {
        return false;
    }
    prefix = dst->rank - src->rank;
    dst_dim = dst->shape[dim];
    if (dst_dim <= 0) {
        return false;
    }
    if (dim < prefix) {
        *out_stride = 0U;
        return true;
    }
    src_dim = src->shape[dim - prefix];
    stride = src->strides[dim - prefix];
    if (src_dim == dst_dim && stride >= 0) {
        *out_stride = (size_t)stride;
        return true;
    }
    if (src_dim == 1) {
        *out_stride = 0U;
        return true;
    }
    return false;
}

static gd_status gd_cuda_broadcast_to_validate(const gd_backend_tensor_view *src,
                                               const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        src->dtype != dst->dtype || !gd_cuda_reduce_dtype_supported(src->dtype) ||
        src->rank > dst->rank || src->rank > GD_CUDA_REDUCE_MAX_DIMS ||
        dst->rank > GD_CUDA_REDUCE_MAX_DIMS || dst->count > (size_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (dim = 0U; dim < dst->rank; ++dim) {
        size_t stride;
        if (!gd_cuda_broadcast_dim(src, dst, dim, &stride)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static void gd_cuda_broadcast_to_fill_args(gd_cuda_broadcast_to_args *args,
                                           const gd_backend_tensor_view *src,
                                           const gd_backend_tensor_view *dst,
                                           float scale)
{
    uint32_t dim;
    memset(args, 0, sizeof(*args));
    args->src_offset = src->offset;
    args->dst_offset = dst->offset;
    args->dst_count = dst->count;
    args->dtype = dst->dtype;
    args->rank = dst->rank;
    args->scale = scale;
    for (dim = 0U; dim < dst->rank; ++dim) {
        size_t stride = 0U;
        args->dst_shape[dim] = (size_t)dst->shape[dim];
        (void)gd_cuda_broadcast_dim(src, dst, dim, &stride);
        args->src_strides[dim] = stride;
    }
}

static gd_status gd_cuda_broadcast_scalar_validate(const gd_backend_tensor_view *src,
                                                   const gd_backend_tensor_view *dst)
{
    bool dtype_pair_ok;
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        src->count != 1U || dst->count > (size_t)UINT32_MAX ||
        !gd_cuda_reduce_dtype_supported(src->dtype) ||
        !gd_cuda_reduce_dtype_supported(dst->dtype)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dtype_pair_ok = src->dtype == dst->dtype ||
                    (src->dtype == GD_CUDA_DTYPE_F32 && dst->dtype == GD_CUDA_DTYPE_F16);
    return dtype_pair_ok ? GD_OK : GD_ERR_INVALID_ARGUMENT;
}

static bool gd_cuda_reduce_shape_compatible(const gd_backend_tensor_view *src,
                                            const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    if (src == NULL || dst == NULL || src->rank > GD_CUDA_REDUCE_MAX_DIMS ||
        dst->rank > src->rank) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t src_dim = src->shape[dim];
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        int64_t dst_stride = dim < prefix ? 0 : dst->strides[dim - prefix];
        if (src_dim <= 0 || dst_dim <= 0 || src->strides[dim] < 0 || dst_stride < 0 ||
            (dst_dim != src_dim && dst_dim != 1)) {
            return false;
        }
    }
    return true;
}

static bool gd_cuda_reduce_is_suffix(const gd_backend_tensor_view *src,
                                     const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    bool suffix_started = false;
    if (src == NULL || dst == NULL || dst->rank > src->rank || dst->count == 0U ||
        src->count % dst->count != 0U) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        if (!suffix_started && dst_dim == 1 && src->shape[dim] != 1) {
            continue;
        }
        suffix_started = true;
        if (dst_dim != src->shape[dim]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_cuda_reduce_validate_views(const gd_backend_tensor_view *src,
                                               const gd_backend_tensor_view *dst)
{
    if (!gd_cuda_tensor_view_range_valid(src) || !gd_cuda_tensor_view_range_valid(dst) ||
        src->dtype != dst->dtype || src->count <= dst->count ||
        src->count > UINT32_MAX || dst->count > UINT32_MAX ||
        !gd_cuda_reduce_shape_compatible(src, dst)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_cuda_reduce_fill_args(gd_cuda_reduce_broadcast_args *args,
                                     const gd_backend_tensor_view *src,
                                     const gd_backend_tensor_view *dst,
                                     float scale)
{
    const uint32_t prefix = src->rank - dst->rank;
    uint32_t dim;
    memset(args, 0, sizeof(*args));
    args->src_offset = src->offset;
    args->dst_offset = dst->offset;
    args->src_count = src->count;
    args->dst_count = dst->count;
    args->dtype = src->dtype;
    args->rank = src->rank;
    args->scale = scale;
    for (dim = 0U; dim < src->rank; ++dim) {
        args->src_shape[dim] = (size_t)src->shape[dim];
        args->src_strides[dim] = (size_t)src->strides[dim];
        if (dim < prefix) {
            args->dst_shape[dim] = 1U;
            args->dst_strides[dim] = 0U;
        } else {
            const uint32_t dst_dim = dim - prefix;
            args->dst_shape[dim] = (size_t)dst->shape[dst_dim];
            args->dst_strides[dim] = (size_t)dst->strides[dst_dim];
        }
    }
}

static bool gd_cuda_mul_reduce_suffix_compatible(const gd_backend_tensor_view *src,
                                                 const gd_backend_tensor_view *dst)
{
    uint32_t dim;
    uint32_t prefix;
    bool suffix_started = false;
    if (src == NULL || dst == NULL || dst->rank > src->rank || dst->count == 0U ||
        src->count <= dst->count || src->count % dst->count != 0U) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        if (!suffix_started && dst_dim == 1 && src->shape[dim] != 1) {
            continue;
        }
        suffix_started = true;
        if (dst_dim != src->shape[dim]) {
            return false;
        }
    }
    return true;
}

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    gd_backend *backend;
    cudaError_t err;
    cublasStatus_t cb;
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
    cb = cublasCreate(&backend->cublas);
    if (cb != CUBLAS_STATUS_SUCCESS) {
        (void)cudaStreamDestroy(backend->transfer_stream);
        (void)cudaStreamDestroy(backend->stream);
        free(backend);
        return gd_cublas_status(cb);
    }
    cb = cublasSetStream(backend->cublas, backend->stream);
    if (cb == CUBLAS_STATUS_SUCCESS) {
        cb = cublasSetMathMode(backend->cublas, CUBLAS_TENSOR_OP_MATH);
    }
    if (cb != CUBLAS_STATUS_SUCCESS) {
        (void)cublasDestroy(backend->cublas);
        (void)cudaStreamDestroy(backend->transfer_stream);
        (void)cudaStreamDestroy(backend->stream);
        free(backend);
        return gd_cublas_status(cb);
    }
    cb = cublasLtCreate(&backend->cublaslt);
    if (cb != CUBLAS_STATUS_SUCCESS) {
        backend->cublaslt = NULL;
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
    }
    if (backend->cublaslt != NULL) {
        (void)cublasLtDestroy(backend->cublaslt);
    }
    if (backend->cublas != NULL) {
        (void)cublasDestroy(backend->cublas);
    }
    if (backend->transfer_stream != NULL) {
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
    if (backend == NULL || x == NULL || w == NULL || y == NULL ||
        !gd_cuda_matrix_bounds_ok(x) || !gd_cuda_matrix_bounds_ok(w) ||
        !gd_cuda_matrix_bounds_ok(y) || x->dtype != w->dtype || x->dtype != y->dtype) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->cols != w->rows || x->rows != y->rows || w->cols != y->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_cuda_matrix_dims_fit_int(y->cols, y->rows, x->cols)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    /* Row-major Y = X * W is column-major Y^T = W^T_storage * X^T_storage.
     * Swapping operands lets cuBLAS consume row-major tensors without copies. */
    return gd_cuda_cublas_gemm(backend,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               w,
                               x,
                               y,
                               (int)y->cols,
                               (int)y->rows,
                               (int)x->cols);
}

gd_status gd_backend_matmul_nt(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    if (backend == NULL || x == NULL || w == NULL || y == NULL ||
        !gd_cuda_matrix_bounds_ok(x) || !gd_cuda_matrix_bounds_ok(w) ||
        !gd_cuda_matrix_bounds_ok(y) || x->dtype != w->dtype || x->dtype != y->dtype) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->cols != w->cols || x->rows != y->rows || w->rows != y->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_cuda_matrix_dims_fit_int(y->cols, y->rows, x->cols)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    /* Row-major Y = X * W^T maps to column-major Y^T = W * X^T_storage. */
    return gd_cuda_cublas_gemm(backend,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               w,
                               x,
                               y,
                               (int)y->cols,
                               (int)y->rows,
                               (int)x->cols);
}

gd_status gd_backend_matmul_tn(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    if (backend == NULL || x == NULL || w == NULL || y == NULL ||
        !gd_cuda_matrix_bounds_ok(x) || !gd_cuda_matrix_bounds_ok(w) ||
        !gd_cuda_matrix_bounds_ok(y) || x->dtype != w->dtype || x->dtype != y->dtype) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->rows != w->rows || x->cols != y->rows || w->cols != y->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_cuda_matrix_dims_fit_int(y->cols, y->rows, x->rows)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    /* Row-major Y = X^T * W maps to column-major Y^T = W^T_storage * X. */
    return gd_cuda_cublas_gemm(backend,
                               CUBLAS_OP_N,
                               CUBLAS_OP_T,
                               w,
                               x,
                               y,
                               (int)y->cols,
                               (int)y->rows,
                               (int)x->rows);
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
    gd_status st;
    if (backend == NULL || x == NULL || w == NULL || y == NULL ||
        !gd_cuda_matrix_bounds_ok(x) || !gd_cuda_matrix_bounds_ok(w) ||
        !gd_cuda_matrix_bounds_ok(y) || x->dtype != w->dtype || x->dtype != y->dtype ||
        (bias != NULL && (bias->dtype != y->dtype || !gd_cuda_vector_bounds_ok(bias)))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->cols != w->rows || x->rows != y->rows || w->cols != y->cols ||
        (bias != NULL && bias->length != y->cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != 1U) {
        return GD_ERR_UNSUPPORTED;
    }
    if (!gd_cuda_matrix_dims_fit_int(y->cols, y->rows, x->cols)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    if (bias == NULL) {
        return gd_backend_matmul(backend, x, w, y);
    }
    st = gd_cuda_cublaslt_linear_bias(backend, x, w, bias, y);
    if (st == GD_OK) {
        return GD_OK;
    }
    if (st != GD_ERR_UNSUPPORTED && st != GD_ERR_INVALID_ARGUMENT) {
        return st;
    }
    st = gd_backend_matmul(backend, x, w, y);
    if (st != GD_OK) {
        return st;
    }
    return gd_cuda_add_row_bias_f16(backend, bias, y);
}

gd_status gd_backend_reduce_rows(gd_backend *backend,
                                 const gd_backend_matrix_view *x,
                                 const gd_backend_vector_view *y)
{
    if (backend == NULL || x == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != 1U || y->dtype != 1U) {
        return GD_ERR_UNSUPPORTED;
    }
    return gd_cuda_reduce_rows_f16(backend, x, y);
}

gd_status gd_backend_accumulate(gd_backend *backend,
                                gd_backend_buffer *dst_buffer,
                                size_t dst_offset,
                                gd_backend_buffer *src_buffer,
                                size_t src_offset,
                                size_t count,
                                uint32_t dtype)
{
    size_t elem_size;
    size_t nbytes;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || dst_buffer == NULL || src_buffer == NULL || count == 0U ||
        !gd_cuda_dtype_elem_size(dtype, &elem_size) ||
        !gd_cuda_count_bytes(count, elem_size, &nbytes) ||
        !gd_cuda_byte_range_valid(dst_buffer, dst_offset, nbytes) ||
        !gd_cuda_byte_range_valid(src_buffer, src_offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    blocks = gd_cuda_blocks_for_count(count, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_accumulate_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                backend->stream>>>((unsigned char *)dst_buffer->ptr,
                                                   dst_offset,
                                                   (const unsigned char *)src_buffer->ptr,
                                                   src_offset,
                                                   count,
                                                   dtype);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
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
    size_t elem_size;
    size_t nbytes;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || dst_buffer == NULL || src_buffer == NULL || count == 0U ||
        !(scale == scale) || !gd_cuda_dtype_elem_size(dtype, &elem_size) ||
        !gd_cuda_count_bytes(count, elem_size, &nbytes) ||
        !gd_cuda_byte_range_valid(dst_buffer, dst_offset, nbytes) ||
        !gd_cuda_byte_range_valid(src_buffer, src_offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    blocks = gd_cuda_blocks_for_count(count, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_scale_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                           backend->stream>>>((unsigned char *)dst_buffer->ptr,
                                              dst_offset,
                                              (const unsigned char *)src_buffer->ptr,
                                              src_offset,
                                              count,
                                              dtype,
                                              scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_reduce_contiguous(gd_backend *backend,
                                       const gd_backend_tensor_view *src,
                                       const gd_backend_tensor_view *dst,
                                       float scale)
{
    gd_cuda_reduce_contiguous_args args;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_reduce_contiguous_validate(src, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    memset(&args, 0, sizeof(args));
    args.src_offset = src->offset;
    args.dst_offset = dst->offset;
    args.src_count = src->count;
    args.dst_count = dst->count;
    args.src_dtype = src->dtype;
    args.dst_dtype = dst->dtype;
    args.scale = scale;
    gd_cuda_reduce_contiguous_kernel<<<(unsigned int)dst->count,
                                        GD_CUDA_REDUCE_BLOCK_THREADS,
                                        0U,
                                        backend->stream>>>(
        (const unsigned char *)src->buffer->ptr,
        (unsigned char *)dst->buffer->ptr,
        args);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_reduce_axis(gd_backend *backend,
                                 const gd_backend_tensor_view *src,
                                 const gd_backend_tensor_view *dst,
                                 uint32_t axis,
                                 float scale)
{
    gd_cuda_reduce_axis_args args;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_reduce_axis_validate(src, dst, axis);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_reduce_axis_fill_args(&args, src, dst, axis, scale);
    if (args.inner_count == 1U) {
        gd_cuda_reduce_axis_last_kernel<<<(unsigned int)dst->count,
                                           GD_CUDA_REDUCE_BLOCK_THREADS,
                                           0U,
                                           backend->stream>>>(
            (const unsigned char *)src->buffer->ptr,
            (unsigned char *)dst->buffer->ptr,
            args);
    } else {
        gd_cuda_reduce_axis_kernel<<<(unsigned int)dst->count,
                                      GD_CUDA_REDUCE_BLOCK_THREADS,
                                      0U,
                                      backend->stream>>>(
            (const unsigned char *)src->buffer->ptr,
            (unsigned char *)dst->buffer->ptr,
            args);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_broadcast_axis(gd_backend *backend,
                                    const gd_backend_tensor_view *src,
                                    const gd_backend_tensor_view *dst,
                                    uint32_t axis,
                                    float scale)
{
    gd_cuda_reduce_axis_args args;
    size_t work_items;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_broadcast_axis_validate(src, dst, axis);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_broadcast_axis_fill_args(&args, src, dst, axis, scale);
    work_items = (dst->count + GD_CUDA_REDUCE_ELEMS_PER_THREAD - 1U) /
                 GD_CUDA_REDUCE_ELEMS_PER_THREAD;
    blocks = gd_cuda_blocks_for_count(work_items, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_broadcast_axis_kernel<<<blocks,
                                    GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                    0U,
                                    backend->stream>>>(
        (const unsigned char *)src->buffer->ptr,
        (unsigned char *)dst->buffer->ptr,
        args);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_broadcast_to(gd_backend *backend,
                                  const gd_backend_tensor_view *src,
                                  const gd_backend_tensor_view *dst,
                                  float scale)
{
    gd_cuda_broadcast_to_args args;
    size_t work_items;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_broadcast_to_validate(src, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_broadcast_to_fill_args(&args, src, dst, scale);
    work_items = (dst->count + GD_CUDA_REDUCE_ELEMS_PER_THREAD - 1U) /
                 GD_CUDA_REDUCE_ELEMS_PER_THREAD;
    blocks = gd_cuda_blocks_for_count(work_items, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_broadcast_to_kernel<<<blocks,
                                  GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                  0U,
                                  backend->stream>>>(
        (const unsigned char *)src->buffer->ptr,
        (unsigned char *)dst->buffer->ptr,
        args);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
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

gd_status gd_backend_broadcast_scalar(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    gd_cuda_broadcast_scalar_args args;
    size_t work_items;
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_broadcast_scalar_validate(src, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    memset(&args, 0, sizeof(args));
    args.src_offset = src->offset;
    args.dst_offset = dst->offset;
    args.dst_count = dst->count;
    args.src_dtype = src->dtype;
    args.dst_dtype = dst->dtype;
    args.scale = scale;
    work_items = (dst->count + GD_CUDA_REDUCE_ELEMS_PER_THREAD - 1U) /
                 GD_CUDA_REDUCE_ELEMS_PER_THREAD;
    blocks = gd_cuda_blocks_for_count(work_items, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_broadcast_scalar_kernel<<<blocks,
                                      GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                      0U,
                                      backend->stream>>>(
        (const unsigned char *)src->buffer->ptr,
        (unsigned char *)dst->buffer->ptr,
        args);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_reduce_broadcast(gd_backend *backend,
                                      const gd_backend_tensor_view *src,
                                      const gd_backend_tensor_view *dst,
                                      float scale)
{
    gd_cuda_reduce_broadcast_args args;
    bool suffix;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_reduce_validate_views(src, dst);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_reduce_fill_args(&args, src, dst, scale);
    suffix = gd_cuda_reduce_is_suffix(src, dst);
    if (suffix) {
        gd_cuda_reduce_broadcast_suffix_kernel<<<(unsigned int)dst->count,
                                                  GD_CUDA_REDUCE_BLOCK_THREADS,
                                                  0U,
                                                  backend->stream>>>(
            (const unsigned char *)src->buffer->ptr,
            (unsigned char *)dst->buffer->ptr,
            args);
    } else {
        const unsigned int blocks = gd_cuda_blocks_for_count(dst->count,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        gd_cuda_reduce_broadcast_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                          backend->stream>>>(
            (const unsigned char *)src->buffer->ptr,
            (unsigned char *)dst->buffer->ptr,
            args);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_mul_backward_direct(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *y,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_x,
                                         const gd_backend_tensor_view *grad_y)
{
    unsigned int blocks;
    gd_status st;
    cudaError_t err;
    if (backend == NULL || x == NULL || y == NULL || grad_out == NULL ||
        grad_x == NULL || grad_y == NULL || x->dtype != 1U || y->dtype != 1U ||
        grad_out->dtype != 1U || grad_x->dtype != 1U || grad_y->dtype != 1U ||
        gd_cuda_validate_flat_same_dtype(x, y) != GD_OK ||
        gd_cuda_validate_flat_same_dtype(x, grad_out) != GD_OK ||
        gd_cuda_validate_flat_same_dtype(x, grad_x) != GD_OK ||
        gd_cuda_validate_flat_same_dtype(y, grad_y) != GD_OK ||
        !gd_cuda_tensor_shapes_equal(x, y) || !gd_cuda_tensor_shapes_equal(x, grad_out) ||
        !gd_cuda_tensor_shapes_equal(x, grad_x) || !gd_cuda_tensor_shapes_equal(y, grad_y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    blocks = gd_cuda_blocks_for_count(grad_out->count, GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
    gd_cuda_mul_backward_direct_kernel<<<blocks, GD_CUDA_DEFAULT_THREADS_PER_BLOCK, 0U,
                                         backend->stream>>>(
        (const unsigned char *)x->buffer->ptr,
        (const unsigned char *)y->buffer->ptr,
        (const unsigned char *)grad_out->buffer->ptr,
        (unsigned char *)grad_x->buffer->ptr,
        (unsigned char *)grad_y->buffer->ptr,
        x->offset,
        y->offset,
        grad_out->offset,
        grad_x->offset,
        grad_y->offset,
        grad_out->count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

gd_status gd_backend_mul_reduce_suffix(gd_backend *backend,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *other,
                                       const gd_backend_tensor_view *dst)
{
    gd_status st;
    cudaError_t err;
    if (backend == NULL || grad_out == NULL || other == NULL || dst == NULL ||
        grad_out->dtype != 1U || other->dtype != 1U || dst->dtype != 1U ||
        gd_cuda_validate_flat_same_dtype(grad_out, other) != GD_OK ||
        !gd_cuda_tensor_view_range_valid(dst) || grad_out->count > UINT32_MAX ||
        dst->count > UINT32_MAX || !gd_cuda_tensor_shapes_equal(grad_out, other) ||
        !gd_cuda_mul_reduce_suffix_compatible(grad_out, dst)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_activate(backend);
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_mul_reduce_suffix_kernel<<<(unsigned int)dst->count,
                                       GD_CUDA_REDUCE_BLOCK_THREADS,
                                       0U,
                                       backend->stream>>>(
        (const unsigned char *)grad_out->buffer->ptr,
        (const unsigned char *)other->buffer->ptr,
        (unsigned char *)dst->buffer->ptr,
        grad_out->offset,
        other->offset,
        dst->offset,
        grad_out->count,
        dst->count);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
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
