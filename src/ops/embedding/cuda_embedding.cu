#include "../../backends/cuda/cuda_backend_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GD_CUDA_EMBEDDING_MAX_DIMS 8U
#define GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD 4U
#define GD_CUDA_EMBEDDING_VEC_BYTES 16U

#define GD_CUDA_DTYPE_F16 1U
#define GD_CUDA_DTYPE_F32 3U
#define GD_CUDA_DTYPE_I32 4U

typedef struct gd_cuda_embedding_launch_args {
    size_t table_offset;
    size_t ids_offset;
    size_t out_offset;
    size_t grad_out_offset;
    size_t grad_table_offset;
    size_t scratch_offset;
    size_t ids_count;
    size_t vocab;
    size_t dim;
    uint32_t dtype;
    uint32_t pad0;
} gd_cuda_embedding_launch_args;

static __device__ __forceinline__ float gd_cuda_embedding_qnan_f32(void)
{
    return __uint_as_float(0x7fc00000U);
}

static __device__ __forceinline__ __half gd_cuda_embedding_qnan_f16(void)
{
    return __float2half(gd_cuda_embedding_qnan_f32());
}

static __global__ void gd_cuda_embedding_forward_kernel(const unsigned char *tablebuf,
                                                        const unsigned char *idsbuf,
                                                        unsigned char *outbuf,
                                                        gd_cuda_embedding_launch_args args)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD;
    const size_t total = args.ids_count * args.dim;
    const int32_t *ids = (const int32_t *)(const void *)(idsbuf + args.ids_offset);
    if (base >= total) {
        return;
    }
    if (args.dtype == GD_CUDA_DTYPE_F16) {
        const __half *table = (const __half *)(const void *)(tablebuf + args.table_offset);
        __half *out = (__half *)(void *)(outbuf + args.out_offset);
#pragma unroll
        for (uint32_t lane = 0U; lane < GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD; ++lane) {
            const size_t i = base + (size_t)lane;
            if (i < total) {
                const size_t row = i / args.dim;
                const size_t col = i - row * args.dim;
                const int32_t id = ids[row];
                out[i] = (id < 0 || (size_t)id >= args.vocab)
                    ? gd_cuda_embedding_qnan_f16()
                    : table[(size_t)id * args.dim + col];
            }
        }
    } else if (args.dtype == GD_CUDA_DTYPE_F32) {
        const float *table = (const float *)(const void *)(tablebuf + args.table_offset);
        float *out = (float *)(void *)(outbuf + args.out_offset);
#pragma unroll
        for (uint32_t lane = 0U; lane < GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD; ++lane) {
            const size_t i = base + (size_t)lane;
            if (i < total) {
                const size_t row = i / args.dim;
                const size_t col = i - row * args.dim;
                const int32_t id = ids[row];
                out[i] = (id < 0 || (size_t)id >= args.vocab)
                    ? gd_cuda_embedding_qnan_f32()
                    : table[(size_t)id * args.dim + col];
            }
        }
    }
}

static __global__ void gd_cuda_embedding_forward_vec16_kernel(const unsigned char *tablebuf,
                                                              const unsigned char *idsbuf,
                                                              unsigned char *outbuf,
                                                              gd_cuda_embedding_launch_args args)
{
    const size_t linear = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    const size_t elem_size = args.dtype == GD_CUDA_DTYPE_F16 ? sizeof(__half) : sizeof(float);
    const size_t row_bytes = args.dim * elem_size;
    const size_t vecs_per_row = row_bytes / GD_CUDA_EMBEDDING_VEC_BYTES;
    const size_t total_vecs = args.ids_count * vecs_per_row;
    const int32_t *ids = (const int32_t *)(const void *)(idsbuf + args.ids_offset);
    const uint32_t nan_word = args.dtype == GD_CUDA_DTYPE_F16 ? 0x7e007e00U : 0x7fc00000U;
    if (linear >= total_vecs) {
        return;
    }
    const size_t row = linear / vecs_per_row;
    const size_t vec = linear - row * vecs_per_row;
    const int32_t id = ids[row];
    uint4 *dst = (uint4 *)(void *)(outbuf + args.out_offset + row * row_bytes +
                                   vec * GD_CUDA_EMBEDDING_VEC_BYTES);
    if (id < 0 || (size_t)id >= args.vocab) {
        *dst = make_uint4(nan_word, nan_word, nan_word, nan_word);
        return;
    }
    const uint4 *src = (const uint4 *)(const void *)(tablebuf + args.table_offset +
                                                     (size_t)id * row_bytes +
                                                     vec * GD_CUDA_EMBEDDING_VEC_BYTES);
    *dst = *src;
}

static __global__ void gd_cuda_embedding_scatter_kernel(const unsigned char *gradbuf,
                                                        const unsigned char *idsbuf,
                                                        unsigned char *targetbuf,
                                                        gd_cuda_embedding_launch_args args)
{
    const size_t i = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
    const size_t total = args.ids_count * args.dim;
    const int32_t *ids = (const int32_t *)(const void *)(idsbuf + args.ids_offset);
    int32_t id;
    size_t row;
    size_t col;
    if (i >= total) {
        return;
    }
    row = i / args.dim;
    id = ids[row];
    if (id < 0 || (size_t)id >= args.vocab) {
        return;
    }
    col = i - row * args.dim;
    if (args.dtype == GD_CUDA_DTYPE_F16) {
        const __half *grad = (const __half *)(const void *)(gradbuf + args.grad_out_offset);
        float *target = (float *)(void *)(targetbuf + args.scratch_offset);
        atomicAdd(target + (size_t)id * args.dim + col, __half2float(grad[i]));
    } else if (args.dtype == GD_CUDA_DTYPE_F32) {
        const float *grad = (const float *)(const void *)(gradbuf + args.grad_out_offset);
        float *target = (float *)(void *)(targetbuf + args.grad_table_offset);
        atomicAdd(target + (size_t)id * args.dim + col, grad[i]);
    }
}

static __global__ void gd_cuda_embedding_cast_f32_to_f16_kernel(const unsigned char *scratchbuf,
                                                                unsigned char *outbuf,
                                                                gd_cuda_embedding_launch_args args)
{
    const size_t base = (((size_t)blockIdx.x * (size_t)blockDim.x) +
                         (size_t)threadIdx.x) * GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD;
    const size_t total = args.vocab * args.dim;
    const float *src = (const float *)(const void *)(scratchbuf + args.scratch_offset);
    __half *dst = (__half *)(void *)(outbuf + args.grad_table_offset);
    if (base >= total) {
        return;
    }
#pragma unroll
    for (uint32_t lane = 0U; lane < GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD; ++lane) {
        const size_t i = base + (size_t)lane;
        if (i < total) {
            dst[i] = __float2half(src[i]);
        }
    }
}

static bool gd_cuda_embedding_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == GD_CUDA_DTYPE_F16) {
        *out_size = 2U;
        return true;
    }
    if (dtype == GD_CUDA_DTYPE_F32 || dtype == GD_CUDA_DTYPE_I32) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static bool gd_cuda_embedding_count_bytes(size_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U ||
        !gd_cuda_embedding_dtype_size(dtype, &elem_size) || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_cuda_embedding_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_cuda_embedding_dtype_size(view->dtype, &elem_size) &&
           (view->offset % elem_size) == 0U &&
           gd_cuda_embedding_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_cuda_embedding_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_CUDA_EMBEDDING_MAX_DIMS) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        const uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride ||
            stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_cuda_embedding_count_matches(const gd_backend_tensor_view *view, size_t count)
{
    return view != NULL && view->count == count;
}

static gd_status gd_cuda_embedding_args_to_size(const gd_backend_embedding_args *args,
                                                size_t *ids_count,
                                                size_t *vocab,
                                                size_t *dim,
                                                size_t *out_count,
                                                size_t *table_count)
{
    if (args == NULL || ids_count == NULL || vocab == NULL || dim == NULL ||
        out_count == NULL || table_count == NULL || args->ids_count == 0U ||
        args->vocab == 0U || args->dim == 0U || args->ids_count > (uint64_t)SIZE_MAX ||
        args->vocab > (uint64_t)SIZE_MAX || args->dim > (uint64_t)SIZE_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *ids_count = (size_t)args->ids_count;
    *vocab = (size_t)args->vocab;
    *dim = (size_t)args->dim;
    if (*ids_count > SIZE_MAX / *dim || *vocab > SIZE_MAX / *dim) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out_count = *ids_count * *dim;
    *table_count = *vocab * *dim;
    if (*out_count > (size_t)UINT32_MAX || *table_count > (size_t)UINT32_MAX ||
        *ids_count > (size_t)UINT32_MAX || *dim > (size_t)UINT32_MAX) {
        return GD_ERR_UNSUPPORTED;
    }
    return GD_OK;
}

static gd_status gd_cuda_embedding_validate_common(const gd_backend_tensor_view *ids,
                                                   const gd_backend_embedding_args *args,
                                                   size_t *ids_count,
                                                   size_t *vocab,
                                                   size_t *dim,
                                                   size_t *out_count,
                                                   size_t *table_count)
{
    gd_status st = gd_cuda_embedding_args_to_size(args,
                                                  ids_count,
                                                  vocab,
                                                  dim,
                                                  out_count,
                                                  table_count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cuda_embedding_view_range_valid(ids) || !gd_cuda_embedding_contiguous_view(ids) ||
        ids->dtype != GD_CUDA_DTYPE_I32 || !gd_cuda_embedding_count_matches(ids, *ids_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cuda_embedding_validate_forward(const gd_backend_tensor_view *table,
                                                    const gd_backend_tensor_view *ids,
                                                    const gd_backend_tensor_view *out,
                                                    const gd_backend_embedding_args *args,
                                                    size_t *ids_count,
                                                    size_t *vocab,
                                                    size_t *dim)
{
    size_t out_count;
    size_t table_count;
    uint32_t i;
    gd_status st = gd_cuda_embedding_validate_common(ids,
                                                     args,
                                                     ids_count,
                                                     vocab,
                                                     dim,
                                                     &out_count,
                                                     &table_count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cuda_embedding_view_range_valid(table) || !gd_cuda_embedding_view_range_valid(out) ||
        !gd_cuda_embedding_contiguous_view(table) || !gd_cuda_embedding_contiguous_view(out) ||
        table->rank != 2U || table->shape[0] <= 0 || table->shape[1] <= 0 ||
        (size_t)table->shape[0] != *vocab || (size_t)table->shape[1] != *dim ||
        out->rank != ids->rank + 1U || out->dtype != table->dtype ||
        (table->dtype != GD_CUDA_DTYPE_F16 && table->dtype != GD_CUDA_DTYPE_F32) ||
        !gd_cuda_embedding_count_matches(table, table_count) ||
        !gd_cuda_embedding_count_matches(out, out_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < ids->rank; ++i) {
        if (out->shape[i] != ids->shape[i]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    if (out->shape[ids->rank] <= 0 || (size_t)out->shape[ids->rank] != *dim) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cuda_embedding_validate_backward(const gd_backend_tensor_view *grad_out,
                                                     const gd_backend_tensor_view *ids,
                                                     const gd_backend_tensor_view *grad_table,
                                                     const gd_backend_tensor_view *scratch,
                                                     const gd_backend_embedding_args *args,
                                                     size_t *ids_count,
                                                     size_t *vocab,
                                                     size_t *dim,
                                                     size_t *out_count,
                                                     size_t *table_count)
{
    uint32_t i;
    gd_status st = gd_cuda_embedding_validate_common(ids,
                                                     args,
                                                     ids_count,
                                                     vocab,
                                                     dim,
                                                     out_count,
                                                     table_count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cuda_embedding_view_range_valid(grad_out) ||
        !gd_cuda_embedding_view_range_valid(grad_table) ||
        !gd_cuda_embedding_contiguous_view(grad_out) ||
        !gd_cuda_embedding_contiguous_view(grad_table) || grad_table->rank != 2U ||
        grad_table->shape[0] <= 0 || grad_table->shape[1] <= 0 ||
        (size_t)grad_table->shape[0] != *vocab ||
        (size_t)grad_table->shape[1] != *dim || grad_out->rank != ids->rank + 1U ||
        grad_out->dtype != grad_table->dtype ||
        (grad_table->dtype != GD_CUDA_DTYPE_F16 && grad_table->dtype != GD_CUDA_DTYPE_F32) ||
        !gd_cuda_embedding_count_matches(grad_table, *table_count) ||
        !gd_cuda_embedding_count_matches(grad_out, *out_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < ids->rank; ++i) {
        if (grad_out->shape[i] != ids->shape[i]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    if (grad_out->shape[ids->rank] <= 0 || (size_t)grad_out->shape[ids->rank] != *dim) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_table->dtype == GD_CUDA_DTYPE_F16) {
        if (scratch == NULL || !gd_cuda_embedding_view_range_valid(scratch) ||
            !gd_cuda_embedding_contiguous_view(scratch) || scratch->dtype != GD_CUDA_DTYPE_F32 ||
            scratch->rank != 2U || scratch->shape[0] <= 0 || scratch->shape[1] <= 0 ||
            (size_t)scratch->shape[0] != *vocab || (size_t)scratch->shape[1] != *dim ||
            !gd_cuda_embedding_count_matches(scratch, *table_count)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    } else if (scratch != NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_cuda_embedding_forward_can_vec16(const gd_backend_tensor_view *table,
                                                const gd_backend_tensor_view *out,
                                                uint32_t dtype,
                                                size_t dim)
{
    size_t elem_size;
    size_t row_bytes;
    if (table == NULL || out == NULL ||
        !gd_cuda_embedding_dtype_size(dtype, &elem_size) || dim > SIZE_MAX / elem_size) {
        return false;
    }
    row_bytes = dim * elem_size;
    return row_bytes != 0U && (row_bytes % GD_CUDA_EMBEDDING_VEC_BYTES) == 0U &&
           (table->offset % GD_CUDA_EMBEDDING_VEC_BYTES) == 0U &&
           (out->offset % GD_CUDA_EMBEDDING_VEC_BYTES) == 0U;
}

static void gd_cuda_embedding_fill_launch_args(gd_cuda_embedding_launch_args *launch,
                                               const gd_backend_tensor_view *table,
                                               const gd_backend_tensor_view *ids,
                                               const gd_backend_tensor_view *out,
                                               const gd_backend_tensor_view *grad_out,
                                               const gd_backend_tensor_view *grad_table,
                                               const gd_backend_tensor_view *scratch,
                                               size_t ids_count,
                                               size_t vocab,
                                               size_t dim,
                                               uint32_t dtype)
{
    memset(launch, 0, sizeof(*launch));
    launch->table_offset = table != NULL ? table->offset : 0U;
    launch->ids_offset = ids != NULL ? ids->offset : 0U;
    launch->out_offset = out != NULL ? out->offset : 0U;
    launch->grad_out_offset = grad_out != NULL ? grad_out->offset : 0U;
    launch->grad_table_offset = grad_table != NULL ? grad_table->offset : 0U;
    launch->scratch_offset = scratch != NULL ? scratch->offset : 0U;
    launch->ids_count = ids_count;
    launch->vocab = vocab;
    launch->dim = dim;
    launch->dtype = dtype;
}

extern "C" gd_status gd_backend_embedding_forward(gd_backend *backend,
                                                    const gd_backend_tensor_view *table,
                                                    const gd_backend_tensor_view *ids,
                                                    const gd_backend_tensor_view *out,
                                                    const gd_backend_embedding_args *args)
{
    gd_cuda_embedding_launch_args launch;
    size_t ids_count;
    size_t vocab;
    size_t dim;
    size_t total;
    gd_status st;
    cudaError_t err;
    bool vec16;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_embedding_validate_forward(table, ids, out, args, &ids_count, &vocab, &dim);
    if (st != GD_OK) {
        return st;
    }
    total = ids_count * dim;
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_embedding_fill_launch_args(&launch,
                                       table,
                                       ids,
                                       out,
                                       NULL,
                                       NULL,
                                       NULL,
                                       ids_count,
                                       vocab,
                                       dim,
                                       table->dtype);
    vec16 = gd_cuda_embedding_forward_can_vec16(table, out, table->dtype, dim);
    if (vec16) {
        const size_t elem_size = table->dtype == GD_CUDA_DTYPE_F16 ? 2U : 4U;
        const size_t vec_count = ids_count * ((dim * elem_size) / GD_CUDA_EMBEDDING_VEC_BYTES);
        const unsigned int blocks = gd_cuda_blocks_for_count(vec_count,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        gd_cuda_embedding_forward_vec16_kernel<<<blocks,
                                                 GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                                 0U,
                                                 gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(table->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(ids->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
            launch);
    } else {
        const size_t work_items = (total + GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD - 1U) /
                                  GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD;
        const unsigned int blocks = gd_cuda_blocks_for_count(work_items,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        gd_cuda_embedding_forward_kernel<<<blocks,
                                           GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                           0U,
                                           gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(table->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(ids->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
            launch);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

extern "C" gd_status gd_backend_embedding_backward(gd_backend *backend,
                                                     const gd_backend_tensor_view *grad_out,
                                                     const gd_backend_tensor_view *ids,
                                                     const gd_backend_tensor_view *grad_table,
                                                     const gd_backend_tensor_view *scratch,
                                                     const gd_backend_embedding_args *args)
{
    gd_cuda_embedding_launch_args launch;
    const gd_backend_tensor_view *target;
    size_t ids_count;
    size_t vocab;
    size_t dim;
    size_t out_count;
    size_t table_count;
    size_t target_bytes;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_embedding_validate_backward(grad_out,
                                             ids,
                                             grad_table,
                                             scratch,
                                             args,
                                             &ids_count,
                                             &vocab,
                                             &dim,
                                             &out_count,
                                             &table_count);
    if (st != GD_OK) {
        return st;
    }
    target = grad_table->dtype == GD_CUDA_DTYPE_F16 ? scratch : grad_table;
    if (!gd_cuda_count_bytes(table_count, sizeof(float), &target_bytes)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_embedding_fill_launch_args(&launch,
                                       NULL,
                                       ids,
                                       NULL,
                                       grad_out,
                                       grad_table,
                                       target,
                                       ids_count,
                                       vocab,
                                       dim,
                                       grad_out->dtype);
    err = cudaMemsetAsync((unsigned char *)gd_cuda_buffer_ptr(target->buffer) + target->offset,
                          0,
                          target_bytes,
                          gd_cuda_stream(backend));
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    {
        const unsigned int blocks = gd_cuda_blocks_for_count(out_count,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        gd_cuda_embedding_scatter_kernel<<<blocks,
                                           GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                           0U,
                                           gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(grad_out->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(ids->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(target->buffer),
            launch);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    if (grad_table->dtype == GD_CUDA_DTYPE_F16) {
        const size_t work_items = (table_count + GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD - 1U) /
                                  GD_CUDA_EMBEDDING_ELEMENTS_PER_THREAD;
        const unsigned int blocks = gd_cuda_blocks_for_count(work_items,
                                                            GD_CUDA_DEFAULT_THREADS_PER_BLOCK);
        gd_cuda_embedding_cast_f32_to_f16_kernel<<<blocks,
                                                   GD_CUDA_DEFAULT_THREADS_PER_BLOCK,
                                                   0U,
                                                   gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(scratch->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(grad_table->buffer),
            launch);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            return gd_cuda_status(err);
        }
    }
    return gd_cuda_finish_if_immediate(backend);
}
