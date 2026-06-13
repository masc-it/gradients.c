#include "../../backends/cuda/cuda_backend_internal.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GD_CUDA_RMS_NORM_MAX_DIMS 8U
#define GD_CUDA_RMS_NORM_MAX_SIMDGROUPS 8U
#define GD_CUDA_RMS_NORM_WGRAD_CHANNELS 128U
#define GD_CUDA_RMS_NORM_WGRAD_ROW_BLOCK_MAX 128U

#define GD_CUDA_RMS_NORM_DTYPE_F16 1U
#define GD_CUDA_RMS_NORM_DTYPE_F32 3U

typedef struct gd_cuda_rms_norm_launch_args {
    size_t x_offset;
    size_t weight_offset;
    size_t out_offset;
    size_t grad_out_offset;
    size_t inv_rms_offset;
    size_t partial_offset;
    size_t rows;
    size_t cols;
    size_t row_blocks;
    float eps;
    uint32_t dtype;
    uint32_t simdgroups;
    uint32_t wgrad_simdgroups;
    uint32_t wgrad_row_block;
} gd_cuda_rms_norm_launch_args;

static __device__ __forceinline__ float gd_cuda_rms_norm_load_f32(const unsigned char *base,
                                                                  size_t offset,
                                                                  uint32_t dtype,
                                                                  size_t index)
{
    if (dtype == GD_CUDA_RMS_NORM_DTYPE_F16) {
        const __half *ptr = (const __half *)(const void *)(base + offset);
        return __half2float(ptr[index]);
    }
    if (dtype == GD_CUDA_RMS_NORM_DTYPE_F32) {
        const float *ptr = (const float *)(const void *)(base + offset);
        return ptr[index];
    }
    return 0.0f;
}

static __device__ __forceinline__ void gd_cuda_rms_norm_store_f32(unsigned char *base,
                                                                  size_t offset,
                                                                  uint32_t dtype,
                                                                  size_t index,
                                                                  float value)
{
    if (dtype == GD_CUDA_RMS_NORM_DTYPE_F16) {
        __half *ptr = (__half *)(void *)(base + offset);
        ptr[index] = __float2half(value);
    } else if (dtype == GD_CUDA_RMS_NORM_DTYPE_F32) {
        float *ptr = (float *)(void *)(base + offset);
        ptr[index] = value;
    }
}

static __device__ __forceinline__ float gd_cuda_rms_norm_warp_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

static __device__ __forceinline__ float gd_cuda_rms_norm_block_sum(float value)
{
    __shared__ float warp_sums[32];
    __shared__ float result;
    const uint32_t lane = (uint32_t)threadIdx.x & 31U;
    const uint32_t warp = (uint32_t)threadIdx.x >> 5U;
    const uint32_t warp_count = ((uint32_t)blockDim.x + 31U) >> 5U;
    value = gd_cuda_rms_norm_warp_sum(value);
    if (lane == 0U) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < warp_count ? warp_sums[lane] : 0.0f;
    if (warp == 0U) {
        value = gd_cuda_rms_norm_warp_sum(value);
        if (lane == 0U) {
            result = value;
        }
    }
    __syncthreads();
    return result;
}

static __global__ void gd_cuda_rms_norm_forward_kernel(const unsigned char *xbuf,
                                                       const unsigned char *weightbuf,
                                                       unsigned char *outbuf,
                                                       unsigned char *invbuf,
                                                       gd_cuda_rms_norm_launch_args args,
                                                       uint32_t write_inv)
{
    const size_t row = (size_t)blockIdx.x;
    const size_t base = row * args.cols;
    float local_ss = 0.0f;
    float ss;
    float inv;
    if (row >= args.rows) {
        return;
    }
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        local_ss += xv * xv;
    }
    ss = gd_cuda_rms_norm_block_sum(local_ss);
    inv = rsqrtf(ss / (float)args.cols + args.eps);
    if (write_inv != 0U && threadIdx.x == 0U) {
        float *inv_out = (float *)(void *)(invbuf + args.inv_rms_offset);
        inv_out[row] = inv;
    }
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        const float wv = gd_cuda_rms_norm_load_f32(weightbuf, args.weight_offset, args.dtype, c);
        gd_cuda_rms_norm_store_f32(outbuf, args.out_offset, args.dtype, base + c, xv * wv * inv);
    }
}

static __global__ void gd_cuda_rms_norm_inv_kernel(const unsigned char *xbuf,
                                                   unsigned char *invbuf,
                                                   gd_cuda_rms_norm_launch_args args)
{
    const size_t row = (size_t)blockIdx.x;
    const size_t base = row * args.cols;
    float local_ss = 0.0f;
    float ss;
    if (row >= args.rows) {
        return;
    }
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        local_ss += xv * xv;
    }
    ss = gd_cuda_rms_norm_block_sum(local_ss);
    if (threadIdx.x == 0U) {
        float *inv = (float *)(void *)(invbuf + args.inv_rms_offset);
        inv[row] = rsqrtf(ss / (float)args.cols + args.eps);
    }
}

static __global__ void gd_cuda_rms_norm_backward_kernel(const unsigned char *xbuf,
                                                        const unsigned char *weightbuf,
                                                        const unsigned char *gradbuf,
                                                        unsigned char *dxbuf,
                                                        gd_cuda_rms_norm_launch_args args)
{
    const size_t row = (size_t)blockIdx.x;
    const size_t base = row * args.cols;
    float local_ss = 0.0f;
    float local_a = 0.0f;
    float ss;
    float a;
    float inv;
    float inv3_over_cols;
    if (row >= args.rows) {
        return;
    }
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        const float wv = gd_cuda_rms_norm_load_f32(weightbuf, args.weight_offset, args.dtype, c);
        const float gv = gd_cuda_rms_norm_load_f32(gradbuf,
                                                   args.grad_out_offset,
                                                   args.dtype,
                                                   base + c);
        local_ss += xv * xv;
        local_a += gv * wv * xv;
    }
    ss = gd_cuda_rms_norm_block_sum(local_ss);
    a = gd_cuda_rms_norm_block_sum(local_a);
    inv = rsqrtf(ss / (float)args.cols + args.eps);
    inv3_over_cols = inv * inv * inv / (float)args.cols;
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        const float wv = gd_cuda_rms_norm_load_f32(weightbuf, args.weight_offset, args.dtype, c);
        const float gv = gd_cuda_rms_norm_load_f32(gradbuf,
                                                   args.grad_out_offset,
                                                   args.dtype,
                                                   base + c);
        const float dx = inv * gv * wv - xv * inv3_over_cols * a;
        gd_cuda_rms_norm_store_f32(dxbuf, args.out_offset, args.dtype, base + c, dx);
    }
}

static __global__ void gd_cuda_rms_norm_backward_stats_kernel(const unsigned char *xbuf,
                                                              const unsigned char *weightbuf,
                                                              const unsigned char *invbuf,
                                                              const unsigned char *gradbuf,
                                                              unsigned char *dxbuf,
                                                              gd_cuda_rms_norm_launch_args args)
{
    const size_t row = (size_t)blockIdx.x;
    const size_t base = row * args.cols;
    const float *inv_ptr = (const float *)(const void *)(invbuf + args.inv_rms_offset);
    float local_a = 0.0f;
    float a;
    float inv;
    float inv3_over_cols;
    if (row >= args.rows) {
        return;
    }
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        const float wv = gd_cuda_rms_norm_load_f32(weightbuf, args.weight_offset, args.dtype, c);
        const float gv = gd_cuda_rms_norm_load_f32(gradbuf,
                                                   args.grad_out_offset,
                                                   args.dtype,
                                                   base + c);
        local_a += gv * wv * xv;
    }
    a = gd_cuda_rms_norm_block_sum(local_a);
    inv = inv_ptr[row];
    inv3_over_cols = inv * inv * inv / (float)args.cols;
    for (size_t c = (size_t)threadIdx.x; c < args.cols; c += (size_t)blockDim.x) {
        const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, base + c);
        const float wv = gd_cuda_rms_norm_load_f32(weightbuf, args.weight_offset, args.dtype, c);
        const float gv = gd_cuda_rms_norm_load_f32(gradbuf,
                                                   args.grad_out_offset,
                                                   args.dtype,
                                                   base + c);
        const float dx = inv * gv * wv - xv * inv3_over_cols * a;
        gd_cuda_rms_norm_store_f32(dxbuf, args.out_offset, args.dtype, base + c, dx);
    }
}

static __global__ void gd_cuda_rms_norm_wgrad_stage_kernel(const unsigned char *xbuf,
                                                           const unsigned char *invbuf,
                                                           const unsigned char *gradbuf,
                                                           unsigned char *partialbuf,
                                                           gd_cuda_rms_norm_launch_args args)
{
    __shared__ float inv_sh[GD_CUDA_RMS_NORM_WGRAD_ROW_BLOCK_MAX];
    const size_t channel_block = (size_t)blockIdx.x;
    const size_t row_block = (size_t)blockIdx.y;
    const size_t tid = (size_t)threadIdx.x;
    const size_t c = channel_block * GD_CUDA_RMS_NORM_WGRAD_CHANNELS + tid;
    const size_t row0 = row_block * (size_t)args.wgrad_row_block;
    size_t tile = args.rows > row0 ? args.rows - row0 : 0U;
    const float *inv = (const float *)(const void *)(invbuf + args.inv_rms_offset);
    float *partial = (float *)(void *)(partialbuf + args.partial_offset);
    if (tile > (size_t)args.wgrad_row_block) {
        tile = (size_t)args.wgrad_row_block;
    }
    if (tid < tile) {
        inv_sh[tid] = inv[row0 + tid];
    }
    __syncthreads();
    if (c < args.cols && row_block < args.row_blocks) {
        float acc = 0.0f;
        for (size_t i = 0U; i < tile; ++i) {
            const size_t idx = (row0 + i) * args.cols + c;
            const float xv = gd_cuda_rms_norm_load_f32(xbuf, args.x_offset, args.dtype, idx);
            const float gv = gd_cuda_rms_norm_load_f32(gradbuf,
                                                       args.grad_out_offset,
                                                       args.dtype,
                                                       idx);
            acc += gv * xv * inv_sh[i];
        }
        partial[row_block * args.cols + c] = acc;
    }
}

static __global__ void gd_cuda_rms_norm_wgrad_reduce_kernel(const unsigned char *partialbuf,
                                                            unsigned char *dwbuf,
                                                            gd_cuda_rms_norm_launch_args args)
{
    const size_t c = (size_t)blockIdx.x;
    const float *partial = (const float *)(const void *)(partialbuf + args.partial_offset);
    float local = 0.0f;
    float sum;
    if (c >= args.cols) {
        return;
    }
    for (size_t rb = (size_t)threadIdx.x; rb < args.row_blocks; rb += (size_t)blockDim.x) {
        local += partial[rb * args.cols + c];
    }
    sum = gd_cuda_rms_norm_block_sum(local);
    if (threadIdx.x == 0U) {
        gd_cuda_rms_norm_store_f32(dwbuf, args.out_offset, args.dtype, c, sum);
    }
}

static bool gd_cuda_rms_norm_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == GD_CUDA_RMS_NORM_DTYPE_F16) {
        *out_size = 2U;
        return true;
    }
    if (dtype == GD_CUDA_RMS_NORM_DTYPE_F32) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static bool gd_cuda_rms_norm_count_bytes(size_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U ||
        !gd_cuda_rms_norm_dtype_size(dtype, &elem_size) || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_cuda_rms_norm_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_cuda_rms_norm_dtype_size(view->dtype, &elem_size) &&
           (view->offset % elem_size) == 0U &&
           gd_cuda_rms_norm_count_bytes(view->count, view->dtype, &nbytes) &&
           gd_cuda_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_cuda_rms_norm_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_CUDA_RMS_NORM_MAX_DIMS) {
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

static bool gd_cuda_rms_norm_count_matches(const gd_backend_tensor_view *view, size_t count)
{
    return view != NULL && view->count == count;
}

static gd_status gd_cuda_rms_norm_args_count(const gd_backend_rms_norm_args *args,
                                             size_t *rows,
                                             size_t *cols,
                                             size_t *row_blocks,
                                             size_t *count)
{
    if (args == NULL || rows == NULL || cols == NULL || row_blocks == NULL || count == NULL ||
        args->rows == 0U || args->cols == 0U || args->rows > (uint64_t)SIZE_MAX ||
        args->cols > (uint64_t)SIZE_MAX || args->row_blocks > (uint64_t)SIZE_MAX ||
        args->rows > (uint64_t)(SIZE_MAX / (size_t)args->cols) ||
        args->rows > (uint64_t)UINT32_MAX || args->cols > (uint64_t)UINT32_MAX ||
        args->row_blocks > (uint64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!(args->eps > 0.0f) || !(args->eps == args->eps) || args->simdgroups == 0U ||
        args->simdgroups > GD_CUDA_RMS_NORM_MAX_SIMDGROUPS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *rows = (size_t)args->rows;
    *cols = (size_t)args->cols;
    *row_blocks = (size_t)args->row_blocks;
    *count = *rows * *cols;
    return GD_OK;
}

static gd_status gd_cuda_rms_norm_validate_common(const gd_backend_tensor_view *x,
                                                  const gd_backend_tensor_view *weight,
                                                  const gd_backend_tensor_view *out,
                                                  const gd_backend_rms_norm_args *args,
                                                  size_t *rows,
                                                  size_t *cols,
                                                  size_t *row_blocks,
                                                  size_t *count)
{
    gd_status st = gd_cuda_rms_norm_args_count(args, rows, cols, row_blocks, count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cuda_rms_norm_view_range_valid(x) ||
        !gd_cuda_rms_norm_view_range_valid(weight) ||
        !gd_cuda_rms_norm_view_range_valid(out) || !gd_cuda_rms_norm_contiguous_view(x) ||
        !gd_cuda_rms_norm_contiguous_view(weight) ||
        !gd_cuda_rms_norm_contiguous_view(out) || x->dtype != weight->dtype ||
        x->dtype != out->dtype || weight->rank != 1U || weight->shape[0] <= 0 ||
        (size_t)weight->shape[0] != *cols || !gd_cuda_rms_norm_count_matches(x, *count) ||
        !gd_cuda_rms_norm_count_matches(out, *count) ||
        !gd_cuda_rms_norm_count_matches(weight, *cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cuda_rms_norm_validate_inv(const gd_backend_tensor_view *inv_rms,
                                               size_t rows)
{
    return inv_rms != NULL && gd_cuda_rms_norm_view_range_valid(inv_rms) &&
                   gd_cuda_rms_norm_contiguous_view(inv_rms) &&
                   inv_rms->dtype == GD_CUDA_RMS_NORM_DTYPE_F32 && inv_rms->rank == 1U &&
                   inv_rms->shape[0] > 0 && (size_t)inv_rms->shape[0] == rows &&
                   gd_cuda_rms_norm_count_matches(inv_rms, rows)
               ? GD_OK
               : GD_ERR_INVALID_ARGUMENT;
}

static gd_status gd_cuda_rms_norm_validate_grad_out(const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *grad_out,
                                                    size_t count)
{
    return x != NULL && grad_out != NULL && gd_cuda_rms_norm_view_range_valid(grad_out) &&
                   gd_cuda_rms_norm_contiguous_view(grad_out) && grad_out->dtype == x->dtype &&
                   gd_cuda_rms_norm_count_matches(grad_out, count)
               ? GD_OK
               : GD_ERR_INVALID_ARGUMENT;
}

static gd_status gd_cuda_rms_norm_validate_inv_only(const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *inv_rms,
                                                    const gd_backend_rms_norm_args *args,
                                                    size_t *rows,
                                                    size_t *cols,
                                                    size_t *row_blocks,
                                                    size_t *count)
{
    gd_status st = gd_cuda_rms_norm_args_count(args, rows, cols, row_blocks, count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_cuda_rms_norm_view_range_valid(x) || !gd_cuda_rms_norm_contiguous_view(x) ||
        !gd_cuda_rms_norm_count_matches(x, *count) ||
        gd_cuda_rms_norm_validate_inv(inv_rms, *rows) != GD_OK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_cuda_rms_norm_validate_weight_backward(const gd_backend_tensor_view *x,
                                                           const gd_backend_tensor_view *inv_rms,
                                                           const gd_backend_tensor_view *grad_out,
                                                           const gd_backend_tensor_view *grad_weight,
                                                           const gd_backend_tensor_view *partial,
                                                           const gd_backend_rms_norm_args *args,
                                                           size_t *rows,
                                                           size_t *cols,
                                                           size_t *row_blocks,
                                                           size_t *count)
{
    size_t partial_count;
    gd_status st = gd_cuda_rms_norm_args_count(args, rows, cols, row_blocks, count);
    if (st != GD_OK) {
        return st;
    }
    if (*row_blocks == 0U || args->wgrad_row_block == 0U ||
        args->wgrad_row_block > GD_CUDA_RMS_NORM_WGRAD_ROW_BLOCK_MAX ||
        args->wgrad_simdgroups == 0U ||
        args->wgrad_simdgroups > GD_CUDA_RMS_NORM_MAX_SIMDGROUPS ||
        !gd_cuda_rms_norm_view_range_valid(x) || !gd_cuda_rms_norm_contiguous_view(x) ||
        !gd_cuda_rms_norm_count_matches(x, *count) ||
        gd_cuda_rms_norm_validate_inv(inv_rms, *rows) != GD_OK ||
        gd_cuda_rms_norm_validate_grad_out(x, grad_out, *count) != GD_OK ||
        !gd_cuda_rms_norm_view_range_valid(grad_weight) ||
        !gd_cuda_rms_norm_contiguous_view(grad_weight) || grad_weight->dtype != x->dtype ||
        grad_weight->rank != 1U || grad_weight->shape[0] <= 0 ||
        (size_t)grad_weight->shape[0] != *cols ||
        !gd_cuda_rms_norm_count_matches(grad_weight, *cols) || partial == NULL ||
        *row_blocks > SIZE_MAX / *cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    partial_count = *row_blocks * *cols;
    if (!gd_cuda_rms_norm_view_range_valid(partial) ||
        !gd_cuda_rms_norm_contiguous_view(partial) ||
        partial->dtype != GD_CUDA_RMS_NORM_DTYPE_F32 || partial->rank != 2U ||
        partial->shape[0] <= 0 || partial->shape[1] <= 0 ||
        (size_t)partial->shape[0] != *row_blocks || (size_t)partial->shape[1] != *cols ||
        !gd_cuda_rms_norm_count_matches(partial, partial_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static unsigned int gd_cuda_rms_norm_row_threads(uint32_t simdgroups)
{
    uint32_t groups = simdgroups == 0U ? 1U : simdgroups;
    if (groups > GD_CUDA_RMS_NORM_MAX_SIMDGROUPS) {
        groups = GD_CUDA_RMS_NORM_MAX_SIMDGROUPS;
    }
    return groups * 32U;
}

static unsigned int gd_cuda_rms_norm_wgrad_threads(uint32_t simdgroups)
{
    return gd_cuda_rms_norm_row_threads(simdgroups);
}

static void gd_cuda_rms_norm_fill_launch_args(gd_cuda_rms_norm_launch_args *launch,
                                              const gd_backend_tensor_view *x,
                                              const gd_backend_tensor_view *weight,
                                              const gd_backend_tensor_view *out,
                                              const gd_backend_tensor_view *grad_out,
                                              const gd_backend_tensor_view *inv_rms,
                                              const gd_backend_tensor_view *partial,
                                              const gd_backend_rms_norm_args *args,
                                              size_t rows,
                                              size_t cols,
                                              size_t row_blocks)
{
    memset(launch, 0, sizeof(*launch));
    launch->x_offset = x != NULL ? x->offset : 0U;
    launch->weight_offset = weight != NULL ? weight->offset : 0U;
    launch->out_offset = out != NULL ? out->offset : 0U;
    launch->grad_out_offset = grad_out != NULL ? grad_out->offset : 0U;
    launch->inv_rms_offset = inv_rms != NULL ? inv_rms->offset : 0U;
    launch->partial_offset = partial != NULL ? partial->offset : 0U;
    launch->rows = rows;
    launch->cols = cols;
    launch->row_blocks = row_blocks;
    launch->eps = args->eps;
    launch->dtype = x != NULL ? x->dtype : (out != NULL ? out->dtype : 0U);
    launch->simdgroups = args->simdgroups;
    launch->wgrad_simdgroups = args->wgrad_simdgroups;
    launch->wgrad_row_block = args->wgrad_row_block;
}

static gd_status gd_cuda_rms_norm_forward_dispatch(gd_backend *backend,
                                                   const gd_backend_tensor_view *x,
                                                   const gd_backend_tensor_view *weight,
                                                   const gd_backend_tensor_view *out,
                                                   const gd_backend_tensor_view *inv_rms,
                                                   const gd_backend_rms_norm_args *args)
{
    gd_cuda_rms_norm_launch_args launch;
    size_t rows;
    size_t cols;
    size_t row_blocks;
    size_t count;
    unsigned int threads;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_rms_norm_validate_common(x, weight, out, args, &rows, &cols, &row_blocks, &count);
    if (st != GD_OK) {
        return st;
    }
    if (inv_rms != NULL) {
        st = gd_cuda_rms_norm_validate_inv(inv_rms, rows);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_rms_norm_fill_launch_args(&launch,
                                      x,
                                      weight,
                                      out,
                                      NULL,
                                      inv_rms,
                                      NULL,
                                      args,
                                      rows,
                                      cols,
                                      row_blocks);
    threads = gd_cuda_rms_norm_row_threads(args->simdgroups);
    gd_cuda_rms_norm_forward_kernel<<<(unsigned int)rows,
                                       threads,
                                       0U,
                                       gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
        (const unsigned char *)gd_cuda_buffer_const_ptr(weight->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(out->buffer),
        inv_rms != NULL ? (unsigned char *)gd_cuda_buffer_ptr(inv_rms->buffer) : NULL,
        launch,
        inv_rms != NULL ? 1U : 0U);
    (void)count;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

extern "C" gd_status gd_backend_rms_norm_forward(gd_backend *backend,
                                                   const gd_backend_tensor_view *x,
                                                   const gd_backend_tensor_view *weight,
                                                   const gd_backend_tensor_view *out,
                                                   const gd_backend_rms_norm_args *args)
{
    return gd_cuda_rms_norm_forward_dispatch(backend, x, weight, out, NULL, args);
}

extern "C" gd_status gd_backend_rms_norm_forward_stats(gd_backend *backend,
                                                         const gd_backend_tensor_view *x,
                                                         const gd_backend_tensor_view *weight,
                                                         const gd_backend_tensor_view *out,
                                                         const gd_backend_tensor_view *inv_rms,
                                                         const gd_backend_rms_norm_args *args)
{
    return gd_cuda_rms_norm_forward_dispatch(backend, x, weight, out, inv_rms, args);
}

extern "C" gd_status gd_backend_rms_norm_inv(gd_backend *backend,
                                               const gd_backend_tensor_view *x,
                                               const gd_backend_tensor_view *inv_rms,
                                               const gd_backend_rms_norm_args *args)
{
    gd_cuda_rms_norm_launch_args launch;
    size_t rows;
    size_t cols;
    size_t row_blocks;
    size_t count;
    unsigned int threads;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_rms_norm_validate_inv_only(x, inv_rms, args, &rows, &cols, &row_blocks, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_rms_norm_fill_launch_args(&launch,
                                      x,
                                      NULL,
                                      NULL,
                                      NULL,
                                      inv_rms,
                                      NULL,
                                      args,
                                      rows,
                                      cols,
                                      row_blocks);
    threads = gd_cuda_rms_norm_row_threads(args->simdgroups);
    gd_cuda_rms_norm_inv_kernel<<<(unsigned int)rows,
                                   threads,
                                   0U,
                                   gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(inv_rms->buffer),
        launch);
    (void)count;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

static gd_status gd_cuda_rms_norm_backward_dispatch(gd_backend *backend,
                                                    const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *weight,
                                                    const gd_backend_tensor_view *inv_rms,
                                                    const gd_backend_tensor_view *grad_out,
                                                    const gd_backend_tensor_view *grad_x,
                                                    const gd_backend_rms_norm_args *args)
{
    gd_cuda_rms_norm_launch_args launch;
    size_t rows;
    size_t cols;
    size_t row_blocks;
    size_t count;
    unsigned int threads;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_rms_norm_validate_common(x,
                                          weight,
                                          grad_x,
                                          args,
                                          &rows,
                                          &cols,
                                          &row_blocks,
                                          &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_rms_norm_validate_grad_out(x, grad_out, count);
    if (st != GD_OK) {
        return st;
    }
    if (inv_rms != NULL) {
        st = gd_cuda_rms_norm_validate_inv(inv_rms, rows);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_rms_norm_fill_launch_args(&launch,
                                      x,
                                      weight,
                                      grad_x,
                                      grad_out,
                                      inv_rms,
                                      NULL,
                                      args,
                                      rows,
                                      cols,
                                      row_blocks);
    threads = gd_cuda_rms_norm_row_threads(args->simdgroups);
    if (inv_rms != NULL) {
        gd_cuda_rms_norm_backward_stats_kernel<<<(unsigned int)rows,
                                                  threads,
                                                  0U,
                                                  gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(weight->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(inv_rms->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(grad_out->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(grad_x->buffer),
            launch);
    } else {
        gd_cuda_rms_norm_backward_kernel<<<(unsigned int)rows,
                                            threads,
                                            0U,
                                            gd_cuda_stream(backend)>>>(
            (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(weight->buffer),
            (const unsigned char *)gd_cuda_buffer_const_ptr(grad_out->buffer),
            (unsigned char *)gd_cuda_buffer_ptr(grad_x->buffer),
            launch);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}

extern "C" gd_status gd_backend_rms_norm_backward(gd_backend *backend,
                                                    const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *weight,
                                                    const gd_backend_tensor_view *grad_out,
                                                    const gd_backend_tensor_view *grad_x,
                                                    const gd_backend_rms_norm_args *args)
{
    return gd_cuda_rms_norm_backward_dispatch(backend, x, weight, NULL, grad_out, grad_x, args);
}

extern "C" gd_status gd_backend_rms_norm_backward_stats(gd_backend *backend,
                                                          const gd_backend_tensor_view *x,
                                                          const gd_backend_tensor_view *weight,
                                                          const gd_backend_tensor_view *inv_rms,
                                                          const gd_backend_tensor_view *grad_out,
                                                          const gd_backend_tensor_view *grad_x,
                                                          const gd_backend_rms_norm_args *args)
{
    return gd_cuda_rms_norm_backward_dispatch(backend,
                                             x,
                                             weight,
                                             inv_rms,
                                             grad_out,
                                             grad_x,
                                             args);
}

extern "C" gd_status gd_backend_rms_norm_weight_backward_stats(gd_backend *backend,
                                                                 const gd_backend_tensor_view *x,
                                                                 const gd_backend_tensor_view *inv_rms,
                                                                 const gd_backend_tensor_view *grad_out,
                                                                 const gd_backend_tensor_view *grad_weight,
                                                                 const gd_backend_tensor_view *partial,
                                                                 const gd_backend_rms_norm_args *args)
{
    gd_cuda_rms_norm_launch_args launch;
    size_t rows;
    size_t cols;
    size_t row_blocks;
    size_t count;
    unsigned int reduce_threads;
    dim3 stage_grid;
    gd_status st;
    cudaError_t err;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cuda_rms_norm_validate_weight_backward(x,
                                                   inv_rms,
                                                   grad_out,
                                                   grad_weight,
                                                   partial,
                                                   args,
                                                   &rows,
                                                   &cols,
                                                   &row_blocks,
                                                   &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cuda_status(cudaSetDevice(backend->device));
    if (st != GD_OK) {
        return st;
    }
    gd_cuda_rms_norm_fill_launch_args(&launch,
                                      x,
                                      NULL,
                                      grad_weight,
                                      grad_out,
                                      inv_rms,
                                      partial,
                                      args,
                                      rows,
                                      cols,
                                      row_blocks);
    stage_grid = dim3(gd_cuda_blocks_for_count(cols, GD_CUDA_RMS_NORM_WGRAD_CHANNELS),
                      (unsigned int)row_blocks,
                      1U);
    gd_cuda_rms_norm_wgrad_stage_kernel<<<stage_grid,
                                           GD_CUDA_RMS_NORM_WGRAD_CHANNELS,
                                           0U,
                                           gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(x->buffer),
        (const unsigned char *)gd_cuda_buffer_const_ptr(inv_rms->buffer),
        (const unsigned char *)gd_cuda_buffer_const_ptr(grad_out->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(partial->buffer),
        launch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    reduce_threads = gd_cuda_rms_norm_wgrad_threads(args->wgrad_simdgroups);
    gd_cuda_rms_norm_wgrad_reduce_kernel<<<(unsigned int)cols,
                                            reduce_threads,
                                            0U,
                                            gd_cuda_stream(backend)>>>(
        (const unsigned char *)gd_cuda_buffer_const_ptr(partial->buffer),
        (unsigned char *)gd_cuda_buffer_ptr(grad_weight->buffer),
        launch);
    (void)count;
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        return gd_cuda_status(err);
    }
    return gd_cuda_finish_if_immediate(backend);
}
