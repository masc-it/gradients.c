#include "../../backends/metal/metal_backend_internal.h"
#include "metal_rope_types.h"

#include <gradients/tensor.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define GD_METAL_ROPE_MAX_THREADS_PER_GROUP 256U

static id<MTLComputePipelineState> gd_rope_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->rope_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->rope_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_rope_backward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->rope_backward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->rope_backward_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_rope_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_rope_byte_range_valid(const gd_backend_buffer *buffer, size_t offset, size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_rope_dtype_size(uint32_t dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return false;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        *out_size = 2U;
        return true;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
        *out_size = 4U;
        return true;
    }
    return false;
}

static bool gd_rope_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U ||
        !gd_rope_dtype_size(view->dtype, &elem_size) || view->count > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = view->count * elem_size;
    return gd_rope_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_rope_contiguous_valid(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride) {
            return false;
        }
        if (stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_rope_same_shape(const gd_backend_tensor_view *a, const gd_backend_tensor_view *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->rank != b->rank || a->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static bool gd_rope_leading_count(const gd_backend_tensor_view *x, uint64_t *out_count)
{
    uint64_t count = 1U;
    uint32_t i;
    if (x == NULL || out_count == NULL || x->rank < 2U) {
        return false;
    }
    for (i = 0U; i + 2U < x->rank; ++i) {
        if (x->shape[i] <= 0 || (uint64_t)x->shape[i] > UINT64_MAX / count) {
            return false;
        }
        count *= (uint64_t)x->shape[i];
    }
    *out_count = count;
    return true;
}

static bool gd_rope_args_valid(const gd_backend_tensor_view *x,
                               const gd_backend_tensor_view *pos_ids,
                               const gd_backend_tensor_view *out,
                               const gd_backend_rope_args *args,
                               gd_metal_rope_args *metal_args)
{
    uint64_t leading_count;
    uint64_t rows;
    uint64_t half_dims;
    uint64_t tail;
    uint64_t lanes;
    uint64_t total_threads;
    int64_t head_dim;
    int64_t heads;
    if (!gd_rope_view_range_valid(x) || !gd_rope_view_range_valid(pos_ids) ||
        !gd_rope_view_range_valid(out) || !gd_rope_contiguous_valid(x) ||
        !gd_rope_contiguous_valid(pos_ids) || !gd_rope_contiguous_valid(out) || args == NULL ||
        metal_args == NULL) {
        return false;
    }
    if ((x->dtype != (uint32_t)GD_DTYPE_F16 && x->dtype != (uint32_t)GD_DTYPE_F32) ||
        out->dtype != x->dtype || pos_ids->dtype != (uint32_t)GD_DTYPE_I32 || x->rank < 2U ||
        !gd_rope_same_shape(x, out)) {
        return false;
    }
    head_dim = x->shape[x->rank - 1U];
    heads = x->shape[x->rank - 2U];
    if (head_dim <= 0 || heads <= 0 || head_dim > (int64_t)UINT32_MAX ||
        heads > (int64_t)UINT32_MAX || args->n_dims == 0U || (args->n_dims & 1U) != 0U ||
        (uint64_t)args->n_dims > (uint64_t)head_dim || args->interleaved > 1U ||
        !isfinite(args->theta) || args->theta <= 0.0f || x->count % (size_t)head_dim != 0U ||
        !gd_rope_leading_count(x, &leading_count) || pos_ids->count != (size_t)leading_count) {
        return false;
    }
    rows = (uint64_t)(x->count / (size_t)head_dim);
    half_dims = (uint64_t)args->n_dims / 2U;
    tail = (uint64_t)head_dim - (uint64_t)args->n_dims;
    lanes = half_dims > tail ? half_dims : tail;
    if (rows == 0U || lanes == 0U || rows > (uint64_t)UINT32_MAX || lanes > (uint64_t)UINT32_MAX ||
        rows > (uint64_t)UINT32_MAX / lanes) {
        return false;
    }
    total_threads = rows * lanes;
    if (total_threads == 0U || total_threads > (uint64_t)UINT32_MAX) {
        return false;
    }
    memset(metal_args, 0, sizeof(*metal_args));
    metal_args->x_offset = (uint64_t)x->offset;
    metal_args->pos_offset = (uint64_t)pos_ids->offset;
    metal_args->out_offset = (uint64_t)out->offset;
    metal_args->rows = (uint32_t)rows;
    metal_args->head_dim = (uint32_t)head_dim;
    metal_args->heads = (uint32_t)heads;
    metal_args->n_dims = args->n_dims;
    metal_args->lanes_per_row = (uint32_t)lanes;
    metal_args->interleaved = args->interleaved;
    metal_args->freq_scale = -2.0f * log2f(args->theta) / (float)args->n_dims;
    return true;
}

static MTLSize gd_rope_grid(const gd_metal_rope_args *args)
{
    return MTLSizeMake((NSUInteger)args->rows * (NSUInteger)args->lanes_per_row, 1U, 1U);
}

static MTLSize gd_rope_threads(const gd_metal_rope_args *args)
{
    NSUInteger threads = (NSUInteger)args->rows * (NSUInteger)args->lanes_per_row;
    if (threads > (NSUInteger)GD_METAL_ROPE_MAX_THREADS_PER_GROUP) {
        threads = (NSUInteger)GD_METAL_ROPE_MAX_THREADS_PER_GROUP;
    }
    return MTLSizeMake(threads, 1U, 1U);
}

static gd_status gd_backend_rope_encode(gd_backend *backend,
                                        const gd_backend_tensor_view *x,
                                        const gd_backend_tensor_view *pos_ids,
                                        const gd_backend_tensor_view *out,
                                        const gd_backend_rope_args *args,
                                        float sin_sign,
                                        bool backward)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_rope_args metal_args;
    bool immediate;
    gd_status st;
    if (backend == NULL || !gd_rope_args_valid(x, pos_ids, out, args, &metal_args)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    metal_args.sin_sign = sin_sign;
    pso = backward ? gd_rope_backward_pso(backend, x->dtype) : gd_rope_forward_pso(backend, x->dtype);
    if (pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_rope_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rope_buffer(pos_ids->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_rope_buffer(out->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:3U];
    [encoder dispatchThreads:gd_rope_grid(&metal_args) threadsPerThreadgroup:gd_rope_threads(&metal_args)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_rope(gd_backend *backend,
                          const gd_backend_tensor_view *x,
                          const gd_backend_tensor_view *pos_ids,
                          const gd_backend_tensor_view *out,
                          const gd_backend_rope_args *args)
{
    return gd_backend_rope_encode(backend, x, pos_ids, out, args, 1.0f, false);
}

gd_status gd_backend_rope_backward(gd_backend *backend,
                                   const gd_backend_tensor_view *grad_out,
                                   const gd_backend_tensor_view *pos_ids,
                                   const gd_backend_tensor_view *grad_x,
                                   const gd_backend_rope_args *args)
{
    return gd_backend_rope_encode(backend, grad_out, pos_ids, grad_x, args, -1.0f, true);
}
