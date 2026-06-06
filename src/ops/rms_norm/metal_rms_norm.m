#include "../../backends/metal/metal_backend_internal.h"
#include "metal_rms_norm_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_rms_norm_forward_pso(gd_backend *backend, uint32_t dtype, bool stats)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)(stats ? backend->rms_norm_forward_stats_f16_pso
                                                            : backend->rms_norm_forward_f16_pso);
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)(stats ? backend->rms_norm_forward_stats_f32_pso
                                                            : backend->rms_norm_forward_f32_pso);
    }
    return nil;
}

static id<MTLComputePipelineState> gd_rms_norm_inv_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_inv_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_inv_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_rms_norm_backward_pso(gd_backend *backend, uint32_t dtype, bool stats)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)(stats ? backend->rms_norm_backward_stats_f16_pso
                                                            : backend->rms_norm_backward_f16_pso);
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)(stats ? backend->rms_norm_backward_stats_f32_pso
                                                            : backend->rms_norm_backward_f32_pso);
    }
    return nil;
}

static id<MTLComputePipelineState> gd_rms_norm_wgrad_stage_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_wgrad_stage_stats_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_wgrad_stage_stats_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_rms_norm_wgrad_reduce_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_wgrad_reduce_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->rms_norm_wgrad_reduce_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_rms_norm_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static size_t gd_rms_norm_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool gd_rms_norm_byte_range_valid(const gd_backend_buffer *buffer,
                                         size_t offset,
                                         size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_rms_norm_count_bytes(uint64_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    elem_size = gd_rms_norm_dtype_size(dtype);
    if (elem_size == 0U || count > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    *out_nbytes = (size_t)count * elem_size;
    return true;
}

static bool gd_rms_norm_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_rms_norm_count_bytes((uint64_t)view->count, view->dtype, &nbytes) &&
           gd_rms_norm_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_rms_norm_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t stride = 1;
    uint32_t i;
    if (view == NULL || view->rank > 8U) {
        return false;
    }
    for (i = view->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        if (view->shape[dim] <= 0 || view->strides[dim] != stride) {
            return false;
        }
        if (view->shape[dim] != 0 && stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_rms_norm_count_matches(const gd_backend_tensor_view *view, uint64_t count)
{
    return view != NULL && view->count == (size_t)count && (uint64_t)view->count == count;
}

static gd_status gd_rms_norm_validate_args(const gd_backend_tensor_view *x,
                                           const gd_backend_tensor_view *weight,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_rms_norm_args *args)
{
    uint64_t count;
    if (x == NULL || weight == NULL || out == NULL || args == NULL ||
        args->rows == 0U || args->cols == 0U || args->simdgroups == 0U ||
        args->simdgroups > GD_METAL_RMS_NORM_MAX_SIMDGROUPS ||
        !(args->eps > 0.0f) || !(args->eps == args->eps) ||
        args->rows > UINT64_MAX / args->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    count = args->rows * args->cols;
    if (!gd_rms_norm_view_range_valid(x) || !gd_rms_norm_view_range_valid(weight) ||
        !gd_rms_norm_view_range_valid(out) || !gd_rms_norm_contiguous_view(x) ||
        !gd_rms_norm_contiguous_view(weight) || !gd_rms_norm_contiguous_view(out) ||
        x->dtype != weight->dtype || x->dtype != out->dtype ||
        gd_rms_norm_dtype_size(x->dtype) == 0U ||
        !gd_rms_norm_count_matches(x, count) || !gd_rms_norm_count_matches(out, count) ||
        weight->rank != 1U || weight->shape[0] <= 0 ||
        (uint64_t)weight->shape[0] != args->cols ||
        !gd_rms_norm_count_matches(weight, args->cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_rms_norm_validate_inv(const gd_backend_tensor_view *x,
                                          const gd_backend_tensor_view *inv_rms,
                                          const gd_backend_rms_norm_args *args)
{
    if (x == NULL || inv_rms == NULL || args == NULL ||
        !gd_rms_norm_view_range_valid(inv_rms) || !gd_rms_norm_contiguous_view(inv_rms) ||
        inv_rms->dtype != (uint32_t)GD_DTYPE_F32 || inv_rms->rank != 1U ||
        inv_rms->shape[0] <= 0 || (uint64_t)inv_rms->shape[0] != args->rows ||
        !gd_rms_norm_count_matches(inv_rms, args->rows) || x->count == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_rms_norm_validate_grad_out(const gd_backend_tensor_view *x,
                                               const gd_backend_tensor_view *grad_out,
                                               const gd_backend_rms_norm_args *args)
{
    if (x == NULL || grad_out == NULL || args == NULL ||
        !gd_rms_norm_view_range_valid(grad_out) || !gd_rms_norm_contiguous_view(grad_out) ||
        grad_out->dtype != x->dtype || !gd_rms_norm_count_matches(grad_out, args->rows * args->cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_rms_norm_validate_partial(const gd_backend_tensor_view *partial,
                                               const gd_backend_rms_norm_args *args)
{
    uint64_t count;
    if (partial == NULL || args == NULL || args->row_blocks == 0U ||
        args->row_blocks > UINT64_MAX / args->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    count = args->row_blocks * args->cols;
    if (!gd_rms_norm_view_range_valid(partial) || !gd_rms_norm_contiguous_view(partial) ||
        partial->dtype != (uint32_t)GD_DTYPE_F32 || partial->rank != 2U ||
        partial->shape[0] <= 0 || partial->shape[1] <= 0 ||
        (uint64_t)partial->shape[0] != args->row_blocks ||
        (uint64_t)partial->shape[1] != args->cols ||
        !gd_rms_norm_count_matches(partial, count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_rms_norm_fill_args(gd_metal_rms_norm_args *metal_args,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *weight,
                                  const gd_backend_tensor_view *out,
                                  const gd_backend_tensor_view *grad_out,
                                  const gd_backend_tensor_view *inv_rms,
                                  const gd_backend_tensor_view *partial,
                                  const gd_backend_rms_norm_args *args)
{
    memset(metal_args, 0, sizeof(*metal_args));
    metal_args->x_offset = (uint64_t)x->offset;
    metal_args->weight_offset = weight != NULL ? (uint64_t)weight->offset : 0U;
    metal_args->out_offset = out != NULL ? (uint64_t)out->offset : 0U;
    metal_args->grad_out_offset = grad_out != NULL ? (uint64_t)grad_out->offset : 0U;
    metal_args->inv_rms_offset = inv_rms != NULL ? (uint64_t)inv_rms->offset : 0U;
    metal_args->partial_offset = partial != NULL ? (uint64_t)partial->offset : 0U;
    metal_args->rows = args->rows;
    metal_args->cols = args->cols;
    metal_args->row_blocks = args->row_blocks;
    metal_args->eps = args->eps;
    metal_args->simdgroups = args->simdgroups;
    metal_args->wgrad_simdgroups = args->wgrad_simdgroups;
    metal_args->wgrad_row_block = args->wgrad_row_block;
}

static MTLSize gd_rms_norm_row_threads(const gd_backend_rms_norm_args *args)
{
    return MTLSizeMake(32U, (NSUInteger)args->simdgroups, 1U);
}

static MTLSize gd_rms_norm_wgrad_reduce_threads(const gd_backend_rms_norm_args *args)
{
    return MTLSizeMake(32U, (NSUInteger)args->wgrad_simdgroups, 1U);
}

static gd_status gd_rms_norm_forward_dispatch(gd_backend *backend,
                                              const gd_backend_tensor_view *x,
                                              const gd_backend_tensor_view *weight,
                                              const gd_backend_tensor_view *out,
                                              const gd_backend_tensor_view *inv_rms,
                                              const gd_backend_rms_norm_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_rms_norm_args metal_args;
    bool immediate;
    bool stats = inv_rms != NULL;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_args(x, weight, out, args);
    if (st != GD_OK) {
        return st;
    }
    if (stats) {
        st = gd_rms_norm_validate_inv(x, inv_rms, args);
        if (st != GD_OK) {
            return st;
        }
    }
    pso = gd_rms_norm_forward_pso(backend, x->dtype, stats);
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
    gd_rms_norm_fill_args(&metal_args, x, weight, out, NULL, inv_rms, NULL, args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_rms_norm_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rms_norm_buffer(weight->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_rms_norm_buffer(out->buffer) offset:0U atIndex:2U];
    if (stats) {
        [encoder setBuffer:gd_rms_norm_buffer(inv_rms->buffer) offset:0U atIndex:3U];
        [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:4U];
    } else {
        [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:3U];
    }
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)args->rows, 1U, 1U)
            threadsPerThreadgroup:gd_rms_norm_row_threads(args)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_rms_norm_forward(gd_backend *backend,
                                      const gd_backend_tensor_view *x,
                                      const gd_backend_tensor_view *weight,
                                      const gd_backend_tensor_view *out,
                                      const gd_backend_rms_norm_args *args)
{
    return gd_rms_norm_forward_dispatch(backend, x, weight, out, NULL, args);
}

gd_status gd_backend_rms_norm_forward_stats(gd_backend *backend,
                                            const gd_backend_tensor_view *x,
                                            const gd_backend_tensor_view *weight,
                                            const gd_backend_tensor_view *out,
                                            const gd_backend_tensor_view *inv_rms,
                                            const gd_backend_rms_norm_args *args)
{
    return gd_rms_norm_forward_dispatch(backend, x, weight, out, inv_rms, args);
}

gd_status gd_backend_rms_norm_inv(gd_backend *backend,
                                  const gd_backend_tensor_view *x,
                                  const gd_backend_tensor_view *inv_rms,
                                  const gd_backend_rms_norm_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_rms_norm_args metal_args;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x == NULL || args == NULL || args->rows == 0U || args->cols == 0U ||
        args->rows > UINT64_MAX / args->cols || !(args->eps > 0.0f) ||
        !(args->eps == args->eps) || !gd_rms_norm_view_range_valid(x) ||
        !gd_rms_norm_contiguous_view(x) || gd_rms_norm_dtype_size(x->dtype) == 0U ||
        !gd_rms_norm_count_matches(x, args->rows * args->cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_inv(x, inv_rms, args);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_rms_norm_inv_pso(backend, x->dtype);
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
    gd_rms_norm_fill_args(&metal_args, x, NULL, NULL, NULL, inv_rms, NULL, args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_rms_norm_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rms_norm_buffer(inv_rms->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:2U];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)args->rows, 1U, 1U)
            threadsPerThreadgroup:gd_rms_norm_row_threads(args)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static gd_status gd_rms_norm_backward_dispatch(gd_backend *backend,
                                               const gd_backend_tensor_view *x,
                                               const gd_backend_tensor_view *weight,
                                               const gd_backend_tensor_view *inv_rms,
                                               const gd_backend_tensor_view *grad_out,
                                               const gd_backend_tensor_view *grad_x,
                                               const gd_backend_rms_norm_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_rms_norm_args metal_args;
    bool immediate;
    bool stats = inv_rms != NULL;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_args(x, weight, grad_x, args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_grad_out(x, grad_out, args);
    if (st != GD_OK) {
        return st;
    }
    if (stats) {
        st = gd_rms_norm_validate_inv(x, inv_rms, args);
        if (st != GD_OK) {
            return st;
        }
    }
    pso = gd_rms_norm_backward_pso(backend, x->dtype, stats);
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
    gd_rms_norm_fill_args(&metal_args, x, weight, grad_x, grad_out, inv_rms, NULL, args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_rms_norm_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rms_norm_buffer(weight->buffer) offset:0U atIndex:1U];
    if (stats) {
        [encoder setBuffer:gd_rms_norm_buffer(inv_rms->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_rms_norm_buffer(grad_out->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_rms_norm_buffer(grad_x->buffer) offset:0U atIndex:4U];
        [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:5U];
    } else {
        [encoder setBuffer:gd_rms_norm_buffer(grad_out->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_rms_norm_buffer(grad_x->buffer) offset:0U atIndex:3U];
        [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:4U];
    }
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)args->rows, 1U, 1U)
            threadsPerThreadgroup:gd_rms_norm_row_threads(args)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_rms_norm_backward(gd_backend *backend,
                                       const gd_backend_tensor_view *x,
                                       const gd_backend_tensor_view *weight,
                                       const gd_backend_tensor_view *grad_out,
                                       const gd_backend_tensor_view *grad_x,
                                       const gd_backend_rms_norm_args *args)
{
    return gd_rms_norm_backward_dispatch(backend, x, weight, NULL, grad_out, grad_x, args);
}

gd_status gd_backend_rms_norm_backward_stats(gd_backend *backend,
                                             const gd_backend_tensor_view *x,
                                             const gd_backend_tensor_view *weight,
                                             const gd_backend_tensor_view *inv_rms,
                                             const gd_backend_tensor_view *grad_out,
                                             const gd_backend_tensor_view *grad_x,
                                             const gd_backend_rms_norm_args *args)
{
    return gd_rms_norm_backward_dispatch(backend, x, weight, inv_rms, grad_out, grad_x, args);
}

gd_status gd_backend_rms_norm_weight_backward_stats(gd_backend *backend,
                                                    const gd_backend_tensor_view *x,
                                                    const gd_backend_tensor_view *inv_rms,
                                                    const gd_backend_tensor_view *grad_out,
                                                    const gd_backend_tensor_view *grad_weight,
                                                    const gd_backend_tensor_view *partial,
                                                    const gd_backend_rms_norm_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> stage_pso;
    id<MTLComputePipelineState> reduce_pso;
    gd_metal_rms_norm_args metal_args;
    NSUInteger channel_blocks;
    bool immediate;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x == NULL || grad_weight == NULL || args == NULL || args->rows == 0U ||
        args->cols == 0U || args->rows > UINT64_MAX / args->cols ||
        args->wgrad_row_block == 0U || args->wgrad_row_block > GD_METAL_RMS_NORM_WGRAD_ROW_BLOCK ||
        args->wgrad_simdgroups == 0U || args->wgrad_simdgroups > GD_METAL_RMS_NORM_MAX_SIMDGROUPS ||
        !gd_rms_norm_view_range_valid(x) || !gd_rms_norm_contiguous_view(x) ||
        gd_rms_norm_dtype_size(x->dtype) == 0U ||
        !gd_rms_norm_count_matches(x, args->rows * args->cols) ||
        !gd_rms_norm_view_range_valid(grad_weight) ||
        !gd_rms_norm_contiguous_view(grad_weight) || grad_weight->dtype != x->dtype ||
        grad_weight->rank != 1U || grad_weight->shape[0] <= 0 ||
        (uint64_t)grad_weight->shape[0] != args->cols ||
        !gd_rms_norm_count_matches(grad_weight, args->cols)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_rms_norm_validate_inv(x, inv_rms, args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_grad_out(x, grad_out, args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rms_norm_validate_partial(partial, args);
    if (st != GD_OK) {
        return st;
    }
    stage_pso = gd_rms_norm_wgrad_stage_pso(backend, x->dtype);
    reduce_pso = gd_rms_norm_wgrad_reduce_pso(backend, x->dtype);
    if (stage_pso == nil || reduce_pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    channel_blocks = (NSUInteger)((args->cols + (uint64_t)GD_METAL_RMS_NORM_WGRAD_CHANNELS - 1U) /
                                  (uint64_t)GD_METAL_RMS_NORM_WGRAD_CHANNELS);
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    gd_rms_norm_fill_args(&metal_args, x, NULL, grad_weight, grad_out, inv_rms, partial, args);
    [encoder setComputePipelineState:stage_pso];
    [encoder setBuffer:gd_rms_norm_buffer(x->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rms_norm_buffer(inv_rms->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_rms_norm_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_rms_norm_buffer(partial->buffer) offset:0U atIndex:3U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:4U];
    [encoder dispatchThreadgroups:MTLSizeMake(channel_blocks, (NSUInteger)args->row_blocks, 1U)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_NORM_WGRAD_CHANNELS, 1U, 1U)];
    [encoder setComputePipelineState:reduce_pso];
    [encoder setBuffer:gd_rms_norm_buffer(partial->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_rms_norm_buffer(grad_weight->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:2U];
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)args->cols, 1U, 1U)
            threadsPerThreadgroup:gd_rms_norm_wgrad_reduce_threads(args)];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
