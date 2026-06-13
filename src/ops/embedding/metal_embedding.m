#include "../../backends/metal/metal_backend_internal.h"
#include "metal_embedding_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <string.h>

static id<MTLComputePipelineState> gd_embedding_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_forward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_forward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_embedding_forward_vec16_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_forward_vec16_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_forward_vec16_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_embedding_scatter_pso(gd_backend *backend, uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_backward_scatter_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->embedding_backward_scatter_f32_pso;
    }
    return nil;
}

static id<MTLBuffer> gd_embedding_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static size_t gd_embedding_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
        return 4U;
    }
    return 0U;
}

static bool gd_embedding_byte_range_valid(const gd_backend_buffer *buffer,
                                          size_t offset,
                                          size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_embedding_count_bytes(uint64_t count, uint32_t dtype, size_t *out_nbytes)
{
    size_t elem_size;
    if (out_nbytes == NULL || count == 0U) {
        return false;
    }
    elem_size = gd_embedding_dtype_size(dtype);
    if (elem_size == 0U || count > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    *out_nbytes = (size_t)count * elem_size;
    return true;
}

static bool gd_embedding_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    return view != NULL && view->buffer != NULL && view->count != 0U &&
           gd_embedding_count_bytes((uint64_t)view->count, view->dtype, &nbytes) &&
           gd_embedding_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_embedding_contiguous_view(const gd_backend_tensor_view *view)
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
        if (stride > INT64_MAX / view->shape[dim]) {
            return false;
        }
        stride *= view->shape[dim];
    }
    return true;
}

static bool gd_embedding_count_matches(const gd_backend_tensor_view *view, uint64_t count)
{
    return view != NULL && view->count == (size_t)count && (uint64_t)view->count == count;
}

static gd_status gd_embedding_validate_common(const gd_backend_tensor_view *ids,
                                              const gd_backend_embedding_args *args)
{
    if (ids == NULL || args == NULL || args->ids_count == 0U || args->vocab == 0U ||
        args->dim == 0U || args->ids_count > (uint64_t)UINT32_MAX ||
        args->dim > (uint64_t)UINT32_MAX || args->ids_count > UINT64_MAX / args->dim ||
        args->vocab > UINT64_MAX / args->dim ||
        args->ids_count * args->dim > (uint64_t)UINT32_MAX ||
        args->vocab * args->dim > (uint64_t)UINT32_MAX ||
        !gd_embedding_view_range_valid(ids) || !gd_embedding_contiguous_view(ids) ||
        ids->dtype != (uint32_t)GD_DTYPE_I32 || !gd_embedding_count_matches(ids, args->ids_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_embedding_validate_forward(const gd_backend_tensor_view *table,
                                               const gd_backend_tensor_view *ids,
                                               const gd_backend_tensor_view *out,
                                               const gd_backend_embedding_args *args)
{
    gd_status st = gd_embedding_validate_common(ids, args);
    uint64_t table_count;
    uint64_t out_count;
    uint32_t i;
    if (st != GD_OK) {
        return st;
    }
    table_count = args->vocab * args->dim;
    out_count = args->ids_count * args->dim;
    if (!gd_embedding_view_range_valid(table) || !gd_embedding_view_range_valid(out) ||
        !gd_embedding_contiguous_view(table) || !gd_embedding_contiguous_view(out) ||
        table->rank != 2U || table->shape[0] <= 0 || table->shape[1] <= 0 ||
        (uint64_t)table->shape[0] != args->vocab ||
        (uint64_t)table->shape[1] != args->dim || out->rank != ids->rank + 1U ||
        out->dtype != table->dtype ||
        (table->dtype != (uint32_t)GD_DTYPE_F16 && table->dtype != (uint32_t)GD_DTYPE_F32) ||
        !gd_embedding_count_matches(table, table_count) ||
        !gd_embedding_count_matches(out, out_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < ids->rank; ++i) {
        if (out->shape[i] != ids->shape[i]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    if (out->shape[ids->rank] <= 0 || (uint64_t)out->shape[ids->rank] != args->dim) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_embedding_validate_backward(const gd_backend_tensor_view *grad_out,
                                                const gd_backend_tensor_view *ids,
                                                const gd_backend_tensor_view *grad_table,
                                                const gd_backend_tensor_view *scratch,
                                                const gd_backend_embedding_args *args)
{
    gd_status st = gd_embedding_validate_common(ids, args);
    uint64_t table_count;
    uint64_t out_count;
    uint32_t i;
    if (st != GD_OK) {
        return st;
    }
    table_count = args->vocab * args->dim;
    out_count = args->ids_count * args->dim;
    if (!gd_embedding_view_range_valid(grad_out) || !gd_embedding_view_range_valid(grad_table) ||
        !gd_embedding_contiguous_view(grad_out) || !gd_embedding_contiguous_view(grad_table) ||
        grad_table->rank != 2U || grad_table->shape[0] <= 0 || grad_table->shape[1] <= 0 ||
        (uint64_t)grad_table->shape[0] != args->vocab ||
        (uint64_t)grad_table->shape[1] != args->dim ||
        grad_out->rank != ids->rank + 1U || grad_out->dtype != grad_table->dtype ||
        (grad_table->dtype != (uint32_t)GD_DTYPE_F16 && grad_table->dtype != (uint32_t)GD_DTYPE_F32) ||
        !gd_embedding_count_matches(grad_table, table_count) ||
        !gd_embedding_count_matches(grad_out, out_count)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < ids->rank; ++i) {
        if (grad_out->shape[i] != ids->shape[i]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    if (grad_out->shape[ids->rank] <= 0 || (uint64_t)grad_out->shape[ids->rank] != args->dim) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_table->dtype == (uint32_t)GD_DTYPE_F16) {
        if (scratch == NULL || !gd_embedding_view_range_valid(scratch) ||
            !gd_embedding_contiguous_view(scratch) || scratch->dtype != (uint32_t)GD_DTYPE_F32 ||
            scratch->rank != 2U || scratch->shape[0] <= 0 || scratch->shape[1] <= 0 ||
            (uint64_t)scratch->shape[0] != args->vocab ||
            (uint64_t)scratch->shape[1] != args->dim ||
            !gd_embedding_count_matches(scratch, table_count)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    } else if (scratch != NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_embedding_fill_metal_args(gd_metal_embedding_args *metal_args,
                                         const gd_backend_tensor_view *table,
                                         const gd_backend_tensor_view *ids,
                                         const gd_backend_tensor_view *out,
                                         const gd_backend_tensor_view *grad_out,
                                         const gd_backend_tensor_view *grad_table,
                                         const gd_backend_tensor_view *scratch,
                                         const gd_backend_embedding_args *args)
{
    memset(metal_args, 0, sizeof(*metal_args));
    metal_args->table_offset = table != NULL ? (uint64_t)table->offset : 0U;
    metal_args->ids_offset = ids != NULL ? (uint64_t)ids->offset : 0U;
    metal_args->out_offset = out != NULL ? (uint64_t)out->offset : 0U;
    metal_args->grad_out_offset = grad_out != NULL ? (uint64_t)grad_out->offset : 0U;
    metal_args->grad_table_offset = grad_table != NULL ? (uint64_t)grad_table->offset : 0U;
    metal_args->scratch_offset = scratch != NULL ? (uint64_t)scratch->offset :
                                (grad_table != NULL ? (uint64_t)grad_table->offset : 0U);
    metal_args->ids_count = args->ids_count;
    metal_args->vocab = args->vocab;
    metal_args->dim = args->dim;
}

static MTLSize gd_embedding_threads_2d(id<MTLComputePipelineState> pso, uint64_t width)
{
    NSUInteger max_threads = [pso maxTotalThreadsPerThreadgroup];
    NSUInteger x = (NSUInteger)(width < (uint64_t)max_threads ? width : (uint64_t)max_threads);
    if (x > 256U) {
        x = 256U;
    }
    if (x == 0U) {
        x = 1U;
    }
    return MTLSizeMake(x, 1U, 1U);
}

static MTLSize gd_embedding_threads_1d(id<MTLComputePipelineState> pso, uint64_t count)
{
    NSUInteger max_threads = [pso maxTotalThreadsPerThreadgroup];
    NSUInteger x = (NSUInteger)(count < (uint64_t)max_threads ? count : (uint64_t)max_threads);
    if (x > 256U) {
        x = 256U;
    }
    if (x == 0U) {
        x = 1U;
    }
    return MTLSizeMake(x, 1U, 1U);
}

static bool gd_embedding_forward_can_vec16(const gd_backend_tensor_view *table,
                                           const gd_backend_tensor_view *out,
                                           const gd_backend_embedding_args *args)
{
    size_t elem_size;
    uint64_t row_bytes;
    if (table == NULL || out == NULL || args == NULL) {
        return false;
    }
    elem_size = gd_embedding_dtype_size(table->dtype);
    if (elem_size == 0U || args->dim > UINT64_MAX / elem_size) {
        return false;
    }
    row_bytes = args->dim * (uint64_t)elem_size;
    return row_bytes != 0U && (row_bytes & 15U) == 0U &&
           ((uint64_t)table->offset & 15U) == 0U && ((uint64_t)out->offset & 15U) == 0U;
}

gd_status gd_backend_embedding_forward(gd_backend *backend,
                                       const gd_backend_tensor_view *table,
                                       const gd_backend_tensor_view *ids,
                                       const gd_backend_tensor_view *out,
                                       const gd_backend_embedding_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_embedding_args metal_args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    bool vec16;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding_validate_forward(table, ids, out, args);
    if (st != GD_OK) {
        return st;
    }
    vec16 = gd_embedding_forward_can_vec16(table, out, args);
    pso = vec16 ? gd_embedding_forward_vec16_pso(backend, table->dtype)
                : gd_embedding_forward_pso(backend, table->dtype);
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
    gd_embedding_fill_metal_args(&metal_args, table, ids, out, NULL, NULL, NULL, args);
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_embedding_buffer(table->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_embedding_buffer(ids->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_embedding_buffer(out->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:3U];
    if (vec16) {
        size_t elem_size = gd_embedding_dtype_size(table->dtype);
        uint64_t vecs = (args->dim * (uint64_t)elem_size) >> 4;
        grid = MTLSizeMake((NSUInteger)vecs, (NSUInteger)args->ids_count, 1U);
        threads = gd_embedding_threads_2d(pso, vecs);
    } else {
        grid = MTLSizeMake((NSUInteger)args->dim, (NSUInteger)args->ids_count, 1U);
        threads = gd_embedding_threads_2d(pso, args->dim);
    }
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_embedding_backward(gd_backend *backend,
                                        const gd_backend_tensor_view *grad_out,
                                        const gd_backend_tensor_view *ids,
                                        const gd_backend_tensor_view *grad_table,
                                        const gd_backend_tensor_view *scratch,
                                        const gd_backend_embedding_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> zero_pso;
    id<MTLComputePipelineState> scatter_pso;
    id<MTLComputePipelineState> cast_pso;
    const gd_backend_tensor_view *target;
    gd_metal_embedding_args metal_args;
    MTLSize grid;
    bool immediate;
    bool use_scratch;
    uint64_t table_count;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding_validate_backward(grad_out, ids, grad_table, scratch, args);
    if (st != GD_OK) {
        return st;
    }
    use_scratch = grad_table->dtype == (uint32_t)GD_DTYPE_F16;
    target = use_scratch ? scratch : grad_table;
    table_count = args->vocab * args->dim;
    zero_pso = (__bridge id<MTLComputePipelineState>)backend->embedding_zero_f32_pso;
    scatter_pso = gd_embedding_scatter_pso(backend, grad_out->dtype);
    cast_pso = (__bridge id<MTLComputePipelineState>)backend->embedding_cast_f32_to_f16_pso;
    if (zero_pso == nil || scatter_pso == nil || (use_scratch && cast_pso == nil)) {
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
    gd_embedding_fill_metal_args(&metal_args, NULL, ids, NULL, grad_out, grad_table, target, args);

    [encoder setComputePipelineState:zero_pso];
    [encoder setBuffer:gd_embedding_buffer(target->buffer) offset:0U atIndex:0U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:1U];
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)table_count, 1U, 1U)
       threadsPerThreadgroup:gd_embedding_threads_1d(zero_pso, table_count)];

    [encoder setComputePipelineState:scatter_pso];
    [encoder setBuffer:gd_embedding_buffer(grad_out->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_embedding_buffer(ids->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_embedding_buffer(target->buffer) offset:0U atIndex:2U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:3U];
    grid = MTLSizeMake((NSUInteger)args->dim, (NSUInteger)args->ids_count, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:gd_embedding_threads_2d(scatter_pso, args->dim)];

    if (use_scratch) {
        [encoder setComputePipelineState:cast_pso];
        [encoder setBuffer:gd_embedding_buffer(scratch->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_embedding_buffer(grad_table->buffer) offset:0U atIndex:1U];
        [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:2U];
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)table_count, 1U, 1U)
           threadsPerThreadgroup:gd_embedding_threads_1d(cast_pso, table_count)];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
