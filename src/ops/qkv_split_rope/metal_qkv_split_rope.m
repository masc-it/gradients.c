#include "../../backends/metal/metal_backend_internal.h"
#include "metal_qkv_split_rope_types.h"

#include <gradients/tensor.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLComputePipelineState> gd_qkv_split_rope_forward_pso(gd_backend *backend)
{
    return backend != NULL ? (__bridge id<MTLComputePipelineState>)backend->qkv_split_rope_forward_f16_pso : nil;
}

static id<MTLComputePipelineState> gd_qkv_split_rope_backward_pso(gd_backend *backend)
{
    return backend != NULL ? (__bridge id<MTLComputePipelineState>)backend->qkv_split_rope_backward_f16_pso : nil;
}

static id<MTLComputePipelineState> gd_qkv_split_rope_forward_table_pso(gd_backend *backend)
{
    return backend != NULL ?
               (__bridge id<MTLComputePipelineState>)backend->qkv_split_rope_forward_table_f16_pso :
               nil;
}

static id<MTLComputePipelineState> gd_qkv_split_rope_backward_table_pso(gd_backend *backend)
{
    return backend != NULL ?
               (__bridge id<MTLComputePipelineState>)backend->qkv_split_rope_backward_table_f16_pso :
               nil;
}

static bool gd_qkv_split_rope_table_disabled(void)
{
    const char *env = getenv("GD_QKV_SPLIT_ROPE_TABLE");
    return env != NULL && env[0] == '0' && env[1] == '\0';
}

static bool gd_qkv_split_rope_table_config_ok(uint32_t head_dim, const gd_backend_rope_args *rope_args)
{
    return !gd_qkv_split_rope_table_disabled() && rope_args != NULL &&
           head_dim == GD_METAL_QKV_SPLIT_ROPE_TABLE_HEAD_DIM &&
           rope_args->n_dims == GD_METAL_QKV_SPLIT_ROPE_TABLE_HEAD_DIM &&
           rope_args->interleaved == 0U &&
           fabsf(rope_args->theta - GD_METAL_QKV_SPLIT_ROPE_TABLE_THETA) <= 1.0e-3f;
}

static void gd_qkv_split_rope_fill_table(float *table)
{
    const float freq_scale = -2.0f * log2f(GD_METAL_QKV_SPLIT_ROPE_TABLE_THETA) /
                             (float)GD_METAL_QKV_SPLIT_ROPE_TABLE_HEAD_DIM;
    uint32_t pos;
    if (table == NULL) {
        return;
    }
    for (pos = 0U; pos < GD_METAL_QKV_SPLIT_ROPE_TABLE_POSITIONS; ++pos) {
        uint32_t pair;
        for (pair = 0U; pair < GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS; ++pair) {
            const float angle = (float)pos * exp2f((float)pair * freq_scale);
            const size_t idx = ((size_t)pos * GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS + (size_t)pair) * 2U;
            table[idx] = cosf(angle);
            table[idx + 1U] = sinf(angle);
        }
    }
}

static id<MTLBuffer> gd_qkv_split_rope_table_buffer(gd_backend *backend)
{
    id<MTLDevice> device;
    id<MTLBuffer> table_buffer;
    const size_t n_floats = (size_t)GD_METAL_QKV_SPLIT_ROPE_TABLE_POSITIONS *
                            (size_t)GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS * 2U;
    const size_t nbytes = n_floats * sizeof(float);
    if (backend == NULL) {
        return nil;
    }
    table_buffer = (__bridge id<MTLBuffer>)backend->qkv_split_rope_table_f32_buffer;
    if (table_buffer != nil) {
        return table_buffer;
    }
    device = (__bridge id<MTLDevice>)backend->device;
    if (device == nil) {
        return nil;
    }
    table_buffer = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    if (table_buffer == nil || [table_buffer contents] == NULL) {
        return nil;
    }
    gd_qkv_split_rope_fill_table((float *)[table_buffer contents]);
    backend->qkv_split_rope_table_f32_buffer = (void *)CFBridgingRetain(table_buffer);
    return (__bridge id<MTLBuffer>)backend->qkv_split_rope_table_f32_buffer;
}

static id<MTLBuffer> gd_qkv_split_rope_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static bool gd_qkv_split_rope_byte_range_valid(const gd_backend_buffer *buffer,
                                                size_t offset,
                                                size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_qkv_split_rope_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

static bool gd_qkv_split_rope_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->count == 0U) {
        return false;
    }
    if (view->dtype == (uint32_t)GD_DTYPE_F16) {
        elem_size = 2U;
    } else if (view->dtype == (uint32_t)GD_DTYPE_I32) {
        elem_size = 4U;
    } else {
        return false;
    }
    if (!gd_qkv_split_rope_count_bytes(view->count, elem_size, &nbytes)) {
        return false;
    }
    return gd_qkv_split_rope_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_qkv_split_rope_contiguous_valid(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    uint32_t axis;
    if (view == NULL || view->rank > GD_MAX_DIMS) {
        return false;
    }
    axis = view->rank;
    while (axis > 0U) {
        uint32_t dim;
        axis -= 1U;
        dim = axis;
        if (view->shape[dim] <= 0 || view->strides[dim] != expected ||
            expected > INT64_MAX / view->shape[dim]) {
            return false;
        }
        expected *= view->shape[dim];
    }
    return view->count == (size_t)expected;
}

static bool gd_qkv_split_rope_output_shape_valid(const gd_backend_tensor_view *out,
                                                  uint64_t tokens,
                                                  uint32_t heads,
                                                  uint32_t head_dim)
{
    return out != NULL && out->dtype == (uint32_t)GD_DTYPE_F16 && out->rank == 3U &&
           out->shape[0] == (int64_t)tokens && out->shape[1] == (int64_t)heads &&
           out->shape[2] == (int64_t)head_dim && gd_qkv_split_rope_contiguous_valid(out) &&
           out->count == (size_t)(tokens * (uint64_t)heads * (uint64_t)head_dim);
}

static bool gd_qkv_split_rope_make_args(const gd_backend_tensor_view *qkv,
                                         const gd_backend_tensor_view *pos_ids,
                                         const gd_backend_tensor_view *q,
                                         const gd_backend_tensor_view *k,
                                         const gd_backend_tensor_view *v,
                                         uint32_t n_heads,
                                         uint32_t head_dim,
                                         const gd_backend_rope_args *rope_args,
                                         float sin_sign,
                                         gd_metal_qkv_split_rope_args *metal_args)
{
    uint64_t tokens;
    uint64_t head_area;
    uint64_t qkv_count;
    uint64_t grid;
    if (!gd_qkv_split_rope_view_range_valid(qkv) || !gd_qkv_split_rope_view_range_valid(pos_ids) ||
        !gd_qkv_split_rope_view_range_valid(q) || !gd_qkv_split_rope_view_range_valid(k) ||
        !gd_qkv_split_rope_view_range_valid(v) || !gd_qkv_split_rope_contiguous_valid(qkv) ||
        !gd_qkv_split_rope_contiguous_valid(pos_ids) || rope_args == NULL || metal_args == NULL ||
        qkv->dtype != (uint32_t)GD_DTYPE_F16 || pos_ids->dtype != (uint32_t)GD_DTYPE_I32 ||
        qkv->rank != 2U || qkv->shape[0] <= 0 || qkv->shape[1] <= 0 || n_heads == 0U ||
        head_dim == 0U || rope_args->n_dims != head_dim || rope_args->interleaved > 1U ||
        !isfinite(rope_args->theta) || rope_args->theta <= 0.0f) {
        return false;
    }
    tokens = (uint64_t)qkv->shape[0];
    if (tokens == 0U || tokens > (uint64_t)UINT32_MAX || pos_ids->count != (size_t)tokens) {
        return false;
    }
    if ((uint64_t)n_heads > UINT64_MAX / (uint64_t)head_dim) {
        return false;
    }
    head_area = (uint64_t)n_heads * (uint64_t)head_dim;
    if (head_area == 0U || head_area > UINT64_MAX / 3U || (uint64_t)qkv->shape[1] != head_area * 3U ||
        tokens > UINT64_MAX / (head_area * 3U)) {
        return false;
    }
    qkv_count = tokens * head_area * 3U;
    if (qkv->count != (size_t)qkv_count ||
        !gd_qkv_split_rope_output_shape_valid(q, tokens, n_heads, head_dim) ||
        !gd_qkv_split_rope_output_shape_valid(k, tokens, n_heads, head_dim) ||
        !gd_qkv_split_rope_output_shape_valid(v, tokens, n_heads, head_dim)) {
        return false;
    }
    grid = tokens * ((uint64_t)head_dim / 2U);
    if ((head_dim & 1U) != 0U || grid == 0U || grid > (uint64_t)UINT32_MAX) {
        return false;
    }
    memset(metal_args, 0, sizeof(*metal_args));
    metal_args->qkv_offset = (uint64_t)qkv->offset;
    metal_args->pos_offset = (uint64_t)pos_ids->offset;
    metal_args->q_offset = (uint64_t)q->offset;
    metal_args->k_offset = (uint64_t)k->offset;
    metal_args->v_offset = (uint64_t)v->offset;
    metal_args->tokens = tokens;
    metal_args->heads = n_heads;
    metal_args->head_dim = head_dim;
    metal_args->n_dims = rope_args->n_dims;
    metal_args->interleaved = rope_args->interleaved;
    metal_args->freq_scale = -2.0f * log2f(rope_args->theta) / (float)rope_args->n_dims;
    metal_args->sin_sign = sin_sign;
    return true;
}

static gd_status gd_qkv_split_rope_encode_forward(gd_backend *backend,
                                                   const gd_backend_tensor_view *qkv,
                                                   const gd_backend_tensor_view *pos_ids,
                                                   const gd_backend_tensor_view *q,
                                                   const gd_backend_tensor_view *k,
                                                   const gd_backend_tensor_view *v,
                                                   uint32_t n_heads,
                                                   uint32_t head_dim,
                                                   const gd_backend_rope_args *rope_args,
                                                   float sin_sign,
                                                   bool backward)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    id<MTLBuffer> rope_table;
    gd_metal_qkv_split_rope_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    NSUInteger thread_count;
    gd_status st;
    if (backend == NULL || !gd_qkv_split_rope_make_args(qkv,
                                                        pos_ids,
                                                        q,
                                                        k,
                                                        v,
                                                        n_heads,
                                                        head_dim,
                                                        rope_args,
                                                        sin_sign,
                                                        &args)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    rope_table = nil;
    pso = backward ? gd_qkv_split_rope_backward_pso(backend) : gd_qkv_split_rope_forward_pso(backend);
    if (gd_qkv_split_rope_table_config_ok(head_dim, rope_args)) {
        id<MTLComputePipelineState> table_pso = backward ? gd_qkv_split_rope_backward_table_pso(backend) :
                                                           gd_qkv_split_rope_forward_table_pso(backend);
        id<MTLBuffer> table_buffer = gd_qkv_split_rope_table_buffer(backend);
        if (table_pso != nil && table_buffer != nil) {
            pso = table_pso;
            rope_table = table_buffer;
        }
    }
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
    thread_count = (NSUInteger)(args.tokens * (uint64_t)(args.head_dim / 2U));
    [encoder setComputePipelineState:pso];
    if (backward) {
        [encoder setBuffer:gd_qkv_split_rope_buffer(q->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(k->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(v->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(pos_ids->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(qkv->buffer) offset:0U atIndex:4U];
    } else {
        [encoder setBuffer:gd_qkv_split_rope_buffer(qkv->buffer) offset:0U atIndex:0U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(pos_ids->buffer) offset:0U atIndex:1U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(q->buffer) offset:0U atIndex:2U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(k->buffer) offset:0U atIndex:3U];
        [encoder setBuffer:gd_qkv_split_rope_buffer(v->buffer) offset:0U atIndex:4U];
    }
    [encoder setBytes:&args length:sizeof(args) atIndex:5U];
    if (rope_table != nil) {
        [encoder setBuffer:rope_table offset:0U atIndex:6U];
    }
    grid = MTLSizeMake(thread_count, 1U, 1U);
    threads = MTLSizeMake(thread_count < GD_METAL_QKV_SPLIT_ROPE_MAX_THREADS_PER_GROUP ?
                              thread_count : GD_METAL_QKV_SPLIT_ROPE_MAX_THREADS_PER_GROUP,
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
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
    return gd_qkv_split_rope_encode_forward(backend,
                                            qkv,
                                            pos_ids,
                                            q,
                                            k,
                                            v,
                                            n_heads,
                                            head_dim,
                                            args,
                                            1.0f,
                                            false);
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
    return gd_qkv_split_rope_encode_forward(backend,
                                            grad_qkv,
                                            pos_ids,
                                            grad_q,
                                            grad_k,
                                            grad_v,
                                            n_heads,
                                            head_dim,
                                            args,
                                            -1.0f,
                                            true);
}
