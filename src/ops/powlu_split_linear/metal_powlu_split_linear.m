#include "../../backends/metal/metal_backend_internal.h"
#include "../_shared/gemm/metal_gemm_types.h"
#include "metal_powlu_split_linear_types.h"

#include <gradients/tensor.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLComputePipelineState> gd_powlu_split_linear_backward_x12_pso(gd_backend *backend)
{
    return backend != NULL ? (__bridge id<MTLComputePipelineState>)backend->powlu_split_linear_backward_x12_f16_reg_pso : nil;
}

static id<MTLBuffer> gd_powlu_split_linear_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static NSUInteger gd_powlu_split_linear_div_up_u32(uint32_t value, uint32_t denom)
{
    return (NSUInteger)(value / denom + (value % denom != 0U ? 1U : 0U));
}

static bool gd_powlu_split_linear_byte_range_valid(const gd_backend_buffer *buffer,
                                                   size_t offset,
                                                   size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_powlu_split_linear_tensor_range_ok(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->dtype != (uint32_t)GD_DTYPE_F16 ||
        view->count == 0U || view->count > (SIZE_MAX / sizeof(uint16_t))) {
        return false;
    }
    nbytes = view->count * sizeof(uint16_t);
    return gd_powlu_split_linear_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_powlu_split_linear_matrix_range_ok(const gd_backend_matrix_view *view)
{
    size_t last_row;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->dtype != (uint32_t)GD_DTYPE_F16 ||
        view->rows == 0U || view->cols == 0U ||
        view->row_bytes < (size_t)view->cols * sizeof(uint16_t) ||
        (size_t)(view->rows - 1U) > SIZE_MAX / view->row_bytes ||
        (size_t)view->cols > SIZE_MAX / sizeof(uint16_t)) {
        return false;
    }
    last_row = (size_t)(view->rows - 1U) * view->row_bytes;
    nbytes = last_row + (size_t)view->cols * sizeof(uint16_t);
    return gd_powlu_split_linear_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_powlu_split_linear_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    int32_t axis;
    if (view == NULL || view->rank == 0U || view->rank > GD_MAX_DIMS) {
        return false;
    }
    for (axis = (int32_t)view->rank - 1; axis >= 0; --axis) {
        if (view->shape[axis] <= 0 || view->strides[axis] != expected ||
            expected > INT64_MAX / view->shape[axis]) {
            return false;
        }
        expected *= view->shape[axis];
    }
    return view->count == (size_t)expected;
}

static bool gd_powlu_split_linear_same_shape(const gd_backend_tensor_view *x,
                                             const gd_backend_tensor_view *y)
{
    uint32_t axis;
    if (x == NULL || y == NULL || x->rank != y->rank || x->count != y->count) {
        return false;
    }
    for (axis = 0U; axis < x->rank; ++axis) {
        if (x->shape[axis] != y->shape[axis]) {
            return false;
        }
    }
    return true;
}

static bool gd_powlu_split_linear_rows_hidden(const gd_backend_tensor_view *x12,
                                              uint32_t *out_rows,
                                              uint32_t *out_hidden)
{
    uint32_t axis;
    uint64_t rows = 1U;
    int64_t last;
    if (x12 == NULL || out_rows == NULL || out_hidden == NULL || x12->rank == 0U) {
        return false;
    }
    last = x12->shape[x12->rank - 1U];
    if (last <= 0 || (last & 1) != 0 || (uint64_t)(last / 2) > (uint64_t)UINT32_MAX) {
        return false;
    }
    for (axis = 0U; axis + 1U < x12->rank; ++axis) {
        if (x12->shape[axis] <= 0 || rows > (uint64_t)UINT32_MAX / (uint64_t)x12->shape[axis]) {
            return false;
        }
        rows *= (uint64_t)x12->shape[axis];
    }
    if (rows == 0U || rows > (uint64_t)UINT32_MAX) {
        return false;
    }
    *out_rows = (uint32_t)rows;
    *out_hidden = (uint32_t)(last / 2);
    return true;
}

static bool gd_powlu_split_linear_disable_fused_x12(void)
{
    const char *env = getenv("GD_POWLU_SPLIT_LINEAR_DISABLE_FUSED_X12");
    return env != NULL && env[0] != '\0' && env[0] != '0';
}

static gd_status gd_powlu_split_linear_backward_x12_validate(const gd_backend_tensor_view *x12,
                                                             const gd_backend_matrix_view *w,
                                                             const gd_backend_matrix_view *grad_out,
                                                             const gd_backend_tensor_view *grad_x12,
                                                             float m,
                                                             uint32_t *out_rows,
                                                             uint32_t *out_hidden)
{
    uint32_t rows;
    uint32_t hidden;
    if (gd_powlu_split_linear_disable_fused_x12()) {
        return GD_ERR_UNSUPPORTED;
    }
    if (!(m == m) || m <= 0.0f || m >= 10.0f ||
        !gd_powlu_split_linear_tensor_range_ok(x12) ||
        !gd_powlu_split_linear_tensor_range_ok(grad_x12) ||
        !gd_powlu_split_linear_matrix_range_ok(w) ||
        !gd_powlu_split_linear_matrix_range_ok(grad_out) ||
        !gd_powlu_split_linear_contiguous_view(x12) ||
        !gd_powlu_split_linear_contiguous_view(grad_x12) ||
        !gd_powlu_split_linear_same_shape(x12, grad_x12) ||
        !gd_powlu_split_linear_rows_hidden(x12, &rows, &hidden)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (w->rows != hidden || w->cols != grad_out->cols || grad_out->rows != rows) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if ((x12->offset % 16U) != 0U || (w->offset % 16U) != 0U ||
        (grad_out->offset % 16U) != 0U || (grad_x12->offset % 16U) != 0U ||
        ((size_t)x12->shape[x12->rank - 1U] * sizeof(uint16_t)) % 16U != 0U ||
        (w->row_bytes % 16U) != 0U || (grad_out->row_bytes % 16U) != 0U ||
        ((size_t)grad_x12->shape[grad_x12->rank - 1U] * sizeof(uint16_t)) % 16U != 0U ||
        (rows % GD_METAL_GEMM_REG_TILE) != 0U ||
        (hidden % GD_METAL_GEMM_REG_TILE) != 0U ||
        (grad_out->cols % 8U) != 0U) {
        return GD_ERR_UNSUPPORTED;
    }
    *out_rows = rows;
    *out_hidden = hidden;
    return GD_OK;
}

gd_status gd_backend_powlu_split_linear_backward_x12(gd_backend *backend,
                                                     const gd_backend_tensor_view *x12,
                                                     const gd_backend_matrix_view *w,
                                                     const gd_backend_matrix_view *grad_out,
                                                     const gd_backend_tensor_view *grad_x12,
                                                     float m)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_powlu_split_linear_bwd_x12_args args;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    uint32_t rows;
    uint32_t hidden;
    gd_status st;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_powlu_split_linear_backward_x12_validate(x12,
                                                     w,
                                                     grad_out,
                                                     grad_x12,
                                                     m,
                                                     &rows,
                                                     &hidden);
    if (st != GD_OK) {
        return st;
    }
    pso = gd_powlu_split_linear_backward_x12_pso(backend);
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
    memset(&args, 0, sizeof(args));
    args.x12_offset = (uint64_t)x12->offset;
    args.w_offset = (uint64_t)w->offset;
    args.grad_offset = (uint64_t)grad_out->offset;
    args.dx12_offset = (uint64_t)grad_x12->offset;
    args.x12_row_bytes = (uint64_t)((size_t)x12->shape[x12->rank - 1U] * sizeof(uint16_t));
    args.w_row_bytes = (uint64_t)w->row_bytes;
    args.grad_row_bytes = (uint64_t)grad_out->row_bytes;
    args.dx12_row_bytes = (uint64_t)((size_t)grad_x12->shape[grad_x12->rank - 1U] * sizeof(uint16_t));
    args.rows = rows;
    args.hidden = hidden;
    args.out_cols = grad_out->cols;
    args.m = m;
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_powlu_split_linear_buffer(x12->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_powlu_split_linear_buffer(w->buffer) offset:0U atIndex:1U];
    [encoder setBuffer:gd_powlu_split_linear_buffer(grad_out->buffer) offset:0U atIndex:2U];
    [encoder setBuffer:gd_powlu_split_linear_buffer(grad_x12->buffer) offset:0U atIndex:3U];
    [encoder setBytes:&args length:sizeof(args) atIndex:4U];
    groups = MTLSizeMake(gd_powlu_split_linear_div_up_u32(hidden,
                                                          GD_METAL_GEMM_REG_TILE * GD_METAL_GEMM_REG_SIMDGROUPS),
                         (NSUInteger)(rows / GD_METAL_GEMM_REG_TILE),
                         1U);
    threads = MTLSizeMake(32U * GD_METAL_GEMM_REG_SIMDGROUPS, 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
