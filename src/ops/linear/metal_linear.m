#include "../../backends/metal/metal_backend_internal.h"

#include <stdint.h>

static bool gd_metal_matrix_bounds_ok(const gd_backend_matrix_view *view)
{
    size_t elem_size;
    size_t last_row;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->rows == 0U || view->cols == 0U) {
        return false;
    }
    elem_size = view->dtype == 1U ? 2U : 0U;
    if (elem_size == 0U || view->row_bytes < (size_t)view->cols * elem_size) {
        return false;
    }
    if ((size_t)(view->rows - 1U) > SIZE_MAX / view->row_bytes) {
        return false;
    }
    last_row = (size_t)(view->rows - 1U) * view->row_bytes;
    if ((size_t)view->cols > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = last_row + (size_t)view->cols * elem_size;
    return view->offset <= view->buffer->nbytes && nbytes <= view->buffer->nbytes - view->offset;
}

static bool gd_metal_vector_bounds_ok(const gd_backend_vector_view *view)
{
    size_t elem_size;
    size_t nbytes;
    if (view == NULL || view->buffer == NULL || view->length == 0U) {
        return false;
    }
    elem_size = view->dtype == 1U ? 2U : 0U;
    if (elem_size == 0U || (size_t)view->length > SIZE_MAX / elem_size) {
        return false;
    }
    nbytes = (size_t)view->length * elem_size;
    return view->offset <= view->buffer->nbytes && nbytes <= view->buffer->nbytes - view->offset;
}

static NSUInteger gd_metal_div_up_u32(uint32_t value, uint32_t denom)
{
    return (NSUInteger)(value / denom + (value % denom != 0U ? 1U : 0U));
}

static bool gd_metal_gemm_reg_ok(const gd_backend_matrix_view *x,
                                 const gd_backend_matrix_view *w,
                                 const gd_backend_vector_view *bias,
                                 const gd_backend_matrix_view *y)
{
    return x != NULL && w != NULL && y != NULL &&
           (x->offset % 16U) == 0U && (w->offset % 16U) == 0U &&
           (bias == NULL || (bias->offset % 2U) == 0U) && (y->offset % 16U) == 0U &&
           (x->row_bytes % 16U) == 0U && (w->row_bytes % 16U) == 0U &&
           (y->row_bytes % 16U) == 0U &&
           (y->rows % GD_METAL_GEMM_REG_TILE) == 0U &&
           (y->cols % GD_METAL_GEMM_REG_TILE) == 0U &&
           (x->cols % 8U) == 0U;
}

gd_status gd_backend_linear(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_vector_view *bias,
                            const gd_backend_matrix_view *y)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_gemm_args args;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != w->dtype || x->dtype != y->dtype ||
        (bias != NULL && bias->dtype != y->dtype) ||
        x->cols != w->rows || x->rows != y->rows || w->cols != y->cols ||
        (bias != NULL && bias->length != y->cols) || !gd_metal_matrix_bounds_ok(x) ||
        !gd_metal_matrix_bounds_ok(w) || !gd_metal_matrix_bounds_ok(y) ||
        (bias != NULL && !gd_metal_vector_bounds_ok(bias))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != 1U) {
        return GD_ERR_UNSUPPORTED;
    }
    if (gd_metal_gemm_reg_ok(x, w, bias, y) && backend->linear_reg_pso != NULL) {
        pso = (__bridge id<MTLComputePipelineState>)backend->linear_reg_pso;
    } else {
        pso = (__bridge id<MTLComputePipelineState>)backend->linear_pso;
    }
    if (pso == nil) {
        return GD_ERR_INTERNAL;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    args.x_offset = (uint64_t)x->offset;
    args.w_offset = (uint64_t)w->offset;
    args.bias_offset = bias != NULL ? (uint64_t)bias->offset : 0U;
    args.y_offset = (uint64_t)y->offset;
    args.x_row_bytes = (uint64_t)x->row_bytes;
    args.w_row_bytes = (uint64_t)w->row_bytes;
    args.y_row_bytes = (uint64_t)y->row_bytes;
    args.rows = y->rows;
    args.cols = y->cols;
    args.inner = x->cols;
    args.has_bias = bias != NULL ? 1U : 0U;
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x->buffer->buffer offset:0U atIndex:0U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)w->buffer->buffer offset:0U atIndex:1U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)(bias != NULL ? bias->buffer->buffer : x->buffer->buffer)
                offset:0U
               atIndex:2U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)y->buffer->buffer offset:0U atIndex:3U];
    [encoder setBytes:&args length:sizeof(args) atIndex:4U];
    if (pso == (__bridge id<MTLComputePipelineState>)backend->linear_reg_pso) {
        groups = MTLSizeMake(gd_metal_div_up_u32(y->cols,
                                                 GD_METAL_GEMM_REG_TILE * GD_METAL_GEMM_REG_SIMDGROUPS),
                             (NSUInteger)(y->rows / GD_METAL_GEMM_REG_TILE),
                             1U);
        threads = MTLSizeMake(32U * GD_METAL_GEMM_REG_SIMDGROUPS, 1U, 1U);
    } else {
        groups = MTLSizeMake(gd_metal_div_up_u32(y->cols, GD_METAL_GEMM_BN),
                             gd_metal_div_up_u32(y->rows, GD_METAL_GEMM_BM),
                             1U);
        threads = MTLSizeMake(GD_METAL_GEMM_BN / GD_METAL_GEMM_TN,
                              GD_METAL_GEMM_BM / GD_METAL_GEMM_TM,
                              1U);
    }
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_reduce_rows(gd_backend *backend,
                                 const gd_backend_matrix_view *x,
                                 const gd_backend_vector_view *y)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_reduce_rows_args args;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    gd_status st;
    if (backend == NULL || x == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != 1U || y->dtype != 1U) {
        return GD_ERR_UNSUPPORTED;
    }
    if (x->cols != y->length || !gd_metal_matrix_bounds_ok(x) || !gd_metal_vector_bounds_ok(y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->reduce_rows_pso == NULL) {
        return GD_ERR_INTERNAL;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.x_offset = (uint64_t)x->offset;
    args.y_offset = (uint64_t)y->offset;
    args.x_row_bytes = (uint64_t)x->row_bytes;
    args.rows = x->rows;
    args.cols = x->cols;
    args.pad0 = 0U;
    [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)backend->reduce_rows_pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x->buffer->buffer offset:0U atIndex:0U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)y->buffer->buffer offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    groups = MTLSizeMake(gd_metal_div_up_u32(y->length, GD_METAL_REDUCE_ROWS_SIMDGROUPS), 1U, 1U);
    threads = MTLSizeMake(32U * GD_METAL_REDUCE_ROWS_SIMDGROUPS, 1U, 1U);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
