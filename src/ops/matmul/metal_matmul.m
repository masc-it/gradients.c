#include "../../backends/metal/metal_backend_internal.h"
#include "metal_matmul_types.h"

#include <stdint.h>

typedef enum gd_metal_matmul_layout {
    GD_METAL_MATMUL_NN = 0,
    GD_METAL_MATMUL_NT = 1,
    GD_METAL_MATMUL_TN = 2,
} gd_metal_matmul_layout;

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

static NSUInteger gd_metal_div_up_u32(uint32_t value, uint32_t denom)
{
    return (NSUInteger)(value / denom + (value % denom != 0U ? 1U : 0U));
}

static bool gd_metal_gemm_reg_ok(const gd_backend_matrix_view *x,
                                 const gd_backend_matrix_view *w,
                                 const gd_backend_matrix_view *y,
                                 uint32_t inner)
{
    return x != NULL && w != NULL && y != NULL &&
           (x->offset % 16U) == 0U && (w->offset % 16U) == 0U && (y->offset % 16U) == 0U &&
           (x->row_bytes % 16U) == 0U && (w->row_bytes % 16U) == 0U && (y->row_bytes % 16U) == 0U &&
           (y->rows % GD_METAL_GEMM_REG_TILE) == 0U &&
           (y->cols % GD_METAL_GEMM_REG_TILE) == 0U &&
           (inner % 8U) == 0U;
}

static bool gd_metal_matmul_layout_ok(const gd_backend_matrix_view *x,
                                      const gd_backend_matrix_view *w,
                                      const gd_backend_matrix_view *y,
                                      gd_metal_matmul_layout layout,
                                      uint32_t *out_inner)
{
    if (x == NULL || w == NULL || y == NULL || out_inner == NULL ||
        x->dtype != w->dtype || x->dtype != y->dtype ||
        !gd_metal_matrix_bounds_ok(x) || !gd_metal_matrix_bounds_ok(w) ||
        !gd_metal_matrix_bounds_ok(y)) {
        return false;
    }
    switch (layout) {
    case GD_METAL_MATMUL_NN:
        if (x->cols != w->rows || x->rows != y->rows || w->cols != y->cols) {
            return false;
        }
        *out_inner = x->cols;
        return true;
    case GD_METAL_MATMUL_NT:
        if (x->cols != w->cols || x->rows != y->rows || w->rows != y->cols) {
            return false;
        }
        *out_inner = x->cols;
        return true;
    case GD_METAL_MATMUL_TN:
        if (x->rows != w->rows || x->cols != y->rows || w->cols != y->cols) {
            return false;
        }
        *out_inner = x->rows;
        return true;
    default:
        return false;
    }
}

static gd_status gd_backend_matmul_dispatch(gd_backend *backend,
                                            const gd_backend_matrix_view *x,
                                            const gd_backend_matrix_view *w,
                                            const gd_backend_matrix_view *y,
                                            gd_metal_matmul_layout layout)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_gemm_args args;
    MTLSize groups;
    MTLSize threads;
    bool immediate;
    bool use_reg;
    uint32_t inner;
    gd_status st;
    void *tiled_pso;
    void *reg_pso;
    if (backend == NULL || x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_metal_matmul_layout_ok(x, w, y, layout, &inner)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != 1U) {
        return GD_ERR_UNSUPPORTED;
    }
    switch (layout) {
    case GD_METAL_MATMUL_NN:
        tiled_pso = backend->matmul_pso;
        reg_pso = backend->matmul_reg_pso;
        break;
    case GD_METAL_MATMUL_NT:
        tiled_pso = backend->matmul_nt_pso;
        reg_pso = backend->matmul_nt_reg_pso;
        break;
    case GD_METAL_MATMUL_TN:
        tiled_pso = backend->matmul_tn_pso;
        reg_pso = backend->matmul_tn_reg_pso;
        break;
    default:
        return GD_ERR_INVALID_ARGUMENT;
    }
    use_reg = gd_metal_gemm_reg_ok(x, w, y, inner) && reg_pso != NULL;
    pso = (__bridge id<MTLComputePipelineState>)(use_reg ? reg_pso : tiled_pso);
    if (pso == nil) {
        return GD_ERR_INTERNAL;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    args.x_offset = (uint64_t)x->offset;
    args.w_offset = (uint64_t)w->offset;
    args.bias_offset = 0U;
    args.y_offset = (uint64_t)y->offset;
    args.x_row_bytes = (uint64_t)x->row_bytes;
    args.w_row_bytes = (uint64_t)w->row_bytes;
    args.y_row_bytes = (uint64_t)y->row_bytes;
    args.rows = y->rows;
    args.cols = y->cols;
    args.inner = inner;
    args.has_bias = 0U;
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x->buffer->buffer offset:0U atIndex:0U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)w->buffer->buffer offset:0U atIndex:1U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)x->buffer->buffer offset:0U atIndex:2U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)y->buffer->buffer offset:0U atIndex:3U];
    [encoder setBytes:&args length:sizeof(args) atIndex:4U];
    if (use_reg) {
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

gd_status gd_backend_matmul(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_matrix_view *y)
{
    return gd_backend_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_NN);
}

gd_status gd_backend_matmul_nt(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    return gd_backend_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_NT);
}

gd_status gd_backend_matmul_tn(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    return gd_backend_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_TN);
}
