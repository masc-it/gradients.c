#include "../../backends/metal/metal_backend_internal.h"
#include "metal_matmul_types.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GD_METAL_MPS_DATA_TYPE_FLOAT16 (0x10000000U | 16U)
#define GD_METAL_MPS_MATMUL_MIN_FLOPS 500000000ULL
#define GD_METAL_MPS_MATMUL_NN_MIN_FLOPS 500000000ULL
#define GD_METAL_MPS_BATCH_MATMUL_MIN_FLOPS 100000000ULL
#define GD_METAL_MPS_BATCH_NN_MATMUL_MIN_FLOPS 100000000ULL

@interface NSObject (GDMetalMPSDynamic)
+ (id)descriptorWithDataType:(uint32_t)dataType shape:(NSArray<NSNumber *> *)shape;
- (void)setPreferPackedRows:(BOOL)preferPackedRows;
- (void)transposeDimension:(NSUInteger)dimension withDimension:(NSUInteger)otherDimension;
- (id)initWithBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset descriptor:(id)descriptor;
- (id)initWithDevice:(id<MTLDevice>)device sourceCount:(NSUInteger)sourceCount;
- (void)encodeToCommandEncoder:(id<MTLComputeCommandEncoder>)encoder
                  commandBuffer:(id<MTLCommandBuffer>)commandBuffer
                   sourceArrays:(NSArray *)sourceArrays
               destinationArray:(id)destinationArray;
@end

/*
 * The public matmul op supports full PyTorch-style batch broadcasting in core
 * code.  The Metal backend receives already-broadcasted batch views: all three
 * views share the output batch shape and broadcasted inputs use zero byte
 * strides on broadcast axes.  Custom kernels use a single dispatch spanning
 * GEMM tiles in x/y and logical batch in z; large dense batched GEMMs can
 * delegate to MPSNDArrayMatrixMultiplication when its library kernel is faster.
 */

typedef enum gd_metal_matmul_layout {
    GD_METAL_MATMUL_NN = 0,
    GD_METAL_MATMUL_NT = 1,
    GD_METAL_MATMUL_TN = 2,
} gd_metal_matmul_layout;

static bool gd_metal_size_add(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool gd_metal_batched_matrix_bounds_ok(const gd_backend_batched_matrix_view *view)
{
    size_t elem_size;
    size_t last_row;
    size_t row_span;
    size_t matrix_bytes;
    size_t max_batch_offset = 0U;
    size_t total_bytes;
    size_t batch_count = 1U;
    uint32_t i;
    if (view == NULL || view->buffer == NULL || view->rows == 0U || view->cols == 0U ||
        view->batch_rank > GD_BACKEND_MAX_BATCH_DIMS || view->batch_count == 0U) {
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
    row_span = (size_t)view->cols * elem_size;
    if (!gd_metal_size_add(last_row, row_span, &matrix_bytes)) {
        return false;
    }
    for (i = 0U; i < view->batch_rank; ++i) {
        size_t dim_extent;
        size_t dim_offset;
        if (view->batch_shape[i] == 0U ||
            batch_count > (size_t)UINT32_MAX / (size_t)view->batch_shape[i]) {
            return false;
        }
        batch_count *= (size_t)view->batch_shape[i];
        dim_extent = (size_t)view->batch_shape[i] - 1U;
        if (dim_extent != 0U && view->batch_strides[i] > SIZE_MAX / dim_extent) {
            return false;
        }
        dim_offset = view->batch_strides[i] * dim_extent;
        if (!gd_metal_size_add(max_batch_offset, dim_offset, &max_batch_offset)) {
            return false;
        }
    }
    if (batch_count != (size_t)view->batch_count ||
        !gd_metal_size_add(max_batch_offset, matrix_bytes, &total_bytes)) {
        return false;
    }
    return view->offset <= view->buffer->nbytes && total_bytes <= view->buffer->nbytes - view->offset;
}

static NSUInteger gd_metal_div_up_u32(uint32_t value, uint32_t denom)
{
    return (NSUInteger)(value / denom + (value % denom != 0U ? 1U : 0U));
}

static bool gd_metal_batched_gemm_reg_ok(const gd_backend_batched_matrix_view *x,
                                         const gd_backend_batched_matrix_view *w,
                                         const gd_backend_batched_matrix_view *y,
                                         uint32_t inner)
{
    uint32_t i;
    if (x == NULL || w == NULL || y == NULL ||
        (x->offset % 16U) != 0U || (w->offset % 16U) != 0U || (y->offset % 16U) != 0U ||
        (x->row_bytes % 16U) != 0U || (w->row_bytes % 16U) != 0U || (y->row_bytes % 16U) != 0U ||
        (y->rows % GD_METAL_GEMM_REG_TILE) != 0U ||
        (y->cols % GD_METAL_GEMM_REG_TILE) != 0U || (inner % 8U) != 0U) {
        return false;
    }
    for (i = 0U; i < y->batch_rank; ++i) {
        if ((x->batch_strides[i] % 16U) != 0U || (w->batch_strides[i] % 16U) != 0U ||
            (y->batch_strides[i] % 16U) != 0U) {
            return false;
        }
    }
    return true;
}

static bool gd_metal_batched_shapes_match(const gd_backend_batched_matrix_view *x,
                                          const gd_backend_batched_matrix_view *w,
                                          const gd_backend_batched_matrix_view *y)
{
    uint32_t i;
    if (x == NULL || w == NULL || y == NULL || x->batch_rank != y->batch_rank ||
        w->batch_rank != y->batch_rank || x->batch_count != y->batch_count ||
        w->batch_count != y->batch_count) {
        return false;
    }
    for (i = 0U; i < y->batch_rank; ++i) {
        if (x->batch_shape[i] != y->batch_shape[i] || w->batch_shape[i] != y->batch_shape[i]) {
            return false;
        }
    }
    return true;
}

static bool gd_metal_matmul_layout_ok(const gd_backend_batched_matrix_view *x,
                                      const gd_backend_batched_matrix_view *w,
                                      const gd_backend_batched_matrix_view *y,
                                      gd_metal_matmul_layout layout,
                                      uint32_t *out_inner)
{
    if (x == NULL || w == NULL || y == NULL || out_inner == NULL ||
        x->dtype != w->dtype || x->dtype != y->dtype ||
        !gd_metal_batched_shapes_match(x, w, y) ||
        !gd_metal_batched_matrix_bounds_ok(x) || !gd_metal_batched_matrix_bounds_ok(w) ||
        !gd_metal_batched_matrix_bounds_ok(y)) {
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

static void gd_metal_fill_gemm_args(gd_metal_gemm_args *args,
                                    const gd_backend_batched_matrix_view *x,
                                    const gd_backend_batched_matrix_view *w,
                                    const gd_backend_batched_matrix_view *y,
                                    uint32_t inner)
{
    uint32_t i;
    memset(args, 0, sizeof(*args));
    args->x_offset = (uint64_t)x->offset;
    args->w_offset = (uint64_t)w->offset;
    args->bias_offset = 0U;
    args->y_offset = (uint64_t)y->offset;
    args->x_row_bytes = (uint64_t)x->row_bytes;
    args->w_row_bytes = (uint64_t)w->row_bytes;
    args->y_row_bytes = (uint64_t)y->row_bytes;
    args->rows = y->rows;
    args->cols = y->cols;
    args->inner = inner;
    args->has_bias = 0U;
    args->batch_rank = y->batch_rank;
    for (i = 0U; i < y->batch_rank; ++i) {
        args->batch_shape[i] = y->batch_shape[i];
        args->x_batch_strides[i] = (uint64_t)x->batch_strides[i];
        args->w_batch_strides[i] = (uint64_t)w->batch_strides[i];
        args->y_batch_strides[i] = (uint64_t)y->batch_strides[i];
    }
}

static bool gd_metal_u64_mul_overflow(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL || (a != 0U && b > UINT64_MAX / a)) {
        return true;
    }
    *out = a * b;
    return false;
}

static uint64_t gd_metal_mps_matmul_min_flops(gd_metal_matmul_layout layout)
{
    const char *env = getenv("GD_MATMUL_MPS_MIN_FLOPS");
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 10);
        if (end != env && *end == '\0') {
            return (uint64_t)value;
        }
    }
    return layout == GD_METAL_MATMUL_NN ? GD_METAL_MPS_MATMUL_NN_MIN_FLOPS
                                        : GD_METAL_MPS_MATMUL_MIN_FLOPS;
}

static uint64_t gd_metal_mps_batch_matmul_min_flops(gd_metal_matmul_layout layout)
{
    const char *env_name = layout == GD_METAL_MATMUL_NN ? "GD_MATMUL_MPS_BATCH_NN_MIN_FLOPS" :
                                                          "GD_MATMUL_MPS_BATCH_MIN_FLOPS";
    const char *env = getenv(env_name);
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 10);
        if (end != env && *end == '\0') {
            return (uint64_t)value;
        }
    }
    return layout == GD_METAL_MATMUL_NN ? GD_METAL_MPS_BATCH_NN_MATMUL_MIN_FLOPS :
                                          GD_METAL_MPS_BATCH_MATMUL_MIN_FLOPS;
}

/*
 * PyTorch's MPS linear path uses MPSNDArrayMatrixMultiplication for large
 * contiguous 2D GEMMs.  Mirror that selectively for packed 2D workloads where
 * the library kernel is consistently faster, including GPT projection forward
 * and backward NT/TN GEMMs. Use dynamic framework loading so the core library
 * does not gain a link-time dependency; non-dense or broadcasted batched cases
 * keep using the custom kernels.
 */
static bool gd_metal_mps_matmul_ok(const gd_backend_batched_matrix_view *x,
                                   const gd_backend_batched_matrix_view *w,
                                   const gd_backend_batched_matrix_view *y,
                                   gd_metal_matmul_layout layout,
                                   uint32_t inner)
{
    uint64_t flops;
    uint64_t tmp;
    if (x == NULL || w == NULL || y == NULL || x->dtype != 1U || w->dtype != 1U || y->dtype != 1U ||
        x->batch_rank != 0U || w->batch_rank != 0U || y->batch_rank != 0U || x->batch_count != 1U ||
        w->batch_count != 1U || y->batch_count != 1U ||
        x->row_bytes != (size_t)x->cols * 2U || w->row_bytes != (size_t)w->cols * 2U ||
        y->row_bytes != (size_t)y->cols * 2U || x->buffer == NULL || w->buffer == NULL ||
        y->buffer == NULL) {
        return false;
    }
    if (gd_metal_u64_mul_overflow((uint64_t)y->rows, (uint64_t)y->cols, &tmp) ||
        gd_metal_u64_mul_overflow(tmp, (uint64_t)inner, &flops) ||
        gd_metal_u64_mul_overflow(flops, 2U, &flops)) {
        return false;
    }
    if (layout == GD_METAL_MATMUL_NN && (uint64_t)y->cols >= (uint64_t)inner * 4ULL &&
        flops < 2000000000ULL) {
        return false;
    }
    return flops >= gd_metal_mps_matmul_min_flops(layout);
}

static bool gd_metal_mps_framework_available(void)
{
    static bool attempted = false;
    static bool available = false;
    if (attempted) {
        return available;
    }
    attempted = true;
    if (NSClassFromString(@"MPSNDArrayMatrixMultiplication") == Nil) {
        NSBundle *bundle = [NSBundle bundleWithPath:@"/System/Library/Frameworks/MetalPerformanceShaders.framework"];
        if (bundle != nil) {
            (void)[bundle load];
        }
    }
    available = NSClassFromString(@"MPSNDArrayMatrixMultiplication") != Nil &&
                NSClassFromString(@"MPSNDArrayDescriptor") != Nil &&
                NSClassFromString(@"MPSNDArray") != Nil;
    return available;
}

static id gd_metal_mps_matrix_array(const gd_backend_batched_matrix_view *view, bool transpose)
{
    Class desc_class;
    Class array_class;
    id desc;
    id<MTLBuffer> buffer;
    if (view == NULL || view->buffer == NULL) {
        return nil;
    }
    desc_class = NSClassFromString(@"MPSNDArrayDescriptor");
    array_class = NSClassFromString(@"MPSNDArray");
    if (desc_class == Nil || array_class == Nil) {
        return nil;
    }
    desc = [desc_class descriptorWithDataType:GD_METAL_MPS_DATA_TYPE_FLOAT16
                                        shape:@[ @((NSUInteger)view->rows), @((NSUInteger)view->cols) ]];
    if (desc == nil) {
        return nil;
    }
    [desc setPreferPackedRows:YES];
    if (transpose) {
        [desc transposeDimension:0 withDimension:1];
    }
    buffer = (__bridge id<MTLBuffer>)view->buffer->buffer;
    return [[array_class alloc] initWithBuffer:buffer offset:(NSUInteger)view->offset descriptor:desc];
}

static id gd_metal_mps_batched_ndarray(const gd_backend_batched_matrix_view *view, bool transpose)
{
    Class desc_class;
    Class array_class;
    id desc;
    id<MTLBuffer> buffer;
    NSUInteger b0 = 1U;
    NSUInteger b1 = 1U;
    if (view == NULL || view->buffer == NULL || view->batch_rank > 2U) {
        return nil;
    }
    if (view->batch_rank == 1U) {
        b1 = (NSUInteger)view->batch_shape[0];
    } else if (view->batch_rank == 2U) {
        b0 = (NSUInteger)view->batch_shape[0];
        b1 = (NSUInteger)view->batch_shape[1];
    }
    desc_class = NSClassFromString(@"MPSNDArrayDescriptor");
    array_class = NSClassFromString(@"MPSNDArray");
    if (desc_class == Nil || array_class == Nil) {
        return nil;
    }
    desc = [desc_class descriptorWithDataType:GD_METAL_MPS_DATA_TYPE_FLOAT16
                                        shape:@[ @(b0), @(b1), @((NSUInteger)view->rows), @((NSUInteger)view->cols) ]];
    if (desc == nil) {
        return nil;
    }
    [desc setPreferPackedRows:YES];
    if (transpose) {
        [desc transposeDimension:0 withDimension:1];
    }
    buffer = (__bridge id<MTLBuffer>)view->buffer->buffer;
    return [[array_class alloc] initWithBuffer:buffer offset:(NSUInteger)view->offset descriptor:desc];
}

static gd_status gd_backend_batched_matmul_dispatch_mps(gd_backend *backend,
                                                        const gd_backend_batched_matrix_view *x,
                                                        const gd_backend_batched_matrix_view *w,
                                                        const gd_backend_batched_matrix_view *y,
                                                        gd_metal_matmul_layout layout)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id kernel;
    id x_array;
    id w_array;
    id y_array;
    bool immediate;
    gd_status st;
    if (backend == NULL || x == NULL || w == NULL || y == NULL || backend->device == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_metal_mps_framework_available()) {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend->mps_matmul_kernel == NULL) {
        Class kernel_class = NSClassFromString(@"MPSNDArrayMatrixMultiplication");
        id new_kernel;
        if (kernel_class == Nil) {
            return GD_ERR_UNSUPPORTED;
        }
        new_kernel = [[kernel_class alloc] initWithDevice:(__bridge id<MTLDevice>)backend->device sourceCount:2];
        if (new_kernel == nil) {
            return GD_ERR_INTERNAL;
        }
        backend->mps_matmul_kernel = (void *)CFBridgingRetain(new_kernel);
    }
    kernel = (__bridge id)backend->mps_matmul_kernel;
    x_array = gd_metal_mps_matrix_array(x, layout == GD_METAL_MATMUL_TN);
    w_array = gd_metal_mps_matrix_array(w, layout == GD_METAL_MATMUL_NT);
    y_array = gd_metal_mps_matrix_array(y, false);
    if (x_array == nil || w_array == nil || y_array == nil || kernel == nil) {
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
    [kernel encodeToCommandEncoder:encoder
                      commandBuffer:command_buffer
                       sourceArrays:@[ x_array, w_array ]
                   destinationArray:y_array];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static gd_status gd_backend_batched_matmul_dispatch_mps_ndarray(gd_backend *backend,
                                                                 const gd_backend_batched_matrix_view *x,
                                                                 const gd_backend_batched_matrix_view *w,
                                                                 const gd_backend_batched_matrix_view *y,
                                                                 gd_metal_matmul_layout layout)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id kernel;
    id x_array;
    id w_array;
    id y_array;
    bool immediate;
    gd_status st;
    if (backend == NULL || x == NULL || w == NULL || y == NULL || backend->device == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_metal_mps_framework_available()) {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend->mps_matmul_kernel == NULL) {
        Class kernel_class = NSClassFromString(@"MPSNDArrayMatrixMultiplication");
        id new_kernel;
        if (kernel_class == Nil) {
            return GD_ERR_UNSUPPORTED;
        }
        new_kernel = [[kernel_class alloc] initWithDevice:(__bridge id<MTLDevice>)backend->device sourceCount:2];
        if (new_kernel == nil) {
            return GD_ERR_INTERNAL;
        }
        backend->mps_matmul_kernel = (void *)CFBridgingRetain(new_kernel);
    }
    kernel = (__bridge id)backend->mps_matmul_kernel;
    x_array = gd_metal_mps_batched_ndarray(x, layout == GD_METAL_MATMUL_TN);
    w_array = gd_metal_mps_batched_ndarray(w, layout == GD_METAL_MATMUL_NT);
    y_array = gd_metal_mps_batched_ndarray(y, false);
    if (x_array == nil || w_array == nil || y_array == nil || kernel == nil) {
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
    [kernel encodeToCommandEncoder:encoder
                      commandBuffer:command_buffer
                       sourceArrays:@[ x_array, w_array ]
                   destinationArray:y_array];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static bool gd_metal_mps_ndarray_dense_batch_ok(const gd_backend_batched_matrix_view *view,
                                                size_t *matrix_bytes_out)
{
    size_t matrix_bytes;
    size_t expected_stride;
    uint32_t axis;
    if (view == NULL || matrix_bytes_out == NULL || view->dtype != 1U || view->buffer == NULL ||
        view->row_bytes != (size_t)view->cols * 2U ||
        (size_t)view->rows > SIZE_MAX / view->row_bytes) {
        return false;
    }
    matrix_bytes = (size_t)view->rows * view->row_bytes;
    if (matrix_bytes == 0U || (matrix_bytes % view->row_bytes) != 0U) {
        return false;
    }
    expected_stride = matrix_bytes;
    axis = view->batch_rank;
    while (axis > 0U) {
        --axis;
        if (view->batch_shape[axis] > 1U) {
            if (view->batch_strides[axis] != expected_stride) {
                return false;
            }
            if (expected_stride > SIZE_MAX / (size_t)view->batch_shape[axis]) {
                return false;
            }
            expected_stride *= (size_t)view->batch_shape[axis];
        }
    }
    *matrix_bytes_out = matrix_bytes;
    return true;
}

static bool gd_metal_mps_batched_matmul_ok(const gd_backend_batched_matrix_view *x,
                                           const gd_backend_batched_matrix_view *w,
                                           const gd_backend_batched_matrix_view *y,
                                           gd_metal_matmul_layout layout,
                                           uint32_t inner)
{
    size_t x_matrix_bytes;
    size_t w_matrix_bytes;
    size_t y_matrix_bytes;
    uint64_t flops;
    uint64_t tmp;
    uint64_t tmp2;
    if (x == NULL || w == NULL || y == NULL ||
        x->batch_rank > 2U || w->batch_rank > 2U || y->batch_rank > 2U ||
        y->batch_count <= 1U || x->batch_count != y->batch_count || w->batch_count != y->batch_count ||
        !gd_metal_mps_ndarray_dense_batch_ok(x, &x_matrix_bytes) ||
        !gd_metal_mps_ndarray_dense_batch_ok(w, &w_matrix_bytes) ||
        !gd_metal_mps_ndarray_dense_batch_ok(y, &y_matrix_bytes)) {
        return false;
    }
    if (gd_metal_u64_mul_overflow((uint64_t)y->rows, (uint64_t)y->cols, &tmp) ||
        gd_metal_u64_mul_overflow(tmp, (uint64_t)inner, &tmp2) ||
        gd_metal_u64_mul_overflow(tmp2, (uint64_t)y->batch_count, &flops) ||
        gd_metal_u64_mul_overflow(flops, 2U, &flops)) {
        return false;
    }
    return flops >= gd_metal_mps_batch_matmul_min_flops(layout);
}

static void gd_metal_compact_unit_batch_dims(gd_backend_batched_matrix_view *x,
                                             gd_backend_batched_matrix_view *w,
                                             gd_backend_batched_matrix_view *y)
{
    uint32_t read_axis;
    uint32_t write_axis = 0U;
    if (x == NULL || w == NULL || y == NULL || x->batch_rank != y->batch_rank || w->batch_rank != y->batch_rank) {
        return;
    }
    for (read_axis = 0U; read_axis < y->batch_rank; ++read_axis) {
        if (y->batch_shape[read_axis] == 1U) {
            continue;
        }
        if (write_axis != read_axis) {
            x->batch_shape[write_axis] = x->batch_shape[read_axis];
            w->batch_shape[write_axis] = w->batch_shape[read_axis];
            y->batch_shape[write_axis] = y->batch_shape[read_axis];
            x->batch_strides[write_axis] = x->batch_strides[read_axis];
            w->batch_strides[write_axis] = w->batch_strides[read_axis];
            y->batch_strides[write_axis] = y->batch_strides[read_axis];
        }
        ++write_axis;
    }
    x->batch_rank = write_axis;
    w->batch_rank = write_axis;
    y->batch_rank = write_axis;
}

static gd_status gd_backend_batched_matmul_dispatch(gd_backend *backend,
                                                    const gd_backend_batched_matrix_view *x,
                                                    const gd_backend_batched_matrix_view *w,
                                                    const gd_backend_batched_matrix_view *y,
                                                    gd_metal_matmul_layout layout)
{
    gd_backend_batched_matrix_view x_compact;
    gd_backend_batched_matrix_view w_compact;
    gd_backend_batched_matrix_view y_compact;
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
    x_compact = *x;
    w_compact = *w;
    y_compact = *y;
    gd_metal_compact_unit_batch_dims(&x_compact, &w_compact, &y_compact);
    x = &x_compact;
    w = &w_compact;
    y = &y_compact;
    if (gd_metal_mps_matmul_ok(x, w, y, layout, inner)) {
        st = gd_backend_batched_matmul_dispatch_mps(backend, x, w, y, layout);
        if (st == GD_OK) {
            return GD_OK;
        }
        if (st != GD_ERR_UNSUPPORTED) {
            return st;
        }
    }
    if (gd_metal_mps_batched_matmul_ok(x, w, y, layout, inner)) {
        st = gd_backend_batched_matmul_dispatch_mps_ndarray(backend, x, w, y, layout);
        if (st == GD_OK) {
            return GD_OK;
        }
        if (st != GD_ERR_UNSUPPORTED) {
            return st;
        }
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
    use_reg = gd_metal_batched_gemm_reg_ok(x, w, y, inner) && reg_pso != NULL;
    pso = (__bridge id<MTLComputePipelineState>)(use_reg ? reg_pso : tiled_pso);
    if (pso == nil) {
        return GD_ERR_INTERNAL;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    gd_metal_fill_gemm_args(&args, x, w, y, inner);
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
                             (NSUInteger)y->batch_count);
        threads = MTLSizeMake(32U * GD_METAL_GEMM_REG_SIMDGROUPS, 1U, 1U);
    } else {
        groups = MTLSizeMake(gd_metal_div_up_u32(y->cols, GD_METAL_GEMM_BN),
                             gd_metal_div_up_u32(y->rows, GD_METAL_GEMM_BM),
                             (NSUInteger)y->batch_count);
        threads = MTLSizeMake(GD_METAL_GEMM_BN / GD_METAL_GEMM_TN,
                              GD_METAL_GEMM_BM / GD_METAL_GEMM_TM,
                              1U);
    }
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

static void gd_backend_batched_matrix_view_from_matrix(const gd_backend_matrix_view *src,
                                                       gd_backend_batched_matrix_view *dst)
{
    memset(dst, 0, sizeof(*dst));
    dst->buffer = src->buffer;
    dst->offset = src->offset;
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->row_bytes = src->row_bytes;
    dst->dtype = src->dtype;
    dst->batch_rank = 0U;
    dst->batch_count = 1U;
}

gd_status gd_backend_batched_matmul(gd_backend *backend,
                                    const gd_backend_batched_matrix_view *x,
                                    const gd_backend_batched_matrix_view *w,
                                    const gd_backend_batched_matrix_view *y)
{
    return gd_backend_batched_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_NN);
}

gd_status gd_backend_batched_matmul_nt(gd_backend *backend,
                                       const gd_backend_batched_matrix_view *x,
                                       const gd_backend_batched_matrix_view *w,
                                       const gd_backend_batched_matrix_view *y)
{
    return gd_backend_batched_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_NT);
}

gd_status gd_backend_batched_matmul_tn(gd_backend *backend,
                                       const gd_backend_batched_matrix_view *x,
                                       const gd_backend_batched_matrix_view *w,
                                       const gd_backend_batched_matrix_view *y)
{
    return gd_backend_batched_matmul_dispatch(backend, x, w, y, GD_METAL_MATMUL_TN);
}

gd_status gd_backend_matmul(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_matrix_view *y)
{
    gd_backend_batched_matrix_view xb;
    gd_backend_batched_matrix_view wb;
    gd_backend_batched_matrix_view yb;
    if (x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_backend_batched_matrix_view_from_matrix(x, &xb);
    gd_backend_batched_matrix_view_from_matrix(w, &wb);
    gd_backend_batched_matrix_view_from_matrix(y, &yb);
    return gd_backend_batched_matmul_dispatch(backend, &xb, &wb, &yb, GD_METAL_MATMUL_NN);
}

gd_status gd_backend_matmul_nt(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    gd_backend_batched_matrix_view xb;
    gd_backend_batched_matrix_view wb;
    gd_backend_batched_matrix_view yb;
    if (x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_backend_batched_matrix_view_from_matrix(x, &xb);
    gd_backend_batched_matrix_view_from_matrix(w, &wb);
    gd_backend_batched_matrix_view_from_matrix(y, &yb);
    return gd_backend_batched_matmul_dispatch(backend, &xb, &wb, &yb, GD_METAL_MATMUL_NT);
}

gd_status gd_backend_matmul_tn(gd_backend *backend,
                               const gd_backend_matrix_view *x,
                               const gd_backend_matrix_view *w,
                               const gd_backend_matrix_view *y)
{
    gd_backend_batched_matrix_view xb;
    gd_backend_batched_matrix_view wb;
    gd_backend_batched_matrix_view yb;
    if (x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_backend_batched_matrix_view_from_matrix(x, &xb);
    gd_backend_batched_matrix_view_from_matrix(w, &wb);
    gd_backend_batched_matrix_view_from_matrix(y, &yb);
    return gd_backend_batched_matmul_dispatch(backend, &xb, &wb, &yb, GD_METAL_MATMUL_TN);
}
