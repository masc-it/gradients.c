#include "../../backends/metal/metal_backend_internal.h"

#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdint.h>

#define GD_LINEAR_BIAS_THREADS 256U

typedef struct gd_metal_linear_bias_args {
    uint64_t y_offset;
    uint64_t bias_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t y_row_bytes;
} gd_metal_linear_bias_args;

static const char *gd_linear_bias_kernel_source(void)
{
    return "#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct gd_linear_bias_args { ulong y_offset; ulong bias_offset; uint rows; uint cols; uint y_row_bytes; };\n"
           "kernel void gd_linear_bias_f16(device uchar *ybuf [[buffer(0)]], device uchar *bbuf [[buffer(1)]], constant gd_linear_bias_args &args [[buffer(2)]], uint gid [[thread_position_in_grid]]) {\n"
           "  uint total = args.rows * args.cols;\n"
           "  if (gid >= total) { return; }\n"
           "  uint row = gid / args.cols;\n"
           "  uint col = gid - row * args.cols;\n"
           "  ulong y_byte = args.y_offset + ulong(row) * ulong(args.y_row_bytes) + ulong(col) * 2ul;\n"
           "  ulong b_byte = args.bias_offset + ulong(col) * 2ul;\n"
           "  device half *yp = reinterpret_cast<device half *>(ybuf + y_byte);\n"
           "  device half *bp = reinterpret_cast<device half *>(bbuf + b_byte);\n"
           "  *yp = half(float(*yp) + float(*bp));\n"
           "}\n";
}

static MPSDataType gd_mps_dtype(uint32_t dtype)
{
    return dtype == 1U ? MPSDataTypeFloat16 : MPSDataTypeInvalid;
}

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

static gd_status gd_metal_linear_bias_pipeline(gd_backend *backend, id<MTLComputePipelineState> *out_pso)
{
    NSError *error = nil;
    NSString *source;
    id<MTLLibrary> library;
    id<MTLFunction> function;
    id<MTLComputePipelineState> pso;
    if (backend == NULL || out_pso == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->linear_bias_pso != NULL) {
        *out_pso = (__bridge id<MTLComputePipelineState>)backend->linear_bias_pso;
        return GD_OK;
    }
    source = [NSString stringWithUTF8String:gd_linear_bias_kernel_source()];
    if (source == nil) {
        return GD_ERR_INTERNAL;
    }
    library = [(__bridge id<MTLDevice>)backend->device newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        return GD_ERR_INTERNAL;
    }
    function = [library newFunctionWithName:@"gd_linear_bias_f16"];
    if (function == nil) {
        return GD_ERR_INTERNAL;
    }
    pso = [(__bridge id<MTLDevice>)backend->device newComputePipelineStateWithFunction:function error:&error];
    if (pso == nil) {
        return GD_ERR_INTERNAL;
    }
    backend->linear_bias_pso = (void *)CFBridgingRetain(pso);
    *out_pso = pso;
    return GD_OK;
}

gd_status gd_backend_linear(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
                            const gd_backend_vector_view *bias,
                            const gd_backend_matrix_view *y)
{
    id<MTLCommandBuffer> command_buffer;
    MPSMatrixDescriptor *x_desc;
    MPSMatrixDescriptor *w_desc;
    MPSMatrixDescriptor *y_desc;
    MPSMatrix *x_mat;
    MPSMatrix *w_mat;
    MPSMatrix *y_mat;
    MPSMatrixMultiplication *gemm;
    id<MTLComputePipelineState> bias_pso;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_linear_bias_args args;
    MTLSize grid;
    MTLSize threads;
    MPSDataType data_type;
    bool immediate;
    gd_status st;
    uint32_t total;
    if (backend == NULL || x == NULL || w == NULL || bias == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!MPSSupportsMTLDevice((__bridge id<MTLDevice>)backend->device)) {
        return GD_ERR_UNSUPPORTED;
    }
    if (x->dtype != w->dtype || x->dtype != y->dtype || bias->dtype != y->dtype ||
        x->cols != w->rows || x->rows != y->rows || w->cols != y->cols ||
        bias->length != y->cols || !gd_metal_matrix_bounds_ok(x) ||
        !gd_metal_matrix_bounds_ok(w) || !gd_metal_matrix_bounds_ok(y) ||
        !gd_metal_vector_bounds_ok(bias)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    data_type = gd_mps_dtype(x->dtype);
    if (data_type == MPSDataTypeInvalid) {
        return GD_ERR_UNSUPPORTED;
    }
    if (y->rows > UINT32_MAX / y->cols) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    total = y->rows * y->cols;
    st = gd_metal_linear_bias_pipeline(backend, &bias_pso);
    if (st != GD_OK) {
        return st;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    x_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)x->rows
                                                   columns:(NSUInteger)x->cols
                                                  rowBytes:(NSUInteger)x->row_bytes
                                                  dataType:data_type];
    w_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)w->rows
                                                   columns:(NSUInteger)w->cols
                                                  rowBytes:(NSUInteger)w->row_bytes
                                                  dataType:data_type];
    y_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)y->rows
                                                   columns:(NSUInteger)y->cols
                                                  rowBytes:(NSUInteger)y->row_bytes
                                                  dataType:data_type];
    x_mat = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)x->buffer->buffer
                                       offset:(NSUInteger)x->offset
                                   descriptor:x_desc];
    w_mat = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)w->buffer->buffer
                                       offset:(NSUInteger)w->offset
                                   descriptor:w_desc];
    y_mat = [[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)y->buffer->buffer
                                       offset:(NSUInteger)y->offset
                                   descriptor:y_desc];
    if (x_mat == nil || w_mat == nil || y_mat == nil) {
        return GD_ERR_INTERNAL;
    }
    gemm = [[MPSMatrixMultiplication alloc] initWithDevice:(__bridge id<MTLDevice>)backend->device
                                             transposeLeft:NO
                                            transposeRight:NO
                                                resultRows:(NSUInteger)y->rows
                                             resultColumns:(NSUInteger)y->cols
                                           interiorColumns:(NSUInteger)x->cols
                                                     alpha:1.0
                                                      beta:0.0];
    if (gemm == nil) {
        return GD_ERR_INTERNAL;
    }
    [gemm encodeToCommandBuffer:command_buffer leftMatrix:x_mat rightMatrix:w_mat resultMatrix:y_mat];

    args.y_offset = y->offset;
    args.bias_offset = bias->offset;
    args.rows = y->rows;
    args.cols = y->cols;
    args.y_row_bytes = (uint32_t)y->row_bytes;
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    [encoder setComputePipelineState:bias_pso];
    [encoder setBuffer:(__bridge id<MTLBuffer>)y->buffer->buffer offset:0U atIndex:0U];
    [encoder setBuffer:(__bridge id<MTLBuffer>)bias->buffer->buffer offset:0U atIndex:1U];
    [encoder setBytes:&args length:sizeof(args) atIndex:2U];
    grid = MTLSizeMake((NSUInteger)total, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(total < GD_LINEAR_BIAS_THREADS ? total : GD_LINEAR_BIAS_THREADS), 1U, 1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
