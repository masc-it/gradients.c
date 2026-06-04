#include "../../backends/metal/metal_backend_internal.h"

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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

gd_status gd_backend_matmul(gd_backend *backend,
                            const gd_backend_matrix_view *x,
                            const gd_backend_matrix_view *w,
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
    MPSDataType data_type;
    bool immediate;
    gd_status st;
    if (backend == NULL || x == NULL || w == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!MPSSupportsMTLDevice((__bridge id<MTLDevice>)backend->device)) {
        return GD_ERR_UNSUPPORTED;
    }
    if (x->dtype != w->dtype || x->dtype != y->dtype || x->cols != w->rows ||
        x->rows != y->rows || w->cols != y->cols ||
        !gd_metal_matrix_bounds_ok(x) || !gd_metal_matrix_bounds_ok(w) ||
        !gd_metal_matrix_bounds_ok(y)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    data_type = gd_mps_dtype(x->dtype);
    if (data_type == MPSDataTypeInvalid) {
        return GD_ERR_UNSUPPORTED;
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
    return gd_metal_finish_immediate(command_buffer, immediate);
}
