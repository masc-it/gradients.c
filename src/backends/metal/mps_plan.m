#import "metal_op.h"

static bool mps_gemm_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16;
}

static MPSDataType mps_gemm_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
}

gd_status _gd_metal_plan_mps_gemm(_gd_metal_plan_ctx *ctx)
{
    GDMetalState *st = ctx->state;
    gd_graph *graph = ctx->graph;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    _gd_op_kind op = node->op;
    int j = ctx->node_id;

    if (st.useMPS && (op == _GD_OP_LINEAR || op == _GD_OP_MATMUL)) {
        const gd_tensor_desc *a_desc = &graph->values[node->inputs[0]].desc;
        const gd_tensor_desc *b_desc = &graph->values[node->inputs[1]].desc;
        const gd_tensor_desc *out_desc = &graph->values[node->outputs[0]].desc;
        bool same_dtype = a_desc->dtype == b_desc->dtype && b_desc->dtype == out_desc->dtype;
        bool mixed_f16_f32 = a_desc->dtype == GD_DTYPE_F16 &&
                             b_desc->dtype == GD_DTYPE_F16 &&
                             out_desc->dtype == GD_DTYPE_F32 &&
                             node->attrs.compute.compute_dtype == GD_DTYPE_F32 &&
                             node->attrs.compute.accum_dtype == GD_DTYPE_INVALID;
        if ((same_dtype || mixed_f16_f32) &&
            mps_gemm_dtype_supported(a_desc->dtype) &&
            mps_gemm_dtype_supported(b_desc->dtype) &&
            mps_gemm_dtype_supported(out_desc->dtype) &&
            a_desc->layout == GD_LAYOUT_CONTIGUOUS && b_desc->layout == GD_LAYOUT_CONTIGUOUS &&
            out_desc->layout == GD_LAYOUT_CONTIGUOUS &&
            a_desc->storage_offset_bytes == 0 && b_desc->storage_offset_bytes == 0 &&
            out_desc->storage_offset_bytes == 0) {
            bool ok = false;
            bool trans_left = false;
            bool trans_right = node->attrs.trans_b ? true : false;
            int result_rows = 0;
            int result_cols = 0;
            int inner = 0;
            NSUInteger batch = 1U;
            NSUInteger a_rows_desc = 0U;
            NSUInteger a_cols_desc = 0U;
            NSUInteger b_rows_desc = 0U;
            NSUInteger b_cols_desc = 0U;
            NSUInteger out_rows_desc = 0U;
            NSUInteger out_cols_desc = 0U;
            if (op == _GD_OP_LINEAR && !node->attrs.has_bias) {
                inner = (int)a_desc->sizes[a_desc->ndim - 1];
                result_cols = (int)out_desc->sizes[out_desc->ndim - 1];
                result_rows = inner > 0 ? (int)(_gd_metal_desc_numel(a_desc) / inner) : 0;
                a_rows_desc = (NSUInteger)result_rows;
                a_cols_desc = (NSUInteger)inner;
                b_rows_desc = (NSUInteger)b_desc->sizes[0];
                b_cols_desc = (NSUInteger)b_desc->sizes[1];
                out_rows_desc = (NSUInteger)result_rows;
                out_cols_desc = (NSUInteger)result_cols;
                ok = true;
            } else if (op == _GD_OP_MATMUL && !node->attrs.trans_a && b_desc->ndim == 2) {
                int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                int b_rows_i = (int)b_desc->sizes[0];
                int b_cols_i = (int)b_desc->sizes[1];
                inner = a_cols_i;
                result_cols = trans_right ? b_rows_i : b_cols_i;
                result_rows = inner > 0 ? (int)(_gd_metal_desc_numel(a_desc) / inner) : 0;
                a_rows_desc = (NSUInteger)result_rows;
                a_cols_desc = (NSUInteger)inner;
                b_rows_desc = (NSUInteger)b_rows_i;
                b_cols_desc = (NSUInteger)b_cols_i;
                out_rows_desc = (NSUInteger)result_rows;
                out_cols_desc = (NSUInteger)result_cols;
                ok = (out_desc->sizes[out_desc->ndim - 1] == result_cols);
            } else if (op == _GD_OP_MATMUL && out_desc->ndim >= 3 &&
                       a_desc->ndim == out_desc->ndim &&
                       b_desc->ndim == out_desc->ndim) {
                bool same_batch = true;
                int batch_ndim = out_desc->ndim - 2;
                int a_rows_i = (int)a_desc->sizes[a_desc->ndim - 2];
                int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                int b_rows_i = (int)b_desc->sizes[b_desc->ndim - 2];
                int b_cols_i = (int)b_desc->sizes[b_desc->ndim - 1];
                int i_batch = 0;
                for (i_batch = 0; i_batch < batch_ndim; ++i_batch) {
                    if (a_desc->sizes[i_batch] != out_desc->sizes[i_batch] ||
                        b_desc->sizes[i_batch] != out_desc->sizes[i_batch]) {
                        same_batch = false;
                        break;
                    }
                    batch *= (NSUInteger)out_desc->sizes[i_batch];
                }
                trans_left = node->attrs.trans_a ? true : false;
                inner = trans_left ? a_rows_i : a_cols_i;
                result_rows = trans_left ? a_cols_i : a_rows_i;
                result_cols = trans_right ? b_rows_i : b_cols_i;
                a_rows_desc = (NSUInteger)a_rows_i;
                a_cols_desc = (NSUInteger)a_cols_i;
                b_rows_desc = (NSUInteger)b_rows_i;
                b_cols_desc = (NSUInteger)b_cols_i;
                out_rows_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 2];
                out_cols_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 1];
                ok = same_batch && inner == (trans_right ? b_cols_i : b_rows_i) &&
                     out_desc->sizes[out_desc->ndim - 2] == result_rows &&
                     out_desc->sizes[out_desc->ndim - 1] == result_cols;
            }
            if (ok && result_rows > 0 && result_cols > 0 && inner > 0 && batch > 0U) {
                NSUInteger a_elem_size = (NSUInteger)gd_dtype_sizeof(a_desc->dtype);
                NSUInteger b_elem_size = (NSUInteger)gd_dtype_sizeof(b_desc->dtype);
                NSUInteger out_elem_size = (NSUInteger)gd_dtype_sizeof(out_desc->dtype);
                MPSDataType a_dtype = mps_gemm_dtype(a_desc->dtype);
                MPSDataType b_dtype = mps_gemm_dtype(b_desc->dtype);
                MPSDataType out_dtype = mps_gemm_dtype(out_desc->dtype);
                NSUInteger a_row_bytes = a_cols_desc * a_elem_size;
                NSUInteger b_row_bytes = b_cols_desc * b_elem_size;
                NSUInteger out_row_bytes = out_cols_desc * out_elem_size;
                MPSMatrixDescriptor *ad = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:a_rows_desc
                                     columns:a_cols_desc
                                    matrices:batch
                                    rowBytes:a_row_bytes
                                 matrixBytes:a_rows_desc * a_row_bytes
                                    dataType:a_dtype];
                MPSMatrixDescriptor *bd = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:b_rows_desc
                                     columns:b_cols_desc
                                    matrices:batch
                                    rowBytes:b_row_bytes
                                 matrixBytes:b_rows_desc * b_row_bytes
                                    dataType:b_dtype];
                MPSMatrixDescriptor *od = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:out_rows_desc
                                     columns:out_cols_desc
                                    matrices:batch
                                    rowBytes:out_row_bytes
                                 matrixBytes:out_rows_desc * out_row_bytes
                                    dataType:out_dtype];
                GDMPSGemmPlan *plan = [GDMPSGemmPlan new];
                plan.kernel = [[MPSMatrixMultiplication alloc]
                    initWithDevice:st.device
                    transposeLeft:trans_left
                    transposeRight:trans_right
                    resultRows:(NSUInteger)result_rows
                    resultColumns:(NSUInteger)result_cols
                    interiorColumns:(NSUInteger)inner
                    alpha:1.0
                    beta:0.0];
                plan.kernel.batchStart = 0U;
                plan.kernel.batchSize = batch;
                plan.left = [[MPSMatrix alloc]
                    initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[0]].storage)
                            offset:0
                        descriptor:ad];
                plan.right = [[MPSMatrix alloc]
                    initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[1]].storage)
                            offset:0
                        descriptor:bd];
                plan.result = [[MPSMatrix alloc]
                    initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->outputs[0]].storage)
                            offset:0
                        descriptor:od];
                if (plan.kernel != nil && plan.left != nil && plan.right != nil &&
                    plan.result != nil) {
                    exe->node_mps[j] = (void *)CFBridgingRetain(plan);
                }
            }
        }
    }
    return GD_OK;
}
