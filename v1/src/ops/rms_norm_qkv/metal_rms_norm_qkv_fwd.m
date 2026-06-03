#import "../../backends/metal/metal_op.h"

@interface GDMPSRmsNormQKVPlan : NSObject
@property (strong) MPSMatrixMultiplication *qKernel;
@property (strong) MPSMatrixMultiplication *kKernel;
@property (strong) MPSMatrixMultiplication *vKernel;
@property (strong) MPSMatrixDescriptor *normDescriptor;
@property (strong) MPSMatrixDescriptor *wqDescriptor;
@property (strong) MPSMatrixDescriptor *wkDescriptor;
@property (strong) MPSMatrixDescriptor *wvDescriptor;
@property (strong) MPSMatrixDescriptor *qDescriptor;
@property (strong) MPSMatrixDescriptor *kDescriptor;
@property (strong) MPSMatrixDescriptor *vDescriptor;
@property (strong) MPSMatrix *norm;
@property (strong) MPSMatrix *wq;
@property (strong) MPSMatrix *wk;
@property (strong) MPSMatrix *wv;
@property (strong) MPSMatrix *q;
@property (strong) MPSMatrix *k;
@property (strong) MPSMatrix *v;
@end

@implementation GDMPSRmsNormQKVPlan
@end

static bool rms_norm_qkv_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16;
}

static MPSDataType rms_norm_qkv_mps_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
}

static void rms_norm_qkv_dims(const gd_tensor_desc *x,
                              const gd_tensor_desc *q,
                              const gd_tensor_desc *k,
                              const gd_tensor_desc *v,
                              int *rows,
                              int *d,
                              int *q_cols,
                              int *k_cols,
                              int *v_cols)
{
    *d = (int)x->sizes[x->ndim - 1];
    *rows = *d > 0 ? (int)(_gd_metal_desc_numel(x) / *d) : 0;
    *q_cols = (int)q->sizes[q->ndim - 1];
    *k_cols = (int)k->sizes[k->ndim - 1];
    *v_cols = (int)v->sizes[v->ndim - 1];
}

static gd_status rms_norm_qkv_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const _gd_node *node = NULL;
    const gd_tensor_desc *x = NULL;
    gd_dtype dtype;
    int i = 0;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal rms_norm_qkv support ctx is NULL");
    }
    node = ctx->node;
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil &&
        (_gd_metal_pipeline_for(ctx->state, node->op) == nil ||
         _gd_metal_pipeline_named(ctx->state, "gd_rms_norm_qkv_norm") == nil)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'rms_norm_qkv'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    x = &ctx->graph->values[node->inputs[0]].desc;
    dtype = x->dtype;
    if (!rms_norm_qkv_dtype_supported(dtype)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rms_norm_qkv supports F32/F16 only");
    }
    if (dtype == GD_DTYPE_F16 && (ctx->state == nil || !ctx->state.useMPS)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 rms_norm_qkv needs GD_METAL_MPS=1");
    }
    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *d = &ctx->graph->values[node->inputs[i]].desc;
        if (d->dtype != dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal rms_norm_qkv requires matching dtypes");
        }
        if (d->layout != GD_LAYOUT_CONTIGUOUS) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal rms_norm_qkv requires contiguous inputs");
        }
    }
    for (i = 0; i < node->n_outputs; ++i) {
        const gd_tensor_desc *d = &ctx->graph->values[node->outputs[i]].desc;
        if (d->dtype != dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal rms_norm_qkv requires matching outputs");
        }
        if (d->layout != GD_LAYOUT_CONTIGUOUS) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal rms_norm_qkv requires contiguous outputs");
        }
    }
    return GD_OK;
}

static MPSMatrixDescriptor *matrix_desc(NSUInteger rows,
                                        NSUInteger cols,
                                        NSUInteger elem_size,
                                        MPSDataType dtype)
{
    NSUInteger row_bytes = cols * elem_size;
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                 columns:cols
                                                matrices:1U
                                                rowBytes:row_bytes
                                             matrixBytes:rows * row_bytes
                                                dataType:dtype];
}

static MPSMatrixMultiplication *mm_kernel(id<MTLDevice> device,
                                          NSUInteger rows,
                                          NSUInteger cols,
                                          NSUInteger inner)
{
    return [[MPSMatrixMultiplication alloc] initWithDevice:device
                                             transposeLeft:NO
                                            transposeRight:NO
                                                resultRows:rows
                                             resultColumns:cols
                                           interiorColumns:inner
                                                     alpha:1.0
                                                      beta:0.0];
}

static gd_status rms_norm_qkv_plan_mps(_gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *x = &ctx->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *wq = &ctx->graph->values[node->inputs[2]].desc;
    const gd_tensor_desc *wk = &ctx->graph->values[node->inputs[3]].desc;
    const gd_tensor_desc *wv = &ctx->graph->values[node->inputs[4]].desc;
    const gd_tensor_desc *q = &ctx->graph->values[node->outputs[1]].desc;
    const gd_tensor_desc *k = &ctx->graph->values[node->outputs[2]].desc;
    const gd_tensor_desc *v = &ctx->graph->values[node->outputs[3]].desc;
    GDMPSRmsNormQKVPlan *plan = nil;
    MPSDataType dtype;
    NSUInteger elem_size = 0U;
    int rows = 0;
    int d = 0;
    int q_cols = 0;
    int k_cols = 0;
    int v_cols = 0;

    if (!ctx->state.useMPS) {
        return GD_OK;
    }
    rms_norm_qkv_dims(x, q, k, v, &rows, &d, &q_cols, &k_cols, &v_cols);
    if (rows <= 0 || d <= 0 || q_cols <= 0 || k_cols <= 0 || v_cols <= 0) {
        return GD_OK;
    }
    dtype = rms_norm_qkv_mps_dtype(x->dtype);
    elem_size = (NSUInteger)gd_dtype_sizeof(x->dtype);
    plan = [GDMPSRmsNormQKVPlan new];
    plan.normDescriptor = matrix_desc((NSUInteger)rows, (NSUInteger)d, elem_size, dtype);
    plan.wqDescriptor = matrix_desc((NSUInteger)wq->sizes[0], (NSUInteger)wq->sizes[1], elem_size, dtype);
    plan.wkDescriptor = matrix_desc((NSUInteger)wk->sizes[0], (NSUInteger)wk->sizes[1], elem_size, dtype);
    plan.wvDescriptor = matrix_desc((NSUInteger)wv->sizes[0], (NSUInteger)wv->sizes[1], elem_size, dtype);
    plan.qDescriptor = matrix_desc((NSUInteger)rows, (NSUInteger)q_cols, elem_size, dtype);
    plan.kDescriptor = matrix_desc((NSUInteger)rows, (NSUInteger)k_cols, elem_size, dtype);
    plan.vDescriptor = matrix_desc((NSUInteger)rows, (NSUInteger)v_cols, elem_size, dtype);
    plan.qKernel = mm_kernel(ctx->state.device, (NSUInteger)rows, (NSUInteger)q_cols, (NSUInteger)d);
    plan.kKernel = mm_kernel(ctx->state.device, (NSUInteger)rows, (NSUInteger)k_cols, (NSUInteger)d);
    plan.vKernel = mm_kernel(ctx->state.device, (NSUInteger)rows, (NSUInteger)v_cols, (NSUInteger)d);
    if (plan.normDescriptor == nil || plan.wqDescriptor == nil || plan.wkDescriptor == nil ||
        plan.wvDescriptor == nil || plan.qDescriptor == nil || plan.kDescriptor == nil ||
        plan.vDescriptor == nil || plan.qKernel == nil || plan.kKernel == nil ||
        plan.vKernel == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to create rms_norm_qkv MPS plan");
    }
    ctx->exe->node_mps[ctx->node_id] = (void *)CFBridgingRetain(plan);
    return GD_OK;
}

static gd_status rms_norm_qkv_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state,
                                                                                   "gd_rms_norm_qkv_norm");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal rms_norm_qkv norm pipeline missing");
    }
    status = rms_norm_qkv_plan_mps(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL) {
        const _gd_node *node = ctx->node;
        const gd_tensor_desc *x = &ctx->graph->values[node->inputs[0]].desc;
        if (x->dtype == GD_DTYPE_F16 && ctx->exe->node_mps[ctx->node_id] == NULL) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 rms_norm_qkv requires an MPS plan");
        }
    }
    return GD_OK;
}

static void fill_params(_gd_executable *exe,
                        const _gd_node *node,
                        gd_metal_rms_norm_qkv_params *p)
{
    const gd_tensor_desc *x = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *q = &exe->graph->values[node->outputs[1]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->outputs[2]].desc;
    const gd_tensor_desc *v = &exe->graph->values[node->outputs[3]].desc;
    int rows = 0;
    int d = 0;
    int q_cols = 0;
    int k_cols = 0;
    int v_cols = 0;

    rms_norm_qkv_dims(x, q, k, v, &rows, &d, &q_cols, &k_cols, &v_cols);
    memset(p, 0, sizeof(*p));
    p->rows = rows;
    p->last = d;
    p->q_cols = q_cols;
    p->k_cols = k_cols;
    p->v_cols = v_cols;
    p->eps = node->attrs.eps;
    p->dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(x->dtype, &p->dtype);
}

static gd_status encode_mps_one(id<MTLCommandBuffer> cmd,
                                __strong id<MTLComputeCommandEncoder> *enc,
                                MPSMatrixMultiplication *kernel,
                                MPSMatrix *left,
                                MPSMatrix *right,
                                MPSMatrix *result)
{
    if (cmd == nil || enc == NULL || *enc == nil || kernel == nil || left == nil ||
        right == nil || result == nil) {
        return _gd_error(GD_ERR_BACKEND, "invalid rms_norm_qkv MPS encode");
    }
    [*enc endEncoding];
    *enc = nil;
    [kernel encodeToCommandBuffer:cmd leftMatrix:left rightMatrix:right resultMatrix:result];
    *enc = [cmd computeCommandEncoder];
    if (*enc == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to resume compute encoder after rms_norm_qkv MPS");
    }
    return GD_OK;
}

static gd_status rms_norm_qkv_encode_mps(_gd_metal_encode_ctx *ctx,
                                         GDMPSRmsNormQKVPlan *plan,
                                         const gd_metal_rms_norm_qkv_params *p)
{
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLBuffer> norm_buf = _gd_metal_value_buffer(exe, node->outputs[0]);
    MPSMatrix *norm = nil;
    MPSMatrix *wq = nil;
    MPSMatrix *wk = nil;
    MPSMatrix *wv = nil;
    MPSMatrix *q = nil;
    MPSMatrix *k = nil;
    MPSMatrix *v = nil;
    gd_status status = GD_OK;

    [*ctx->encoder setComputePipelineState:ctx->pso2];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [*ctx->encoder setBuffer:norm_buf offset:0 atIndex:2];
    [*ctx->encoder setBytes:p length:sizeof(*p) atIndex:3];
    if (p->rows > 0) {
        _gd_metal_dispatch_reduce_groups(*ctx->encoder, (NSUInteger)p->rows);
    }

    norm = [[MPSMatrix alloc] initWithBuffer:norm_buf offset:0 descriptor:plan.normDescriptor];
    wq = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->inputs[2])
                                    offset:0 descriptor:plan.wqDescriptor];
    wk = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->inputs[3])
                                    offset:0 descriptor:plan.wkDescriptor];
    wv = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->inputs[4])
                                    offset:0 descriptor:plan.wvDescriptor];
    q = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->outputs[1])
                                   offset:0 descriptor:plan.qDescriptor];
    k = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->outputs[2])
                                   offset:0 descriptor:plan.kDescriptor];
    v = [[MPSMatrix alloc] initWithBuffer:_gd_metal_value_buffer(exe, node->outputs[3])
                                   offset:0 descriptor:plan.vDescriptor];
    if (norm == nil || wq == nil || wk == nil || wv == nil || q == nil || k == nil || v == nil) {
        return _gd_error(GD_ERR_BACKEND, "failed to create rms_norm_qkv MPS matrices");
    }
    plan.norm = norm;
    plan.wq = wq;
    plan.wk = wk;
    plan.wv = wv;
    plan.q = q;
    plan.k = k;
    plan.v = v;
    status = encode_mps_one(ctx->command_buffer, ctx->encoder, plan.qKernel, plan.norm, plan.wq, plan.q);
    if (status != GD_OK) { return status; }
    status = encode_mps_one(ctx->command_buffer, ctx->encoder, plan.kKernel, plan.norm, plan.wk, plan.k);
    if (status != GD_OK) { return status; }
    return encode_mps_one(ctx->command_buffer, ctx->encoder, plan.vKernel, plan.norm, plan.wv, plan.v);
}

static gd_status rms_norm_qkv_encode(_gd_metal_encode_ctx *ctx)
{
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    GDMPSRmsNormQKVPlan *mps = exe->node_mps != NULL
                                    ? (__bridge GDMPSRmsNormQKVPlan *)exe->node_mps[ctx->node_id]
                                    : nil;
    gd_metal_rms_norm_qkv_params p;
    int total = 0;

    fill_params(exe, node, &p);
    if (mps != nil) {
        return rms_norm_qkv_encode_mps(ctx, mps, &p);
    }

    total = p.rows * (p.last + p.q_cols + p.k_cols + p.v_cols);
    [*ctx->encoder setComputePipelineState:ctx->pso];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:5];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:6];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:7];
    [*ctx->encoder setBuffer:_gd_metal_value_buffer(exe, node->outputs[3]) offset:0 atIndex:8];
    [*ctx->encoder setBytes:&p length:sizeof(p) atIndex:9];
    if (total > 0) {
        _gd_metal_dispatch_1d(*ctx->encoder, ctx->pso, (NSUInteger)total);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_rms_norm_qkv = {
    .kind = _GD_OP_RMS_NORM_QKV,
    .name = "rms_norm_qkv",
    .support = rms_norm_qkv_support,
    .plan = rms_norm_qkv_plan,
    .encode = rms_norm_qkv_encode,
};
