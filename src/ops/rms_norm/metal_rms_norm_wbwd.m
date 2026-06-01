#import "../../backends/metal/metal_op.h"

static gd_status rms_norm_wbwd_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    const gd_tensor_desc *x_desc = NULL;
    int last = 0;
    int rows = 0;
    int row_blocks = 0;

    if (status != GD_OK) {
        return status;
    }
    x_desc = &ctx->graph->values[ctx->node->inputs[0]].desc;
    last = (int)x_desc->sizes[x_desc->ndim - 1];
    rows = last > 0 ? (int)(_gd_metal_desc_numel(x_desc) / last) : 0;
    row_blocks = (rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_rms_norm_wbwd_reduce");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal rms_norm_wbwd reduce pipeline missing");
    }
    if (row_blocks > 0 && last > 0) {
        ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)row_blocks * (size_t)last *
                                                     sizeof(float);
    }
    return GD_OK;
}

static gd_status rms_norm_wbwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> reduce_pso = ctx->pso2;
    id<MTLBuffer> scratch = ctx->scratch;

    /* dweight: inputs x(0), go(1); output dweight [last]. dims from x. */
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)x_desc->sizes[x_desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(x_desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    p.dtype = GD_METAL_DT_F32;
    if (p.rows <= 0 || p.last <= 0) {
        return GD_OK;
    }
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal rms_norm_wbwd scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger row_blocks = ((NSUInteger)p.rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        NSUInteger channel_blocks = ((NSUInteger)p.last + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        [enc dispatchThreadgroups:MTLSizeMake(row_blocks * channel_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_TG, 1, 1)];
    }
    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    _gd_metal_dispatch_1d(enc, reduce_pso, (NSUInteger)p.last);
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_rms_norm_wbwd = {
    .kind = _GD_OP_RMS_NORM_WBWD,
    .name = "rms_norm_wbwd",
    .plan = rms_norm_wbwd_plan,
    .encode = rms_norm_wbwd_encode,
};
