#import "../../backends/metal/metal_op.h"

static gd_status embedding_bwd_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_support_default(ctx);

    if (status != GD_OK || ctx->graph == NULL) {
        return status;
    }
    if (ctx->graph->values[ctx->node->inputs[1]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding_bwd needs I32 ids in v1");
    }
    if (ctx->graph->values[ctx->node->inputs[0]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[ctx->node->outputs[0]].desc.dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding_bwd needs F32 grad/output");
    }
    return GD_OK;
}

static gd_status embedding_bwd_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_embedding_bwd_scatter");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal embedding_bwd scatter pipeline missing");
    }
    return GD_OK;
}

static gd_status embedding_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> scatter_pso = ctx->pso2;

    const gd_tensor_desc *table = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding_bwd needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(_gd_metal_desc_numel(go) / p.dim) : 0;
    if (scatter_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal embedding_bwd scatter pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    {
        int total = p.vocab * p.dim;
        if (total > 0) {
            _gd_metal_dispatch_1d(enc, pso, (NSUInteger)total);
        }
    }
    [enc setComputePipelineState:scatter_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        int total = p.n * p.dim;
        if (total > 0) {
            _gd_metal_dispatch_1d(enc, scatter_pso, (NSUInteger)total);
        }
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_embedding_bwd = {
    .kind = _GD_OP_EMBEDDING_BWD,
    .name = "embedding_bwd",
    .support = embedding_bwd_support,
    .plan = embedding_bwd_plan,
    .encode = embedding_bwd_encode,
};
