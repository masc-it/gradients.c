#import "../../backends/metal/metal_op.h"

static gd_status embedding_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *table = NULL;
    const gd_tensor_desc *ids = NULL;
    const gd_tensor_desc *out = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal embedding support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'embedding'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    table = &ctx->graph->values[ctx->node->inputs[0]].desc;
    ids = &ctx->graph->values[ctx->node->inputs[1]].desc;
    out = &ctx->graph->values[ctx->node->outputs[0]].desc;
    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding needs I32 ids in v1");
    }
    if (table->dtype != out->dtype ||
        (out->dtype != GD_DTYPE_F32 && out->dtype != GD_DTYPE_F16)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding supports F32/F16 tables only");
    }
    return GD_OK;
}

static gd_status embedding_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *table = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(numel / p.dim) : 0;
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &p.dtype);
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_embedding = {
    .kind = _GD_OP_EMBEDDING,
    .name = "embedding",
    .support = embedding_support,
    .encode = embedding_encode,
};
