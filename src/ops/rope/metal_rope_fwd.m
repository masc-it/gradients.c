#import "../../backends/metal/metal_op.h"

static gd_status rope_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *x = NULL;
    const gd_tensor_desc *pos = NULL;
    const gd_tensor_desc *out = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal rope support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'rope'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    x = &ctx->graph->values[ctx->node->inputs[0]].desc;
    pos = &ctx->graph->values[ctx->node->inputs[1]].desc;
    out = &ctx->graph->values[ctx->node->outputs[0]].desc;
    if (pos->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope needs I32 position ids in v1");
    }
    if (x->dtype != out->dtype || (out->dtype != GD_DTYPE_F32 && out->dtype != GD_DTYPE_F16)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope supports F32/F16 inputs only");
    }
    return GD_OK;
}

static gd_status rope_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    float sin_sign = 1.0F;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *pos = &exe->graph->values[node->inputs[1]].desc;
    int64_t head_dim = out_desc->sizes[out_desc->ndim - 1];
    gd_metal_rope_params p;

    if (pos->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope needs I32 position ids in v1");
    }
    p.head_dim = (int)head_dim;
    p.heads = (int)out_desc->sizes[out_desc->ndim - 2];
    p.rows = p.head_dim > 0 ? (int)(_gd_metal_desc_numel(out_desc) / head_dim) : 0;
    p.n_dims = node->attrs.rope_n_dims;
    p.interleaved = node->attrs.rope_interleaved;
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &p.dtype);
    p.theta = node->attrs.rope_theta;
    p.sin_sign = sin_sign;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.rows > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)p.rows);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_rope = {
    .kind = _GD_OP_ROPE,
    .name = "rope",
    .support = rope_support,
    .encode = rope_encode,
};
