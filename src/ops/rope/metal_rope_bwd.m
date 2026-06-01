#import "../../backends/metal/metal_op.h"

static gd_status rope_bwd_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_support_default(ctx);

    if (status != GD_OK || ctx->graph == NULL) {
        return status;
    }
    if (ctx->graph->values[ctx->node->inputs[1]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope_bwd needs I32 position ids in v1");
    }
    return GD_OK;
}

static gd_status rope_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    float sin_sign = -1.0F;

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

const _gd_metal_op _gd_metal_op_rope_bwd = {
    .kind = _GD_OP_ROPE_BWD,
    .name = "rope_bwd",
    .support = rope_bwd_support,
    .encode = rope_bwd_encode,
};
