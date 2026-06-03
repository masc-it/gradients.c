#import "../../backends/metal/metal_op.h"

static gd_status amp_unscale_grad_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;

    if (ctx->graph == NULL) {
        return GD_OK;
    }
    if (ctx->graph->values[node->inputs[0]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[1]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[2]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_DTYPE, "metal amp_unscale_grad expects F32 grad/scale and I32 found_inf");
    }
    return GD_OK;
}

static gd_status amp_unscale_grad_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *grad = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(grad);
    gd_metal_amp_params p = {(int)numel, 0};

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_amp_unscale_grad = {
    .kind = _GD_OP_AMP_UNSCALE_GRAD,
    .name = "amp_unscale_grad",
    .support = amp_unscale_grad_support,
    .encode = amp_unscale_grad_encode,
};
