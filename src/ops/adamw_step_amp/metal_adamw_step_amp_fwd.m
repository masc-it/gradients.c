#import "../../backends/metal/metal_op.h"

static gd_status adamw_step_amp_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;

    if (ctx->graph == NULL) {
        return GD_OK;
    }
    if (ctx->graph->values[node->inputs[0]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[1]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[2]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[3]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[4]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[5]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_DTYPE, "metal adamw_step_amp expects F32 tensors and I32 found_inf");
    }
    if (node->n_inputs == 7 &&
        ctx->graph->values[node->inputs[6]].desc.dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_DTYPE, "metal adamw_step_amp lr tensor must be F32");
    }
    return GD_OK;
}

static gd_status adamw_step_amp_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(param);
    gd_metal_adamw_params p;

    p.numel = (int)numel;
    p.use_lr_tensor = node->n_inputs == 7 ? 1 : 0;
    p.lr = node->attrs.lr;
    p.lr_scale = node->attrs.scale;
    p.beta1 = node->attrs.beta1;
    p.beta2 = node->attrs.beta2;
    p.eps = node->attrs.eps;
    p.weight_decay = node->attrs.weight_decay;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[5]) offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[node->n_inputs == 7 ? 6 : 4])
            offset:0
           atIndex:7];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_adamw_step_amp = {
    .kind = _GD_OP_ADAMW_STEP_AMP,
    .name = "adamw_step_amp",
    .support = adamw_step_amp_support,
    .encode = adamw_step_amp_encode,
};
