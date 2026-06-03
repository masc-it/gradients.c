#import "../../backends/metal/metal_op.h"

static gd_status amp_step_inc_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;

    if (ctx->graph == NULL) {
        return GD_OK;
    }
    if (ctx->graph->values[node->inputs[0]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[1]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_DTYPE, "metal amp_step_inc expects F32 step and I32 found_inf");
    }
    return GD_OK;
}

static gd_status amp_step_inc_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    _gd_metal_dispatch_1d(enc, pso, 1);
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_amp_step_inc = {
    .kind = _GD_OP_AMP_STEP_INC,
    .name = "amp_step_inc",
    .support = amp_step_inc_support,
    .encode = amp_step_inc_encode,
};
