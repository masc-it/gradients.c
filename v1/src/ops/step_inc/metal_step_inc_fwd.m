#import "../../backends/metal/metal_op.h"

static gd_status step_inc_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    _gd_metal_dispatch_1d(enc, pso, 1);
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_step_inc = {
    .kind = _GD_OP_STEP_INC,
    .name = "step_inc",
    .encode = step_inc_encode,
};
