#import "../../backends/metal/metal_op.h"

static gd_status gelu_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    bool is_bwd = true;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_gelu_params p;
    int idx = 0;

    p.numel = (int)numel;
    p.tanh_approx = node->attrs.gelu_tanh ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    idx = 1;
    if (is_bwd) {
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        idx = 2;
    }
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:(NSUInteger)idx];
    [enc setBytes:&p length:sizeof(p) atIndex:(NSUInteger)(idx + 1)];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_gelu_bwd = {
    .kind = _GD_OP_GELU_BWD,
    .name = "gelu_bwd",
    .encode = gelu_bwd_encode,
};
