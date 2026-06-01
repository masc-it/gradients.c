#import "../../backends/metal/metal_op.h"

static gd_status powlu_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_powlu_params p;

    p.numel = (int)numel;
    p.m = node->attrs.powlu_m;
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &p.dtype);
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_powlu_bwd = {
    .kind = _GD_OP_POWLU_BWD,
    .name = "powlu_bwd",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = powlu_bwd_encode,
};
