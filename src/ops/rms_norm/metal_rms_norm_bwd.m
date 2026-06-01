#import "../../backends/metal/metal_op.h"

static gd_status rms_norm_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    /* dx: inputs x(0), weight(1), go(2); output dx (x shape). */
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.rows > 0) {
        _gd_metal_dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_rms_norm_bwd = {
    .kind = _GD_OP_RMS_NORM_BWD,
    .name = "rms_norm_bwd",
    .encode = rms_norm_bwd_encode,
};
