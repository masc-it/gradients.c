#import "../../backends/metal/metal_op.h"

static gd_status reduce_to_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *target = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_to_params p;
    int i = 0;

    memset(&p, 0, sizeof(p));
    p.target_ndim = target->ndim;
    p.target_numel = (int)_gd_metal_desc_numel(target);
    p.go_ndim = go->ndim;
    p.go_numel = (int)_gd_metal_desc_numel(go);
    for (i = 0; i < target->ndim; ++i) {
        p.target_sizes[i] = (int)target->sizes[i];
    }
    for (i = 0; i < go->ndim; ++i) {
        p.go_sizes[i] = (int)go->sizes[i];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.target_numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)p.target_numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_reduce_to = {
    .kind = _GD_OP_REDUCE_TO,
    .name = "reduce_to",
    .encode = reduce_to_encode,
};
