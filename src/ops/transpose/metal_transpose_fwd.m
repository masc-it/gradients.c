#import "../../backends/metal/metal_op.h"

static gd_status transpose_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_transpose_params p;
    int64_t stride = 1;
    int k = 0;

    memset(&p, 0, sizeof(p));
    p.ndim = out_desc->ndim;
    p.numel = (int)numel;
    for (k = out_desc->ndim - 1; k >= 0; --k) {
        p.in_strides[k] = (int)stride;
        stride *= in_desc->sizes[k];
    }
    for (k = 0; k < out_desc->ndim; ++k) {
        p.out_sizes[k] = (int)out_desc->sizes[k];
        p.perm[k] = node->attrs.perm[k];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_transpose = {
    .kind = _GD_OP_TRANSPOSE,
    .name = "transpose",
    .encode = transpose_encode,
};
