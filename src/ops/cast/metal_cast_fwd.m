#import "../../backends/metal/metal_op.h"

static gd_status cast_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_support_default(ctx);
    int dtype_code = 0;

    if (status != GD_OK || ctx->graph == NULL) {
        return status;
    }
    status = _gd_metal_dtype_code(ctx->graph->values[ctx->node->inputs[0]].desc.dtype,
                                  &dtype_code);
    if (status != GD_OK) {
        return status;
    }
    return _gd_metal_dtype_code(ctx->graph->values[ctx->node->outputs[0]].desc.dtype,
                                &dtype_code);
}

static gd_status cast_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_cast_params params;
    gd_status status = GD_OK;

    params.numel = (int)numel;
    status = _gd_metal_dtype_code(in_desc->dtype, &params.src_dtype);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_metal_dtype_code(out_desc->dtype, &params.dst_dtype);
    if (status != GD_OK) {
        return status;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_cast = {
    .kind = _GD_OP_CAST,
    .name = "cast",
    .support = cast_support,
    .encode = cast_encode,
};
