#import "../../backends/metal/metal_op.h"

static gd_status amp_refresh_param_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *param = NULL;
    const gd_tensor_desc *master = NULL;
    int dim = 0;

    if (ctx->graph == NULL) {
        return GD_OK;
    }
    param = &ctx->graph->values[node->inputs[0]].desc;
    master = &ctx->graph->values[node->inputs[1]].desc;
    if ((param->dtype != GD_DTYPE_F32 && param->dtype != GD_DTYPE_F16) ||
        master->dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[2]].desc.dtype != GD_DTYPE_I32 ||
        param->ndim != master->ndim) {
        return _gd_error(GD_ERR_DTYPE, "metal amp_refresh_param dtype mismatch");
    }
    for (dim = 0; dim < param->ndim; ++dim) {
        if (param->sizes[dim] != master->sizes[dim]) {
            return _gd_error(GD_ERR_SHAPE, "metal amp_refresh_param shape mismatch");
        }
    }
    return GD_OK;
}

static gd_status amp_refresh_param_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(param);
    gd_metal_amp_params p;
    gd_status status = GD_OK;

    p.numel = (int)numel;
    status = _gd_metal_dtype_code(param->dtype, &p.dtype);
    if (status != GD_OK) {
        return status;
    }
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

const _gd_metal_op _gd_metal_op_amp_refresh_param = {
    .kind = _GD_OP_AMP_REFRESH_PARAM,
    .name = "amp_refresh_param",
    .support = amp_refresh_param_support,
    .encode = amp_refresh_param_encode,
};
