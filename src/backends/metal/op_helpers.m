#import "metal_op.h"

static bool metal_support_dtype_f32_f16(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16;
}

gd_status _gd_metal_support_f32_f16_same_dtype(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    gd_dtype dtype = GD_DTYPE_INVALID;
    int i = 0;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        char msg[96];
        (void)snprintf(msg, sizeof(msg), "metal has no kernel for op '%s'",
                       _gd_op_kind_name(ctx->node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    if (ctx->node->n_outputs > 0) {
        dtype = ctx->graph->values[ctx->node->outputs[0]].desc.dtype;
    } else if (ctx->node->n_inputs > 0) {
        dtype = ctx->graph->values[ctx->node->inputs[0]].desc.dtype;
    }
    if (!metal_support_dtype_f32_f16(dtype)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal op supports F32/F16 floating tensors only");
    }
    for (i = 0; i < ctx->node->n_inputs; ++i) {
        if (ctx->graph->values[ctx->node->inputs[i]].desc.dtype != dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal op requires matching F32/F16 dtypes");
        }
    }
    for (i = 0; i < ctx->node->n_outputs; ++i) {
        if (ctx->graph->values[ctx->node->outputs[i]].desc.dtype != dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal op requires matching F32/F16 dtypes");
        }
    }
    return GD_OK;
}

void _gd_metal_split_around_dim(const gd_tensor_desc *desc, int dim,
                                int *outer, int *d, int *inner)
{
    int i = 0;

    *outer = 1;
    *inner = 1;
    *d = (int)desc->sizes[dim];
    for (i = 0; i < dim; ++i) {
        *outer *= (int)desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        *inner *= (int)desc->sizes[i];
    }
}

gd_status _gd_metal_encode_binary(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *a_desc = &ctx->exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &ctx->exe->graph->values[node->inputs[1]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_ew_params params;

    _gd_metal_build_ew_params(&params, out_desc, a_desc, b_desc);
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)numel);
    }
    return GD_OK;
}

gd_status _gd_metal_encode_unary(_gd_metal_encode_ctx *ctx, float scale)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_unary_params params;

    params.numel = (int)numel;
    params.scale = scale;
    params.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &params.dtype);
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)numel);
    }
    return GD_OK;
}

gd_status _gd_metal_encode_unary_bwd(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_unary_params p;

    p.numel = (int)numel;
    p.scale = 0.0F;
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(out_desc->dtype, &p.dtype);
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)numel);
    }
    return GD_OK;
}

gd_status _gd_metal_encode_reduce(_gd_metal_encode_ctx *ctx, bool mean)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *x_desc = &ctx->exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_params p;

    _gd_metal_split_around_dim(x_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = mean ? 1 : 0;
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)(p.outer * p.inner));
    }
    return GD_OK;
}

gd_status _gd_metal_encode_softmax(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    _gd_metal_split_around_dim(desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(desc->dtype, &p.dtype);
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)(p.outer * p.inner));
    }
    return GD_OK;
}

gd_status _gd_metal_encode_softmax_bwd(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    _gd_metal_split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.dtype = GD_METAL_DT_F32;
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)(p.outer * p.inner));
    }
    return GD_OK;
}

gd_status _gd_metal_encode_sum_bwd(_gd_metal_encode_ctx *ctx, bool mean)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    gd_metal_reduce_params p;

    _gd_metal_split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = mean ? 1 : 0;
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)(p.outer * p.inner));
    }
    return GD_OK;
}
