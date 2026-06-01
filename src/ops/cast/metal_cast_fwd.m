#import "../../backends/metal/metal_op.h"

static gd_status cast_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    int dtype_code = 0;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal cast support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'cast'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
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
    bool use_x4 = false;

    params.numel = (int)numel;
    status = _gd_metal_dtype_code(in_desc->dtype, &params.src_dtype);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_metal_dtype_code(out_desc->dtype, &params.dst_dtype);
    if (status != GD_OK) {
        return status;
    }
    if (in_desc->dtype == GD_DTYPE_F16 && out_desc->dtype == GD_DTYPE_F32) {
        pso = _gd_metal_pipeline_named(ctx->state, "gd_cast_f16_to_f32_x4");
        if (pso == nil) {
            return _gd_error(GD_ERR_BACKEND, "metal cast F16->F32 pipeline missing");
        }
        use_x4 = true;
    } else if (in_desc->dtype == GD_DTYPE_F32 && out_desc->dtype == GD_DTYPE_F16) {
        pso = _gd_metal_pipeline_named(ctx->state, "gd_cast_f32_to_f16_x4");
        if (pso == nil) {
            return _gd_error(GD_ERR_BACKEND, "metal cast F32->F16 pipeline missing");
        }
        use_x4 = true;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        NSUInteger work_items = use_x4 ? (NSUInteger)((numel + 3) / 4) : (NSUInteger)numel;
        _gd_metal_dispatch_1d(enc, pso, work_items);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_cast = {
    .kind = _GD_OP_CAST,
    .name = "cast",
    .support = cast_support,
    .encode = cast_encode,
};
