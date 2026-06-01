#import "../../backends/metal/metal_op.h"

static gd_status cross_entropy_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal cross_entropy support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'cross_entropy'");
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *logits = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *targets = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *out = &ctx->graph->values[ctx->node->outputs[0]].desc;
        if (targets->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy needs I32 targets in v1");
        }
        if (logits->dtype != GD_DTYPE_F32 && logits->dtype != GD_DTYPE_F16) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy supports F32/F16 logits only");
        }
        if (out->dtype != GD_DTYPE_F32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy outputs F32 loss");
        }
    }
    return GD_OK;
}

static gd_status cross_entropy_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    const gd_tensor_desc *logits = NULL;
    int dim = 0;
    int64_t positions = 0;

    if (status != GD_OK) {
        return status;
    }
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_cross_entropy_reduce");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal cross_entropy reduce kernel missing");
    }
    logits = &ctx->graph->values[ctx->node->inputs[0]].desc;
    dim = ctx->node->attrs.dim;
    if (dim < 0) {
        dim += logits->ndim;
    }
    if (dim >= 0 && dim < logits->ndim && logits->sizes[dim] > 0) {
        positions = _gd_metal_desc_numel(logits) / logits->sizes[dim];
    }
    if (positions > 0) {
        ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)positions * sizeof(float);
    }
    return GD_OK;
}

static gd_status cross_entropy_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> reduce_pso = ctx->pso2;
    id<MTLBuffer> scratch = ctx->scratch;

    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy needs I32 targets in v1");
    }
    _gd_metal_split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    p.dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(logits->dtype, &p.dtype);
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal cross_entropy scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        if (p.positions > 0) {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.positions, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        }
    }

    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    {
        NSUInteger tg = reduce_pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_cross_entropy = {
    .kind = _GD_OP_CROSS_ENTROPY,
    .name = "cross_entropy",
    .support = cross_entropy_support,
    .plan = cross_entropy_plan,
    .encode = cross_entropy_encode,
};
