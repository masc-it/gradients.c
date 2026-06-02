#import "../../backends/metal/metal_op.h"

static gd_status cross_entropy_bwd_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_support_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *targets = &ctx->graph->values[ctx->node->inputs[1]].desc;
        if (targets->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy_bwd needs I32 targets in v1");
        }
    }
    return GD_OK;
}

static gd_status cross_entropy_bwd_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->node->attrs.has_ignore_index) {
        ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state,
                                                                                       "gd_cross_entropy_count_valid");
        if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
            return _gd_error(GD_ERR_BACKEND, "metal cross_entropy count kernel missing");
        }
        ctx->exe->node_scratch_bytes[ctx->node_id] = sizeof(float);
    }
    return GD_OK;
}

static gd_status cross_entropy_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    id<MTLComputePipelineState> count_pso = ctx->pso2;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLBuffer> scratch = ctx->scratch;

    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy_bwd needs I32 targets");
    }
    _gd_metal_split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    p.dtype = GD_METAL_DT_F32;
    p.has_ignore_index = node->attrs.has_ignore_index ? 1 : 0;
    p.ignore_index = node->attrs.ignore_index;

    if (node->attrs.has_ignore_index) {
        if (count_pso == nil || scratch == nil) {
            return _gd_error(GD_ERR_BACKEND, "metal cross_entropy_bwd count scratch missing");
        }
        [enc setComputePipelineState:count_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc setBuffer:(node->attrs.has_ignore_index ? scratch : _gd_metal_value_buffer(exe, node->inputs[2]))
           offset:0
          atIndex:5];
    /* One threadgroup per position; threads cooperate over the class dim (see
     * gd_cross_entropy_bwd). Power-of-two threadgroup width for the reductions. */
    if (p.outer * p.inner > 0) {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(p.outer * p.inner), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_cross_entropy_bwd = {
    .kind = _GD_OP_CROSS_ENTROPY_BWD,
    .name = "cross_entropy_bwd",
    .support = cross_entropy_bwd_support,
    .plan = cross_entropy_bwd_plan,
    .encode = cross_entropy_bwd_encode,
};
