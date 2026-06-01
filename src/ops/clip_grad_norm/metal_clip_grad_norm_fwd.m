#import "../../backends/metal/metal_op.h"

static gd_status clip_grad_norm_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    int in_i = 0;
    int64_t total_groups = 0;
    int64_t reduce_groups = 0;

    if (status != GD_OK) {
        return status;
    }
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_clip_norm_reduce");
    ctx->exe->node_pso3[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_clip_norm_scale");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL || ctx->exe->node_pso3[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal clip_grad_norm kernels missing");
    }
    for (in_i = 0; in_i < ctx->node->n_inputs; ++in_i) {
        const gd_tensor_desc *gd = &ctx->graph->values[ctx->node->inputs[in_i]].desc;
        int64_t numel = _gd_metal_desc_numel(gd);
        total_groups += (numel + GD_METAL_CLIP_NORM_CHUNK - 1) /
                        GD_METAL_CLIP_NORM_CHUNK;
    }
    if (total_groups <= 0) {
        return _gd_error(GD_ERR_SHAPE, "clip_grad_norm has no elements");
    }
    reduce_groups = (total_groups + GD_METAL_CLIP_NORM_TG - 1) / GD_METAL_CLIP_NORM_TG;
    ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)(total_groups + reduce_groups + 1) *
                                                 sizeof(float);
    return GD_OK;
}

static gd_status clip_grad_norm_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> partial_pso = ctx->pso;
    id<MTLComputePipelineState> reduce_pso = ctx->pso2;
    id<MTLComputePipelineState> scale_pso = ctx->pso3;
    id<MTLBuffer> scratch = ctx->scratch;

    gd_metal_clip_norm_params p;
    int i = 0;
    int scratch_offset = 0;
    int total_groups = 0;
    int reduce_groups_max = 0;
    int scale_index = 0;
    int src_offset = 0;
    int dst_offset = 0;
    int src_count = 0;

    if (partial_pso == nil || reduce_pso == nil || scale_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal clip_grad_norm pipeline/scratch missing");
    }
    if (node->n_inputs <= 0 || node->n_outputs != 1) {
        return _gd_error(GD_ERR_INTERNAL, "clip_grad_norm expects inputs and one output");
    }
    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        int64_t groups64 = (numel64 + GD_METAL_CLIP_NORM_CHUNK - 1) /
                           GD_METAL_CLIP_NORM_CHUNK;
        if (numel64 > INT_MAX || groups64 > INT_MAX || groups64 <= 0) {
            return _gd_error(GD_ERR_SHAPE, "clip_grad_norm input too large");
        }
        if (total_groups > INT_MAX - (int)groups64) {
            return _gd_error(GD_ERR_SHAPE, "clip_grad_norm has too many groups");
        }
        total_groups += (int)groups64;
    }
    if (total_groups <= 0) {
        return _gd_error(GD_ERR_SHAPE, "clip_grad_norm has no elements");
    }
    reduce_groups_max = (total_groups + GD_METAL_CLIP_NORM_TG - 1) /
                        GD_METAL_CLIP_NORM_TG;
    scale_index = total_groups + reduce_groups_max;

    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        int groups = (int)((numel64 + GD_METAL_CLIP_NORM_CHUNK - 1) /
                           GD_METAL_CLIP_NORM_CHUNK);
        memset(&p, 0, sizeof(p));
        p.numel = (int)numel64;
        p.scratch_offset = scratch_offset;
        p.scale_index = scale_index;
        p.max_norm = node->attrs.scale;
        p.eps = node->attrs.eps;
        [enc setComputePipelineState:partial_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[i]) offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_CLIP_NORM_TG, 1, 1)];
        scratch_offset += groups;
    }

    src_offset = 0;
    dst_offset = total_groups;
    src_count = total_groups;
    for (;;) {
        int groups = (src_count + GD_METAL_CLIP_NORM_TG - 1) / GD_METAL_CLIP_NORM_TG;
        memset(&p, 0, sizeof(p));
        p.numel = src_count;
        p.scratch_offset = src_offset;
        p.dst_offset = dst_offset;
        p.total_groups = groups;
        p.scale_index = scale_index;
        p.finalize = groups == 1 ? 1 : 0;
        p.max_norm = node->attrs.scale;
        p.eps = node->attrs.eps;
        [enc setComputePipelineState:reduce_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_CLIP_NORM_TG, 1, 1)];
        if (groups == 1) {
            break;
        }
        src_count = groups;
        src_offset = dst_offset;
        dst_offset = src_offset == 0 ? total_groups : 0;
    }

    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        memset(&p, 0, sizeof(p));
        p.numel = (int)numel64;
        p.scale_index = scale_index;
        [enc setComputePipelineState:scale_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[i]) offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        _gd_metal_dispatch_1d(enc, scale_pso, (NSUInteger)numel64);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_clip_grad_norm = {
    .kind = _GD_OP_CLIP_GRAD_NORM,
    .name = "clip_grad_norm",
    .plan = clip_grad_norm_plan,
    .encode = clip_grad_norm_encode,
};
