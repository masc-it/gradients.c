#import "../../backends/metal/metal_op.h"

static void fill_sdpa_params(gd_metal_sdpa_params *p,
                             const gd_tensor_desc *q_desc,
                             const gd_tensor_desc *k_desc,
                             const gd_tensor_desc *bias_desc,
                             const _gd_node *node)
{
    p->B = (int)q_desc->sizes[0];
    p->Tq = (int)q_desc->sizes[1];
    p->Hq = (int)q_desc->sizes[2];
    p->Dh = (int)q_desc->sizes[3];
    p->Tk = (int)k_desc->sizes[1];
    p->Hkv = (int)k_desc->sizes[2];
    p->scale = node->attrs.attn_scale;
    p->causal = node->attrs.causal;
    p->window = node->attrs.sliding_window;
    p->prefix_len = node->attrs.prefix_len;
    p->has_bias = node->attrs.has_bias ? 1 : 0;
    p->Bb = bias_desc != NULL ? (int)bias_desc->sizes[0] : 1;
    p->Hb = bias_desc != NULL ? (int)bias_desc->sizes[1] : 1;
    p->Tqb = bias_desc != NULL ? (int)bias_desc->sizes[2] : 1;
    p->Tkb = bias_desc != NULL ? (int)bias_desc->sizes[3] : 1;
    p->n_splits = 1;
    p->split_len = p->Tk;
}

static gd_status sdpa_kernel(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> splitk_pso = ctx->pso2;
    id<MTLComputePipelineState> combine_pso = ctx->pso3;
    id<MTLBuffer> scratch = ctx->scratch;

    const gd_tensor_desc *q = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *bias = node->attrs.has_bias
                                     ? &exe->graph->values[node->inputs[3]].desc
                                     : NULL;
    /* bias is always bound; the kernel only reads it when has_bias. Use q as a
     * valid placeholder buffer when absent. */
    int bias_input = node->attrs.has_bias ? node->inputs[3] : node->inputs[0];
    gd_metal_sdpa_params p;
    int n_qb = 0;

    fill_sdpa_params(&p, q, k, bias, node);
    n_qb = (p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;

    /* Split-K (flash-decoding) path for long context: partition the key range
     * across threadgroups to shorten the heaviest query block's critical path,
     * then merge partials. Eligible only when the scratch/pipelines were
     * allocated at compile (Dh <= DHT and >1 split). */
    if (p.Dh <= GD_METAL_SDPA_DHT && scratch != nil &&
        splitk_pso != nil && combine_pso != nil) {
        p.n_splits = _gd_metal_sdpa_num_splits(p.Tk);
        p.split_len = _gd_metal_sdpa_split_len(p.Tk, p.n_splits);
        /* Pass 1: per-split partial online-softmax into scratch. */
        [enc setComputePipelineState:splitk_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
        [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:3];
        [enc setBuffer:scratch offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        NSUInteger sgroups = (NSUInteger)(p.B * p.Hq * n_qb * p.n_splits);
        if (sgroups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(sgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
        }
        /* Pass 2: combine the splits into the output. */
        [enc setComputePipelineState:combine_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        int grid = p.B * p.Hq * p.Tq;
        if (grid > 0) {
            _gd_metal_dispatch_1d(enc, combine_pso, (NSUInteger)grid);
        }
        return GD_OK;
    }

    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];

    if (p.Dh <= GD_METAL_SDPA_DHT) {
        /* Tiled: one threadgroup per (b, hq, query-block) of GD_METAL_SDPA_BQ. */
        NSUInteger groups = (NSUInteger)(p.B * p.Hq * n_qb);
        [enc setComputePipelineState:pso];
        if (groups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
        }
    } else {
        /* Fallback to the per-(b,hq,i) reference kernel for large head_dim. */
        id<MTLComputePipelineState> ref = _gd_metal_pipeline_named(ctx->state, "gd_sdpa");
        int grid = p.B * p.Hq * p.Tq;
        [enc setComputePipelineState:ref];
        if (grid > 0) {
            _gd_metal_dispatch_1d(enc, ref, (NSUInteger)grid);
        }
    }
    return GD_OK;
}

static gd_status sdpa_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    {
        const gd_tensor_desc *qd = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *kd = &ctx->graph->values[ctx->node->inputs[1]].desc;
        int Dh = (int)qd->sizes[3];
        int Tk = (int)kd->sizes[1];
        int nsplit = _gd_metal_sdpa_num_splits(Tk);
        if (Dh <= GD_METAL_SDPA_DHT && nsplit > 1) {
            int64_t npart = qd->sizes[0] * qd->sizes[2] * qd->sizes[1] *
                            nsplit * (Dh + 2);
            ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_sdpa_splitk");
            ctx->exe->node_pso3[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_sdpa_combine");
            if (ctx->exe->node_pso2[ctx->node_id] == NULL || ctx->exe->node_pso3[ctx->node_id] == NULL) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa split-K pipeline missing");
            }
            ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)npart * sizeof(float);
        }
    }
    return GD_OK;
}

static gd_status sdpa_encode(_gd_metal_encode_ctx *ctx)
{
    return sdpa_kernel(ctx);
}

const _gd_metal_op _gd_metal_op_sdpa = {
    .kind = _GD_OP_SDPA,
    .name = "sdpa",
    .plan = sdpa_plan,
    .encode = sdpa_encode,
};
