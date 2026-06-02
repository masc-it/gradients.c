#import "../../backends/metal/metal_op.h"

static void fill_sdpa_varlen_bwd_params(gd_metal_sdpa_varlen_params *p,
                                        const gd_tensor_desc *q,
                                        const gd_tensor_desc *k,
                                        const gd_tensor_desc *cu,
                                        const _gd_node *node)
{
    p->total_tokens = (int)q->sizes[0];
    p->B = (int)cu->sizes[0] - 1;
    p->Hq = (int)q->sizes[1];
    p->Hkv = (int)k->sizes[1];
    p->Dh = (int)q->sizes[2];
    p->max_seqlen = node->attrs.max_seqlen;
    if (p->max_seqlen <= 0) {
        p->max_seqlen = p->total_tokens;
    }
    p->n_qb_max = (p->max_seqlen + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;
    p->scale = node->attrs.attn_scale;
    p->causal = node->attrs.causal;
    p->window = node->attrs.sliding_window;
    p->prefix_len = node->attrs.prefix_len;
    p->dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(q->dtype, &p->dtype);
    p->n_splits = 1;
}

static bool sdpa_varlen_bwd_use_f16_dh64_prefix_window(const gd_tensor_desc *q,
                                                        const _gd_node *node)
{
    return q->dtype == GD_DTYPE_F16 && q->sizes[2] == 64 && node->attrs.causal != 0 &&
           node->attrs.prefix_len > 0 && node->attrs.sliding_window > 0;
}

static gd_status sdpa_varlen_bwd_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *go = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *q = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *k = &ctx->graph->values[ctx->node->inputs[2]].desc;
        const gd_tensor_desc *v = &ctx->graph->values[ctx->node->inputs[3]].desc;
        const gd_tensor_desc *cu = &ctx->graph->values[ctx->node->inputs[4]].desc;
        if (q->dtype != GD_DTYPE_F32 && q->dtype != GD_DTYPE_F16) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen_bwd supports F32/F16 q/k/v only");
        }
        if (go->dtype != q->dtype || k->dtype != q->dtype || v->dtype != q->dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen_bwd requires matching q/k/v/go dtype");
        }
        if (cu->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen_bwd requires I32 cu_seqlens");
        }
        if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || go->ndim != 3 || cu->ndim != 1) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen_bwd expects 3D q/k/v/go and 1D cu");
        }
        if (q->sizes[2] > GD_METAL_SDPA_DHT) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen_bwd supports head_dim <= 64");
        }
    }
    return GD_OK;
}

static gd_status sdpa_varlen_bwd_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    {
        const gd_tensor_desc *q = &ctx->graph->values[ctx->node->inputs[1]].desc;
        bool fast = sdpa_varlen_bwd_use_f16_dh64_prefix_window(q, ctx->node);
        int max_seqlen = ctx->node->attrs.max_seqlen > 0 ? ctx->node->attrs.max_seqlen
                                                          : (int)q->sizes[0];
        int nsplit = fast ? _gd_metal_sdpa_num_splits(max_seqlen) : 1;
        const char *dkv_name = fast
            ? (nsplit > 1 ? "gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16"
                          : "gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16")
            : "gd_sdpa_varlen_bwd_dkv";
        const char *stats_name = fast ? "gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16"
                                      : "gd_sdpa_varlen_bwd_stats";
        if (fast && nsplit > 1) {
            ctx->exe->node_pso[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(
                ctx->state, "gd_sdpa_varlen_bwd_dkv_reduce_f16");
        }
        ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, dkv_name);
        ctx->exe->node_pso3[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, stats_name);
    }
    if (ctx->exe->node_pso2[ctx->node_id] == NULL || ctx->exe->node_pso3[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen_bwd pipeline missing");
    }
    {
        const gd_tensor_desc *q = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *k = &ctx->graph->values[ctx->node->inputs[2]].desc;
        bool fast = sdpa_varlen_bwd_use_f16_dh64_prefix_window(q, ctx->node);
        int max_seqlen = ctx->node->attrs.max_seqlen > 0 ? ctx->node->attrs.max_seqlen
                                                          : (int)q->sizes[0];
        int nsplit = fast ? _gd_metal_sdpa_num_splits(max_seqlen) : 1;
        int64_t stats = q->sizes[0] * q->sizes[1] * 3;
        int64_t dkv_part = fast && nsplit > 1 ? q->sizes[0] * k->sizes[1] *
                            (int64_t)nsplit * 2 * q->sizes[2] : 0;
        ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)(stats + dkv_part) * sizeof(float);
    }
    return GD_OK;
}

static gd_status sdpa_varlen_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> dq_pso = ctx->pso;
    id<MTLComputePipelineState> dkv_pso = ctx->pso2;
    id<MTLComputePipelineState> stats_pso = ctx->pso3;
    id<MTLBuffer> stats_buf = ctx->scratch;
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[2]].desc;
    const gd_tensor_desc *cu = &exe->graph->values[node->inputs[4]].desc;
    gd_metal_sdpa_varlen_params p;
    NSUInteger q_groups = 0;
    NSUInteger kv_groups = 0;

    if (dq_pso == nil || dkv_pso == nil || stats_pso == nil || stats_buf == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen_bwd plan missing");
    }
    fill_sdpa_varlen_bwd_params(&p, q, k, cu, node);
    if (p.Dh > GD_METAL_SDPA_DHT) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen_bwd supports head_dim <= 64");
    }
    if (sdpa_varlen_bwd_use_f16_dh64_prefix_window(q, node)) {
        int n_qb = (p.max_seqlen + GD_METAL_SDPA_CAUSAL_QROWS - 1) / GD_METAL_SDPA_CAUSAL_QROWS;
        int n_kb = (p.max_seqlen + GD_METAL_SDPA_DKV_WIDE_KEYS - 1) / GD_METAL_SDPA_DKV_WIDE_KEYS;
        int nsplit = _gd_metal_sdpa_num_splits(p.max_seqlen);
        NSUInteger stats_bytes = (NSUInteger)p.total_tokens * (NSUInteger)p.Hq * 3U * sizeof(float);
        p.n_splits = nsplit;
        q_groups = (NSUInteger)(p.B * p.Hq * n_qb);
        kv_groups = (NSUInteger)(p.B * p.Hkv * n_kb);

        [enc setComputePipelineState:stats_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:5];
        [enc setBytes:&p length:sizeof(p) atIndex:6];
        [enc setBuffer:stats_buf offset:0 atIndex:7];
        if (q_groups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(q_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_CAUSAL_THREADS, 1, 1)];
        }

        [enc setComputePipelineState:dkv_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
        if (nsplit > 1) {
            [enc setBuffer:stats_buf offset:stats_bytes atIndex:5];
            [enc setBytes:&p length:sizeof(p) atIndex:6];
            [enc setBuffer:stats_buf offset:0 atIndex:7];
            if (kv_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(kv_groups * (NSUInteger)nsplit, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_DKV_WIDE_THREADS, 1, 1)];
            }

            [enc setComputePipelineState:dq_pso];
            [enc setBuffer:stats_buf offset:stats_bytes atIndex:0];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:1];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            if (p.total_tokens > 0) {
                _gd_metal_dispatch_1d(enc, dq_pso,
                    (NSUInteger)p.total_tokens * (NSUInteger)p.Hkv * GD_METAL_SDPA_DHT);
            }
        } else {
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:5];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:6];
            [enc setBytes:&p length:sizeof(p) atIndex:7];
            [enc setBuffer:stats_buf offset:0 atIndex:8];
            if (kv_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(kv_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_DKV_WIDE_THREADS, 1, 1)];
            }
        }
        return GD_OK;
    }

    q_groups = (NSUInteger)(p.B * p.Hq * p.n_qb_max);
    kv_groups = (NSUInteger)(p.B * p.Hkv * p.n_qb_max);

    [enc setComputePipelineState:stats_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBuffer:stats_buf offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake(q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    [enc setComputePipelineState:dq_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc setBuffer:stats_buf offset:0 atIndex:7];
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake(q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    [enc setComputePipelineState:dkv_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:5];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:6];
    [enc setBytes:&p length:sizeof(p) atIndex:7];
    [enc setBuffer:stats_buf offset:0 atIndex:8];
    if (kv_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake(kv_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_sdpa_varlen_bwd = {
    .kind = _GD_OP_SDPA_VARLEN_BWD,
    .name = "sdpa_varlen_bwd",
    .support = sdpa_varlen_bwd_support,
    .plan = sdpa_varlen_bwd_plan,
    .encode = sdpa_varlen_bwd_encode,
};
