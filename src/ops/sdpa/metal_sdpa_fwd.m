#import "../../backends/metal/metal_op.h"

#include <stdlib.h>

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
    p->dtype = GD_METAL_DT_F32;
    (void)_gd_metal_dtype_code(q_desc->dtype, &p->dtype);
}

static bool sdpa_is_causal_no_bias(const gd_metal_sdpa_params *p)
{
    const char *fast = getenv("GD_METAL_SDPA_CAUSAL_FAST");

    if (fast != NULL && fast[0] == '0') {
        return false;
    }
    return p != NULL && p->causal != 0 && p->prefix_len == 0 && p->has_bias == 0;
}

static gd_status sdpa_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    gd_dtype dtype = GD_DTYPE_INVALID;
    int i = 0;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal sdpa support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'sdpa'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    dtype = ctx->graph->values[ctx->node->outputs[0]].desc.dtype;
    if (dtype != GD_DTYPE_F32 && dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa supports F32/F16 tensors only");
    }
    if (dtype == GD_DTYPE_F16) {
        if (_gd_metal_pipeline_named(ctx->state, "gd_sdpa_tiled_f16") == nil ||
            _gd_metal_pipeline_named(ctx->state, "gd_sdpa_splitk_f16") == nil ||
            _gd_metal_pipeline_named(ctx->state, "gd_sdpa_combine_f16") == nil ||
            _gd_metal_pipeline_named(ctx->state, "gd_sdpa_f16") == nil) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa F16 kernels missing");
        }
    }
    for (i = 0; i < 3; ++i) {
        if (ctx->graph->values[ctx->node->inputs[i]].desc.dtype != dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa requires matching q/k/v dtype");
        }
    }
    if (ctx->node->attrs.has_bias &&
        ctx->graph->values[ctx->node->inputs[3]].desc.dtype != dtype) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa requires matching bias dtype");
    }
    return GD_OK;
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
    bool f16 = q->dtype == GD_DTYPE_F16;
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
        bool lane_split = false;
        if (f16) {
            splitk_pso = _gd_metal_pipeline_named(ctx->state, "gd_sdpa_splitk_f16");
            combine_pso = _gd_metal_pipeline_named(ctx->state, "gd_sdpa_combine_f16");
            if (splitk_pso == nil || combine_pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa F16 split-K pipeline missing");
            }
        } else if (sdpa_is_causal_no_bias(&p)) {
            if (p.window > 0) {
                id<MTLComputePipelineState> window_fast =
                    _gd_metal_pipeline_named(ctx->state,
                                             "gd_sdpa_splitk_causal_window_lane8");
                if (window_fast != nil) {
                    splitk_pso = window_fast;
                    lane_split = true;
                }
            } else {
                id<MTLComputePipelineState> lane_fast = _gd_metal_pipeline_named(ctx->state,
                                                                                "gd_sdpa_splitk_causal_lane8");
                id<MTLComputePipelineState> fast = _gd_metal_pipeline_named(ctx->state,
                                                                            "gd_sdpa_splitk_causal");
                if (lane_fast != nil) {
                    splitk_pso = lane_fast;
                    lane_split = true;
                } else if (fast != nil) {
                    splitk_pso = fast;
                }
            }
        }
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
        int split_q = lane_split ? GD_METAL_SDPA_CAUSAL_QROWS : GD_METAL_SDPA_BQ;
        int split_n_qb = (p.Tq + split_q - 1) / split_q;
        NSUInteger sgroups = (NSUInteger)(p.B * p.Hq * split_n_qb * p.n_splits);
        if (sgroups > 0) {
            NSUInteger split_threads = lane_split ? GD_METAL_SDPA_CAUSAL_THREADS : GD_METAL_SDPA_BQ;
            [enc dispatchThreadgroups:MTLSizeMake(sgroups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(split_threads, 1, 1)];
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
        if (f16) {
            pso = _gd_metal_pipeline_named(ctx->state, "gd_sdpa_tiled_f16");
            if (pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa F16 tiled pipeline missing");
            }
        } else if (sdpa_is_causal_no_bias(&p)) {
            id<MTLComputePipelineState> fast = _gd_metal_pipeline_named(ctx->state, "gd_sdpa_tiled_causal");
            if (fast != nil) {
                pso = fast;
            }
        }
        [enc setComputePipelineState:pso];
        if (groups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
        }
    } else {
        /* Fallback to the per-(b,hq,i) reference kernel for large head_dim. */
        id<MTLComputePipelineState> ref = _gd_metal_pipeline_named(ctx->state,
                                                                   f16 ? "gd_sdpa_f16" : "gd_sdpa");
        int grid = p.B * p.Hq * p.Tq;
        if (ref == nil) {
            return _gd_error(GD_ERR_BACKEND, "metal sdpa reference pipeline missing");
        }
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
            const char *split_name = qd->dtype == GD_DTYPE_F16 ? "gd_sdpa_splitk_f16" : "gd_sdpa_splitk";
            const char *combine_name = qd->dtype == GD_DTYPE_F16 ? "gd_sdpa_combine_f16" : "gd_sdpa_combine";
            ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, split_name);
            ctx->exe->node_pso3[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, combine_name);
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
    .support = sdpa_support,
    .plan = sdpa_plan,
    .encode = sdpa_encode,
};
