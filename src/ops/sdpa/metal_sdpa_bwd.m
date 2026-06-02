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

static gd_status sdpa_bwd_kernel(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> dq_pso = ctx->pso;
    id<MTLComputePipelineState> dkv_pso = ctx->pso2;
    id<MTLComputePipelineState> stats_pso = ctx->pso3;
    id<MTLBuffer> stats_buf = ctx->scratch;

    /* inputs: go(0), q(1), k(2), v(3); outputs: dq(0), dk(1), dv(2). */
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[2]].desc;
    const gd_tensor_desc *bias = node->attrs.has_bias
                                     ? &exe->graph->values[node->inputs[4]].desc
                                     : NULL;
    int bias_input = node->attrs.has_bias ? node->inputs[4] : node->inputs[1];
    gd_metal_sdpa_params p;
    bool f16 = q->dtype == GD_DTYPE_F16;
    int q_groups = 0;
    int kv_groups = 0;

    if (dkv_pso == nil || stats_pso == nil || stats_buf == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd pipeline/scratch missing");
    }
    fill_sdpa_params(&p, q, k, bias, node);
    /* The tiled backward keeps per-thread q/k/v/dO/acc rows in registers, capped
     * at GD_METAL_SDPA_DHT. Fail loud for larger head_dim (CPU_REF is the
     * correctness fallback) rather than overrun the register arrays. */
    if (p.Dh > GD_METAL_SDPA_DHT) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd supports head_dim <= 64");
    }

    /* Long-context split-K backward: like the forward, each pass is critical-path
     * bound on the heaviest block, so split the scanned dimension across
     * threadgroups into per-split partials, then reduce. stats/dq split the key
     * range (per query block); dkv splits the query range (per key block). */
    {
        bool f16_window_split = f16 && sdpa_is_causal_no_bias(&p) && p.window > 0;
        int nsplit = (f16 && !f16_window_split) ? 1 :
            _gd_metal_sdpa_num_splits(p.Tk > p.Tq ? p.Tk : p.Tq);
        if (nsplit > 1) {
            gd_metal_sdpa_bwd_layout L = _gd_metal_sdpa_bwd_scratch_layout(p.B, p.Hq, p.Hkv, p.Tq, p.Tk,
                                                        p.Dh, nsplit);
            NSUInteger stat_b = (NSUInteger)L.stats_off * sizeof(float);
            NSUInteger spart_b = (NSUInteger)L.stats_part_off * sizeof(float);
            NSUInteger dqpart_b = (NSUInteger)L.dq_part_off * sizeof(float);
            NSUInteger dkvpart_b = (NSUInteger)L.dkv_part_off * sizeof(float);
            int dkv_keys = GD_METAL_SDPA_DKV_KEYS;
            NSUInteger dkv_threads = GD_METAL_SDPA_BQ;
            NSUInteger qsplit_groups = 0;
            NSUInteger ksplit_groups = 0;
            NSUInteger q_rows = (NSUInteger)(p.B * p.Hq * p.Tq);
            NSUInteger kv_rows = (NSUInteger)(p.B * p.Hkv * p.Tk);
            GDMetalState *st = ctx->state;
            id<MTLComputePipelineState> sdq_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats_dq_split");
            id<MTLComputePipelineState> sdqc_pso = _gd_metal_pipeline_named(st, f16 ? "gd_sdpa_bwd_stats_dq_combine_f16" : "gd_sdpa_bwd_stats_dq_combine");
            id<MTLComputePipelineState> dks_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv_split");
            id<MTLComputePipelineState> dkr_pso = _gd_metal_pipeline_named(st, f16 ? "gd_sdpa_bwd_dkv_reduce_f16" : "gd_sdpa_bwd_dkv_reduce");
            bool sdq_lane_split = false;
            NSUInteger sdq_threads = GD_METAL_SDPA_BQ;
            if (sdpa_is_causal_no_bias(&p)) {
                if (p.window > 0) {
                    const char *sdq_window_name = f16 && p.Dh == 64
                        ? "gd_sdpa_bwd_stats_dq_split_causal_window_lane8_dh64_f16"
                        : (f16 ? "gd_sdpa_bwd_stats_dq_split_causal_window_lane8_f16"
                               : "gd_sdpa_bwd_stats_dq_split_causal_window_lane8");
                    const char *dks_window_name = f16 && p.Dh == 64
                        ? "gd_sdpa_bwd_dkv_split_causal_window_k16_dh64_f16"
                        : (f16 ? "gd_sdpa_bwd_dkv_split_causal_window_k16_f16"
                               : "gd_sdpa_bwd_dkv_split_causal_window_k16");
                    id<MTLComputePipelineState> fast_sdq_window =
                        _gd_metal_pipeline_named(st, sdq_window_name);
                    id<MTLComputePipelineState> fast_dks_window =
                        _gd_metal_pipeline_named(st, dks_window_name);
                    if (fast_sdq_window != nil) {
                        sdq_pso = fast_sdq_window;
                        sdq_lane_split = true;
                        sdq_threads = GD_METAL_SDPA_CAUSAL_THREADS;
                    }
                    if (fast_dks_window != nil) {
                        dks_pso = fast_dks_window;
                        dkv_keys = GD_METAL_SDPA_DKV_WIDE_KEYS;
                        dkv_threads = GD_METAL_SDPA_DKV_WIDE_THREADS;
                    }
                } else {
                    id<MTLComputePipelineState> fast_sdq_lane =
                        _gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats_dq_split_causal_lane8");
                    id<MTLComputePipelineState> fast_sdq =
                        _gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats_dq_split_causal");
                    id<MTLComputePipelineState> fast_dks_wide =
                        _gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv_split_causal_k16");
                    id<MTLComputePipelineState> fast_dks =
                        _gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv_split_causal");
                    if (fast_sdq_lane != nil) {
                        sdq_pso = fast_sdq_lane;
                        sdq_lane_split = true;
                        sdq_threads = GD_METAL_SDPA_CAUSAL_THREADS;
                    } else if (fast_sdq != nil) {
                        sdq_pso = fast_sdq;
                    }
                    if (fast_dks_wide != nil) {
                        dks_pso = fast_dks_wide;
                        dkv_keys = GD_METAL_SDPA_DKV_WIDE_KEYS;
                        dkv_threads = GD_METAL_SDPA_DKV_WIDE_THREADS;
                    } else if (fast_dks != nil) {
                        dks_pso = fast_dks;
                    }
                }
            }
            id<MTLBuffer> go_b = _gd_metal_value_buffer(exe, node->inputs[0]);
            id<MTLBuffer> q_b = _gd_metal_value_buffer(exe, node->inputs[1]);
            id<MTLBuffer> k_b = _gd_metal_value_buffer(exe, node->inputs[2]);
            id<MTLBuffer> v_b = _gd_metal_value_buffer(exe, node->inputs[3]);
            id<MTLBuffer> bias_b = _gd_metal_value_buffer(exe, bias_input);
            if (sdq_pso == nil || sdqc_pso == nil || dks_pso == nil || dkr_pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd split-K pipeline missing");
            }
            p.n_splits = nsplit;
            int sdq_qrows = sdq_lane_split ? GD_METAL_SDPA_CAUSAL_QROWS : GD_METAL_SDPA_BQ;
            int sdq_n_qb = (p.Tq + sdq_qrows - 1) / sdq_qrows;
            int n_kb = (p.Tk + dkv_keys - 1) / dkv_keys;
            qsplit_groups = (NSUInteger)(p.B * p.Hq * sdq_n_qb * nsplit);
            ksplit_groups = (NSUInteger)(p.B * p.Hkv * n_kb * nsplit);

            /* Fused stats+dq: per-split (m,l,raw,acc,ksum) -> final stats+dq.
             * This removes the old second key scan in the dq split pass. */
            [enc setComputePipelineState:sdq_pso];
            [enc setBuffer:go_b offset:0 atIndex:0];
            [enc setBuffer:q_b offset:0 atIndex:1];
            [enc setBuffer:k_b offset:0 atIndex:2];
            [enc setBuffer:v_b offset:0 atIndex:3];
            [enc setBuffer:bias_b offset:0 atIndex:4];
            [enc setBuffer:stats_buf offset:spart_b atIndex:5];
            [enc setBuffer:stats_buf offset:dqpart_b atIndex:6];
            [enc setBytes:&p length:sizeof(p) atIndex:7];
            if (qsplit_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(qsplit_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(sdq_threads, 1, 1)];
            }
            [enc setComputePipelineState:sdqc_pso];
            [enc setBuffer:stats_buf offset:spart_b atIndex:0];
            [enc setBuffer:stats_buf offset:dqpart_b atIndex:1];
            [enc setBuffer:stats_buf offset:stat_b atIndex:2];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
            [enc setBytes:&p length:sizeof(p) atIndex:4];
            if (q_rows > 0) {
                _gd_metal_dispatch_1d(enc, sdqc_pso, q_rows * (NSUInteger)p.Dh);
            }

            /* dk/dv: per-split partial sums -> reduce. */
            [enc setComputePipelineState:dks_pso];
            [enc setBuffer:go_b offset:0 atIndex:0];
            [enc setBuffer:q_b offset:0 atIndex:1];
            [enc setBuffer:k_b offset:0 atIndex:2];
            [enc setBuffer:v_b offset:0 atIndex:3];
            [enc setBuffer:bias_b offset:0 atIndex:4];
            [enc setBuffer:stats_buf offset:dkvpart_b atIndex:5];
            [enc setBytes:&p length:sizeof(p) atIndex:6];
            [enc setBuffer:stats_buf offset:stat_b atIndex:7];
            if (ksplit_groups > 0) {
                [enc dispatchThreadgroups:MTLSizeMake(ksplit_groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(dkv_threads, 1, 1)];
            }
            [enc setComputePipelineState:dkr_pso];
            [enc setBuffer:stats_buf offset:dkvpart_b atIndex:0];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:1];
            [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:2];
            [enc setBytes:&p length:sizeof(p) atIndex:3];
            if (kv_rows > 0) {
                _gd_metal_dispatch_1d(enc, dkr_pso, kv_rows * (NSUInteger)p.Dh);
            }
            return GD_OK;
        }
    }

    /* Threadgroups: query blocks for stats/dq, key blocks for dk/dv; both use
     * GD_METAL_SDPA_BQ threads per group. */
    q_groups = p.B * p.Hq * ((p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ);
    kv_groups = p.B * p.Hkv * ((p.Tk + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ);

    /* Pass 1: per-query softmax stats (m, l, D) into scratch. */
    [enc setComputePipelineState:stats_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:stats_buf offset:0 atIndex:5];                          /* stats */
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    /* Pass 2: dq (one pass over keys using stats). */
    [enc setComputePipelineState:dq_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:5]; /* dq */
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc setBuffer:stats_buf offset:0 atIndex:7];                          /* stats */
    if (q_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)q_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }

    /* Pass 3: dk/dv (one pass over queries per kv slot using stats). */
    [enc setComputePipelineState:dkv_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0]; /* go */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1]; /* q */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2]; /* k */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3]; /* v */
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:4];      /* bias */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:5]; /* dk */
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[2]) offset:0 atIndex:6]; /* dv */
    [enc setBytes:&p length:sizeof(p) atIndex:7];
    [enc setBuffer:stats_buf offset:0 atIndex:8];                          /* stats */
    if (kv_groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)kv_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }
    return GD_OK;
}

static gd_status sdpa_bwd_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_sdpa_bwd_dkv");
    ctx->exe->node_pso3[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(ctx->state, "gd_sdpa_bwd_stats");
    if (ctx->exe->node_pso2[ctx->node_id] == NULL || ctx->exe->node_pso3[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd pipeline missing");
    }
    {
        const gd_tensor_desc *qd = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *kd = &ctx->graph->values[ctx->node->inputs[2]].desc;
        int B = (int)qd->sizes[0];
        int Tq = (int)qd->sizes[1];
        int Hq = (int)qd->sizes[2];
        int Dh = (int)qd->sizes[3];
        int Tk = (int)kd->sizes[1];
        int Hkv = (int)kd->sizes[2];
        bool f16_window_split = qd->dtype == GD_DTYPE_F16 && ctx->node->attrs.causal &&
                                ctx->node->attrs.prefix_len == 0 &&
                                !ctx->node->attrs.has_bias &&
                                ctx->node->attrs.sliding_window > 0;
        int nsplit = (qd->dtype == GD_DTYPE_F16 && !f16_window_split) ? 1 :
            _gd_metal_sdpa_num_splits(Tk > Tq ? Tk : Tq);
        int64_t nfloats = Dh <= GD_METAL_SDPA_DHT && nsplit > 1
                              ? _gd_metal_sdpa_bwd_scratch_layout(B, Hq, Hkv, Tq, Tk, Dh, nsplit).total
                              : (int64_t)B * Hq * Tq * 3;
        ctx->exe->node_scratch_bytes[ctx->node_id] = (size_t)nfloats * sizeof(float);
    }
    return GD_OK;
}

static gd_status sdpa_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return sdpa_bwd_kernel(ctx);
}

const _gd_metal_op _gd_metal_op_sdpa_bwd = {
    .kind = _GD_OP_SDPA_BWD,
    .name = "sdpa_bwd",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .plan = sdpa_bwd_plan,
    .encode = sdpa_bwd_encode,
};
