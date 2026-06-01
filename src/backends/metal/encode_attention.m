#import "metal_internal.h"

gd_status _gd_metal_encode_lm_cross_entropy(_gd_backend *self,
                                         id<MTLCommandBuffer> cmd,
                                         id<MTLComputeCommandEncoder> *enc,
                                         id<MTLBuffer> scratch,
                                         _gd_executable *exe,
                                         const _gd_node *node)
{
    GDMetalState *st = _gd_metal_state(self);
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[2]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(_gd_metal_desc_numel(x_desc) / D) : 0;
    int chunk_max = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    gd_metal_lmce_scratch_layout L = _gd_metal_lmce_fwd_scratch_layout_for(rows, chunk_max);
    id<MTLBuffer> x_b = _gd_metal_value_buffer(exe, node->inputs[0]);
    id<MTLBuffer> w_b = _gd_metal_value_buffer(exe, node->inputs[1]);
    id<MTLBuffer> t_b = _gd_metal_value_buffer(exe, node->inputs[2]);
    id<MTLBuffer> out_b = _gd_metal_value_buffer(exe, node->outputs[0]);
    id<MTLBuffer> m_b = nil;
    id<MTLBuffer> l_b = nil;
    id<MTLComputePipelineState> fwd_pso = _gd_metal_pipeline_named(st, "gd_lmce_fwd_chunk");
    id<MTLComputePipelineState> loss_pso = _gd_metal_pipeline_named(st, "gd_lmce_loss_rows");
    id<MTLComputePipelineState> reduce_pso = _gd_metal_pipeline_named(st, "gd_cross_entropy_reduce");

    if (!st.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs GD_METAL_MPS=1");
    }
    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs I32 targets");
    }
    if (node->n_outputs != 3) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy expects stats outputs");
    }
    m_b = _gd_metal_value_buffer(exe, node->outputs[1]);
    l_b = _gd_metal_value_buffer(exe, node->outputs[2]);
    if (scratch == nil || fwd_pso == nil || loss_pso == nil || reduce_pso == nil ||
        rows <= 0 || D <= 0 || V <= 0) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy plan missing");
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = _gd_metal_mps_matrix(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *wm = _gd_metal_mps_matrix(w_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *lm = _gd_metal_mps_matrix(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float));
        gd_status status = _gd_metal_encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, c0 == 0 ? 1 : 0};
        [*enc setComputePipelineState:fwd_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:m_b offset:0 atIndex:2];
        [*enc setBuffer:l_b offset:0 atIndex:3];
        [*enc setBuffer:scratch offset:L.target_logit_off atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        _gd_metal_dispatch_1d(*enc, fwd_pso, (NSUInteger)rows);
    }

    {
        gd_metal_lmce_params p = {rows, D, V, 0, chunk_max, 0};
        gd_metal_ce_params rp = {1, 1, V, rows};
        [*enc setComputePipelineState:loss_pso];
        [*enc setBuffer:m_b offset:0 atIndex:0];
        [*enc setBuffer:l_b offset:0 atIndex:1];
        [*enc setBuffer:scratch offset:L.target_logit_off atIndex:2];
        [*enc setBuffer:scratch offset:L.losses_off atIndex:3];
        [*enc setBytes:&p length:sizeof(p) atIndex:4];
        _gd_metal_dispatch_1d(*enc, loss_pso, (NSUInteger)rows);

        [*enc setComputePipelineState:reduce_pso];
        [*enc setBuffer:scratch offset:L.losses_off atIndex:0];
        [*enc setBuffer:out_b offset:0 atIndex:1];
        [*enc setBytes:&rp length:sizeof(rp) atIndex:2];
        [*enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    return GD_OK;
}

gd_status _gd_metal_encode_lm_cross_entropy_bwd(_gd_backend *self,
                                             id<MTLCommandBuffer> cmd,
                                             id<MTLComputeCommandEncoder> *enc,
                                             id<MTLBuffer> scratch,
                                             _gd_executable *exe,
                                             const _gd_node *node)
{
    GDMetalState *st = _gd_metal_state(self);
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[2]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(_gd_metal_desc_numel(x_desc) / D) : 0;
    int chunk_max = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    gd_metal_lmce_scratch_layout L = {0};
    id<MTLBuffer> x_b = _gd_metal_value_buffer(exe, node->inputs[0]);
    id<MTLBuffer> w_b = _gd_metal_value_buffer(exe, node->inputs[1]);
    id<MTLBuffer> t_b = _gd_metal_value_buffer(exe, node->inputs[2]);
    id<MTLBuffer> go_b = _gd_metal_value_buffer(exe, node->inputs[3]);
    id<MTLBuffer> m_b = nil;
    id<MTLBuffer> l_b = nil;
    id<MTLBuffer> dx_b = _gd_metal_value_buffer(exe, node->outputs[0]);
    id<MTLBuffer> dw_b = _gd_metal_value_buffer(exe, node->outputs[1]);
    id<MTLComputePipelineState> dlogits_pso = _gd_metal_pipeline_named(st, "gd_lmce_dlogits_chunk");

    if (!st.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs GD_METAL_MPS=1");
    }
    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs I32 targets");
    }
    if (node->n_inputs != 6) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy_bwd expects stats inputs");
    }
    m_b = _gd_metal_value_buffer(exe, node->inputs[4]);
    l_b = _gd_metal_value_buffer(exe, node->inputs[5]);
    if (scratch == nil || dlogits_pso == nil || rows <= 0 || D <= 0 || V <= 0) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy_bwd plan missing");
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = _gd_metal_mps_matrix(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *wm = _gd_metal_mps_matrix(w_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * sizeof(float));
        MPSMatrix *lm = _gd_metal_mps_matrix(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float));
        gd_status status = _gd_metal_encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, 0};
        [*enc setComputePipelineState:dlogits_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:go_b offset:0 atIndex:2];
        [*enc setBuffer:m_b offset:0 atIndex:3];
        [*enc setBuffer:l_b offset:0 atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        _gd_metal_dispatch_1d(*enc, dlogits_pso, (NSUInteger)rows * (NSUInteger)csz);

        MPSMatrix *dxm = _gd_metal_mps_matrix(dx_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                    (NSUInteger)D * sizeof(float));
        MPSMatrix *dwm = _gd_metal_mps_matrix(dw_b, (NSUInteger)c0 * (NSUInteger)D * sizeof(float),
                                    (NSUInteger)csz, (NSUInteger)D,
                                    (NSUInteger)D * sizeof(float));
        status = _gd_metal_encode_mps_mm(cmd, enc, st.device, lm, wm, dxm,
                               false, false, (NSUInteger)rows, (NSUInteger)D,
                               (NSUInteger)csz, c0 == 0 ? 0.0 : 1.0);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_metal_encode_mps_mm(cmd, enc, st.device, lm, xm, dwm,
                               true, false, (NSUInteger)csz, (NSUInteger)D,
                               (NSUInteger)rows, 0.0);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

/* reduce_to: sum go (broadcast shape) into the target output shape. */

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

void _gd_metal_encode_sdpa(_gd_backend *self,
                        id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        id<MTLComputePipelineState> splitk_pso,
                        id<MTLComputePipelineState> combine_pso,
                        id<MTLBuffer> scratch,
                        _gd_executable *exe,
                        const _gd_node *node)
{
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
        return;
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
        id<MTLComputePipelineState> ref = _gd_metal_pipeline_named(_gd_metal_state(self), "gd_sdpa");
        int grid = p.B * p.Hq * p.Tq;
        [enc setComputePipelineState:ref];
        if (grid > 0) {
            _gd_metal_dispatch_1d(enc, ref, (NSUInteger)grid);
        }
    }
}

/* SDPA backward: dq kernel over query rows, then dk/dv kernel over kv rows. */
gd_status _gd_metal_encode_sdpa_bwd(_gd_backend *self,
                                 id<MTLComputeCommandEncoder> enc,
                                 id<MTLComputePipelineState> dq_pso,
                                 id<MTLComputePipelineState> dkv_pso,
                                 id<MTLComputePipelineState> stats_pso,
                                 id<MTLBuffer> stats_buf,
                                 _gd_executable *exe,
                                 const _gd_node *node)
{
    /* inputs: go(0), q(1), k(2), v(3); outputs: dq(0), dk(1), dv(2). */
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[2]].desc;
    const gd_tensor_desc *bias = node->attrs.has_bias
                                     ? &exe->graph->values[node->inputs[4]].desc
                                     : NULL;
    int bias_input = node->attrs.has_bias ? node->inputs[4] : node->inputs[1];
    gd_metal_sdpa_params p;
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
        int nsplit = _gd_metal_sdpa_num_splits(p.Tk > p.Tq ? p.Tk : p.Tq);
        if (nsplit > 1) {
            gd_metal_sdpa_bwd_layout L = _gd_metal_sdpa_bwd_scratch_layout(p.B, p.Hq, p.Hkv, p.Tq, p.Tk,
                                                        p.Dh, nsplit);
            NSUInteger stat_b = (NSUInteger)L.stats_off * sizeof(float);
            NSUInteger spart_b = (NSUInteger)L.stats_part_off * sizeof(float);
            NSUInteger dqpart_b = (NSUInteger)L.dq_part_off * sizeof(float);
            NSUInteger dkvpart_b = (NSUInteger)L.dkv_part_off * sizeof(float);
            int n_qb = (p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;
            int n_kb = (p.Tk + GD_METAL_SDPA_DKV_KEYS - 1) / GD_METAL_SDPA_DKV_KEYS;
            NSUInteger qsplit_groups = (NSUInteger)(p.B * p.Hq * n_qb * nsplit);
            NSUInteger ksplit_groups = (NSUInteger)(p.B * p.Hkv * n_kb * nsplit);
            NSUInteger q_rows = (NSUInteger)(p.B * p.Hq * p.Tq);
            NSUInteger kv_rows = (NSUInteger)(p.B * p.Hkv * p.Tk);
            GDMetalState *st = _gd_metal_state(self);
            id<MTLComputePipelineState> sdq_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats_dq_split");
            id<MTLComputePipelineState> sdqc_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats_dq_combine");
            id<MTLComputePipelineState> dks_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv_split");
            id<MTLComputePipelineState> dkr_pso = _gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv_reduce");
            id<MTLBuffer> go_b = _gd_metal_value_buffer(exe, node->inputs[0]);
            id<MTLBuffer> q_b = _gd_metal_value_buffer(exe, node->inputs[1]);
            id<MTLBuffer> k_b = _gd_metal_value_buffer(exe, node->inputs[2]);
            id<MTLBuffer> v_b = _gd_metal_value_buffer(exe, node->inputs[3]);
            id<MTLBuffer> bias_b = _gd_metal_value_buffer(exe, bias_input);
            MTLSize tg = MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1);

            if (sdq_pso == nil || sdqc_pso == nil || dks_pso == nil || dkr_pso == nil) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa_bwd split-K pipeline missing");
            }
            p.n_splits = nsplit;

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
                [enc dispatchThreadgroups:MTLSizeMake(qsplit_groups, 1, 1) threadsPerThreadgroup:tg];
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
                [enc dispatchThreadgroups:MTLSizeMake(ksplit_groups, 1, 1) threadsPerThreadgroup:tg];
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
