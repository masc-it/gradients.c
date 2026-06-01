#import "../../backends/metal/metal_op.h"

static gd_status lm_cross_entropy_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state == nil || !ctx->state.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs GD_METAL_MPS=1");
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *targets = &ctx->graph->values[ctx->node->inputs[2]].desc;
        if (targets->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy needs I32 targets");
        }
    }
    return GD_OK;
}

static gd_status lm_cross_entropy_plan(_gd_metal_plan_ctx *ctx)
{
    const gd_tensor_desc *x_desc = &ctx->graph->values[ctx->node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &ctx->graph->values[ctx->node->inputs[1]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(_gd_metal_desc_numel(x_desc) / D) : 0;
    int chunk = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    if (rows > 0 && chunk > 0) {
        ctx->exe->node_scratch_bytes[ctx->node_id] = _gd_metal_lmce_fwd_scratch_layout_for(rows, chunk).total;
    }
    return GD_OK;
}

static gd_status lm_cross_entropy_encode(_gd_metal_encode_ctx *ctx)
{
    _gd_backend *self = ctx->backend;
    id<MTLCommandBuffer> cmd = ctx->command_buffer;
    __strong id<MTLComputeCommandEncoder> *enc = ctx->encoder;
    id<MTLBuffer> scratch = ctx->scratch;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

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
        gd_metal_ce_params rp = {1, 1, V, rows, GD_METAL_DT_F32};
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

const _gd_metal_op _gd_metal_op_lm_cross_entropy = {
    .kind = _GD_OP_LM_CROSS_ENTROPY,
    .name = "lm_cross_entropy",
    .support = lm_cross_entropy_support,
    .plan = lm_cross_entropy_plan,
    .encode = lm_cross_entropy_encode,
};
