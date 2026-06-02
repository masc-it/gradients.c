#import "../../backends/metal/metal_op.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int gd_vlm_debug_metal_enabled(void)
{
    const char *v = getenv("GD_VLM_DEBUG_METAL");
    if (v == NULL || v[0] == '\0') {
        return 0;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0;
}

static gd_status lm_cross_entropy_bwd_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state == nil || !ctx->state.useMPS) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs GD_METAL_MPS=1");
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *hidden = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *weight = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *targets = &ctx->graph->values[ctx->node->inputs[2]].desc;
        const gd_tensor_desc *go = &ctx->graph->values[ctx->node->inputs[3]].desc;
        const gd_tensor_desc *dhidden = &ctx->graph->values[ctx->node->outputs[0]].desc;
        const gd_tensor_desc *dweight = &ctx->graph->values[ctx->node->outputs[1]].desc;
        if (hidden->dtype != GD_DTYPE_F32 && hidden->dtype != GD_DTYPE_F16) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd supports F32/F16 hidden");
        }
        if (weight->dtype != hidden->dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs matching hidden/weight dtype");
        }
        if (dhidden->dtype != hidden->dtype || dweight->dtype != GD_DTYPE_F32 ||
            go->dtype != GD_DTYPE_F32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd output dtype mismatch");
        }
        if (targets->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal lm_cross_entropy_bwd needs I32 targets");
        }
    }
    return GD_OK;
}

static gd_status lm_cross_entropy_bwd_plan(_gd_metal_plan_ctx *ctx)
{
    const gd_tensor_desc *x_desc = &ctx->graph->values[ctx->node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &ctx->graph->values[ctx->node->inputs[1]].desc;
    int D = (int)x_desc->sizes[x_desc->ndim - 1];
    int V = (int)w_desc->sizes[0];
    int rows = D > 0 ? (int)(_gd_metal_desc_numel(x_desc) / D) : 0;
    int chunk = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
    if (rows > 0 && chunk > 0) {
        const gd_tensor_desc *dx_desc = &ctx->graph->values[ctx->node->outputs[0]].desc;
        size_t bytes = _gd_metal_lmce_bwd_scratch_bytes_for(rows, chunk);
        if (dx_desc->dtype == GD_DTYPE_F16) {
            bytes += (size_t)rows * (size_t)D * sizeof(float);
        }
        if (ctx->node->attrs.has_ignore_index) {
            bytes += sizeof(float);
        }
        ctx->exe->node_scratch_bytes[ctx->node_id] = bytes;
    }
    return GD_OK;
}

static gd_status lm_cross_entropy_bwd_encode(_gd_metal_encode_ctx *ctx)
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
    const gd_tensor_desc *dx_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *dw_desc = &exe->graph->values[node->outputs[1]].desc;
    gd_metal_lmce_scratch_layout L = {0};
    NSUInteger input_elem = (NSUInteger)gd_dtype_sizeof(x_desc->dtype);
    NSUInteger dx_elem = (NSUInteger)gd_dtype_sizeof(dx_desc->dtype);
    NSUInteger dw_elem = (NSUInteger)gd_dtype_sizeof(dw_desc->dtype);
    MPSDataType input_type = x_desc->dtype == GD_DTYPE_F16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    MPSDataType dx_type = dx_desc->dtype == GD_DTYPE_F16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    MPSDataType dw_type = dw_desc->dtype == GD_DTYPE_F16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
    id<MTLBuffer> x_b = _gd_metal_value_buffer(exe, node->inputs[0]);
    id<MTLBuffer> w_b = _gd_metal_value_buffer(exe, node->inputs[1]);
    id<MTLBuffer> t_b = _gd_metal_value_buffer(exe, node->inputs[2]);
    id<MTLBuffer> go_b = _gd_metal_value_buffer(exe, node->inputs[3]);
    id<MTLBuffer> m_b = nil;
    id<MTLBuffer> l_b = nil;
    id<MTLBuffer> dx_b = _gd_metal_value_buffer(exe, node->outputs[0]);
    id<MTLBuffer> dw_b = _gd_metal_value_buffer(exe, node->outputs[1]);
    id<MTLComputePipelineState> dlogits_pso = _gd_metal_pipeline_named(st, "gd_lmce_dlogits_chunk");
    id<MTLComputePipelineState> dx_store_pso = _gd_metal_pipeline_named(st, "gd_lmce_store_dx_f16");
    id<MTLComputePipelineState> count_pso = _gd_metal_pipeline_named(st, "gd_cross_entropy_count_valid");
    NSUInteger dx_tmp_off = (NSUInteger)_gd_metal_lmce_bwd_scratch_bytes_for(rows, chunk_max);
    NSUInteger count_off = dx_tmp_off + (dx_desc->dtype == GD_DTYPE_F16
                                      ? (NSUInteger)rows * (NSUInteger)D * sizeof(float)
                                      : 0U);
    int has_ignore_index = node->attrs.has_ignore_index ? 1 : 0;
    int ignore_index = node->attrs.ignore_index;

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
    if (dx_desc->dtype == GD_DTYPE_F16 && dx_store_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy_bwd F16 dx kernel missing");
    }
    if (has_ignore_index && count_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal lm_cross_entropy_bwd count kernel missing");
    }
    if (gd_vlm_debug_metal_enabled()) {
        fprintf(stderr,
                "vlm_debug_metal node=%d lmce_bwd rows=%d D=%d V=%d chunk=%d x_dtype=%s dx_dtype=%s dw_dtype=%s ignore=%d ignore_index=%d\n",
                ctx->node_id, rows, D, V, chunk_max, gd_dtype_name(x_desc->dtype),
                gd_dtype_name(dx_desc->dtype), gd_dtype_name(dw_desc->dtype),
                has_ignore_index, ignore_index);
    }
    if (has_ignore_index) {
        gd_metal_ce_params cp = {1, 1, V, rows, GD_METAL_DT_F32,
                                 has_ignore_index, ignore_index};
        [*enc setComputePipelineState:count_pso];
        [*enc setBuffer:t_b offset:0 atIndex:0];
        [*enc setBuffer:scratch offset:count_off atIndex:1];
        [*enc setBytes:&cp length:sizeof(cp) atIndex:2];
        [*enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }

    for (int c0 = 0; c0 < V; c0 += chunk_max) {
        int csz = V - c0;
        if (csz > chunk_max) {
            csz = chunk_max;
        }
        MPSMatrix *xm = _gd_metal_mps_matrix_typed(x_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                   (NSUInteger)D * input_elem, input_type);
        MPSMatrix *wm = _gd_metal_mps_matrix_typed(w_b, (NSUInteger)c0 * (NSUInteger)D * input_elem,
                                   (NSUInteger)csz, (NSUInteger)D,
                                   (NSUInteger)D * input_elem, input_type);
        MPSMatrix *lm = _gd_metal_mps_matrix_typed(scratch, L.logits_off, (NSUInteger)rows,
                                   (NSUInteger)csz, (NSUInteger)csz * sizeof(float),
                                   MPSDataTypeFloat32);
        gd_status status = _gd_metal_encode_mps_mm(cmd, enc, st.device, xm, wm, lm,
                                         false, true, (NSUInteger)rows,
                                         (NSUInteger)csz, (NSUInteger)D, 0.0);
        if (status != GD_OK) {
            return status;
        }
        gd_metal_lmce_params p = {rows, D, V, c0, csz, 0,
                                  has_ignore_index, ignore_index};
        [*enc setComputePipelineState:dlogits_pso];
        [*enc setBuffer:scratch offset:L.logits_off atIndex:0];
        [*enc setBuffer:t_b offset:0 atIndex:1];
        [*enc setBuffer:go_b offset:0 atIndex:2];
        [*enc setBuffer:m_b offset:0 atIndex:3];
        [*enc setBuffer:l_b offset:0 atIndex:4];
        [*enc setBytes:&p length:sizeof(p) atIndex:5];
        [*enc setBuffer:(has_ignore_index ? scratch : go_b)
                 offset:(has_ignore_index ? count_off : 0U)
                atIndex:6];
        _gd_metal_dispatch_1d(*enc, dlogits_pso, (NSUInteger)rows * (NSUInteger)csz);

        MPSMatrix *dxm = dx_desc->dtype == GD_DTYPE_F16
                              ? _gd_metal_mps_matrix_typed(scratch, dx_tmp_off,
                                    (NSUInteger)rows, (NSUInteger)D,
                                    (NSUInteger)D * sizeof(float), MPSDataTypeFloat32)
                              : _gd_metal_mps_matrix_typed(dx_b, 0, (NSUInteger)rows, (NSUInteger)D,
                                    (NSUInteger)D * dx_elem, dx_type);
        MPSMatrix *dwm = _gd_metal_mps_matrix_typed(dw_b, (NSUInteger)c0 * (NSUInteger)D * dw_elem,
                                    (NSUInteger)csz, (NSUInteger)D,
                                    (NSUInteger)D * dw_elem, dw_type);
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
    if (dx_desc->dtype == GD_DTYPE_F16) {
        gd_metal_lmce_params p = {rows, D, V, 0, chunk_max, 0,
                                  has_ignore_index, ignore_index};
        [*enc setComputePipelineState:dx_store_pso];
        [*enc setBuffer:scratch offset:dx_tmp_off atIndex:0];
        [*enc setBuffer:dx_b offset:0 atIndex:1];
        [*enc setBytes:&p length:sizeof(p) atIndex:2];
        _gd_metal_dispatch_1d(*enc, dx_store_pso, (NSUInteger)rows * (NSUInteger)D);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_lm_cross_entropy_bwd = {
    .kind = _GD_OP_LM_CROSS_ENTROPY_BWD,
    .name = "lm_cross_entropy_bwd",
    .support = lm_cross_entropy_bwd_support,
    .plan = lm_cross_entropy_bwd_plan,
    .encode = lm_cross_entropy_bwd_encode,
};
