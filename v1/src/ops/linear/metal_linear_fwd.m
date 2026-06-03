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

static bool linear_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16;
}

static bool linear_node_uses_f16(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *x_desc = &ctx->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *w_desc = &ctx->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *out_desc = &ctx->graph->values[node->outputs[0]].desc;

    return x_desc->dtype == GD_DTYPE_F16 || w_desc->dtype == GD_DTYPE_F16 ||
           out_desc->dtype == GD_DTYPE_F16;
}

static gd_status linear_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const _gd_node *node = NULL;
    const gd_tensor_desc *x_desc = NULL;
    const gd_tensor_desc *w_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal linear support ctx is NULL");
    }
    node = ctx->node;
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'linear'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    x_desc = &ctx->graph->values[node->inputs[0]].desc;
    w_desc = &ctx->graph->values[node->inputs[1]].desc;
    out_desc = &ctx->graph->values[node->outputs[0]].desc;
    if (!linear_dtype_supported(x_desc->dtype) || !linear_dtype_supported(w_desc->dtype) ||
        !linear_dtype_supported(out_desc->dtype)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal linear supports F32/F16 only");
    }
    if (x_desc->dtype != w_desc->dtype || w_desc->dtype != out_desc->dtype) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal linear requires matching input/output dtypes");
    }
    if (node->attrs.has_bias) {
        const gd_tensor_desc *bias_desc = &ctx->graph->values[node->inputs[2]].desc;
        if (bias_desc->dtype != out_desc->dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal linear bias dtype must match output");
        }
    }
    if (out_desc->dtype == GD_DTYPE_F16 && (ctx->state == nil || !ctx->state.useMPS)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 linear needs GD_METAL_MPS=1");
    }
    return GD_OK;
}

static gd_status linear_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_metal_plan_mps_gemm(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL && linear_node_uses_f16(ctx) &&
        ctx->exe->node_mps[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 linear requires an MPS GEMM plan");
    }
    return GD_OK;
}

static gd_status linear_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    GDMPSGemmPlan *mps = exe->node_mps != NULL
                              ? (__bridge GDMPSGemmPlan *)exe->node_mps[ctx->node_id]
                              : nil;

    /* `linear_plan` prebuilds MPS descriptors for contiguous no-bias GEMMs.
     * Use that path when available; otherwise fall back to the portable Metal
     * kernel for unsupported F32 shapes (bias, offsets, non-contiguous layouts).
     * F16 never uses the portable F32 kernel; planning fails if no MPS plan exists. */
    if (mps != nil) {
        if (gd_vlm_debug_metal_enabled()) {
            const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
            const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
            const gd_tensor_desc *w_desc = &exe->graph->values[node->inputs[1]].desc;
            int in_features = (int)x_desc->sizes[x_desc->ndim - 1];
            int rows = in_features > 0 ? (int)(_gd_metal_desc_numel(x_desc) / in_features) : 0;
            fprintf(stderr,
                    "vlm_debug_metal node=%d linear_mps rows=%d in=%d out=%d x_dtype=%s w_dtype=%s out_dtype=%s trans_b=%d\n",
                    ctx->node_id, rows, in_features,
                    (int)out_desc->sizes[out_desc->ndim - 1], gd_dtype_name(x_desc->dtype),
                    gd_dtype_name(w_desc->dtype), gd_dtype_name(out_desc->dtype),
                    node->attrs.trans_b ? 1 : 0);
        }
        return _gd_metal_encode_mps_gemm(ctx->command_buffer, ctx->encoder, exe, mps);
    }

    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_linear_params p;
    int bias_input = node->attrs.has_bias ? node->inputs[2] : node->inputs[0];

    p.in_features = (int)x_desc->sizes[x_desc->ndim - 1];
    p.out_features = (int)out_desc->sizes[out_desc->ndim - 1];
    p.rows = p.in_features > 0 ? (int)(_gd_metal_desc_numel(x_desc) / p.in_features) : 0;
    p.trans_w = node->attrs.trans_b ? 1 : 0;
    p.has_bias = node->attrs.has_bias ? 1 : 0;

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    /* bias is always bound (placeholder when absent; the kernel guards reads). */
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.out_features > 0 && p.rows > 0) {
        _gd_metal_dispatch_gemm_tiles(enc, (NSUInteger)p.out_features, (NSUInteger)p.rows, 1);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_linear = {
    .kind = _GD_OP_LINEAR,
    .name = "linear",
    .support = linear_support,
    .plan = linear_plan,
    .encode = linear_encode,
};
