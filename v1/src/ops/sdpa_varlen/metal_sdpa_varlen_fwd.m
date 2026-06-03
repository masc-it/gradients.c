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

static void fill_sdpa_varlen_params(gd_metal_sdpa_varlen_params *p,
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

static gd_status sdpa_varlen_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *q = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *k = &ctx->graph->values[ctx->node->inputs[1]].desc;
        const gd_tensor_desc *v = &ctx->graph->values[ctx->node->inputs[2]].desc;
        const gd_tensor_desc *cu = &ctx->graph->values[ctx->node->inputs[3]].desc;
        if (q->dtype != GD_DTYPE_F32 && q->dtype != GD_DTYPE_F16) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen supports F32/F16 q/k/v only");
        }
        if (k->dtype != q->dtype || v->dtype != q->dtype) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen requires matching q/k/v dtype");
        }
        if (cu->dtype != GD_DTYPE_I32) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen requires I32 cu_seqlens");
        }
        if (q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || cu->ndim != 1) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen expects q/k/v [N,H,Dh] and cu [B+1]");
        }
        if (q->sizes[2] > GD_METAL_SDPA_DHT) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen supports head_dim <= 64");
        }
        if (ctx->node->attrs.max_seqlen < 0) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_varlen max_seqlen must be non-negative");
        }
    }
    return GD_OK;
}

static bool sdpa_varlen_use_f16_dh64_prefix_window(const gd_tensor_desc *q,
                                                    const _gd_node *node)
{
    const char *fast = getenv("GD_METAL_SDPA_VARLEN_FAST");
    if (fast != NULL && (strcmp(fast, "0") == 0 || strcmp(fast, "false") == 0 ||
                         strcmp(fast, "FALSE") == 0 || strcmp(fast, "off") == 0 ||
                         strcmp(fast, "OFF") == 0)) {
        return false;
    }
    return q->dtype == GD_DTYPE_F16 && q->sizes[2] == 64 && node->attrs.causal != 0 &&
           node->attrs.prefix_len > 0 && node->attrs.sliding_window > 0;
}

static gd_status sdpa_varlen_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *q = &ctx->graph->values[ctx->node->inputs[0]].desc;
        if (sdpa_varlen_use_f16_dh64_prefix_window(q, ctx->node)) {
            ctx->exe->node_pso2[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_named(
                ctx->state, "gd_sdpa_varlen_prefix_window_lane8_dh64_f16");
            if (ctx->exe->node_pso2[ctx->node_id] == NULL) {
                return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen f16 Dh64 pipeline missing");
            }
        }
    }
    return GD_OK;
}

static gd_status sdpa_varlen_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    id<MTLComputePipelineState> pso = ctx->pso;
    id<MTLComputePipelineState> fast_pso = ctx->pso2;
    const gd_tensor_desc *q = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *k = &exe->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *cu = &exe->graph->values[node->inputs[3]].desc;
    gd_metal_sdpa_varlen_params p;
    NSUInteger groups = 0;

    if (pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen pipeline missing");
    }
    fill_sdpa_varlen_params(&p, q, k, cu, node);
    if (p.Dh > GD_METAL_SDPA_DHT) {
        return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen supports head_dim <= 64");
    }
    if (sdpa_varlen_use_f16_dh64_prefix_window(q, node)) {
        int n_qb = (p.max_seqlen + GD_METAL_SDPA_CAUSAL_QROWS - 1) / GD_METAL_SDPA_CAUSAL_QROWS;
        if (fast_pso == nil) {
            return _gd_error(GD_ERR_BACKEND, "metal sdpa_varlen f16 Dh64 pipeline missing");
        }
        groups = (NSUInteger)(p.B * p.Hq * n_qb);
        if (gd_vlm_debug_metal_enabled()) {
            fprintf(stderr,
                    "vlm_debug_metal node=%d sdpa_varlen_fwd fast=1 tokens=%d B=%d Hq=%d Hkv=%d Dh=%d max_seq=%d n_qb=%d prefix=%d window=%d groups=%llu dtype=%d\n",
                    ctx->node_id, p.total_tokens, p.B, p.Hq, p.Hkv, p.Dh,
                    p.max_seqlen, n_qb, p.prefix_len, p.window,
                    (unsigned long long)groups, p.dtype);
        }
        [enc setComputePipelineState:fast_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        if (groups > 0) {
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_CAUSAL_THREADS, 1, 1)];
        }
        return GD_OK;
    }
    groups = (NSUInteger)(p.B * p.Hq * p.n_qb_max);
    if (gd_vlm_debug_metal_enabled()) {
        fprintf(stderr,
                "vlm_debug_metal node=%d sdpa_varlen_fwd fast=0 tokens=%d B=%d Hq=%d Hkv=%d Dh=%d max_seq=%d n_qb=%d prefix=%d window=%d groups=%llu dtype=%d\n",
                ctx->node_id, p.total_tokens, p.B, p.Hq, p.Hkv, p.Dh,
                p.max_seqlen, p.n_qb_max, p.prefix_len, p.window,
                (unsigned long long)groups, p.dtype);
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (groups > 0) {
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_sdpa_varlen = {
    .kind = _GD_OP_SDPA_VARLEN,
    .name = "sdpa_varlen",
    .support = sdpa_varlen_support,
    .plan = sdpa_varlen_plan,
    .encode = sdpa_varlen_encode,
};
