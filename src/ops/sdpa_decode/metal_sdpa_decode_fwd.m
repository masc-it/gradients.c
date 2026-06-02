#import "../../backends/metal/metal_op.h"

#include <limits.h>

static gd_status sdpa_decode_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const _gd_node *node = NULL;
    const gd_tensor_desc *q = NULL;
    const gd_tensor_desc *k = NULL;
    const gd_tensor_desc *v = NULL;
    const gd_tensor_desc *pos = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal sdpa_decode support ctx is NULL");
    }
    node = ctx->node;
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) { return status; }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'sdpa_decode'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    q = &ctx->graph->values[node->inputs[0]].desc;
    k = &ctx->graph->values[node->inputs[1]].desc;
    v = &ctx->graph->values[node->inputs[2]].desc;
    pos = &ctx->graph->values[node->inputs[3]].desc;
    if ((q->dtype != GD_DTYPE_F32 && q->dtype != GD_DTYPE_F16) ||
        q->dtype != k->dtype || q->dtype != v->dtype) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_decode requires matching F32/F16 q/k/v");
    }
    if (pos->dtype != GD_DTYPE_I32 || pos->ndim != 0) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_decode requires I32 scalar cache_pos");
    }
    if (q->ndim != 4 || k->ndim != 4 || v->ndim != 4) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_decode requires 4D tensors");
    }
    if (q->sizes[3] > GD_METAL_SDPA_DHT) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_decode supports head_dim <= 64");
    }
    if (q->sizes[0] > INT_MAX || q->sizes[1] > INT_MAX || k->sizes[1] > INT_MAX ||
        q->sizes[2] > INT_MAX || k->sizes[2] > INT_MAX || q->sizes[3] > INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal sdpa_decode dimension exceeds int32 range");
    }
    return GD_OK;
}

static gd_status sdpa_decode_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *q = &ctx->exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *k = &ctx->exe->graph->values[node->inputs[1]].desc;
    gd_metal_sdpa_decode_params p;
    int n_qb = 0;
    int dtype = GD_METAL_DT_F32;
    gd_status status = _gd_metal_dtype_code(q->dtype, &dtype);
    if (status != GD_OK) { return status; }

    p.B = (int)q->sizes[0];
    p.Tq = (int)q->sizes[1];
    p.Tmax = (int)k->sizes[1];
    p.Hq = (int)q->sizes[2];
    p.Hkv = (int)k->sizes[2];
    p.Dh = (int)q->sizes[3];
    p.scale = node->attrs.attn_scale;
    p.window = node->attrs.sliding_window;
    p.prefix_len = node->attrs.prefix_len;
    p.dtype = dtype;
    n_qb = (p.Tq + GD_METAL_SDPA_BQ - 1) / GD_METAL_SDPA_BQ;

    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (n_qb > 0) {
        NSUInteger groups = (NSUInteger)(p.B * p.Hq * n_qb);
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_SDPA_BQ, 1, 1)];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_sdpa_decode = {
    .kind = _GD_OP_SDPA_DECODE,
    .name = "sdpa_decode",
    .support = sdpa_decode_support,
    .encode = sdpa_decode_encode,
};
