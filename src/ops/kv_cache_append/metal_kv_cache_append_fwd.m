#import "../../backends/metal/metal_op.h"

#include <limits.h>

static gd_status kv_cache_append_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const _gd_node *node = NULL;
    const gd_tensor_desc *kc = NULL;
    const gd_tensor_desc *vc = NULL;
    const gd_tensor_desc *pos = NULL;
    const gd_tensor_desc *kn = NULL;
    const gd_tensor_desc *vn = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal kv_cache_append support ctx is NULL");
    }
    node = ctx->node;
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) { return status; }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'kv_cache_append'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    kc = &ctx->graph->values[node->inputs[0]].desc;
    vc = &ctx->graph->values[node->inputs[1]].desc;
    pos = &ctx->graph->values[node->inputs[2]].desc;
    kn = &ctx->graph->values[node->inputs[3]].desc;
    vn = &ctx->graph->values[node->inputs[4]].desc;
    if ((kc->dtype != GD_DTYPE_F32 && kc->dtype != GD_DTYPE_F16) ||
        kc->dtype != vc->dtype || kc->dtype != kn->dtype || kc->dtype != vn->dtype) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal kv_cache_append requires matching F32/F16 tensors");
    }
    if (pos->dtype != GD_DTYPE_I32 || pos->ndim != 0) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal kv_cache_append requires I32 scalar cache_pos");
    }
    if (kc->ndim != 4 || vc->ndim != 4 || kn->ndim != 4 || vn->ndim != 4) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal kv_cache_append requires 4D tensors");
    }
    if (kc->sizes[0] > INT_MAX || kc->sizes[1] > INT_MAX || kn->sizes[1] > INT_MAX ||
        kc->sizes[2] > INT_MAX || kc->sizes[3] > INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal kv_cache_append dimension exceeds int32 range");
    }
    return GD_OK;
}

static gd_status kv_cache_append_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *kc = &ctx->exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *kn = &ctx->exe->graph->values[node->inputs[3]].desc;
    gd_metal_kv_cache_append_params p;
    size_t elem = gd_dtype_sizeof(kc->dtype);
    size_t row_bytes = 0U;
    size_t total = 0U;

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "metal kv_cache_append dtype has unknown size");
    }
    row_bytes = (size_t)(kc->sizes[2] * kc->sizes[3]) * elem;
    total = (size_t)(kn->sizes[0] * kn->sizes[1]) * row_bytes;
    if (row_bytes > (size_t)INT_MAX || total > (size_t)INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal kv_cache_append byte count exceeds int32 range");
    }
    p.B = (int)kc->sizes[0];
    p.Tmax = (int)kc->sizes[1];
    p.Tnew = (int)kn->sizes[1];
    p.row_bytes = (int)row_bytes;

    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (total > 0U) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)total);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_kv_cache_append = {
    .kind = _GD_OP_KV_CACHE_APPEND,
    .name = "kv_cache_append",
    .support = kv_cache_append_support,
    .encode = kv_cache_append_encode,
};
