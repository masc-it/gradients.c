#import "../../backends/metal/metal_op.h"

#include <limits.h>

static gd_status copy_encode(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *in_desc = &ctx->exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    gd_metal_unary_params p;
    size_t nbytes = 0U;
    size_t alignment = 0U;
    gd_status status = GD_OK;

    if (ctx->node->n_inputs == 1 && ctx->node->n_outputs == 1 &&
        ctx->exe->values[ctx->node->inputs[0]].storage == ctx->exe->values[ctx->node->outputs[0]].storage) {
        _gd_profile_record_event(ctx->backend->ctx, ctx->backend, _GD_PROFILE_EVENT_COPY_ALIAS,
                                 0U, 0U, 1U);
        return GD_OK;
    }
    if (in_desc->dtype != out_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "metal copy requires matching dtypes");
    }
    status = gd_tensor_desc_nbytes(out_desc, &nbytes, &alignment);
    if (status != GD_OK) {
        return status;
    }
    if (nbytes > (size_t)INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal copy byte count exceeds int32 range");
    }
    p.numel = (int)nbytes;
    p.scale = 0.0F;
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (nbytes > 0U) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)nbytes);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_copy = {
    .kind = _GD_OP_COPY,
    .name = "copy",
    .encode = copy_encode,
};
