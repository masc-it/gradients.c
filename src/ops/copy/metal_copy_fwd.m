#import "../../backends/metal/metal_op.h"

static gd_status copy_encode(_gd_metal_encode_ctx *ctx)
{
    if (ctx->node->n_inputs == 1 && ctx->node->n_outputs == 1 &&
        ctx->exe->values[ctx->node->inputs[0]].storage == ctx->exe->values[ctx->node->outputs[0]].storage) {
        _gd_profile_record_event(ctx->backend->ctx, ctx->backend, _GD_PROFILE_EVENT_COPY_ALIAS,
                                 0U, 0U, 1U);
        return GD_OK;
    }
    return _gd_metal_encode_unary(ctx, 0.0F);
}

const _gd_metal_op _gd_metal_op_copy = {
    .kind = _GD_OP_COPY,
    .name = "copy",
    .encode = copy_encode,
};
