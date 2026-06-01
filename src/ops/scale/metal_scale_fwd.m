#import "../../backends/metal/metal_op.h"

static gd_status scale_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary(ctx, ctx->node->attrs.scale);
}

const _gd_metal_op _gd_metal_op_scale = {
    .kind = _GD_OP_SCALE,
    .name = "scale",
    .encode = scale_encode,
};
