#import "../../backends/metal/metal_op.h"

static gd_status scale_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary(ctx, ctx->node->attrs.scale);
}

const _gd_metal_op _gd_metal_op_scale = {
    .kind = _GD_OP_SCALE,
    .name = "scale",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = scale_encode,
};
