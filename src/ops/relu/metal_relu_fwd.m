#import "../../backends/metal/metal_op.h"

static gd_status relu_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary(ctx, 0.0F);
}

const _gd_metal_op _gd_metal_op_relu = {
    .kind = _GD_OP_RELU,
    .name = "relu",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = relu_encode,
};
