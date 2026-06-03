#import "../../backends/metal/metal_op.h"

static gd_status softmax_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_softmax(ctx);
}

const _gd_metal_op _gd_metal_op_softmax = {
    .kind = _GD_OP_SOFTMAX,
    .name = "softmax",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = softmax_encode,
};
