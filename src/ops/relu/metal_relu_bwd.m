#import "../../backends/metal/metal_op.h"

static gd_status relu_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary_bwd(ctx);
}

const _gd_metal_op _gd_metal_op_relu_bwd = {
    .kind = _GD_OP_RELU_BWD,
    .name = "relu_bwd",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = relu_bwd_encode,
};
