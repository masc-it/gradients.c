#import "../../backends/metal/metal_op.h"

static gd_status mul_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_binary(ctx);
}

const _gd_metal_op _gd_metal_op_mul = {
    .kind = _GD_OP_MUL,
    .name = "mul",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = mul_encode,
};
