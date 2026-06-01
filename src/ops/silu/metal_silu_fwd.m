#import "../../backends/metal/metal_op.h"

static gd_status silu_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary(ctx, 0.0F);
}

const _gd_metal_op _gd_metal_op_silu = {
    .kind = _GD_OP_SILU,
    .name = "silu",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = silu_encode,
};
