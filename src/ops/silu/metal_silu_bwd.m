#import "../../backends/metal/metal_op.h"

static gd_status silu_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_unary_bwd(ctx);
}

const _gd_metal_op _gd_metal_op_silu_bwd = {
    .kind = _GD_OP_SILU_BWD,
    .name = "silu_bwd",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = silu_bwd_encode,
};
