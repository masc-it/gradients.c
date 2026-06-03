#import "../../backends/metal/metal_op.h"

static gd_status softmax_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_softmax_bwd(ctx);
}

const _gd_metal_op _gd_metal_op_softmax_bwd = {
    .kind = _GD_OP_SOFTMAX_BWD,
    .name = "softmax_bwd",
    .encode = softmax_bwd_encode,
};
