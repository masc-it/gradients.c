#import "../../backends/metal/metal_op.h"

static gd_status sum_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_reduce(ctx, false);
}

const _gd_metal_op _gd_metal_op_sum = {
    .kind = _GD_OP_SUM,
    .name = "sum",
    .encode = sum_encode,
};
