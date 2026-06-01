#import "../../backends/metal/metal_op.h"

static gd_status mean_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_reduce(ctx, true);
}

const _gd_metal_op _gd_metal_op_mean = {
    .kind = _GD_OP_MEAN,
    .name = "mean",
    .encode = mean_encode,
};
