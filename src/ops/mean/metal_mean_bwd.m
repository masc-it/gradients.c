#import "../../backends/metal/metal_op.h"

static gd_status mean_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_sum_bwd(ctx, true);
}

const _gd_metal_op _gd_metal_op_mean_bwd = {
    .kind = _GD_OP_MEAN_BWD,
    .name = "mean_bwd",
    .encode = mean_bwd_encode,
};
