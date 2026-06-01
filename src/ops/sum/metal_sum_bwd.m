#import "../../backends/metal/metal_op.h"

static gd_status sum_bwd_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_sum_bwd(ctx, false);
}

const _gd_metal_op _gd_metal_op_sum_bwd = {
    .kind = _GD_OP_SUM_BWD,
    .name = "sum_bwd",
    .encode = sum_bwd_encode,
};
