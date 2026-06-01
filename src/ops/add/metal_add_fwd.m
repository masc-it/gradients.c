#import "../../backends/metal/metal_op.h"

static gd_status add_encode(_gd_metal_encode_ctx *ctx)
{
    return _gd_metal_encode_binary(ctx);
}

const _gd_metal_op _gd_metal_op_add = {
    .kind = _GD_OP_ADD,
    .name = "add",
    .encode = add_encode,
};
