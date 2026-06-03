#import "../../backends/metal/metal_op.h"

static gd_status assert_close_unsupported(void)
{
    return _gd_error(GD_ERR_UNSUPPORTED, "metal assert_close is CPU-only debug op in v1");
}

static gd_status assert_close_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    return assert_close_unsupported();
}

static gd_status assert_close_encode(_gd_metal_encode_ctx *ctx)
{
    (void)ctx;
    return assert_close_unsupported();
}

const _gd_metal_op _gd_metal_op_assert_close = {
    .kind = _GD_OP_ASSERT_CLOSE,
    .name = "assert_close",
    .support = assert_close_support,
    .encode = assert_close_encode,
};
