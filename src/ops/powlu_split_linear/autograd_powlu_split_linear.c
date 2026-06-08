#include "../autograd_impl.h"

/* Backward implementation notes:
 * See docs/guides/metal_tips.md before implementing Metal hot paths.
 * Custom backend mode: generated backend stubs are omitted; add custom backend declarations/PSOs manually.
 */
static gd_status gd_powlu_split_linear_autograd_backward(gd_bwd_ctx *bwd,
                                            const gd_tape_node *node)
{
    (void)bwd;
    (void)node;
    return GD_ERR_UNSUPPORTED;
}

const gd_autograd_rule gd_bwd_rule_powlu_split_linear = {
    .kind = GD_OP_POWLU_SPLIT_LINEAR,
    .name = "powlu_split_linear",
    .backward = gd_powlu_split_linear_autograd_backward,
};
