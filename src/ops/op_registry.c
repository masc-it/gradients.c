#include "autograd_impl.h"

/* Generated from src/ops/<op>/autograd_<op>.c in the full capsule flow. */
extern const gd_autograd_rule gd_bwd_rule_matmul;
extern const gd_autograd_rule gd_bwd_rule_linear;

static const gd_autograd_rule *const gd_bwd_rules[GD_OP_COUNT] = {
    [GD_OP_MATMUL] = &gd_bwd_rule_matmul,
    [GD_OP_LINEAR] = &gd_bwd_rule_linear,
};

const gd_autograd_rule *gd_autograd_rule_for(gd_op_kind kind)
{
    if (kind <= GD_OP_INVALID || kind >= GD_OP_COUNT) {
        return NULL;
    }
    return gd_bwd_rules[kind];
}
