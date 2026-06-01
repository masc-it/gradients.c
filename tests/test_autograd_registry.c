#include "../src/autograd/autograd_internal.h"

#include <stdio.h>
#include <string.h>

#define CHECK_TRUE(expr)                                                           \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int test_public_differentiable_ops_have_rules(void)
{
    int i = 0;

    CHECK_TRUE(_gd_bwd_rule_for(_GD_OP_INVALID) == NULL);
    CHECK_TRUE(_gd_bwd_rule_for((_gd_op_kind)_GD_OP_COUNT) == NULL);

    for (i = 1; i < _GD_OP_COUNT; ++i) {
        _gd_op_kind kind = (_gd_op_kind)i;
        const _gd_op_def *def = _gd_op_def_for(kind);
        const _gd_bwd_rule *rule = NULL;

        CHECK_TRUE(def != NULL);
        if ((def->flags & GD_OPF_PUBLIC) == 0u || (def->flags & GD_OPF_DIFF) == 0u) {
            continue;
        }
        rule = _gd_bwd_rule_for(kind);
        CHECK_TRUE(rule != NULL);
        CHECK_TRUE(rule->op == kind);
        CHECK_TRUE(rule->fn != NULL || rule->unsupported_reason != NULL);
        if (kind != _GD_OP_CAST) {
            CHECK_TRUE(rule->fn != NULL);
            CHECK_TRUE(rule->unsupported_reason == NULL);
        }
    }
    return 0;
}

static int test_known_internal_and_unsupported_rules(void)
{
    const _gd_bwd_rule *copy_rule = _gd_bwd_rule_for(_GD_OP_COPY);
    const _gd_bwd_rule *cast_rule = _gd_bwd_rule_for(_GD_OP_CAST);

    CHECK_TRUE(copy_rule != NULL);
    CHECK_TRUE(copy_rule->fn != NULL);
    CHECK_TRUE(copy_rule->unsupported_reason == NULL);

    CHECK_TRUE(cast_rule != NULL);
    CHECK_TRUE(cast_rule->fn == NULL);
    CHECK_TRUE(cast_rule->unsupported_reason != NULL);
    CHECK_TRUE(strcmp(cast_rule->unsupported_reason,
                      "cast backward is not supported in v1") == 0);
    return 0;
}

int main(void)
{
    if (test_public_differentiable_ops_have_rules() != 0) {
        return 1;
    }
    if (test_known_internal_and_unsupported_rules() != 0) {
        return 1;
    }
    return 0;
}
