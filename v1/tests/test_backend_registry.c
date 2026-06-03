#include "../src/backends/cpu_ref/cpu_op.h"

#include <stdio.h>

#define CHECK_TRUE(expr)                                                           \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_STATUS(expr, expected)                                               \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != (expected)) {                                               \
            fprintf(stderr, "%s got %s expected %s\n", #expr,                    \
                    gd_status_name(status_), gd_status_name(expected));            \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int op_is_pseudo(_gd_op_kind kind)
{
    const _gd_op_def *def = _gd_op_def_for(kind);
    return def != NULL && (def->flags & GD_OPF_PSEUDO) != 0U;
}

static int test_cpu_registry_coverage(void)
{
    int i = 0;

    CHECK_TRUE(_gd_cpu_op_for(_GD_OP_INVALID) == NULL);
    CHECK_TRUE(_gd_cpu_op_for((_gd_op_kind)_GD_OP_COUNT) == NULL);

    for (i = 1; i < _GD_OP_COUNT; ++i) {
        _gd_op_kind kind = (_gd_op_kind)i;
        const _gd_op_def *def = _gd_op_def_for(kind);
        const _gd_cpu_op *op = _gd_cpu_op_for(kind);
        _gd_node node = {0};

        CHECK_TRUE(def != NULL);
        if (op_is_pseudo(kind)) {
            CHECK_TRUE(op == NULL);
            continue;
        }

        CHECK_TRUE(op != NULL);
        CHECK_TRUE(op->kind == kind);
        CHECK_TRUE(op->name != NULL);
        CHECK_TRUE(op->run != NULL);

        node.op = kind;
        node.n_inputs = def->min_inputs;
        node.n_outputs = def->n_outputs;
        CHECK_STATUS(op->support != NULL ? op->support(&node) : _gd_cpu_support_default(&node),
                     GD_OK);
    }
    return 0;
}

static int test_cpu_registry_missing_op_guards(void)
{
    CHECK_TRUE(op_is_pseudo(_GD_OP_BACKWARD));
    CHECK_TRUE(op_is_pseudo(_GD_OP_ZERO_GRAD));
    CHECK_TRUE(_gd_cpu_op_for(_GD_OP_BACKWARD) == NULL);
    CHECK_TRUE(_gd_cpu_op_for(_GD_OP_ZERO_GRAD) == NULL);
    return 0;
}

int main(void)
{
    if (test_cpu_registry_coverage() != 0) {
        return 1;
    }
    if (test_cpu_registry_missing_op_guards() != 0) {
        return 1;
    }
    return 0;
}
