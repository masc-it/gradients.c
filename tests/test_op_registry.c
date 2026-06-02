#include "../src/ops/op_impl.h"

#include <stdio.h>
#include <string.h>

#define CHECK_TRUE(expr)                                                           \
    do {                                                                           \
        if (!(expr)) {                                                             \
            fprintf(stderr, "%s failed\n", #expr);                               \
            return 1;                                                              \
        }                                                                          \
    } while (0)

#define CHECK_STATUS(expr, expected)                                                \
    do {                                                                           \
        gd_status status_ = (expr);                                                \
        if (status_ != (expected)) {                                                \
            fprintf(stderr, "%s got %s expected %s\n", #expr,                    \
                    gd_status_name(status_), gd_status_name(expected));             \
            return 1;                                                              \
        }                                                                          \
    } while (0)

typedef struct expected_op {
    _gd_op_kind kind;
    const char *name;
} expected_op;

static const expected_op g_expected_ops[] = {
    {_GD_OP_ADAMW_STEP, "adamw_step"},
    {_GD_OP_ADD, "add"},
    {_GD_OP_AMP_CLIP_GRAD_NORM, "amp_clip_grad_norm"},
    {_GD_OP_AMP_STEP_INC, "amp_step_inc"},
    {_GD_OP_AMP_UNSCALE_GRAD, "amp_unscale_grad"},
    {_GD_OP_ASSERT_CLOSE, "assert_close"},
    {_GD_OP_ASSERT_FINITE, "assert_finite"},
    {_GD_OP_BACKWARD, "backward"},
    {_GD_OP_CAST, "cast"},
    {_GD_OP_CLIP_GRAD_NORM, "clip_grad_norm"},
    {_GD_OP_CONCAT, "concat"},
    {_GD_OP_COPY, "copy"},
    {_GD_OP_CROSS_ENTROPY_BWD, "cross_entropy_bwd"},
    {_GD_OP_CROSS_ENTROPY, "cross_entropy"},
    {_GD_OP_DROPOUT_BWD, "dropout_bwd"},
    {_GD_OP_DROPOUT, "dropout"},
    {_GD_OP_EMBEDDING_BWD, "embedding_bwd"},
    {_GD_OP_EMBEDDING, "embedding"},
    {_GD_OP_GELU_BWD, "gelu_bwd"},
    {_GD_OP_GELU, "gelu"},
    {_GD_OP_LINEAR, "linear"},
    {_GD_OP_LM_CROSS_ENTROPY_BWD, "lm_cross_entropy_bwd"},
    {_GD_OP_LM_CROSS_ENTROPY, "lm_cross_entropy"},
    {_GD_OP_MATMUL, "matmul"},
    {_GD_OP_MEAN_BWD, "mean_bwd"},
    {_GD_OP_MEAN, "mean"},
    {_GD_OP_MUL, "mul"},
    {_GD_OP_POWLU_BWD, "powlu_bwd"},
    {_GD_OP_POWLU, "powlu"},
    {_GD_OP_REDUCE_TO, "reduce_to"},
    {_GD_OP_RELU_BWD, "relu_bwd"},
    {_GD_OP_RELU, "relu"},
    {_GD_OP_RMS_NORM_BWD, "rms_norm_bwd"},
    {_GD_OP_RMS_NORM, "rms_norm"},
    {_GD_OP_RMS_NORM_WBWD, "rms_norm_wbwd"},
    {_GD_OP_ROPE_BWD, "rope_bwd"},
    {_GD_OP_ROPE, "rope"},
    {_GD_OP_SCALE, "scale"},
    {_GD_OP_SDPA_BWD, "sdpa_bwd"},
    {_GD_OP_SDPA, "sdpa"},
    {_GD_OP_SILU_BWD, "silu_bwd"},
    {_GD_OP_SILU, "silu"},
    {_GD_OP_SOFTMAX_BWD, "softmax_bwd"},
    {_GD_OP_SOFTMAX, "softmax"},
    {_GD_OP_STEP_INC, "step_inc"},
    {_GD_OP_SUM_BWD, "sum_bwd"},
    {_GD_OP_SUM, "sum"},
    {_GD_OP_TRANSPOSE, "transpose"},
    {_GD_OP_ZERO_GRAD, "zero_grad"},
    {_GD_OP_SDPA_VARLEN, "sdpa_varlen"},
    {_GD_OP_SDPA_VARLEN_BWD, "sdpa_varlen_bwd"},
    {_GD_OP_SLICE, "slice"},
    {_GD_OP_SLICE_BWD, "slice_bwd"},
};

static int test_registry_contents(void)
{
    int i = 0;
    int n_expected = (int)(sizeof(g_expected_ops) / sizeof(g_expected_ops[0]));

    CHECK_TRUE(_GD_OP_COUNT == n_expected + 1);
    CHECK_TRUE(_gd_op_def_for(_GD_OP_INVALID) == NULL);
    CHECK_TRUE(strcmp(_gd_op_kind_name(_GD_OP_INVALID), "invalid") == 0);
    CHECK_TRUE(strcmp(_gd_op_kind_name((_gd_op_kind)_GD_OP_COUNT), "unknown") == 0);

    for (i = 0; i < n_expected; ++i) {
        const expected_op *expected = &g_expected_ops[i];
        const _gd_op_def *def = _gd_op_def_for(expected->kind);

        CHECK_TRUE(def != NULL);
        CHECK_TRUE(def->kind == expected->kind);
        CHECK_TRUE(def->name != NULL && def->name[0] != '\0');
        CHECK_TRUE(strcmp(def->name, expected->name) == 0);
        CHECK_TRUE(strcmp(_gd_op_kind_name(expected->kind), expected->name) == 0);
        CHECK_TRUE(def->min_inputs >= 0);
        CHECK_TRUE(def->max_inputs >= def->min_inputs);
        CHECK_TRUE(def->max_inputs <= 256);
        CHECK_TRUE(def->n_outputs >= 0);
        CHECK_TRUE(def->n_outputs <= 256);
        CHECK_TRUE(def->meta != NULL);
    }
    for (i = 1; i < _GD_OP_COUNT; ++i) {
        CHECK_TRUE(_gd_op_def_for((_gd_op_kind)i) != NULL);
    }
    return 0;
}

static int test_registry_helpers(void)
{
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_ADD, 2, 1), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_ADD, 1, 1), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_LINEAR, 2, 1), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_LINEAR, 3, 1), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_LINEAR, 4, 1), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_CONCAT, 1, 1), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_CONCAT, 256, 1), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_CONCAT, 0, 1), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_CONCAT, 257, 1), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_ASSERT_FINITE, 1, 0), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_ASSERT_FINITE, 1, 1), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_SDPA_BWD, 4, 3), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_SDPA_BWD, 5, 3), GD_OK);
    CHECK_STATUS(_gd_op_validate_arity(_GD_OP_INVALID, 0, 0), GD_ERR_INVALID_ARGUMENT);

    CHECK_TRUE(_gd_op_is_differentiable(_GD_OP_ADD));
    CHECK_TRUE(_gd_op_is_differentiable(_GD_OP_CAST));
    CHECK_TRUE(_gd_op_is_differentiable(_GD_OP_COPY));
    CHECK_TRUE(!_gd_op_is_differentiable(_GD_OP_ADAMW_STEP));
    CHECK_TRUE(!_gd_op_is_differentiable(_GD_OP_ASSERT_CLOSE));
    return 0;
}

int main(void)
{
    if (test_registry_contents() != 0) {
        return 1;
    }
    if (test_registry_helpers() != 0) {
        return 1;
    }
    return 0;
}
