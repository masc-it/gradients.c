#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_cross_entropy failed: %s (%s:%d)\n", (msg), \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)
#define CHECK_STATUS(expr, status) CHECK((expr) == (status), #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static float f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    int32_t exp = (int32_t)(((uint32_t)bits >> 10) & 0x1fU);
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1;
        }
        mant &= 0x3ffU;
        exp += 1;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

static gd_memory_config ce_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 4U * 1024U * 1024U;
    cfg.state_bytes = 64U * 1024U;
    cfg.scratch_slot_bytes = 8U * 1024U * 1024U;
    cfg.data_slot_bytes = 64U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void create_context_or_skip(const gd_memory_config *cfg, gd_context **out_ctx)
{
    gd_status st = gd_context_create(cfg, out_ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_cross_entropy: skipped (no supported GPU backend)\n");
        exit(0);
    }
    CHECK_OK(st);
}

static void expect_f32(float got, float want, float tol, const char *what)
{
    CHECK(abs_f32(got - want) <= tol, what);
}

static void test_cross_entropy_rejects_f32_logits(void)
{
    enum { N = 2, C = 4 };
    const int64_t logits_shape[2] = {N, C};
    const int64_t target_shape[1] = {N};
    const int32_t labels[N] = {0, 3};
    gd_context *ctx = NULL;
    gd_memory_config cfg = ce_config();
    gd_tensor logits;
    gd_tensor targets;
    gd_tensor loss;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, 2U, logits_shape, 256U, &logits));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, 1U, target_shape, 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &targets, labels, sizeof(labels)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin(ctx, GD_SCOPE_INFER));
    CHECK_STATUS(gd_cross_entropy(ctx, &logits, &targets, &loss), GD_ERR_UNSUPPORTED);
    CHECK_OK(gd_end(ctx));
    gd_context_destroy(ctx);
}

static void test_cross_entropy_f16_zero_logits_small(void)
{
    enum { N = 2, C = 8, COUNT = N * C };
    const int64_t logits_shape[2] = {N, C};
    const int64_t target_shape[1] = {N};
    const int32_t labels[N] = {2, 5};
    const float want_loss = 2.0794415417f; /* log(8). */
    const float non_target = 1.0f / ((float)N * (float)C);
    const float target = (1.0f / (float)C - 1.0f) / (float)N;
    gd_context *ctx = NULL;
    gd_memory_config cfg = ce_config();
    gd_tensor logits;
    gd_tensor targets;
    gd_tensor loss;
    gd_tensor dlogits;
    float got_loss = 0.0f;
    uint16_t got_grad[COUNT];
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, logits_shape, 256U, &logits));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, 1U, target_shape, 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &targets, labels, sizeof(labels)));
    logits.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_cross_entropy(ctx, &logits, &targets, &loss));
    CHECK(loss.rank == 0U && loss.dtype == GD_DTYPE_F32, "cross_entropy scalar f32 output");
    CHECK_OK(gd_tensor_read(ctx, &loss, &got_loss, sizeof(got_loss)));
    expect_f32(got_loss, want_loss, 1.0e-6f, "cross_entropy f16 zero logits loss");
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &logits, &dlogits));
    CHECK(dlogits.dtype == GD_DTYPE_F16, "cross_entropy f16 grad dtype");
    CHECK_OK(gd_tensor_read(ctx, &dlogits, got_grad, sizeof(got_grad)));
    for (i = 0U; i < COUNT; ++i) {
        float want = non_target;
        if ((i == 2U) || (i == (uint32_t)(C + 5))) {
            want = target;
        }
        expect_f32(f16_bits_to_f32(got_grad[i]), want, 1.0e-4f, "cross_entropy f16 grad");
    }
    CHECK_OK(gd_end(ctx));
    gd_context_destroy(ctx);
}

static void test_cross_entropy_f16_direct_backward(void)
{
    enum { N = 2, C = 8, COUNT = N * C };
    const int64_t logits_shape[2] = {N, C};
    const int64_t target_shape[1] = {N};
    const int32_t labels[N] = {2, 5};
    const float non_target = 1.0f / ((float)N * (float)C);
    const float target = (1.0f / (float)C - 1.0f) / (float)N;
    gd_context *ctx = NULL;
    gd_memory_config cfg = ce_config();
    gd_tensor logits;
    gd_tensor targets;
    gd_tensor grad_seed;
    gd_tensor dlogits;
    uint16_t got_grad[COUNT];
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, logits_shape, 256U, &logits));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, 1U, target_shape, 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &targets, labels, sizeof(labels)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_INFER));
    CHECK_OK(gd_tensor_ones(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, 0U, NULL, 256U, &grad_seed));
    CHECK_OK(gd_cross_entropy_backward(ctx, &logits, &targets, &grad_seed, &dlogits, NULL));
    CHECK_OK(gd_tensor_read(ctx, &dlogits, got_grad, sizeof(got_grad)));
    for (i = 0U; i < COUNT; ++i) {
        float want = non_target;
        if ((i == 2U) || (i == (uint32_t)(C + 5))) {
            want = target;
        }
        expect_f32(f16_bits_to_f32(got_grad[i]), want, 1.0e-4f, "cross_entropy direct backward f16 grad");
    }
    CHECK_OK(gd_end(ctx));
    gd_context_destroy(ctx);
}

static void test_cross_entropy_f16_thousands_classes(void)
{
    enum { N = 3, C = 2048, COUNT = N * C };
    const int64_t logits_shape[2] = {N, C};
    const int64_t target_shape[1] = {N};
    const int32_t labels[N] = {0, 1023, 2047};
    const float want_loss = 7.6246190071f; /* log(2048). */
    const float non_target = 1.0f / ((float)N * (float)C);
    const float target = (1.0f / (float)C - 1.0f) / (float)N;
    gd_context *ctx = NULL;
    gd_memory_config cfg = ce_config();
    gd_tensor logits;
    gd_tensor targets;
    gd_tensor loss;
    gd_tensor dlogits;
    float got_loss = 0.0f;
    uint16_t *got_grad;
    create_context_or_skip(&cfg, &ctx);
    got_grad = (uint16_t *)calloc((size_t)COUNT, sizeof(*got_grad));
    CHECK(got_grad != NULL, "gradient host allocation");
    CHECK_OK(gd_tensor_zeros(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, logits_shape, 256U, &logits));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, 1U, target_shape, 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &targets, labels, sizeof(labels)));
    logits.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_cross_entropy(ctx, &logits, &targets, &loss));
    CHECK_OK(gd_tensor_read(ctx, &loss, &got_loss, sizeof(got_loss)));
    expect_f32(got_loss, want_loss, 1.0e-5f, "cross_entropy f16 thousands classes loss");
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &logits, &dlogits));
    CHECK_OK(gd_tensor_read(ctx, &dlogits, got_grad, (size_t)COUNT * sizeof(got_grad[0])));
    expect_f32(f16_bits_to_f32(got_grad[0]), target, 5.0e-4f, "cross_entropy thousands grad target row0");
    expect_f32(f16_bits_to_f32(got_grad[1]), non_target, 2.0e-6f, "cross_entropy thousands grad non-target row0");
    expect_f32(f16_bits_to_f32(got_grad[C + 1023]), target, 5.0e-4f, "cross_entropy thousands grad target row1");
    expect_f32(f16_bits_to_f32(got_grad[(2U * C) + 2047U]), target, 5.0e-4f, "cross_entropy thousands grad target row2");
    expect_f32(f16_bits_to_f32(got_grad[(2U * C) + 17U]), non_target, 2.0e-6f, "cross_entropy thousands grad non-target row2");
    CHECK_OK(gd_end(ctx));

    free(got_grad);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_cross_entropy_rejects_f32_logits();
    test_cross_entropy_f16_zero_logits_small();
    test_cross_entropy_f16_direct_backward();
    test_cross_entropy_f16_thousands_classes();
    printf("test_cross_entropy: ok\n");
    return 0;
}
