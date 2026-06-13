#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_concat failed: %s (%s:%d)\n", (msg),        \
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

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_concat failed: %s have=%.8f want=%.8f tol=%.8f\n",
                msg,
                (double)have,
                (double)want,
                (double)tol);
        exit(1);
    }
}

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static gd_memory_config concat_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 6U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 12U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void test_concat_f32_axis1_forward_backward(void)
{
    const int64_t a_shape[2] = {2, 2};
    const int64_t b_shape[2] = {2, 3};
    const float a_data[4] = {1.0f, 2.0f, 6.0f, 7.0f};
    const float b_data[6] = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    const float grad_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float want[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                            6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float want_da[4] = {1.0f, 2.0f, 6.0f, 7.0f};
    const float want_db[6] = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    gd_memory_config cfg = concat_config(sizeof(want) * 4U);
    gd_context *ctx = NULL;
    gd_tensor a;
    gd_tensor b;
    gd_tensor grad;
    gd_tensor out;
    gd_tensor grads[2];
    const gd_tensor *inputs[2];
    float got[10];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, a_shape), a_data, 4U, false, &a));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, b_shape), b_data, 6U, false, &b));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, (const int64_t[]){2, 5}), grad_data, 10U, false, &grad));
    CHECK_OK(gd_context_seal_params(ctx));
    inputs[0] = &a;
    inputs[1] = &b;

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_concat(ctx, inputs, 2U, 1, &out));
    CHECK(out.rank == 2U && out.shape[0] == 2 && out.shape[1] == 5, "axis1 output shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &out, got, 10U));
    for (i = 0U; i < 10U; ++i) {
        check_close(got[i], want[i], 1.0e-6f, "f32 concat axis1 forward");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_concat_backward(ctx, &grad, inputs, 2U, 1, grads));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &grads[0], got, 4U));
    for (i = 0U; i < 4U; ++i) {
        check_close(got[i], want_da[i], 1.0e-6f, "f32 concat direct da");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &grads[1], got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], want_db[i], 1.0e-6f, "f32 concat direct db");
    }
    gd_context_destroy(ctx);
}

static void test_concat_f16_axis0_and_u8(void)
{
    const int64_t f16_a_shape[2] = {2, 2};
    const int64_t f16_b_shape[2] = {1, 2};
    const int64_t u8_a_shape[1] = {3};
    const int64_t u8_b_shape[1] = {2};
    const float f16_a[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    const float f16_b[2] = {4.5f, 5.5f};
    const float f16_want[6] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    const uint8_t u8_a[3] = {1U, 2U, 3U};
    const uint8_t u8_b[2] = {4U, 5U};
    const uint8_t u8_want[5] = {1U, 2U, 3U, 4U, 5U};
    gd_memory_config cfg = concat_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor a;
    gd_tensor b;
    gd_tensor ua;
    gd_tensor ub;
    gd_tensor out;
    const gd_tensor *inputs[2];
    float got_f32[6];
    uint8_t got_u8[5];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, f16_a_shape), f16_a, 4U, false, &a));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, f16_b_shape), f16_b, 2U, false, &b));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(1U, u8_a_shape), 256U, &ua));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(1U, u8_b_shape), 256U, &ub));
    CHECK_OK(gd_tensor_write(ctx, &ua, u8_a, sizeof(u8_a)));
    CHECK_OK(gd_tensor_write(ctx, &ub, u8_b, sizeof(u8_b)));
    CHECK_OK(gd_context_seal_params(ctx));

    inputs[0] = &a;
    inputs[1] = &b;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_concat(ctx, inputs, 2U, 0, &out));
    CHECK(out.rank == 2U && out.shape[0] == 3 && out.shape[1] == 2, "f16 axis0 output shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &out, got_f32, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got_f32[i], f16_want[i], 1.0e-3f, "f16 concat axis0 forward");
    }

    inputs[0] = &ua;
    inputs[1] = &ub;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_concat(ctx, inputs, 2U, -1, &out));
    CHECK(out.rank == 1U && out.shape[0] == 5, "u8 output shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &out, got_u8, sizeof(got_u8)));
    for (i = 0U; i < 5U; ++i) {
        CHECK(got_u8[i] == u8_want[i], "u8 concat forward");
    }
    gd_context_destroy(ctx);
}

static void test_concat_rank3_negative_axis(void)
{
    const int64_t a_shape[3] = {2, 3, 1};
    const int64_t b_shape[3] = {2, 3, 2};
    const float a_data[6] = {1.0f, 4.0f, 7.0f, 10.0f, 13.0f, 16.0f};
    const float b_data[12] = {2.0f, 3.0f, 5.0f, 6.0f, 8.0f, 9.0f,
                              11.0f, 12.0f, 14.0f, 15.0f, 17.0f, 18.0f};
    const float want[18] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f};
    gd_memory_config cfg = concat_config(sizeof(want) * 4U);
    gd_context *ctx = NULL;
    gd_tensor a;
    gd_tensor b;
    gd_tensor out;
    const gd_tensor *inputs[2];
    float got[18];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, a_shape), a_data, 6U, false, &a));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(3U, b_shape), b_data, 12U, false, &b));
    CHECK_OK(gd_context_seal_params(ctx));
    inputs[0] = &a;
    inputs[1] = &b;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_concat(ctx, inputs, 2U, -1, &out));
    CHECK(out.rank == 3U && out.shape[0] == 2 && out.shape[1] == 3 && out.shape[2] == 3,
          "rank3 output shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &out, got, 18U));
    for (i = 0U; i < 18U; ++i) {
        check_close(got[i], want[i], 1.0e-6f, "rank3 concat forward");
    }
    gd_context_destroy(ctx);
}

static void test_concat_autograd(void)
{
    const int64_t a_shape[2] = {2, 2};
    const int64_t b_shape[2] = {2, 3};
    const float a_data[4] = {1.0f, 2.0f, 6.0f, 7.0f};
    const float b_data[6] = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    gd_memory_config cfg = concat_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor a;
    gd_tensor b;
    gd_tensor out;
    gd_tensor loss;
    gd_tensor da;
    gd_tensor db;
    const gd_tensor *inputs[2];
    float got[6];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, a_shape), a_data, 4U, true, &a));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, b_shape), b_data, 6U, true, &b));
    CHECK_OK(gd_context_seal_params(ctx));
    inputs[0] = &a;
    inputs[1] = &b;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_concat(ctx, inputs, 2U, 1, &out));
    CHECK_OK(gd_reduce_sum(ctx, &out, &loss));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &a, &da));
    CHECK_OK(gd_tensor_grad(ctx, &b, &db));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &da, got, 4U));
    for (i = 0U; i < 4U; ++i) {
        check_close(got[i], 1.0f, 1.0e-6f, "concat autograd da");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &db, got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], 1.0f, 1.0e-6f, "concat autograd db");
    }
    gd_context_destroy(ctx);
}

int main(void)
{
    test_concat_f32_axis1_forward_backward();
    test_concat_f16_axis0_and_u8();
    test_concat_rank3_negative_axis();
    test_concat_autograd();
    printf("test_concat: ok\n");
    return 0;
}
