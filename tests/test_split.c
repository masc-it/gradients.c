#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_split failed: %s (%s:%d)\n", (msg),         \
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
                "test_split failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config split_config(size_t bytes)
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

static void test_split_f32_axis1_forward_backward(void)
{
    const int64_t x_shape[2] = {2, 5};
    const int64_t sizes[2] = {2, 3};
    const int64_t g0_shape[2] = {2, 2};
    const int64_t g1_shape[2] = {2, 3};
    const float x_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float g0_data[4] = {11.0f, 12.0f, 16.0f, 17.0f};
    const float g1_data[6] = {13.0f, 14.0f, 15.0f, 18.0f, 19.0f, 20.0f};
    const float want0[4] = {1.0f, 2.0f, 6.0f, 7.0f};
    const float want1[6] = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    const float want_dx[10] = {11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
                               16.0f, 17.0f, 18.0f, 19.0f, 20.0f};
    gd_memory_config cfg = split_config(sizeof(want_dx) * 4U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor outputs[2];
    gd_tensor g0;
    gd_tensor g1;
    gd_tensor dx;
    const gd_tensor *grad_outputs[2];
    float got[10];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), x_data, 10U, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, g0_shape), g0_data, 4U, false, &g0));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, g1_shape), g1_data, 6U, false, &g1));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_split(ctx, &x, sizes, 2U, 1, outputs));
    CHECK(outputs[0].rank == 2U && outputs[0].shape[0] == 2 && outputs[0].shape[1] == 2,
          "axis1 output0 shape");
    CHECK(outputs[1].rank == 2U && outputs[1].shape[0] == 2 && outputs[1].shape[1] == 3,
          "axis1 output1 shape");
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &outputs[0], got, 4U));
    for (i = 0U; i < 4U; ++i) {
        check_close(got[i], want0[i], 1.0e-6f, "f32 split output0");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &outputs[1], got, 6U));
    for (i = 0U; i < 6U; ++i) {
        check_close(got[i], want1[i], 1.0e-6f, "f32 split output1");
    }

    grad_outputs[0] = &g0;
    grad_outputs[1] = &g1;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_split_backward(ctx, &x, grad_outputs, sizes, 2U, 1, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 10U));
    for (i = 0U; i < 10U; ++i) {
        check_close(got[i], want_dx[i], 1.0e-6f, "f32 split direct dx");
    }
    gd_context_destroy(ctx);
}

static void test_split_f16_negative_axis_and_u8(void)
{
    const int64_t f16_shape[2] = {3, 2};
    const int64_t f16_sizes[2] = {1, 2};
    const int64_t u8_shape[1] = {5};
    const int64_t u8_sizes[2] = {2, 3};
    const float f16_data[6] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    const float f16_want0[2] = {0.5f, 1.5f};
    const float f16_want1[4] = {2.5f, 3.5f, 4.5f, 5.5f};
    const uint8_t u8_data[5] = {1U, 2U, 3U, 4U, 5U};
    const uint8_t u8_want0[2] = {1U, 2U};
    const uint8_t u8_want1[3] = {3U, 4U, 5U};
    gd_memory_config cfg = split_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor ux;
    gd_tensor outputs[2];
    float got_f32[4];
    uint8_t got_u8[3];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(2U, f16_shape), f16_data, 6U, false, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_U8,
                             gd_shape_make(1U, u8_shape), 256U, &ux));
    CHECK_OK(gd_tensor_write(ctx, &ux, u8_data, sizeof(u8_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_split(ctx, &x, f16_sizes, 2U, -2, outputs));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &outputs[0], got_f32, 2U));
    for (i = 0U; i < 2U; ++i) {
        check_close(got_f32[i], f16_want0[i], 1.0e-3f, "f16 split output0");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &outputs[1], got_f32, 4U));
    for (i = 0U; i < 4U; ++i) {
        check_close(got_f32[i], f16_want1[i], 1.0e-3f, "f16 split output1");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_split(ctx, &ux, u8_sizes, 2U, -1, outputs));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &outputs[0], got_u8, sizeof(u8_want0)));
    for (i = 0U; i < 2U; ++i) {
        CHECK(got_u8[i] == u8_want0[i], "u8 split output0");
    }
    CHECK_OK(gd_tensor_read(ctx, &outputs[1], got_u8, sizeof(u8_want1)));
    for (i = 0U; i < 3U; ++i) {
        CHECK(got_u8[i] == u8_want1[i], "u8 split output1");
    }
    gd_context_destroy(ctx);
}

static void test_split_autograd_partial_outputs(void)
{
    const int64_t x_shape[2] = {2, 5};
    const int64_t sizes[3] = {2, 1, 2};
    const float x_data[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    const float want_dx[10] = {1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
                               1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    gd_memory_config cfg = split_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor outputs[3];
    gd_tensor dx;
    const gd_tensor *seed_outputs[2];
    float got[10];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), x_data, 10U, true, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_split(ctx, &x, sizes, 3U, 1, outputs));
    seed_outputs[0] = &outputs[0];
    seed_outputs[1] = &outputs[2];
    CHECK_OK(gd_backward_many(ctx, 2U, seed_outputs, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 10U));
    for (i = 0U; i < 10U; ++i) {
        check_close(got[i], want_dx[i], 1.0e-6f, "split partial-output autograd dx");
    }
    gd_context_destroy(ctx);
}

static void test_split_validation(void)
{
    const int64_t x_shape[2] = {2, 4};
    const int64_t bad_sum[2] = {1, 2};
    const int64_t bad_zero[2] = {2, 0};
    const int64_t ok_sizes[2] = {2, 2};
    const float x_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    gd_memory_config cfg = split_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor outputs[2];
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), x_data, 8U, false, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_split(ctx, &x, bad_sum, 2U, 1, outputs), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_split(ctx, &x, bad_zero, 2U, 1, outputs), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_split(ctx, &x, ok_sizes, 2U, 2, outputs), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_split(ctx, &x, ok_sizes, 0U, 1, outputs), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_split(ctx, &x, ok_sizes, 2U, 1, NULL), GD_ERR_INVALID_ARGUMENT);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_split_f32_axis1_forward_backward();
    test_split_f16_negative_axis_and_u8();
    test_split_autograd_partial_outputs();
    test_split_validation();
    printf("test_split: ok\n");
    return 0;
}
