#include <gradients/gradients.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_rms_norm failed: %s (%s:%d)\n", (msg),     \
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

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static gd_memory_config rms_config(size_t bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(bytes * 8U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(bytes * 24U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_rms_norm failed: %s have=%.8f want=%.8f tol=%.8f\n",
                msg,
                (double)have,
                (double)want,
                (double)tol);
        exit(1);
    }
}

static void rms_reference(const float *x,
                          const float *weight,
                          const float *grad_out,
                          uint32_t rows,
                          uint32_t cols,
                          float eps,
                          float *out,
                          float *dx,
                          float *dw)
{
    uint32_t r;
    uint32_t c;
    if (dw != NULL) {
        for (c = 0U; c < cols; ++c) {
            dw[c] = 0.0f;
        }
    }
    for (r = 0U; r < rows; ++r) {
        float ss = 0.0f;
        float inv;
        float a = 0.0f;
        for (c = 0U; c < cols; ++c) {
            float xv = x[(size_t)r * cols + c];
            ss += xv * xv;
        }
        inv = 1.0f / sqrtf(ss / (float)cols + eps);
        if (out != NULL) {
            for (c = 0U; c < cols; ++c) {
                size_t idx = (size_t)r * cols + c;
                out[idx] = x[idx] * weight[c] * inv;
            }
        }
        if (grad_out != NULL) {
            for (c = 0U; c < cols; ++c) {
                size_t idx = (size_t)r * cols + c;
                a += grad_out[idx] * weight[c] * x[idx];
                if (dw != NULL) {
                    dw[c] += grad_out[idx] * x[idx] * inv;
                }
            }
            if (dx != NULL) {
                float inv3_over_cols = inv * inv * inv / (float)cols;
                for (c = 0U; c < cols; ++c) {
                    size_t idx = (size_t)r * cols + c;
                    dx[idx] = inv * grad_out[idx] * weight[c] - x[idx] * inv3_over_cols * a;
                }
            }
        }
    }
}

static void test_rms_norm_f32_forward_backward(void)
{
    const int64_t x_shape[2] = {2, 4};
    const int64_t w_shape[1] = {4};
    const float eps = 1.0e-5f;
    const float x_data[8] = {0.5f, -1.0f, 2.0f, 3.0f, -2.0f, 1.5f, 0.25f, -0.75f};
    const float w_data[4] = {1.0f, 0.5f, -1.5f, 2.0f};
    const float g_data[8] = {0.25f, -0.5f, 0.75f, 1.0f, -1.25f, 0.5f, 0.125f, -0.25f};
    float want_y[8];
    float want_dx[8];
    float want_dw[4];
    float got[8];
    gd_memory_config cfg = rms_config(sizeof(x_data) * 8U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    uint32_t i;
    rms_reference(x_data, w_data, g_data, 2U, 4U, eps, want_y, want_dx, want_dw);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), x_data, 8U, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(1U, w_shape), w_data, 4U, false, &w));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), g_data, 8U, false, &g));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_rms_norm(ctx, &x, &w, eps, &y));
    CHECK_OK(gd_rms_norm_backward(ctx, &x, &w, &g, eps, &dx, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, 8U));
    for (i = 0U; i < 8U; ++i) {
        check_close(got[i], want_y[i], 2.0e-5f, "f32 rms_norm forward");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 8U));
    for (i = 0U; i < 8U; ++i) {
        check_close(got[i], want_dx[i], 4.0e-5f, "f32 rms_norm dx");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got, 4U));
    for (i = 0U; i < 4U; ++i) {
        check_close(got[i], want_dw[i], 4.0e-5f, "f32 rms_norm dweight");
    }
    gd_context_destroy(ctx);
}

static void test_rms_norm_f16_autograd(void)
{
    const int64_t x_shape[3] = {2, 3, 5};
    const int64_t w_shape[1] = {5};
    const float eps = 1.0e-5f;
    float x_data[30];
    float g_data[30];
    const float w_data[5] = {0.75f, -1.25f, 0.5f, 1.5f, -0.25f};
    float want_dx[30];
    float want_dw[5];
    float got[30];
    gd_memory_config cfg = rms_config(sizeof(x_data) * 12U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    uint32_t i;
    for (i = 0U; i < 30U; ++i) {
        x_data[i] = ((float)((int)(i % 11U) - 5)) * 0.2f;
        g_data[i] = ((float)((int)(i % 7U) - 3)) * 0.125f;
    }
    rms_reference(x_data, w_data, g_data, 6U, 5U, eps, NULL, want_dx, want_dw);
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, x_shape), x_data, 30U, true, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, w_shape), w_data, 5U, true, &w));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(3U, x_shape), g_data, 30U, false, &g));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_rms_norm(ctx, &x, &w, eps, &y));
    CHECK_OK(gd_backward(ctx, &y, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    CHECK_OK(gd_tensor_grad(ctx, &w, &dw));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, 30U));
    for (i = 0U; i < 30U; ++i) {
        check_close(got[i], want_dx[i], 4.0e-3f, "f16 rms_norm autograd dx");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got, 5U));
    for (i = 0U; i < 5U; ++i) {
        check_close(got[i], want_dw[i], 6.0e-3f, "f16 rms_norm autograd dweight");
    }
    gd_context_destroy(ctx);
}

static void test_rms_norm_validation(void)
{
    const int64_t x_shape[2] = {2, 4};
    const int64_t bad_w_shape[1] = {3};
    const int64_t good_w_shape[1] = {4};
    const float x_data[8] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    const float w_data[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    gd_memory_config cfg = rms_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w_bad;
    gd_tensor w_good;
    gd_tensor y;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(2U, x_shape), x_data, 8U, false, &x));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32,
                                gd_shape_make(1U, bad_w_shape), w_data, 3U, false, &w_bad));
    CHECK_OK(gd_tensor_from_f32(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                                gd_shape_make(1U, good_w_shape), w_data, 4U, false, &w_good));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_rms_norm(ctx, &x, &w_bad, 1.0e-5f, &y), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_rms_norm(ctx, &x, &w_good, 1.0e-5f, &y), GD_ERR_UNSUPPORTED);
    CHECK_STATUS(gd_rms_norm(ctx, &x, &w_bad, 0.0f, &y), GD_ERR_INVALID_ARGUMENT);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_rms_norm_f32_forward_backward();
    test_rms_norm_f16_autograd();
    test_rms_norm_validation();
    printf("test_rms_norm: ok\n");
    return 0;
}
