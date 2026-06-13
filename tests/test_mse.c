#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_mse failed: %s (%s:%d)\n", (msg),           \
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
                "test_mse failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config mse_config(size_t tensor_bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(tensor_bytes * 4U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 8U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static float mse_ref(const float *x, const float *y, size_t count)
{
    float sum = 0.0f;
    size_t i;
    for (i = 0U; i < count; ++i) {
        float d = x[i] - y[i];
        sum += d * d;
    }
    return sum / (float)count;
}

static void test_mse_f32_forward_backward(void)
{
    enum { N = 6 };
    const int64_t shape[2] = {2, 3};
    const float x_data[N] = {1.0f, -2.0f, 0.5f, 3.0f, -4.0f, 0.25f};
    const float y_data[N] = {0.0f, -1.5f, 1.0f, 1.0f, -5.0f, -0.75f};
    const float grad_seed = 1.5f;
    float got_scalar = 0.0f;
    float got_x[N];
    float got_y[N];
    gd_memory_config cfg = mse_config(sizeof(x_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor seed;
    gd_tensor loss;
    gd_tensor dx;
    gd_tensor dy;
    gd_tensor auto_dx;
    gd_tensor auto_dy;
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &y));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(0U, NULL), 256U, &seed));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &y, y_data, sizeof(y_data)));
    CHECK_OK(gd_tensor_write(ctx, &seed, &grad_seed, sizeof(grad_seed)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mse(ctx, &x, &y, &loss));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &loss, &got_scalar, sizeof(got_scalar)));
    check_close(got_scalar, mse_ref(x_data, y_data, N), 2.0e-6f, "f32 mse forward");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mse_backward(ctx, &x, &y, &seed, &dx, &dy));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &dx, got_x, sizeof(got_x)));
    CHECK_OK(gd_tensor_read(ctx, &dy, got_y, sizeof(got_y)));
    for (i = 0U; i < N; ++i) {
        float want = grad_seed * (2.0f / (float)N) * (x_data[i] - y_data[i]);
        check_close(got_x[i], want, 2.0e-6f, "f32 mse direct dx");
        check_close(got_y[i], -want, 2.0e-6f, "f32 mse direct dy");
    }

    x.requires_grad = true;
    y.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mse(ctx, &x, &y, &loss));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &auto_dx));
    CHECK_OK(gd_tensor_grad(ctx, &y, &auto_dy));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &auto_dx, got_x, sizeof(got_x)));
    CHECK_OK(gd_tensor_read(ctx, &auto_dy, got_y, sizeof(got_y)));
    for (i = 0U; i < N; ++i) {
        float want = (2.0f / (float)N) * (x_data[i] - y_data[i]);
        check_close(got_x[i], want, 2.0e-6f, "f32 mse autograd dx");
        check_close(got_y[i], -want, 2.0e-6f, "f32 mse autograd dy");
    }
    gd_context_destroy(ctx);
}

static void test_mse_f16_forward_backward(void)
{
    enum { N = 8 };
    const int64_t shape[1] = {N};
    const float x_data[N] = {1.0f, 0.5f, -1.0f, 2.0f, -2.0f, 0.25f, -0.5f, 1.5f};
    const float y_data[N] = {0.0f, -0.5f, -1.5f, 1.0f, -1.0f, -0.25f, 0.5f, 0.0f};
    const float seed_data = 0.75f;
    float got_scalar = 0.0f;
    float got[N];
    gd_memory_config cfg = mse_config((size_t)N * sizeof(uint16_t));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor seed;
    gd_tensor loss;
    gd_tensor dx;
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &y));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(0U, NULL), 256U, &seed));
    CHECK_OK(gd_tensor_write_f32(ctx, &x, x_data, N));
    CHECK_OK(gd_tensor_write_f32(ctx, &y, y_data, N));
    CHECK_OK(gd_tensor_write(ctx, &seed, &seed_data, sizeof(seed_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mse(ctx, &x, &y, &loss));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &loss, &got_scalar, sizeof(got_scalar)));
    check_close(got_scalar, mse_ref(x_data, y_data, N), 2.0e-4f, "f16 mse forward");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mse_backward(ctx, &x, &y, &seed, &dx, NULL));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx, got, N));
    for (i = 0U; i < N; ++i) {
        float want = seed_data * (2.0f / (float)N) * (x_data[i] - y_data[i]);
        check_close(got[i], want, 1.5e-3f, "f16 mse direct dx");
    }
    gd_context_destroy(ctx);
}

static void test_mse_large_partial_reduce(void)
{
    enum { N = 5000 };
    const int64_t shape[1] = {N};
    float *x_data = (float *)malloc((size_t)N * sizeof(float));
    float *y_data = (float *)calloc((size_t)N, sizeof(float));
    gd_memory_config cfg = mse_config((size_t)N * sizeof(float));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    gd_tensor loss;
    float got = 0.0f;
    int i;
    CHECK(x_data != NULL && y_data != NULL, "allocate large fixture");
    for (i = 0; i < N; ++i) {
        x_data[i] = 1.0f;
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 256U, &y));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, (size_t)N * sizeof(float)));
    CHECK_OK(gd_tensor_write(ctx, &y, y_data, (size_t)N * sizeof(float)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_mse(ctx, &x, &y, &loss));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read(ctx, &loss, &got, sizeof(got)));
    check_close(got, 1.0f, 2.0e-6f, "large partial reduce mse");
    gd_context_destroy(ctx);
    free(x_data);
    free(y_data);
}

int main(void)
{
    gd_context *probe = NULL;
    gd_memory_config cfg = mse_config(1024U);
    gd_status st = gd_context_create(&cfg, &probe);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_mse: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);
    gd_context_destroy(probe);

    test_mse_f32_forward_backward();
    test_mse_f16_forward_backward();
    test_mse_large_partial_reduce();
    printf("test_mse: ok\n");
    return 0;
}
