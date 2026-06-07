#include <gradients/gradients.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_powlu failed: %s (%s:%d)\n", (msg),       \
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
                "test_powlu failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config powlu_config(size_t tensor_bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(tensor_bytes * 4U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 16U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static float sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    {
        float e = expf(x);
        return e / (1.0f + e);
    }
}

static float powlu_gate(float z, float m)
{
    float s = sigmoid_stable(z);
    if (z <= 0.0f) {
        return z * s;
    }
    {
        float r = sqrtf(z);
        float a = m / (r + 1.0f);
        return powf(z, a) * s;
    }
}

static float powlu_gate_grad(float z, float m)
{
    float s = sigmoid_stable(z);
    if (z <= 0.0f) {
        return s * (1.0f + z * (1.0f - s));
    }
    {
        float r = sqrtf(z);
        float rp1 = r + 1.0f;
        float a = m / rp1;
        float g = powf(z, a);
        float da = -m / (2.0f * r * rp1 * rp1);
        float lz = logf(z > 1.17549435e-38f ? z : 1.17549435e-38f);
        return g * s * (a / z + da * lz + (1.0f - s));
    }
}

static void ref_powlu_forward(const float *x1,
                              const float *x2,
                              size_t count,
                              float m,
                              float *out)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        out[i] = x1[i] * powlu_gate(x2[i], m);
    }
}

static void ref_powlu_backward(const float *x1,
                               const float *x2,
                               const float *grad,
                               size_t count,
                               float m,
                               float *dx1,
                               float *dx2)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        dx1[i] = grad[i] * powlu_gate(x2[i], m);
        dx2[i] = grad[i] * x1[i] * powlu_gate_grad(x2[i], m);
    }
}

static void test_powlu_f16_forward_backward_autograd(void)
{
    enum { N = 17 };
    const int64_t shape[2] = {1, N};
    const float x1_data[N] = {
        -1.50f, -0.75f, -0.25f, 0.00f, 0.25f, 0.50f, 0.75f, 1.00f, 1.25f,
        1.50f, -1.00f, 2.00f, -2.00f, 0.375f, -0.625f, 1.75f, -1.25f,
    };
    const float x2_data[N] = {
        -4.00f, -2.00f, -1.00f, -0.50f, -0.125f, 0.00f, 0.125f, 0.25f, 0.50f,
        1.00f, 1.50f, 2.00f, 3.00f, 4.00f, 6.00f, 8.00f, 0.75f,
    };
    const float grad_data[N] = {
        0.25f, -0.375f, 0.50f, -0.625f, 0.75f, -0.875f, 1.00f, -1.125f, 1.25f,
        -1.375f, 1.50f, -1.625f, 1.75f, -1.875f, 2.00f, -2.125f, 2.25f,
    };
    float x1q[N];
    float x2q[N];
    float gradq[N];
    float want_y[N];
    float want_dx1[N];
    float want_dx2[N];
    float got[N];
    gd_memory_config cfg = powlu_config((size_t)N * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x1;
    gd_tensor x2;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dx1;
    gd_tensor dx2;
    gd_tensor y_auto;
    gd_tensor dx1_auto;
    gd_tensor dx2_auto;
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x1));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x2));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x1, x1_data, N));
    CHECK_OK(gd_tensor_write_f32(ctx, &x2, x2_data, N));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, N));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x1, x1q, N));
    CHECK_OK(gd_tensor_read_f32(ctx, &x2, x2q, N));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, gradq, N));
    ref_powlu_forward(x1q, x2q, N, 3.0f, want_y);
    ref_powlu_backward(x1q, x2q, gradq, N, 3.0f, want_dx1, want_dx2);

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu(ctx, &x1, &x2, 3.0f, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, N));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], want_y[i], 8.0e-3f, "f16 powlu forward");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_backward(ctx, &x1, &x2, &grad, 3.0f, &dx1, &dx2));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx1, got, N));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], want_dx1[i], 8.0e-3f, "f16 powlu direct dx1");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dx2, got, N));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], want_dx2[i], 1.2e-2f, "f16 powlu direct dx2");
    }

    x1.requires_grad = true;
    x2.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu(ctx, &x1, &x2, 3.0f, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x1, &dx1_auto));
    CHECK_OK(gd_tensor_grad(ctx, &x2, &dx2_auto));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx1_auto, got, N));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], want_dx1[i], 8.0e-3f, "f16 powlu autograd dx1");
    }
    CHECK_OK(gd_tensor_read_f32(ctx, &dx2_auto, got, N));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], want_dx2[i], 1.2e-2f, "f16 powlu autograd dx2");
    }
    gd_context_destroy(ctx);
}

static void test_powlu_optional_grad_and_m2(void)
{
    enum { N = 5 };
    const int64_t shape[1] = {N};
    const float x1_data[N] = {0.25f, -0.5f, 1.0f, -1.5f, 2.0f};
    const float x2_data[N] = {-1.0f, 0.25f, 0.5f, 1.0f, 2.0f};
    const float grad_data[N] = {1.0f, -0.75f, 0.5f, -0.25f, 0.125f};
    float x1q[N];
    float x2q[N];
    float gradq[N];
    float want_dx1[N];
    float want_dx2[N];
    float got[N];
    gd_memory_config cfg = powlu_config((size_t)N * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x1;
    gd_tensor x2;
    gd_tensor grad;
    gd_tensor dx2;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x1));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &x2));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x1, x1_data, N));
    CHECK_OK(gd_tensor_write_f32(ctx, &x2, x2_data, N));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, N));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x1, x1q, N));
    CHECK_OK(gd_tensor_read_f32(ctx, &x2, x2q, N));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, gradq, N));
    ref_powlu_backward(x1q, x2q, gradq, N, 2.0f, want_dx1, want_dx2);
    (void)want_dx1;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_backward(ctx, &x1, &x2, &grad, 2.0f, NULL, &dx2));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx2, got, N));
    for (uint32_t i = 0U; i < N; ++i) {
        check_close(got[i], want_dx2[i], 1.2e-2f, "f16 powlu optional dx2");
    }
    gd_context_destroy(ctx);
}

static void test_powlu_split_f16_forward_backward_autograd(void)
{
    enum { ROWS = 3, H = 5, XN = ROWS * H * 2, YN = ROWS * H };
    const int64_t x_shape[2] = {ROWS, H * 2};
    const int64_t y_shape[2] = {ROWS, H};
    const float x12_data[XN] = {
        -1.50f, -0.75f, -0.25f, 0.00f, 0.25f, -4.00f, -2.00f, -1.00f, -0.50f, -0.125f,
        0.50f, 0.75f, 1.00f, 1.25f, 1.50f, 0.00f, 0.125f, 0.25f, 0.50f, 1.00f,
        -1.00f, 2.00f, -2.00f, 0.375f, -0.625f, 1.50f, 2.00f, 3.00f, 4.00f, 6.00f,
    };
    const float grad_data[YN] = {
        0.25f, -0.375f, 0.50f, -0.625f, 0.75f,
        -0.875f, 1.00f, -1.125f, 1.25f, -1.375f,
        1.50f, -1.625f, 1.75f, -1.875f, 2.00f,
    };
    float x12q[XN];
    float gradq[YN];
    float want_y[YN];
    float want_dx12[XN];
    float got_x[XN];
    float got_y[YN];
    gd_memory_config cfg = powlu_config((size_t)XN * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x12;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dx12;
    gd_tensor y_auto;
    gd_tensor dx12_auto;
    uint32_t row;
    uint32_t col;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x12));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, y_shape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x12, x12_data, XN));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, YN));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x12, x12q, XN));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, gradq, YN));
    memset(want_dx12, 0, sizeof(want_dx12));
    for (row = 0U; row < ROWS; ++row) {
        for (col = 0U; col < H; ++col) {
            const uint32_t yi = row * H + col;
            const uint32_t x1i = row * H * 2U + col;
            const uint32_t x2i = x1i + H;
            const float gate = powlu_gate(x12q[x2i], 3.0f);
            want_y[yi] = x12q[x1i] * gate;
            want_dx12[x1i] = gradq[yi] * gate;
            want_dx12[x2i] = gradq[yi] * x12q[x1i] * powlu_gate_grad(x12q[x2i], 3.0f);
        }
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_split(ctx, &x12, 3.0f, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got_y, YN));
    for (uint32_t i = 0U; i < YN; ++i) {
        check_close(got_y[i], want_y[i], 8.0e-3f, "f16 powlu_split forward");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_split_backward(ctx, &x12, &grad, 3.0f, &dx12));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, got_x, XN));
    for (uint32_t i = 0U; i < XN; ++i) {
        check_close(got_x[i], want_dx12[i], 1.2e-2f, "f16 powlu_split direct backward");
    }

    x12.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_powlu_split(ctx, &x12, 3.0f, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x12, &dx12_auto));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12_auto, got_x, XN));
    for (uint32_t i = 0U; i < XN; ++i) {
        check_close(got_x[i], want_dx12[i], 1.2e-2f, "f16 powlu_split autograd backward");
    }
    gd_context_destroy(ctx);
}

static void test_powlu_validation(void)
{
    const int64_t shape_a[1] = {4};
    const int64_t shape_b[1] = {5};
    gd_memory_config cfg = powlu_config(5U * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x1;
    gd_tensor x2;
    gd_tensor bad_shape;
    gd_tensor bad_dtype;
    gd_tensor out;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape_a), 256U, &x1));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape_a), 256U, &x2));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, shape_b), 256U, &bad_shape));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape_a), 256U, &bad_dtype));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_powlu(ctx, &x1, &x2, 0.0f, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_powlu(ctx, &x1, &x2, 10.0f, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_powlu(ctx, &x1, &bad_shape, 3.0f, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_powlu(ctx, &x1, &bad_dtype, 3.0f, &out), GD_ERR_UNSUPPORTED);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_powlu_f16_forward_backward_autograd();
    test_powlu_optional_grad_and_m2();
    test_powlu_split_f16_forward_backward_autograd();
    test_powlu_validation();
    printf("test_powlu: ok\n");
    return 0;
}
