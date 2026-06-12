#include <gradients/gradients.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_swiglu failed: %s (%s:%d)\n", (msg),      \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static float abs_f32(float x) { return x < 0.0f ? -x : x; }

static void check_close(float have, float want, float tol, const char *msg)
{
    if (abs_f32(have - want) > tol) {
        fprintf(stderr,
                "test_swiglu failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config swiglu_config(size_t tensor_bytes)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = align_up(tensor_bytes * 8U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 32U + 1024U * 1024U, 4096U);
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

static float swiglu_gate(float z)
{
    return z * sigmoid_stable(z);
}

static float swiglu_gate_grad(float z)
{
    float s = sigmoid_stable(z);
    return s * (1.0f + z * (1.0f - s));
}

static void test_swiglu_split_forward_backward(void)
{
    enum { ROWS = 2, H = 8, XN = ROWS * 2 * H, YN = ROWS * H };
    const int64_t xshape[2] = {ROWS, 2 * H};
    const int64_t yshape[2] = {ROWS, H};
    float x12_data[XN];
    float grad_data[YN];
    float x12q[XN];
    float gradq[YN];
    float want_y[YN];
    float want_dx12[XN];
    float got_x[XN];
    float got_y[YN];
    gd_memory_config cfg = swiglu_config((size_t)XN * sizeof(unsigned short));
    gd_context *ctx = NULL;
    gd_tensor x12;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dx12;
    gd_tensor y_auto;
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    for (i = 0U; i < XN; ++i) {
        x12_data[i] = ((float)((int)(i % 17U) - 8)) * 0.375f;
    }
    for (i = 0U; i < YN; ++i) {
        grad_data[i] = ((float)((int)(i % 11U) - 5)) * 0.25f;
    }
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, xshape), 256U, &x12));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, yshape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x12, x12_data, XN));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, YN));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x12, x12q, XN));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, gradq, YN));
    for (i = 0U; i < YN; ++i) {
        const uint32_t row = i / H;
        const uint32_t col = i % H;
        const uint32_t x1i = row * 2U * H + col;
        const uint32_t x2i = x1i + H;
        const float g = swiglu_gate(x12q[x2i]);
        want_y[i] = x12q[x1i] * g;
        want_dx12[x1i] = gradq[i] * g;
        want_dx12[x2i] = gradq[i] * x12q[x1i] * swiglu_gate_grad(x12q[x2i]);
    }
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_swiglu_split(ctx, &x12, &y));
    CHECK_OK(gd_swiglu_split_backward(ctx, &x12, &grad, &dx12));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got_y, YN));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, got_x, XN));
    for (i = 0U; i < YN; ++i) {
        const uint32_t row = i / H;
        const uint32_t col = i % H;
        const uint32_t x1i = row * 2U * H + col;
        const uint32_t x2i = x1i + H;
        check_close(got_y[i], want_y[i], 8.0e-3f, "swiglu_split forward");
        check_close(got_x[x1i], want_dx12[x1i], 8.0e-3f, "swiglu_split dx1");
        check_close(got_x[x2i], want_dx12[x2i], 1.0e-2f, "swiglu_split dx2");
    }
    x12.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_swiglu_split(ctx, &x12, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &grad));
    CHECK_OK(gd_tensor_grad(ctx, &x12, &dx12));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, got_x, XN));
    for (i = 0U; i < YN; ++i) {
        const uint32_t row = i / H;
        const uint32_t col = i % H;
        const uint32_t x1i = row * 2U * H + col;
        const uint32_t x2i = x1i + H;
        check_close(got_x[x1i], want_dx12[x1i], 8.0e-3f, "swiglu_split autograd dx1");
        check_close(got_x[x2i], want_dx12[x2i], 1.0e-2f, "swiglu_split autograd dx2");
    }
    gd_context_destroy(ctx);
}

static void test_swiglu_split_linear(void)
{
    enum { ROWS = 3, H = 4, O = 3, XN = ROWS * 2 * H, WN = H * O, YN = ROWS * O };
    const int64_t xshape[2] = {ROWS, 2 * H};
    const int64_t wshape[2] = {H, O};
    const int64_t yshape[2] = {ROWS, O};
    float x12_data[XN];
    float w_data[WN];
    float grad_data[YN];
    float x12q[XN];
    float wq[WN];
    float gradq[YN];
    float act[ROWS * H];
    float want_y[YN];
    float want_dw[WN];
    float want_dx12[XN];
    float got[YN > XN ? YN : XN];
    gd_memory_config cfg = swiglu_config(4096U);
    gd_context *ctx = NULL;
    gd_tensor x12;
    gd_tensor w;
    gd_tensor grad;
    gd_tensor y;
    gd_tensor dx12;
    gd_tensor dw;
    uint32_t i;
    uint32_t r;
    uint32_t h;
    uint32_t o;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    for (i = 0U; i < XN; ++i) { x12_data[i] = ((float)((int)(i % 13U) - 6)) * 0.25f; }
    for (i = 0U; i < WN; ++i) { w_data[i] = ((float)((int)(i % 7U) - 3)) * 0.125f; }
    for (i = 0U; i < YN; ++i) { grad_data[i] = ((float)((int)(i % 5U) - 2)) * 0.5f; }
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, xshape), 256U, &x12));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, wshape), 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, yshape), 256U, &grad));
    CHECK_OK(gd_tensor_write_f32(ctx, &x12, x12_data, XN));
    CHECK_OK(gd_tensor_write_f32(ctx, &w, w_data, WN));
    CHECK_OK(gd_tensor_write_f32(ctx, &grad, grad_data, YN));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &x12, x12q, XN));
    CHECK_OK(gd_tensor_read_f32(ctx, &w, wq, WN));
    CHECK_OK(gd_tensor_read_f32(ctx, &grad, gradq, YN));
    memset(want_y, 0, sizeof(want_y));
    memset(want_dw, 0, sizeof(want_dw));
    memset(want_dx12, 0, sizeof(want_dx12));
    for (r = 0U; r < ROWS; ++r) {
        for (h = 0U; h < H; ++h) {
            const uint32_t x1i = r * 2U * H + h;
            const uint32_t x2i = x1i + H;
            act[r * H + h] = x12q[x1i] * swiglu_gate(x12q[x2i]);
        }
    }
    for (r = 0U; r < ROWS; ++r) {
        for (o = 0U; o < O; ++o) {
            float acc = 0.0f;
            for (h = 0U; h < H; ++h) {
                acc += act[r * H + h] * wq[h * O + o];
                want_dw[h * O + o] += act[r * H + h] * gradq[r * O + o];
            }
            want_y[r * O + o] = acc;
        }
    }
    for (r = 0U; r < ROWS; ++r) {
        for (h = 0U; h < H; ++h) {
            float dact = 0.0f;
            const uint32_t x1i = r * 2U * H + h;
            const uint32_t x2i = x1i + H;
            for (o = 0U; o < O; ++o) {
                dact += gradq[r * O + o] * wq[h * O + o];
            }
            want_dx12[x1i] = dact * swiglu_gate(x12q[x2i]);
            want_dx12[x2i] = dact * x12q[x1i] * swiglu_gate_grad(x12q[x2i]);
        }
    }
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_swiglu_split_linear(ctx, &x12, &w, NULL, &y));
    CHECK_OK(gd_swiglu_split_linear_backward(ctx, &x12, &w, NULL, &grad, &dx12, &dw, NULL));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK_OK(gd_tensor_read_f32(ctx, &y, got, YN));
    for (i = 0U; i < YN; ++i) { check_close(got[i], want_y[i], 1.5e-2f, "swiglu_split_linear forward"); }
    CHECK_OK(gd_tensor_read_f32(ctx, &dx12, got, XN));
    for (i = 0U; i < XN; ++i) { check_close(got[i], want_dx12[i], 2.0e-2f, "swiglu_split_linear dx12"); }
    CHECK_OK(gd_tensor_read_f32(ctx, &dw, got, WN));
    for (i = 0U; i < WN; ++i) { check_close(got[i], want_dw[i], 2.0e-2f, "swiglu_split_linear dw"); }
    gd_context_destroy(ctx);
}

int main(void)
{
    test_swiglu_split_forward_backward();
    test_swiglu_split_linear();
    printf("test_swiglu: ok\n");
    return 0;
}
