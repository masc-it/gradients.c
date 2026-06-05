#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_sigmoid_ops failed: %s (%s:%d)\n", (msg),  \
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
                "test_sigmoid_ops failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config sigmoid_config(size_t tensor_bytes)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(tensor_bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 8U + 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static uint16_t f32_to_f16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;
    uint32_t out_exp;
    uint32_t out_mant;
    v.f = value;
    sign = (v.u >> 16) & 0x8000U;
    exp = (int32_t)((v.u >> 23) & 0xffU) - 127;
    mant = v.u & 0x7fffffU;
    if (((v.u >> 23) & 0xffU) == 0xffU) {
        return (uint16_t)(sign | (mant == 0U ? 0x7c00U : 0x7e00U));
    }
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (exp < -14) {
        uint32_t shifted;
        uint32_t remainder;
        uint32_t halfway;
        int32_t shift = -14 - exp;
        if (shift > 24) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        shifted = mant >> (uint32_t)(shift + 13);
        remainder = mant & ((1U << (uint32_t)(shift + 13)) - 1U);
        halfway = 1U << (uint32_t)(shift + 12);
        if (remainder > halfway || (remainder == halfway && (shifted & 1U) != 0U)) {
            shifted += 1U;
        }
        return (uint16_t)(sign | shifted);
    }
    out_exp = (uint32_t)(exp + 15);
    out_mant = mant >> 13;
    {
        uint32_t remainder = mant & 0x1fffU;
        if (remainder > 0x1000U || (remainder == 0x1000U && (out_mant & 1U) != 0U)) {
            out_mant += 1U;
            if (out_mant == 0x400U) {
                out_mant = 0U;
                out_exp += 1U;
                if (out_exp >= 31U) {
                    return (uint16_t)(sign | 0x7c00U);
                }
            }
        }
    }
    return (uint16_t)(sign | (out_exp << 10) | out_mant);
}

static float f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)bits >> 10) & 0x1fU;
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0U) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1U;
        }
        mant &= 0x3ffU;
        exp += 1U;
    } else if (exp == 31U) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    return v.f;
}

static float sigmoid_ref(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void test_sigmoid_f32_forward_backward(void)
{
    enum { N = 9 };
    const int64_t shape[2] = {3, 3};
    const float x_data[N] = {-8.0f, -2.0f, -0.5f, 0.0f, 0.5f, 2.0f, 8.0f, 12.0f, -12.0f};
    const float g_data[N] = {0.25f, -0.5f, 0.75f, 1.0f, -1.25f, 1.5f, -1.75f, 2.0f, -2.25f};
    float got[N];
    gd_memory_config cfg = sigmoid_config(sizeof(x_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor y_auto;
    gd_tensor dx_auto;
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid(ctx, &x, &y));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        check_close(got[i], sigmoid_ref(x_data[i]), 6.0e-6f, "f32 sigmoid forward");
    }

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid_backward(ctx, &x, &g, &dx));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        float s = sigmoid_ref(x_data[i]);
        check_close(got[i], g_data[i] * s * (1.0f - s), 8.0e-6f, "f32 sigmoid direct backward");
    }

    x.requires_grad = true;
    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid(ctx, &x, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx_auto, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        float s = sigmoid_ref(x_data[i]);
        check_close(got[i], g_data[i] * s * (1.0f - s), 8.0e-6f, "f32 sigmoid autograd backward");
    }

    gd_context_destroy(ctx);
}

static void test_sigmoid_f16_forward_backward(void)
{
    enum { N = 17 };
    const int64_t shape[2] = {1, N};
    const float x_values[N] = {
        -12.0f, -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.125f, 0.0f, 0.125f,
        0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 12.0f, 0.25f, -0.25f,
    };
    const float g_values[N] = {
        0.25f, -0.375f, 0.5f, -0.625f, 0.75f, -0.875f, 1.0f, -1.125f, 1.25f,
        -1.375f, 1.5f, -1.625f, 1.75f, -1.875f, 2.0f, -2.125f, 2.25f,
    };
    uint16_t x_data[N];
    uint16_t g_data[N];
    uint16_t got[N];
    gd_memory_config cfg = sigmoid_config(sizeof(x_data));
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor y_auto;
    gd_tensor dx_auto;
    uint32_t i;
    for (i = 0U; i < N; ++i) {
        x_data[i] = f32_to_f16_bits(x_values[i]);
        g_data[i] = f32_to_f16_bits(g_values[i]);
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid(ctx, &x, &y));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        float want = f16_bits_to_f32(f32_to_f16_bits(sigmoid_ref(f16_bits_to_f32(x_data[i]))));
        check_close(f16_bits_to_f32(got[i]), want, 1.2e-3f, "f16 sigmoid forward");
    }

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid_backward(ctx, &x, &g, &dx));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        float x_f = f16_bits_to_f32(x_data[i]);
        float g_f = f16_bits_to_f32(g_data[i]);
        float s = sigmoid_ref(x_f);
        float want = f16_bits_to_f32(f32_to_f16_bits(g_f * s * (1.0f - s)));
        check_close(f16_bits_to_f32(got[i]), want, 1.5e-3f, "f16 sigmoid direct backward");
    }

    x.requires_grad = true;
    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sigmoid(ctx, &x, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_end(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx_auto, got, sizeof(got)));
    for (i = 0U; i < N; ++i) {
        float saved = f16_bits_to_f32(f32_to_f16_bits(sigmoid_ref(f16_bits_to_f32(x_data[i]))));
        float g_f = f16_bits_to_f32(g_data[i]);
        float want = f16_bits_to_f32(f32_to_f16_bits(g_f * saved * (1.0f - saved)));
        check_close(f16_bits_to_f32(got[i]), want, 1.5e-3f, "f16 sigmoid autograd backward");
    }

    gd_context_destroy(ctx);
}

static void test_sigmoid_rejects_i32(void)
{
    const int64_t shape[1] = {1};
    gd_memory_config cfg = sigmoid_config(4U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor y;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, shape), 256U, &x));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_sigmoid(ctx, &x, &y), GD_ERR_UNSUPPORTED);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_sigmoid_f32_forward_backward();
    test_sigmoid_f16_forward_backward();
    test_sigmoid_rejects_i32();
    printf("test_sigmoid_ops: ok\n");
    return 0;
}
