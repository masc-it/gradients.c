#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_dropout_ops failed: %s (%s:%d)\n", (msg),  \
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
                "test_dropout_ops failed: %s have=%.8f want=%.8f tol=%.8f\n",
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

static gd_memory_config dropout_config(size_t tensor_bytes, size_t mask_bytes)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = align_up(tensor_bytes * 3U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = align_up(tensor_bytes * 10U + mask_bytes * 4U + 1024U * 1024U, 4096U);
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

static uint64_t dropout_splitmix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static int dropout_keep(uint64_t seed, uint32_t index, float p)
{
    uint64_t r = dropout_splitmix64(seed + (uint64_t)index);
    uint32_t mant = (uint32_t)((r >> 40) & UINT64_C(0xffffff));
    float u = (float)mant * (1.0f / 16777216.0f);
    return u >= p;
}

static void test_dropout_f32_training_forward_backward(void)
{
    enum { N = 17 };
    const int64_t shape[2] = {1, N};
    const float x_data[N] = {
        -1.25f, -1.0f, -0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f,
        1.0f, 1.25f, 1.5f, -1.5f, 2.0f, -2.0f, 3.0f, -3.0f,
    };
    const float g_data[N] = {
        0.125f, -0.25f, 0.375f, -0.5f, 0.625f, -0.75f, 0.875f, -1.0f,
        1.125f, -1.25f, 1.375f, -1.5f, 1.625f, -1.75f, 1.875f, -2.0f, 2.125f,
    };
    const float p = 0.35f;
    const float scale = 1.0f / (1.0f - p);
    const uint64_t seed = UINT64_C(0x123456789abcdef0);
    gd_memory_config cfg = dropout_config(sizeof(x_data), (size_t)N);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor y_auto;
    gd_tensor dx_auto;
    float got[N];
    uint32_t i;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, p, true, seed, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    CHECK(y.id != x.id, "training dropout materializes output");
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float want = dropout_keep(seed, i, p) ? x_data[i] * scale : 0.0f;
        check_close(got[i], want, 1.0e-6f, "f32 dropout forward");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout_backward(ctx, &x, &g, p, seed, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float want = dropout_keep(seed, i, p) ? g_data[i] * scale : 0.0f;
        check_close(got[i], want, 1.0e-6f, "f32 dropout direct backward");
    }

    x.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, p, true, seed, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx_auto, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float want = dropout_keep(seed, i, p) ? g_data[i] * scale : 0.0f;
        check_close(got[i], want, 1.0e-6f, "f32 dropout autograd backward");
    }

    gd_context_destroy(ctx);
}

static void test_dropout_f16_training_forward_backward(void)
{
    enum { N = 19 };
    const int64_t shape[2] = {1, N};
    const float x_values[N] = {
        -2.0f, -1.5f, -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f,
        2.0f, -3.0f, 3.0f, -4.0f, 4.0f, 0.125f, -0.125f, 0.75f, -0.75f,
    };
    const float g_values[N] = {
        0.5f, -0.625f, 0.75f, -0.875f, 1.0f, -1.125f, 1.25f, -1.375f, 1.5f,
        -1.625f, 1.75f, -1.875f, 2.0f, -2.125f, 2.25f, -2.375f, 2.5f, -2.625f, 2.75f,
    };
    const float p = 0.20f;
    const float scale = 1.0f / (1.0f - p);
    const uint64_t seed = UINT64_C(0x0ddc0ffee1234567);
    uint16_t x_data[N];
    uint16_t g_data[N];
    uint16_t got[N];
    gd_memory_config cfg = dropout_config(sizeof(x_data), (size_t)N);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor y_auto;
    gd_tensor dx_auto;
    uint32_t i;
    for (i = 0U; i < (uint32_t)N; ++i) {
        x_data[i] = f32_to_f16_bits(x_values[i]);
        g_data[i] = f32_to_f16_bits(g_values[i]);
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, p, true, seed, &y));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float x_f = f16_bits_to_f32(x_data[i]);
        float want = dropout_keep(seed, i, p) ? f16_bits_to_f32(f32_to_f16_bits(x_f * scale)) : 0.0f;
        check_close(f16_bits_to_f32(got[i]), want, 1.0e-3f, "f16 dropout forward");
    }

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout_backward(ctx, &x, &g, p, seed, &dx));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float g_f = f16_bits_to_f32(g_data[i]);
        float want = dropout_keep(seed, i, p) ? f16_bits_to_f32(f32_to_f16_bits(g_f * scale)) : 0.0f;
        check_close(f16_bits_to_f32(got[i]), want, 1.0e-3f, "f16 dropout direct backward");
    }

    x.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, p, true, seed, &y_auto));
    CHECK_OK(gd_backward(ctx, &y_auto, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx_auto));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx_auto, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float g_f = f16_bits_to_f32(g_data[i]);
        float want = dropout_keep(seed, i, p) ? f16_bits_to_f32(f32_to_f16_bits(g_f * scale)) : 0.0f;
        check_close(f16_bits_to_f32(got[i]), want, 1.0e-3f, "f16 dropout autograd backward");
    }

    gd_context_destroy(ctx);
}

static void test_dropout_add_f16_training_autograd(void)
{
    enum { N = 19 };
    const int64_t shape[2] = {1, N};
    const float residual_values[N] = {
        0.25f, -0.25f, 0.50f, -0.50f, 0.75f, -0.75f, 1.00f, -1.00f, 1.25f, -1.25f,
        1.50f, -1.50f, 1.75f, -1.75f, 2.00f, -2.00f, 0.125f, -0.125f, 0.0f,
    };
    const float x_values[N] = {
        -2.0f, -1.5f, -1.0f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f,
        2.0f, -3.0f, 3.0f, -4.0f, 4.0f, 0.125f, -0.125f, 0.75f, -0.75f,
    };
    const float g_values[N] = {
        0.5f, -0.625f, 0.75f, -0.875f, 1.0f, -1.125f, 1.25f, -1.375f, 1.5f,
        -1.625f, 1.75f, -1.875f, 2.0f, -2.125f, 2.25f, -2.375f, 2.5f, -2.625f, 2.75f,
    };
    const float p = 0.20f;
    const float scale = 1.0f / (1.0f - p);
    const uint64_t seed = UINT64_C(0xabcddcba98761234);
    uint16_t residual_data[N];
    uint16_t x_data[N];
    uint16_t g_data[N];
    uint16_t got[N];
    gd_memory_config cfg = dropout_config((size_t)N * sizeof(uint16_t), (size_t)N);
    gd_context *ctx = NULL;
    gd_tensor residual;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y;
    gd_tensor grad_residual;
    gd_tensor grad_x;
    uint32_t i;
    for (i = 0U; i < (uint32_t)N; ++i) {
        residual_data[i] = f32_to_f16_bits(residual_values[i]);
        x_data[i] = f32_to_f16_bits(x_values[i]);
        g_data[i] = f32_to_f16_bits(g_values[i]);
    }
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &residual));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &residual, residual_data, sizeof(residual_data)));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    residual.requires_grad = true;
    x.requires_grad = true;
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout_add(ctx, &residual, &x, p, true, seed, &y));
    CHECK_OK(gd_backward(ctx, &y, &g));
    CHECK_OK(gd_tensor_grad(ctx, &residual, &grad_residual));
    CHECK_OK(gd_tensor_grad(ctx, &x, &grad_x));
    CHECK_OK(gd_end_step(ctx));
    CHECK_OK(gd_synchronize(ctx));

    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float residual_f = f16_bits_to_f32(residual_data[i]);
        float x_f = f16_bits_to_f32(x_data[i]);
        float drop_f = dropout_keep(seed, i, p) ? f16_bits_to_f32(f32_to_f16_bits(x_f * scale)) : 0.0f;
        float want = f16_bits_to_f32(f32_to_f16_bits(residual_f + drop_f));
        check_close(f16_bits_to_f32(got[i]), want, 1.0e-3f, "f16 dropout_add forward");
    }
    CHECK_OK(gd_tensor_read(ctx, &grad_residual, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        check_close(f16_bits_to_f32(got[i]), f16_bits_to_f32(g_data[i]), 1.0e-3f, "f16 dropout_add residual grad");
    }
    CHECK_OK(gd_tensor_read(ctx, &grad_x, got, sizeof(got)));
    for (i = 0U; i < (uint32_t)N; ++i) {
        float g_f = f16_bits_to_f32(g_data[i]);
        float want = dropout_keep(seed, i, p) ? f16_bits_to_f32(f32_to_f16_bits(g_f * scale)) : 0.0f;
        check_close(f16_bits_to_f32(got[i]), want, 1.0e-3f, "f16 dropout_add branch grad");
    }

    gd_context_destroy(ctx);
}

static void test_dropout_identity_paths(void)
{
    enum { N = 5 };
    const int64_t shape[1] = {N};
    const float x_data[N] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    const float g_data[N] = {0.25f, -0.5f, 0.75f, -1.0f, 1.25f};
    gd_memory_config cfg = dropout_config(sizeof(x_data), (size_t)N);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor g;
    gd_tensor y_eval;
    gd_tensor y_p0;
    gd_tensor dx_p0;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, 0.75f, false, 11U, &y_eval));
    CHECK_OK(gd_end_step(ctx));
    CHECK(y_eval.id == x.id, "eval dropout aliases input");

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_dropout(ctx, &x, 0.0f, true, 22U, &y_p0));
    CHECK_OK(gd_dropout_backward(ctx, &x, &g, 0.0f, 22U, &dx_p0));
    CHECK_OK(gd_end_step(ctx));
    CHECK(y_p0.id == x.id, "p=0 dropout aliases input");
    CHECK(dx_p0.id == g.id, "p=0 dropout backward aliases grad_out");

    gd_context_destroy(ctx);
}

static void test_dropout_rejects_invalid_inputs(void)
{
    const int64_t shape[1] = {4};
    gd_memory_config cfg = dropout_config(4U * sizeof(float), 4U);
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor xi;
    gd_tensor out;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, shape), 256U, &xi));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_STATUS(gd_dropout(ctx, &x, -0.1f, true, 1U, &out), GD_ERR_INVALID_ARGUMENT);
    gd_context_clear_error(ctx);
    CHECK_STATUS(gd_dropout(ctx, &x, 1.0f, true, 1U, &out), GD_ERR_INVALID_ARGUMENT);
    gd_context_clear_error(ctx);
    CHECK_STATUS(gd_dropout(ctx, &xi, 0.5f, true, 1U, &out), GD_ERR_UNSUPPORTED);
    gd_context_destroy(ctx);
}

int main(void)
{
    test_dropout_f32_training_forward_backward();
    test_dropout_f16_training_forward_backward();
    test_dropout_add_f16_training_autograd();
    test_dropout_identity_paths();
    test_dropout_rejects_invalid_inputs();
    printf("test_dropout_ops: ok\n");
    return 0;
}
