#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_binary_ops failed: %s (%s:%d)\n", (msg),   \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static gd_memory_config binary_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 16384U;
    cfg.state_bytes = 4096U;
    cfg.scratch_slot_bytes = 65536U;
    cfg.data_slot_bytes = 4096U;
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

static void pack_f16(const float *src, uint16_t *dst, uint32_t count)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        dst[i] = f32_to_f16_bits(src[i]);
    }
}

static void expect_tensor_f16(gd_context *ctx,
                              const gd_tensor *tensor,
                              const float *want,
                              uint32_t count,
                              const char *what)
{
    uint16_t got[16];
    uint32_t i;
    CHECK(count <= 16U, "test helper count");
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, tensor, got, (size_t)count * sizeof(got[0])));
    for (i = 0U; i < count; ++i) {
        float have = f16_bits_to_f32(got[i]);
        CHECK(abs_f32(have - want[i]) <= 1.0e-3f, what);
    }
}

static void run_binary_ops_test(void)
{
    enum { COUNT = 6 };
    const int64_t shape[2] = {2, 3};
    const int64_t bias_shape[1] = {3};
    const float x_f32[COUNT] = {-1.0f, -0.5f, 0.25f, 0.75f, 1.0f, 1.5f};
    const float y_f32[COUNT] = {0.5f, -0.25f, 0.75f, -1.0f, 2.0f, -0.5f};
    const float b_f32[3] = {0.5f, -0.25f, 2.0f};
    const float g_f32[COUNT] = {0.25f, -0.5f, 1.0f, -1.5f, 0.75f, 0.125f};
    float want[COUNT];
    uint16_t x_h[COUNT];
    uint16_t y_h[COUNT];
    uint16_t g_h[COUNT];
    uint16_t b_h[3];
    gd_context *ctx = NULL;
    gd_memory_config cfg = binary_config();
    gd_tensor x;
    gd_tensor y;
    gd_tensor g;
    gd_tensor b;
    gd_tensor x32;
    gd_tensor y32;
    gd_tensor b32;
    gd_tensor out;
    gd_tensor dx;
    gd_tensor dy;
    uint32_t i;
    gd_status st;

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_binary_ops: skipped (no supported GPU backend)\n");
        return;
    }
    CHECK_OK(st);

    pack_f16(x_f32, x_h, COUNT);
    pack_f16(y_f32, y_h, COUNT);
    pack_f16(g_f32, g_h, COUNT);
    pack_f16(b_f32, b_h, 3U);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &y));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, shape), 256U, &g));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, bias_shape), 256U, &b));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &x32));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(2U, shape), 256U, &y32));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, gd_shape_make(1U, bias_shape), 256U, &b32));
    CHECK_OK(gd_tensor_write(ctx, &x, x_h, sizeof(x_h)));
    CHECK_OK(gd_tensor_write(ctx, &y, y_h, sizeof(y_h)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_h, sizeof(g_h)));
    CHECK_OK(gd_tensor_write(ctx, &b, b_h, sizeof(b_h)));
    CHECK_OK(gd_tensor_write(ctx, &x32, x_f32, sizeof(x_f32)));
    CHECK_OK(gd_tensor_write(ctx, &y32, y_f32, sizeof(y_f32)));
    CHECK_OK(gd_tensor_write(ctx, &b32, b_f32, sizeof(b_f32)));
    x.requires_grad = true;
    y.requires_grad = true;
    b.requires_grad = true;
    x32.requires_grad = true;
    b32.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_add(ctx, &x, &y, &out));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = x_f32[i] + y_f32[i];
    }
    expect_tensor_f16(ctx, &out, want, COUNT, "add f16 forward");

    CHECK_OK(gd_sub(ctx, &x, &y, &out));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = x_f32[i] - y_f32[i];
    }
    expect_tensor_f16(ctx, &out, want, COUNT, "sub f16 forward");

    CHECK_OK(gd_mul(ctx, &x, &y, &out));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = x_f32[i] * y_f32[i];
    }
    expect_tensor_f16(ctx, &out, want, COUNT, "mul f16 forward");

    CHECK_OK(gd_add(ctx, &x, &b, &out));
    CHECK(out.rank == 2U && out.shape[0] == 2 && out.shape[1] == 3, "broadcast add output shape");
    for (i = 0U; i < COUNT; ++i) {
        want[i] = x_f32[i] + b_f32[i % 3U];
    }
    expect_tensor_f16(ctx, &out, want, COUNT, "add f16 broadcast forward");

    st = gd_add(ctx, &x32, &y32, &out);
    CHECK(st == GD_ERR_UNSUPPORTED, "add f32 unsupported");
    gd_context_clear_error(ctx);

    st = gd_add(ctx, &x32, &b32, &out);
    CHECK(st == GD_ERR_UNSUPPORTED, "add f32 broadcast unsupported");
    gd_context_clear_error(ctx);

    st = gd_mul(ctx, &x32, &b32, &out);
    CHECK(st == GD_ERR_UNSUPPORTED, "mul f32 unsupported");
    gd_context_clear_error(ctx);

    CHECK_OK(gd_mul(ctx, &x, &b, &out));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = x_f32[i] * b_f32[i % 3U];
    }
    expect_tensor_f16(ctx, &out, want, COUNT, "mul f16 broadcast forward");

    CHECK_OK(gd_add_backward(ctx, &x, &b, &g, &dx, &dy));
    expect_tensor_f16(ctx, &dx, g_f32, COUNT, "add broadcast backward dx");
    for (i = 0U; i < 3U; ++i) {
        want[i] = g_f32[i] + g_f32[i + 3U];
    }
    expect_tensor_f16(ctx, &dy, want, 3U, "add broadcast backward dy");

    CHECK_OK(gd_sub_backward(ctx, &x, &y, &g, &dx, &dy));
    expect_tensor_f16(ctx, &dx, g_f32, COUNT, "sub backward dx");
    for (i = 0U; i < COUNT; ++i) {
        want[i] = -g_f32[i];
    }
    expect_tensor_f16(ctx, &dy, want, COUNT, "sub backward dy");

    CHECK_OK(gd_mul_backward(ctx, &x, &y, &g, &dx, &dy));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = g_f32[i] * y_f32[i];
    }
    expect_tensor_f16(ctx, &dx, want, COUNT, "mul backward dx");
    for (i = 0U; i < COUNT; ++i) {
        want[i] = g_f32[i] * x_f32[i];
    }
    expect_tensor_f16(ctx, &dy, want, COUNT, "mul backward dy");

    CHECK_OK(gd_mul_backward(ctx, &x, &b, &g, &dx, &dy));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = g_f32[i] * b_f32[i % 3U];
    }
    expect_tensor_f16(ctx, &dx, want, COUNT, "mul broadcast backward dx");
    for (i = 0U; i < 3U; ++i) {
        want[i] = g_f32[i] * x_f32[i] + g_f32[i + 3U] * x_f32[i + 3U];
    }
    expect_tensor_f16(ctx, &dy, want, 3U, "mul broadcast backward dy");
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mul(ctx, &x, &x, &out));
    CHECK_OK(gd_backward(ctx, &out, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want[i] = 2.0f * x_f32[i] * g_f32[i];
    }
    expect_tensor_f16(ctx, &dx, want, COUNT, "mul autograd fanin accumulates twice");
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_mul(ctx, &x, &b, &out));
    CHECK_OK(gd_backward(ctx, &out, &g));
    CHECK_OK(gd_tensor_grad(ctx, &b, &dy));
    for (i = 0U; i < 3U; ++i) {
        want[i] = g_f32[i] * x_f32[i] + g_f32[i + 3U] * x_f32[i + 3U];
    }
    expect_tensor_f16(ctx, &dy, want, 3U, "mul broadcast autograd dy");
    CHECK_OK(gd_end_step(ctx));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_add(ctx, &x, &b, &out));
    CHECK_OK(gd_backward(ctx, &out, &g));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    expect_tensor_f16(ctx, &dx, g_f32, COUNT, "add broadcast autograd dx");
    CHECK_OK(gd_tensor_grad(ctx, &b, &dy));
    CHECK(dy.rank == 1U && dy.shape[0] == 3, "add broadcast autograd dy shape");
    for (i = 0U; i < 3U; ++i) {
        want[i] = g_f32[i] + g_f32[i + 3U];
    }
    expect_tensor_f16(ctx, &dy, want, 3U, "add broadcast autograd dy");
    CHECK_OK(gd_end_step(ctx));

    gd_context_destroy(ctx);
}

int main(void)
{
    run_binary_ops_test();
    printf("test_binary_ops: ok\n");
    return 0;
}
