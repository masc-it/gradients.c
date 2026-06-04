#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_reduce_ops failed: %s (%s:%d)\n", (msg),   \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static gd_memory_config reduce_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 32768U;
    cfg.state_bytes = 4096U;
    cfg.scratch_slot_bytes = 131072U;
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

static void expect_f32_tensor(gd_context *ctx,
                              const gd_tensor *tensor,
                              const float *want,
                              uint32_t count,
                              float tol,
                              const char *what)
{
    float got[32];
    uint32_t i;
    CHECK(count <= 32U, "helper count");
    CHECK_OK(gd_tensor_read(ctx, tensor, got, (size_t)count * sizeof(got[0])));
    for (i = 0U; i < count; ++i) {
        CHECK(abs_f32(got[i] - want[i]) <= tol, what);
    }
}

static void create_context_or_skip(const gd_memory_config *cfg, gd_context **out_ctx)
{
    gd_status st = gd_context_create(cfg, out_ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_reduce_ops: skipped (no supported GPU backend)\n");
        exit(0);
    }
    CHECK_OK(st);
}

static void test_reduce_forward_backward(void)
{
    enum { COUNT = 12 };
    const int64_t shape[2] = {3, 4};
    const float x_data[COUNT] = {
        -1.0f, -0.5f, 0.25f, 0.75f,
         1.0f,  1.5f, 2.00f, -2.0f,
         0.5f, -1.5f, 0.125f, 0.375f,
    };
    float want_grad[COUNT];
    gd_context *ctx = NULL;
    gd_memory_config cfg = reduce_config();
    gd_tensor x;
    gd_tensor sum;
    gd_tensor mean;
    gd_tensor dx;
    float got_sum = 0.0f;
    float got_mean = 0.0f;
    float ref_sum = 0.0f;
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, 2U, shape, 256U, &x));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    x.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));

    for (i = 0U; i < COUNT; ++i) {
        ref_sum += x_data[i];
    }

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_sum(ctx, &x, &sum));
    CHECK(sum.rank == 0U, "reduce_sum output is scalar");
    CHECK_OK(gd_tensor_read(ctx, &sum, &got_sum, sizeof(got_sum)));
    CHECK(abs_f32(got_sum - ref_sum) <= 1.0e-5f, "reduce_sum f32 forward");
    CHECK_OK(gd_reduce_mean(ctx, &x, &mean));
    CHECK(mean.rank == 0U, "reduce_mean output is scalar");
    CHECK_OK(gd_tensor_read(ctx, &mean, &got_mean, sizeof(got_mean)));
    CHECK(abs_f32(got_mean - ref_sum / (float)COUNT) <= 1.0e-6f, "reduce_mean f32 forward");
    CHECK_OK(gd_reduce_mean_backward(ctx, &x, &sum, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want_grad[i] = ref_sum / (float)COUNT;
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_mean backward broadcast scale");
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_mean(ctx, &x, &mean));
    CHECK_OK(gd_backward(ctx, &mean, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want_grad[i] = 1.0f / (float)COUNT;
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_mean autograd gradient");
    CHECK_OK(gd_end(ctx));

    gd_context_destroy(ctx);
}

static void test_reduce_axis_forward_backward(void)
{
    enum { COUNT = 6 };
    const int64_t shape[2] = {2, 3};
    const float x_data[COUNT] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float want_row_sum[2] = {6.0f, 15.0f};
    const float want_col_mean[3] = {2.5f, 3.5f, 4.5f};
    float want_grad[COUNT];
    gd_context *ctx = NULL;
    gd_memory_config cfg = reduce_config();
    gd_tensor x;
    gd_tensor out;
    gd_tensor dx;
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, 2U, shape, 256U, &x));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    x.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_sum_axis(ctx, &x, 1, false, &out));
    CHECK(out.rank == 1U && out.shape[0] == 2, "reduce_sum_axis output shape");
    expect_f32_tensor(ctx, &out, want_row_sum, 2U, 1.0e-6f, "reduce_sum_axis rows");
    CHECK_OK(gd_reduce_sum_axis_backward(ctx, &x, &out, 1, false, &dx));
    for (i = 0U; i < 3U; ++i) {
        want_grad[i] = want_row_sum[0];
        want_grad[i + 3U] = want_row_sum[1];
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_sum_axis direct backward");
    CHECK_OK(gd_backward(ctx, &out, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want_grad[i] = 1.0f;
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_sum_axis autograd grad");
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_mean_axis(ctx, &x, 0, true, &out));
    CHECK(out.rank == 2U && out.shape[0] == 1 && out.shape[1] == 3,
          "reduce_mean_axis keepdims output shape");
    expect_f32_tensor(ctx, &out, want_col_mean, 3U, 1.0e-6f, "reduce_mean_axis cols keepdims");
    CHECK_OK(gd_backward(ctx, &out, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want_grad[i] = 0.5f;
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_mean_axis keepdims autograd grad");
    CHECK_OK(gd_end(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_mean_axis(ctx, &x, -1, false, &out));
    CHECK(out.rank == 1U && out.shape[0] == 2, "reduce_mean_axis negative axis shape");
    CHECK_OK(gd_backward(ctx, &out, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    for (i = 0U; i < COUNT; ++i) {
        want_grad[i] = 1.0f / 3.0f;
    }
    expect_f32_tensor(ctx, &dx, want_grad, COUNT, 1.0e-6f, "reduce_mean_axis negative autograd grad");
    CHECK_OK(gd_end(ctx));

    gd_context_destroy(ctx);
}

static void test_mse_graph(void)
{
    enum { COUNT = 6 };
    const int64_t shape[2] = {2, 3};
    const float pred_data[COUNT] = {0.0f, 0.25f, 0.75f, 1.0f, -0.5f, 1.5f};
    const float target_data[COUNT] = {0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f};
    float want_grad[COUNT];
    gd_context *ctx = NULL;
    gd_memory_config cfg = reduce_config();
    gd_tensor pred;
    gd_tensor target;
    gd_tensor diff;
    gd_tensor sq;
    gd_tensor loss;
    gd_tensor dpred;
    float got_loss = 0.0f;
    float want_loss = 0.0f;
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, 2U, shape, 256U, &pred));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F32, 2U, shape, 256U, &target));
    CHECK_OK(gd_tensor_write(ctx, &pred, pred_data, sizeof(pred_data)));
    CHECK_OK(gd_tensor_write(ctx, &target, target_data, sizeof(target_data)));
    pred.requires_grad = true;
    target.requires_grad = false;
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_sub(ctx, &pred, &target, &diff));
    CHECK_OK(gd_mul(ctx, &diff, &diff, &sq));
    CHECK_OK(gd_reduce_mean(ctx, &sq, &loss));
    CHECK_OK(gd_tensor_read(ctx, &loss, &got_loss, sizeof(got_loss)));
    for (i = 0U; i < COUNT; ++i) {
        float d = pred_data[i] - target_data[i];
        want_loss += d * d / (float)COUNT;
        want_grad[i] = 2.0f * d / (float)COUNT;
    }
    CHECK(abs_f32(got_loss - want_loss) <= 1.0e-6f, "mse graph loss");
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &pred, &dpred));
    expect_f32_tensor(ctx, &dpred, want_grad, COUNT, 1.0e-6f, "mse graph autograd grad");
    CHECK_OK(gd_end(ctx));

    gd_context_destroy(ctx);
}

static void test_reduce_f16(void)
{
    enum { COUNT = 8 };
    const int64_t shape[1] = {COUNT};
    const float x_f32[COUNT] = {0.5f, -1.0f, 0.25f, 0.75f, 1.5f, -0.5f, 0.125f, 0.375f};
    uint16_t x_h[COUNT];
    uint16_t got_h = 0U;
    float want = 0.0f;
    gd_context *ctx = NULL;
    gd_memory_config cfg = reduce_config();
    gd_tensor x;
    gd_tensor sum;
    uint32_t i;
    create_context_or_skip(&cfg, &ctx);
    pack_f16(x_f32, x_h, COUNT);
    for (i = 0U; i < COUNT; ++i) {
        want += x_f32[i];
    }
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 1U, shape, 256U, &x));
    CHECK_OK(gd_tensor_write(ctx, &x, x_h, sizeof(x_h)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_reduce_sum(ctx, &x, &sum));
    CHECK_OK(gd_tensor_read(ctx, &sum, &got_h, sizeof(got_h)));
    CHECK(abs_f32(f16_bits_to_f32(got_h) - want) <= 1.0e-3f, "reduce_sum f16 forward");
    CHECK_OK(gd_end(ctx));
    gd_context_destroy(ctx);
}

int main(void)
{
    test_reduce_forward_backward();
    test_reduce_axis_forward_backward();
    test_mse_graph();
    test_reduce_f16();
    printf("test_reduce_ops: ok\n");
    return 0;
}
