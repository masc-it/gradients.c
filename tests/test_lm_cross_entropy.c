#include <gradients/gradients.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                               \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "test_lm_cross_entropy failed: %s (%s:%d)\n", (msg),     \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
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

static gd_memory_config lmce_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 1024U * 1024U;
    cfg.state_bytes = 64U * 1024U;
    cfg.scratch_slot_bytes = 4U * 1024U * 1024U;
    cfg.data_slot_bytes = 64U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void create_context_or_skip(const gd_memory_config *cfg, gd_context **out_ctx)
{
    gd_status st = gd_context_create(cfg, out_ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_lm_cross_entropy: skipped (no supported GPU backend)\n");
        exit(0);
    }
    CHECK_OK(st);
}

static void fill_hidden(uint16_t *dst, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        float v = -0.7f + 0.03125f * (float)((i * 13U + 5U) % 37U);
        dst[i] = f32_to_f16_bits(v);
    }
}

static void fill_weight(uint16_t *dst, size_t count)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        float v = -0.4f + 0.02734375f * (float)((i * 11U + 3U) % 41U);
        dst[i] = f32_to_f16_bits(v);
    }
}

static void run_case(bool fused, float *loss_out, uint16_t *dh_out, uint16_t *dw_out)
{
    enum { ROWS = 6, DIM = 8, VOCAB = 2048 };
    const int64_t hidden_shape[2] = {ROWS, DIM};
    const int64_t weight_shape[2] = {VOCAB, DIM};
    const int64_t target_shape[1] = {ROWS};
    const int32_t targets_h[ROWS] = {0, 1000, 3, 2047, 1024, 1};
    uint16_t hidden_h[ROWS * DIM];
    uint16_t weight_h[VOCAB * DIM];
    gd_context *ctx = NULL;
    gd_memory_config cfg = lmce_config();
    gd_tensor hidden;
    gd_tensor weight;
    gd_tensor targets;
    gd_tensor logits;
    gd_tensor loss;
    gd_tensor dh;
    gd_tensor dw;
    create_context_or_skip(&cfg, &ctx);
    fill_hidden(hidden_h, ROWS * DIM);
    fill_weight(weight_h, VOCAB * DIM);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, hidden_shape), 256U, &hidden));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, weight_shape), 256U, &weight));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, target_shape), 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &hidden, hidden_h, sizeof(hidden_h)));
    CHECK_OK(gd_tensor_write(ctx, &weight, weight_h, sizeof(weight_h)));
    CHECK_OK(gd_tensor_write(ctx, &targets, targets_h, sizeof(targets_h)));
    hidden.requires_grad = true;
    weight.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    if (fused) {
        CHECK_OK(gd_lm_cross_entropy(ctx, &hidden, &weight, &targets, &loss));
    } else {
        CHECK_OK(gd_linear_transposed_weight(ctx, &hidden, &weight, NULL, &logits));
        CHECK_OK(gd_cross_entropy(ctx, &logits, &targets, &loss));
    }
    CHECK_OK(gd_tensor_read(ctx, &loss, loss_out, sizeof(*loss_out)));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &hidden, &dh));
    CHECK_OK(gd_tensor_grad(ctx, &weight, &dw));
    CHECK_OK(gd_tensor_read(ctx, &dh, dh_out, (size_t)ROWS * (size_t)DIM * sizeof(dh_out[0])));
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_out, (size_t)VOCAB * (size_t)DIM * sizeof(dw_out[0])));
    CHECK_OK(gd_end_step(ctx));
    gd_context_destroy(ctx);
}

static void test_fused_matches_materialized(void)
{
    enum { ROWS = 6, DIM = 8, VOCAB = 2048, HCOUNT = ROWS * DIM, WCOUNT = VOCAB * DIM };
    float baseline_loss = 0.0f;
    float fused_loss = 0.0f;
    uint16_t baseline_dh[HCOUNT];
    uint16_t fused_dh[HCOUNT];
    uint16_t baseline_dw[WCOUNT];
    uint16_t fused_dw[WCOUNT];
    uint32_t i;
    memset(baseline_dh, 0, sizeof(baseline_dh));
    memset(fused_dh, 0, sizeof(fused_dh));
    memset(baseline_dw, 0, sizeof(baseline_dw));
    memset(fused_dw, 0, sizeof(fused_dw));
    run_case(false, &baseline_loss, baseline_dh, baseline_dw);
    run_case(true, &fused_loss, fused_dh, fused_dw);
    CHECK(abs_f32(baseline_loss - fused_loss) <= 1.0e-6f, "lm_cross_entropy loss matches materialized path");
    for (i = 0U; i < HCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(baseline_dh[i]) - f16_bits_to_f32(fused_dh[i])) <= 2.0e-4f,
              "lm_cross_entropy hidden grad matches");
    }
    for (i = 0U; i < WCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(baseline_dw[i]) - f16_bits_to_f32(fused_dw[i])) <= 2.0e-4f,
              "lm_cross_entropy weight grad matches");
    }
}

int main(void)
{
    test_fused_matches_materialized();
    printf("test_lm_cross_entropy: ok\n");
    return 0;
}
