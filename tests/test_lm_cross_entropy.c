#include <gradients/gradients.h>

#include <math.h>
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

enum {
    SOFT_ROWS = 6,
    SOFT_DIM = 8,
    SOFT_VOCAB = 32,
    SOFT_HCOUNT = SOFT_ROWS * SOFT_DIM,
    SOFT_WCOUNT = SOFT_VOCAB * SOFT_DIM,
};

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

static float softcap_tanh_f32(float x)
{
    const float ax = abs_f32(x);
    if (ax < 0.000244140625f) {
        return x;
    }
    const float e = expf(-2.0f * ax);
    const float y = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -y : y;
}

static float softcap_f32(float x, float cap)
{
    return cap > 0.0f ? cap * softcap_tanh_f32(x / cap) : x;
}

static float softcap_grad_f32(float soft_logit, float cap)
{
    float y;
    float grad;
    if (cap <= 0.0f) {
        return 1.0f;
    }
    y = soft_logit / cap;
    grad = 1.0f - y * y;
    return grad > 0.0f ? grad : 0.0f;
}

static void compute_softcapped_oracle(const uint16_t *hidden_h,
                                      const uint16_t *weight_h,
                                      const int32_t *targets_h,
                                      float cap,
                                      float *loss_out,
                                      uint16_t *dh_out,
                                      uint16_t *dw_out)
{
    float logits[SOFT_ROWS * SOFT_VOCAB];
    float soft_logits[SOFT_ROWS * SOFT_VOCAB];
    uint16_t dlogits_h[SOFT_ROWS * SOFT_VOCAB];
    float loss_sum = 0.0f;
    int valid_rows = 0;
    int r;
    int v;
    int d;
    for (r = 0; r < SOFT_ROWS; ++r) {
        for (v = 0; v < SOFT_VOCAB; ++v) {
            float acc = 0.0f;
            for (d = 0; d < SOFT_DIM; ++d) {
                const size_t hidx = (size_t)r * (size_t)SOFT_DIM + (size_t)d;
                const size_t widx = (size_t)v * (size_t)SOFT_DIM + (size_t)d;
                acc += f16_bits_to_f32(hidden_h[hidx]) * f16_bits_to_f32(weight_h[widx]);
            }
            logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v] =
                f16_bits_to_f32(f32_to_f16_bits(acc));
            soft_logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v] =
                softcap_f32(logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v], cap);
        }
    }
    for (r = 0; r < SOFT_ROWS; ++r) {
        if (targets_h[r] >= 0 && targets_h[r] < SOFT_VOCAB) {
            valid_rows += 1;
        }
    }
    CHECK(valid_rows > 0, "oracle needs at least one valid target");
    for (r = 0; r < SOFT_ROWS; ++r) {
        const int32_t target = targets_h[r];
        float row_max = soft_logits[(size_t)r * (size_t)SOFT_VOCAB];
        float row_sum = 0.0f;
        if (target < 0 || target >= SOFT_VOCAB) {
            for (v = 0; v < SOFT_VOCAB; ++v) {
                dlogits_h[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v] = f32_to_f16_bits(0.0f);
            }
            continue;
        }
        for (v = 1; v < SOFT_VOCAB; ++v) {
            const float value = soft_logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v];
            if (value > row_max) {
                row_max = value;
            }
        }
        for (v = 0; v < SOFT_VOCAB; ++v) {
            row_sum += expf(soft_logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)v] - row_max);
        }
        loss_sum += logf(row_sum) + row_max - soft_logits[(size_t)r * (size_t)SOFT_VOCAB + (size_t)target];
        for (v = 0; v < SOFT_VOCAB; ++v) {
            const size_t idx = (size_t)r * (size_t)SOFT_VOCAB + (size_t)v;
            float p = expf(soft_logits[idx] - row_max) / row_sum;
            if (v == target) {
                p -= 1.0f;
            }
            p *= (1.0f / (float)valid_rows) * softcap_grad_f32(soft_logits[idx], cap);
            dlogits_h[idx] = f32_to_f16_bits(p);
        }
    }
    *loss_out = loss_sum / (float)valid_rows;
    for (r = 0; r < SOFT_ROWS; ++r) {
        for (d = 0; d < SOFT_DIM; ++d) {
            float acc = 0.0f;
            for (v = 0; v < SOFT_VOCAB; ++v) {
                const size_t gidx = (size_t)r * (size_t)SOFT_VOCAB + (size_t)v;
                const size_t widx = (size_t)v * (size_t)SOFT_DIM + (size_t)d;
                acc += f16_bits_to_f32(dlogits_h[gidx]) * f16_bits_to_f32(weight_h[widx]);
            }
            dh_out[(size_t)r * (size_t)SOFT_DIM + (size_t)d] = f32_to_f16_bits(acc);
        }
    }
    for (v = 0; v < SOFT_VOCAB; ++v) {
        for (d = 0; d < SOFT_DIM; ++d) {
            float acc = 0.0f;
            for (r = 0; r < SOFT_ROWS; ++r) {
                const size_t gidx = (size_t)r * (size_t)SOFT_VOCAB + (size_t)v;
                const size_t hidx = (size_t)r * (size_t)SOFT_DIM + (size_t)d;
                acc += f16_bits_to_f32(dlogits_h[gidx]) * f16_bits_to_f32(hidden_h[hidx]);
            }
            dw_out[(size_t)v * (size_t)SOFT_DIM + (size_t)d] = f32_to_f16_bits(acc);
        }
    }
}

static void run_softcapped_case_with_targets(const int32_t *targets_arg,
                                             float cap,
                                             float *loss_out,
                                             uint16_t *dh_out,
                                             uint16_t *dw_out)
{
    const int64_t hidden_shape[2] = {SOFT_ROWS, SOFT_DIM};
    const int64_t weight_shape[2] = {SOFT_VOCAB, SOFT_DIM};
    const int64_t target_shape[1] = {SOFT_ROWS};
    const int32_t default_targets[SOFT_ROWS] = {0, 15, 3, 31, 16, 1};
    const int32_t *targets_h = targets_arg != NULL ? targets_arg : default_targets;
    uint16_t hidden_h[SOFT_HCOUNT];
    uint16_t weight_h[SOFT_WCOUNT];
    gd_context *ctx = NULL;
    gd_memory_config cfg = lmce_config();
    gd_tensor hidden;
    gd_tensor weight;
    gd_tensor targets;
    gd_tensor loss;
    gd_tensor dh;
    gd_tensor dw;
    create_context_or_skip(&cfg, &ctx);
    fill_hidden(hidden_h, SOFT_HCOUNT);
    fill_weight(weight_h, SOFT_WCOUNT);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, hidden_shape), 256U, &hidden));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, weight_shape), 256U, &weight));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, target_shape), 256U, &targets));
    CHECK_OK(gd_tensor_write(ctx, &hidden, hidden_h, sizeof(hidden_h)));
    CHECK_OK(gd_tensor_write(ctx, &weight, weight_h, sizeof(weight_h)));
    CHECK_OK(gd_tensor_write(ctx, &targets, targets_h, (size_t)SOFT_ROWS * sizeof(targets_h[0])));
    hidden.requires_grad = true;
    weight.requires_grad = true;
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_lm_cross_entropy_softcapped(ctx, &hidden, &weight, &targets, cap, &loss));
    CHECK_OK(gd_tensor_read(ctx, &loss, loss_out, sizeof(*loss_out)));
    CHECK_OK(gd_backward(ctx, &loss, NULL));
    CHECK_OK(gd_tensor_grad(ctx, &hidden, &dh));
    CHECK_OK(gd_tensor_grad(ctx, &weight, &dw));
    CHECK_OK(gd_tensor_read(ctx, &dh, dh_out, (size_t)SOFT_HCOUNT * sizeof(dh_out[0])));
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_out, (size_t)SOFT_WCOUNT * sizeof(dw_out[0])));
    CHECK_OK(gd_end_step(ctx));
    gd_context_destroy(ctx);
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

static void test_ignore_index_matches_oracle(void)
{
    const float cap = 0.0f;
    const int32_t targets_h[SOFT_ROWS] = {0, -1, 3, 31, -1, 1};
    float actual_loss = 0.0f;
    float expected_loss = 0.0f;
    uint16_t hidden_h[SOFT_HCOUNT];
    uint16_t weight_h[SOFT_WCOUNT];
    uint16_t actual_dh[SOFT_HCOUNT];
    uint16_t expected_dh[SOFT_HCOUNT];
    uint16_t actual_dw[SOFT_WCOUNT];
    uint16_t expected_dw[SOFT_WCOUNT];
    uint32_t i;
    memset(actual_dh, 0, sizeof(actual_dh));
    memset(expected_dh, 0, sizeof(expected_dh));
    memset(actual_dw, 0, sizeof(actual_dw));
    memset(expected_dw, 0, sizeof(expected_dw));
    fill_hidden(hidden_h, SOFT_HCOUNT);
    fill_weight(weight_h, SOFT_WCOUNT);
    compute_softcapped_oracle(hidden_h,
                              weight_h,
                              targets_h,
                              cap,
                              &expected_loss,
                              expected_dh,
                              expected_dw);
    run_softcapped_case_with_targets(targets_h, cap, &actual_loss, actual_dh, actual_dw);
    CHECK(abs_f32(actual_loss - expected_loss) <= 1.5e-3f,
          "ignore-index lm_cross_entropy loss matches oracle");
    for (i = 0U; i < SOFT_HCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(actual_dh[i]) - f16_bits_to_f32(expected_dh[i])) <= 1.5e-3f,
              "ignore-index lm_cross_entropy hidden grad matches oracle");
    }
    for (i = 0U; i < SOFT_WCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(actual_dw[i]) - f16_bits_to_f32(expected_dw[i])) <= 1.5e-3f,
              "ignore-index lm_cross_entropy weight grad matches oracle");
    }
}

static void test_softcapped_matches_oracle(void)
{
    const float cap = 0.75f;
    const int32_t targets_h[SOFT_ROWS] = {0, 15, 3, 31, 16, 1};
    float actual_loss = 0.0f;
    float expected_loss = 0.0f;
    uint16_t hidden_h[SOFT_HCOUNT];
    uint16_t weight_h[SOFT_WCOUNT];
    uint16_t actual_dh[SOFT_HCOUNT];
    uint16_t expected_dh[SOFT_HCOUNT];
    uint16_t actual_dw[SOFT_WCOUNT];
    uint16_t expected_dw[SOFT_WCOUNT];
    uint32_t i;
    memset(actual_dh, 0, sizeof(actual_dh));
    memset(expected_dh, 0, sizeof(expected_dh));
    memset(actual_dw, 0, sizeof(actual_dw));
    memset(expected_dw, 0, sizeof(expected_dw));
    fill_hidden(hidden_h, SOFT_HCOUNT);
    fill_weight(weight_h, SOFT_WCOUNT);
    compute_softcapped_oracle(hidden_h,
                              weight_h,
                              targets_h,
                              cap,
                              &expected_loss,
                              expected_dh,
                              expected_dw);
    run_softcapped_case_with_targets(NULL, cap, &actual_loss, actual_dh, actual_dw);
    CHECK(abs_f32(actual_loss - expected_loss) <= 1.5e-3f,
          "softcapped lm_cross_entropy loss matches oracle");
    for (i = 0U; i < SOFT_HCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(actual_dh[i]) - f16_bits_to_f32(expected_dh[i])) <= 1.5e-3f,
              "softcapped lm_cross_entropy hidden grad matches oracle");
    }
    for (i = 0U; i < SOFT_WCOUNT; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(actual_dw[i]) - f16_bits_to_f32(expected_dw[i])) <= 1.5e-3f,
              "softcapped lm_cross_entropy weight grad matches oracle");
    }
}

int main(void)
{
    test_fused_matches_materialized();
    test_ignore_index_matches_oracle();
    test_softcapped_matches_oracle();
    printf("test_lm_cross_entropy: ok\n");
    return 0;
}
