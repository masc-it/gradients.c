#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "test_sdpa_varlen failed: %s (%s:%d)\n", (msg),   \
                    __FILE__, __LINE__);                                        \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_OK(expr) CHECK((expr) == GD_OK, #expr)

static gd_memory_config test_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 1U << 20;
    cfg.state_bytes = 1U << 18;
    cfg.scratch_slot_bytes = 1U << 20;
    cfg.data_slot_bytes = 1U << 16;
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

static float absf32(float x)
{
    return x < 0.0f ? -x : x;
}

static int allowed(int i, int j, int causal, int window, int prefix_len)
{
    if (causal) {
        if (prefix_len > 0) {
            if (i < prefix_len) {
                if (j >= prefix_len) {
                    return 0;
                }
            } else if (j > i) {
                return 0;
            }
        } else if (j > i) {
            return 0;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (i >= prefix_len && j >= prefix_len && (i - j) >= window) {
                return 0;
            }
        } else if ((i - j) >= window) {
            return 0;
        }
    }
    return 1;
}

static void fill_tokens(uint16_t *x, uint32_t count, float base)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        x[i] = f32_to_f16_bits(base + 0.03125f * (float)((i * 17U + 3U) % 19U) -
                              0.015625f * (float)(i % 5U));
    }
}

static void ref_varlen_fwd(const uint16_t *q,
                           const uint16_t *k,
                           const uint16_t *v,
                           const int32_t *cu,
                           uint32_t B,
                           uint32_t Hq,
                           uint32_t Hkv,
                           uint32_t Dh,
                           float scale,
                           int causal,
                           int window,
                           int prefix_len,
                           float *out)
{
    uint32_t b;
    uint32_t hq;
    memset(out, 0, (size_t)cu[B] * (size_t)Hq * (size_t)Dh * sizeof(float));
    for (b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t end = (uint32_t)cu[b + 1U];
        uint32_t T = end - start;
        for (hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / (Hq / Hkv);
            uint32_t i;
            for (i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                double m = -HUGE_VAL;
                double sum = 0.0;
                uint32_t j;
                for (j = 0U; j < T; ++j) {
                    if (allowed((int)i, (int)j, causal, window, prefix_len)) {
                        uint32_t kg = start + j;
                        double s = 0.0;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
                                 (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
                        }
                        s *= (double)scale;
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0U; j < T; ++j) {
                    if (allowed((int)i, (int)j, causal, window, prefix_len)) {
                        uint32_t kg = start + j;
                        double s = 0.0;
                        double e;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
                                 (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
                        }
                        e = exp((double)scale * s - m);
                        sum += e;
                        for (c = 0U; c < Dh; ++c) {
                            out[(qg * Hq + hq) * Dh + c] +=
                                (float)(e * (double)f16_bits_to_f32(v[(kg * Hkv + hkv) * Dh + c]));
                        }
                    }
                }
                if (sum > 0.0) {
                    uint32_t c;
                    for (c = 0U; c < Dh; ++c) {
                        out[(qg * Hq + hq) * Dh + c] =
                            (float)((double)out[(qg * Hq + hq) * Dh + c] / sum);
                    }
                }
            }
        }
    }
}

static void ref_varlen_bwd(const uint16_t *go,
                           const uint16_t *q,
                           const uint16_t *k,
                           const uint16_t *v,
                           const int32_t *cu,
                           uint32_t B,
                           uint32_t Hq,
                           uint32_t Hkv,
                           uint32_t Dh,
                           float scale,
                           int causal,
                           int window,
                           int prefix_len,
                           float *dq,
                           float *dk,
                           float *dv)
{
    const uint32_t N = (uint32_t)cu[B];
    const size_t rows = (size_t)N * (size_t)Hq;
    float *m;
    float *l;
    float *D;
    size_t stat_index;
    uint32_t b;
    CHECK(Hq == 0U || rows / (size_t)Hq == (size_t)N, "reference stats size overflow");
    CHECK(rows <= SIZE_MAX / sizeof(*m), "reference stats byte overflow");
    m = (float *)malloc(rows * sizeof(*m));
    l = (float *)malloc(rows * sizeof(*l));
    D = (float *)malloc(rows * sizeof(*D));
    CHECK(m != NULL && l != NULL && D != NULL, "reference stats allocation");
    memset(dq, 0, (size_t)N * Hq * Dh * sizeof(float));
    memset(dk, 0, (size_t)N * Hkv * Dh * sizeof(float));
    memset(dv, 0, (size_t)N * Hkv * Dh * sizeof(float));
    for (stat_index = 0U; stat_index < rows; ++stat_index) {
        m[stat_index] = -INFINITY;
        l[stat_index] = 0.0f;
        D[stat_index] = 0.0f;
    }
    for (b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        uint32_t hq;
        for (hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / (Hq / Hkv);
            uint32_t i;
            for (i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * Hq + hq;
                double mm = -HUGE_VAL;
                double ll = 0.0;
                double raw = 0.0;
                uint32_t j;
                for (j = 0U; j < T; ++j) {
                    if (allowed((int)i, (int)j, causal, window, prefix_len)) {
                        uint32_t kg = start + j;
                        double s = 0.0;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
                                 (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
                        }
                        s *= (double)scale;
                        if (s > mm) {
                            mm = s;
                        }
                    }
                }
                for (j = 0U; j < T; ++j) {
                    if (allowed((int)i, (int)j, causal, window, prefix_len)) {
                        uint32_t kg = start + j;
                        double s = 0.0;
                        double dp = 0.0;
                        double e;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
                                 (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
                            dp += (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[(kg * Hkv + hkv) * Dh + c]);
                        }
                        e = exp((double)scale * s - mm);
                        ll += e;
                        raw += e * dp;
                    }
                }
                m[row] = (float)mm;
                l[row] = (float)ll;
                D[row] = ll > 0.0 ? (float)(raw / ll) : 0.0f;
            }
        }
    }
    for (b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        uint32_t hq;
        for (hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / (Hq / Hkv);
            uint32_t i;
            for (i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * Hq + hq;
                uint32_t j;
                if (l[row] <= 0.0f) {
                    continue;
                }
                for (j = 0U; j < T; ++j) {
                    if (allowed((int)i, (int)j, causal, window, prefix_len)) {
                        uint32_t kg = start + j;
                        double s = 0.0;
                        double dp = 0.0;
                        double pj;
                        double ds;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
                                 (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
                            dp += (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[(kg * Hkv + hkv) * Dh + c]);
                        }
                        pj = exp((double)scale * s - (double)m[row]) / (double)l[row];
                        ds = pj * (dp - (double)D[row]);
                        for (c = 0U; c < Dh; ++c) {
                            dq[(qg * Hq + hq) * Dh + c] +=
                                (float)((double)scale * ds *
                                        (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]));
                            dk[(kg * Hkv + hkv) * Dh + c] +=
                                (float)((double)scale * ds *
                                        (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]));
                            dv[(kg * Hkv + hkv) * Dh + c] +=
                                (float)(pj * (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]));
                        }
                    }
                }
            }
        }
    }
    free(m);
    free(l);
    free(D);
}

static void test_varlen_fwd_bwd(gd_context *ctx)
{
    enum { N = 5, Hq = 2, Hkv = 1, Dh = 4, B = 2 };
    const int64_t q_shape[3] = {N, Hq, Dh};
    const int64_t k_shape[3] = {N, Hkv, Dh};
    const int64_t cu_shape[1] = {B + 1};
    uint16_t q_data[N * Hq * Dh];
    uint16_t k_data[N * Hkv * Dh];
    uint16_t v_data[N * Hkv * Dh];
    uint16_t go_data[N * Hq * Dh];
    int32_t cu_data[B + 1] = {0, 2, 5};
    uint16_t got[N * Hq * Dh];
    uint16_t dq_got[N * Hq * Dh];
    uint16_t dk_got[N * Hkv * Dh];
    uint16_t dv_got[N * Hkv * Dh];
    float ref[N * Hq * Dh];
    float dq_ref[N * Hq * Dh];
    float dk_ref[N * Hkv * Dh];
    float dv_ref[N * Hkv * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor cu;
    gd_tensor go;
    gd_tensor out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_sdpa_varlen_config cfg;
    uint32_t i;
    fill_tokens(q_data, N * Hq * Dh, -0.25f);
    fill_tokens(k_data, N * Hkv * Dh, 0.125f);
    fill_tokens(v_data, N * Hkv * Dh, -0.375f);
    fill_tokens(go_data, N * Hq * Dh, 0.0625f);
    cfg.scale = 0.5f;
    cfg.causal = true;
    cfg.sliding_window = 0;
    cfg.prefix_len = 0;
    cfg.max_seqlen = 3;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &go));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));
    CHECK_OK(gd_tensor_write(ctx, &go, go_data, sizeof(go_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_sdpa_varlen(ctx, &q, &k, &v, &cu, &cfg, &out));
    CHECK_OK(gd_tensor_read(ctx, &out, got, sizeof(got)));
    ref_varlen_fwd(q_data, k_data, v_data, cu_data, B, Hq, Hkv, Dh, cfg.scale, 1, 0, 0, ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(got[i]) - ref[i]) < 2.0e-3f, "varlen forward mismatch");
    }

    CHECK_OK(gd_sdpa_varlen_backward(ctx, &q, &k, &v, &cu, &go, &cfg, &dq, &dk, &dv));
    CHECK_OK(gd_tensor_read(ctx, &dq, dq_got, sizeof(dq_got)));
    CHECK_OK(gd_tensor_read(ctx, &dk, dk_got, sizeof(dk_got)));
    CHECK_OK(gd_tensor_read(ctx, &dv, dv_got, sizeof(dv_got)));
    ref_varlen_bwd(go_data, q_data, k_data, v_data, cu_data, B, Hq, Hkv, Dh, cfg.scale, 1, 0, 0,
                   dq_ref, dk_ref, dv_ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dq_got[i]) - dq_ref[i]) < 2.5e-3f, "varlen dq mismatch");
    }
    for (i = 0U; i < N * Hkv * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dk_got[i]) - dk_ref[i]) < 2.5e-3f, "varlen dk mismatch");
        CHECK(absf32(f16_bits_to_f32(dv_got[i]) - dv_ref[i]) < 2.5e-3f, "varlen dv mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}


static void test_varlen_dh64_window_case(gd_context *ctx, int prefix_len)
{
    enum { N = 160, Hq = 2, Hkv = 1, Dh = 64, B = 1 };
    const int64_t q_shape[3] = {N, Hq, Dh};
    const int64_t k_shape[3] = {N, Hkv, Dh};
    const int64_t cu_shape[1] = {B + 1};
    uint16_t q_data[N * Hq * Dh];
    uint16_t k_data[N * Hkv * Dh];
    uint16_t v_data[N * Hkv * Dh];
    uint16_t go_data[N * Hq * Dh];
    int32_t cu_data[B + 1] = {0, N};
    uint16_t got[N * Hq * Dh];
    uint16_t dq_got[N * Hq * Dh];
    uint16_t dk_got[N * Hkv * Dh];
    uint16_t dv_got[N * Hkv * Dh];
    float ref[N * Hq * Dh];
    float dq_ref[N * Hq * Dh];
    float dk_ref[N * Hkv * Dh];
    float dv_ref[N * Hkv * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor cu;
    gd_tensor go;
    gd_tensor out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_sdpa_varlen_config cfg;
    uint32_t i;
    fill_tokens(q_data, N * Hq * Dh, -0.0625f);
    fill_tokens(k_data, N * Hkv * Dh, 0.03125f);
    fill_tokens(v_data, N * Hkv * Dh, -0.09375f);
    fill_tokens(go_data, N * Hq * Dh, 0.015625f);
    cfg.scale = 0.125f;
    cfg.causal = true;
    cfg.sliding_window = 8;
    cfg.prefix_len = prefix_len;
    cfg.max_seqlen = N;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &go));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));
    CHECK_OK(gd_tensor_write(ctx, &go, go_data, sizeof(go_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_sdpa_varlen(ctx, &q, &k, &v, &cu, &cfg, &out));
    CHECK_OK(gd_tensor_read(ctx, &out, got, sizeof(got)));
    ref_varlen_fwd(q_data, k_data, v_data, cu_data, B, Hq, Hkv, Dh, cfg.scale, 1,
                   cfg.sliding_window, cfg.prefix_len, ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(got[i]) - ref[i]) < 2.5e-3f,
              "dh64 prefix/window forward mismatch");
    }
    CHECK_OK(gd_sdpa_varlen_backward(ctx, &q, &k, &v, &cu, &go, &cfg, &dq, &dk, &dv));
    CHECK_OK(gd_tensor_read(ctx, &dq, dq_got, sizeof(dq_got)));
    CHECK_OK(gd_tensor_read(ctx, &dk, dk_got, sizeof(dk_got)));
    CHECK_OK(gd_tensor_read(ctx, &dv, dv_got, sizeof(dv_got)));
    ref_varlen_bwd(go_data, q_data, k_data, v_data, cu_data, B, Hq, Hkv, Dh, cfg.scale, 1,
                   cfg.sliding_window, cfg.prefix_len, dq_ref, dk_ref, dv_ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dq_got[i]) - dq_ref[i]) < 4.0e-3f,
              "dh64 prefix/window dq mismatch");
    }
    for (i = 0U; i < N * Hkv * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dk_got[i]) - dk_ref[i]) < 4.0e-3f,
              "dh64 prefix/window dk mismatch");
        CHECK(absf32(f16_bits_to_f32(dv_got[i]) - dv_ref[i]) < 4.0e-3f,
              "dh64 prefix/window dv mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}

static void test_varlen_dh64_long_sequence_window_no_split(gd_context *ctx)
{
    enum { N = 520, Hq = 1, Hkv = 1, Dh = 64, B = 1 };
    const int64_t q_shape[3] = {N, Hq, Dh};
    const int64_t k_shape[3] = {N, Hkv, Dh};
    const int64_t cu_shape[1] = {B + 1};
    uint16_t q_data[N * Hq * Dh];
    uint16_t k_data[N * Hkv * Dh];
    uint16_t v_data[N * Hkv * Dh];
    uint16_t go_data[N * Hq * Dh];
    int32_t cu_data[B + 1] = {0, N};
    uint16_t dq_got[N * Hq * Dh];
    uint16_t dk_got[N * Hkv * Dh];
    uint16_t dv_got[N * Hkv * Dh];
    float dq_ref[N * Hq * Dh];
    float dk_ref[N * Hkv * Dh];
    float dv_ref[N * Hkv * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor cu;
    gd_tensor go;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_sdpa_varlen_config cfg;
    uint32_t i;
    fill_tokens(q_data, N * Hq * Dh, -0.046875f);
    fill_tokens(k_data, N * Hkv * Dh, 0.0234375f);
    fill_tokens(v_data, N * Hkv * Dh, -0.078125f);
    fill_tokens(go_data, N * Hq * Dh, 0.01171875f);
    cfg.scale = 0.125f;
    cfg.causal = true;
    cfg.sliding_window = 32;
    cfg.prefix_len = 0;
    cfg.max_seqlen = N;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, k_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &go));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));
    CHECK_OK(gd_tensor_write(ctx, &go, go_data, sizeof(go_data)));

    /* T exceeds the default split threshold, but the prefix-free local dK/dV
     * query span is only window + key_block - 1.  This covers the window-aware
     * path that keeps n_splits at 1 and avoids the partial dK/dV buffer. */
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_sdpa_varlen_backward(ctx, &q, &k, &v, &cu, &go, &cfg, &dq, &dk, &dv));
    CHECK_OK(gd_tensor_read(ctx, &dq, dq_got, sizeof(dq_got)));
    CHECK_OK(gd_tensor_read(ctx, &dk, dk_got, sizeof(dk_got)));
    CHECK_OK(gd_tensor_read(ctx, &dv, dv_got, sizeof(dv_got)));
    ref_varlen_bwd(go_data, q_data, k_data, v_data, cu_data, B, Hq, Hkv, Dh, cfg.scale, 1,
                   cfg.sliding_window, cfg.prefix_len, dq_ref, dk_ref, dv_ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dq_got[i]) - dq_ref[i]) < 4.0e-3f,
              "dh64 long window dq mismatch");
    }
    for (i = 0U; i < N * Hkv * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(dk_got[i]) - dk_ref[i]) < 4.0e-3f,
              "dh64 long window dk mismatch");
        CHECK(absf32(f16_bits_to_f32(dv_got[i]) - dv_ref[i]) < 4.0e-3f,
              "dh64 long window dv mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}

static int decode_allowed(int qpos, int j, int window, int prefix_len)
{
    return allowed(qpos, j, 1, window, prefix_len);
}

static void ref_decode(const uint16_t *q,
                       const uint16_t *k,
                       const uint16_t *v,
                       int cache_pos,
                       uint32_t B,
                       uint32_t Tq,
                       uint32_t Tmax,
                       uint32_t Hq,
                       uint32_t Hkv,
                       uint32_t Dh,
                       float scale,
                       int window,
                       int prefix_len,
                       float *out)
{
    uint32_t b;
    memset(out, 0, B * Tq * Hq * Dh * sizeof(float));
    for (b = 0U; b < B; ++b) {
        uint32_t hq;
        for (hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / (Hq / Hkv);
            uint32_t i;
            for (i = 0U; i < Tq; ++i) {
                int qpos = cache_pos + (int)i;
                uint32_t qbase = ((b * Tq + i) * Hq + hq) * Dh;
                uint32_t live = (uint32_t)(cache_pos + (int)Tq);
                double m = -HUGE_VAL;
                double sum = 0.0;
                uint32_t j;
                if (live > Tmax) {
                    live = Tmax;
                }
                for (j = 0U; j < live; ++j) {
                    if (decode_allowed(qpos, (int)j, window, prefix_len)) {
                        uint32_t kbase = ((b * Tmax + j) * Hkv + hkv) * Dh;
                        double s = 0.0;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[qbase + c]) *
                                 (double)f16_bits_to_f32(k[kbase + c]);
                        }
                        s *= (double)scale;
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0U; j < live; ++j) {
                    if (decode_allowed(qpos, (int)j, window, prefix_len)) {
                        uint32_t kbase = ((b * Tmax + j) * Hkv + hkv) * Dh;
                        double s = 0.0;
                        double e;
                        uint32_t c;
                        for (c = 0U; c < Dh; ++c) {
                            s += (double)f16_bits_to_f32(q[qbase + c]) *
                                 (double)f16_bits_to_f32(k[kbase + c]);
                        }
                        e = exp((double)scale * s - m);
                        sum += e;
                        for (c = 0U; c < Dh; ++c) {
                            out[qbase + c] += (float)(e * (double)f16_bits_to_f32(v[kbase + c]));
                        }
                    }
                }
                if (sum > 0.0) {
                    uint32_t c;
                    for (c = 0U; c < Dh; ++c) {
                        out[qbase + c] = (float)((double)out[qbase + c] / sum);
                    }
                }
            }
        }
    }
}

static void test_decode(gd_context *ctx)
{
    enum { B = 1, Tq = 2, Tmax = 5, Hq = 2, Hkv = 1, Dh = 4 };
    const int64_t q_shape[4] = {B, Tq, Hq, Dh};
    const int64_t k_shape[4] = {B, Tmax, Hkv, Dh};
    uint16_t q_data[B * Tq * Hq * Dh];
    uint16_t k_data[B * Tmax * Hkv * Dh];
    uint16_t v_data[B * Tmax * Hkv * Dh];
    int32_t pos_data = 2;
    uint16_t got[B * Tq * Hq * Dh];
    float ref[B * Tq * Hq * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor pos;
    gd_tensor out;
    gd_sdpa_decode_config cfg;
    uint32_t i;
    fill_tokens(q_data, B * Tq * Hq * Dh, -0.125f);
    fill_tokens(k_data, B * Tmax * Hkv * Dh, 0.25f);
    fill_tokens(v_data, B * Tmax * Hkv * Dh, -0.5f);
    cfg.scale = 0.5f;
    cfg.sliding_window = 0;
    cfg.prefix_len = 0;
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, k_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, k_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, GD_SCALAR_SHAPE, 256U, &pos));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &pos, &pos_data, sizeof(pos_data)));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    CHECK_OK(gd_sdpa_decode(ctx, &q, &k, &v, &pos, &cfg, &out));
    CHECK_OK(gd_tensor_read(ctx, &out, got, sizeof(got)));
    ref_decode(q_data, k_data, v_data, pos_data, B, Tq, Tmax, Hq, Hkv, Dh, cfg.scale, 0, 0, ref);
    for (i = 0U; i < B * Tq * Hq * Dh; ++i) {
        CHECK(absf32(f16_bits_to_f32(got[i]) - ref[i]) < 2.0e-3f, "decode forward mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = test_config();
    {
        gd_status st = gd_context_create(&cfg, &ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("test_sdpa_varlen: skipped (no supported GPU backend)\n");
            return 0;
        }
        CHECK_OK(st);
    }
    test_varlen_fwd_bwd(ctx);
    test_varlen_dh64_window_case(ctx, 4);
    test_varlen_dh64_window_case(ctx, 0);
    test_varlen_dh64_long_sequence_window_no_split(ctx);
    test_decode(ctx);
    gd_context_destroy(ctx);
    printf("test_sdpa_varlen: ok\n");
    return 0;
}
