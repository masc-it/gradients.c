#include <gradients/gradients.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_minimax_m3_sparse_attention failed: %s (%s:%d)\n", \
                    (msg), __FILE__, __LINE__);                                \
            exit(1);                                                           \
        }                                                                      \
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
    union { float f; uint32_t u; } v;
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
        if (exp < -24) {
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
    union { uint32_t u; float f; } v;
    if (exp == 0U) {
        int32_t e = -14;
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            e -= 1;
        }
        mant &= 0x3ffU;
        v.u = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        return v.f;
    }
    if (exp == 31U) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    return v.f;
}

static float abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static void fill_tokens(uint16_t *x, uint32_t count, float base)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        x[i] = f32_to_f16_bits(base + 0.046875f * (float)((i * 13U + 5U) % 17U) -
                              0.03125f * (float)(i % 7U));
    }
}

static double dot_h(const uint16_t *a,
                    const uint16_t *b,
                    uint32_t ia,
                    uint32_t ib,
                    uint32_t h,
                    uint32_t H,
                    uint32_t Dh)
{
    double s = 0.0;
    uint32_t c;
    for (c = 0U; c < Dh; ++c) {
        s += (double)f16_bits_to_f32(a[(ia * H + h) * Dh + c]) *
             (double)f16_bits_to_f32(b[(ib * H + h) * Dh + c]);
    }
    return s;
}

static void ref_topk(const uint16_t *q,
                     const uint16_t *k,
                     const int32_t *cu,
                     uint32_t B,
                     uint32_t H,
                     uint32_t Dh,
                     uint32_t block_size,
                     uint32_t topk,
                     uint32_t init_blocks,
                     uint32_t local_blocks,
                     int32_t *out)
{
    uint32_t h;
    uint32_t n;
    memset(out, 0xff, (size_t)H * (size_t)cu[B] * (size_t)topk * sizeof(out[0]));
    for (h = 0U; h < H; ++h) {
        for (n = 0U; n < (uint32_t)cu[B]; ++n) {
            uint32_t t;
            for (t = 0U; t < topk; ++t) {
                out[(h * (uint32_t)cu[B] + n) * topk + t] = -1;
            }
        }
    }
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (h = 0U; h < H; ++h) {
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t valid_blocks = (i + block_size) / block_size;
                uint32_t local_start = valid_blocks > local_blocks ? valid_blocks - local_blocks : 0U;
                double best_score[4] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
                int32_t best_idx[4] = {-1, -1, -1, -1};
                CHECK(topk <= 4U, "ref topk capacity");
                for (uint32_t blk = 0U; blk < valid_blocks; ++blk) {
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    double score = -HUGE_VAL;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double s = dot_h(q, k, start + i, start + j, h, H, Dh);
                        if (s > score) {
                            score = s;
                        }
                    }
                    if (blk < init_blocks) {
                        score = 1.0e30;
                    }
                    if (local_blocks > 0U && blk >= local_start && blk < valid_blocks) {
                        score = 1.0e29;
                    }
                    for (uint32_t slot = 0U; slot < topk; ++slot) {
                        if (score > best_score[slot]) {
                            for (uint32_t move = topk - 1U; move > slot; --move) {
                                best_score[move] = best_score[move - 1U];
                                best_idx[move] = best_idx[move - 1U];
                            }
                            best_score[slot] = score;
                            best_idx[slot] = (int32_t)blk;
                            break;
                        }
                    }
                }
                for (uint32_t t = 0U; t < topk; ++t) {
                    out[(h * (uint32_t)cu[B] + start + i) * topk + t] =
                        t < valid_blocks ? best_idx[t] : -1;
                }
            }
        }
    }
}

static int contains_block(const int32_t *topk,
                          uint32_t N,
                          uint32_t topk_count,
                          uint32_t h,
                          uint32_t qg,
                          uint32_t blk)
{
    uint32_t t;
    for (t = 0U; t < topk_count; ++t) {
        if (topk[(h * N + qg) * topk_count + t] == (int32_t)blk) {
            return 1;
        }
    }
    return 0;
}

static void ref_sparse_fwd(const uint16_t *q,
                           const uint16_t *k,
                           const uint16_t *v,
                           const int32_t *cu,
                           const int32_t *topk,
                           uint32_t B,
                           uint32_t H,
                           uint32_t Dh,
                           uint32_t block_size,
                           uint32_t topk_count,
                           float scale,
                           float *out)
{
    uint32_t N = (uint32_t)cu[B];
    memset(out, 0, (size_t)N * H * Dh * sizeof(out[0]));
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t h = 0U; h < H; ++h) {
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                double m = -HUGE_VAL;
                double l = 0.0;
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(h * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double s = (double)scale * dot_h(q, k, qg, start + j, h, H, Dh);
                        if (s > m) { m = s; }
                    }
                }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(h * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double e = exp((double)scale * dot_h(q, k, qg, start + j, h, H, Dh) - m);
                        l += e;
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            out[(qg * H + h) * Dh + c] +=
                                (float)(e * (double)f16_bits_to_f32(v[((start + j) * H + h) * Dh + c]));
                        }
                    }
                }
                if (l > 0.0) {
                    for (uint32_t c = 0U; c < Dh; ++c) {
                        out[(qg * H + h) * Dh + c] = (float)((double)out[(qg * H + h) * Dh + c] / l);
                    }
                }
            }
        }
    }
}

static void ref_sparse_bwd(const uint16_t *go,
                           const uint16_t *q,
                           const uint16_t *k,
                           const uint16_t *v,
                           const int32_t *cu,
                           const int32_t *topk,
                           uint32_t B,
                           uint32_t H,
                           uint32_t Dh,
                           uint32_t block_size,
                           uint32_t topk_count,
                           float scale,
                           float *dq,
                           float *dk,
                           float *dv)
{
    uint32_t N = (uint32_t)cu[B];
    float m[64];
    float l[64];
    float D[64];
    CHECK(N * H <= 64U, "ref stats capacity");
    memset(dq, 0, (size_t)N * H * Dh * sizeof(dq[0]));
    memset(dk, 0, (size_t)N * H * Dh * sizeof(dk[0]));
    memset(dv, 0, (size_t)N * H * Dh * sizeof(dv[0]));
    for (uint32_t row = 0U; row < N * H; ++row) {
        m[row] = -INFINITY;
        l[row] = 0.0f;
        D[row] = 0.0f;
    }
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t h = 0U; h < H; ++h) {
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * H + h;
                double mm = -HUGE_VAL;
                double ll = 0.0;
                double raw = 0.0;
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(h * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double s = (double)scale * dot_h(q, k, qg, start + j, h, H, Dh);
                        if (s > mm) { mm = s; }
                    }
                }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(h * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double dp = 0.0;
                        double e = exp((double)scale * dot_h(q, k, qg, start + j, h, H, Dh) - mm);
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dp += (double)f16_bits_to_f32(go[(qg * H + h) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[((start + j) * H + h) * Dh + c]);
                        }
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
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t h = 0U; h < H; ++h) {
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * H + h;
                if (l[row] <= 0.0f) { continue; }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(h * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        uint32_t kg = start + j;
                        double dp = 0.0;
                        double pj = exp((double)scale * dot_h(q, k, qg, kg, h, H, Dh) - (double)m[row]) /
                                    (double)l[row];
                        double ds;
                        CHECK(contains_block(topk, N, topk_count, h, qg, blk), "selected block missing");
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dp += (double)f16_bits_to_f32(go[(qg * H + h) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[(kg * H + h) * Dh + c]);
                        }
                        ds = pj * (dp - (double)D[row]);
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dq[(qg * H + h) * Dh + c] +=
                                (float)((double)scale * ds * (double)f16_bits_to_f32(k[(kg * H + h) * Dh + c]));
                            dk[(kg * H + h) * Dh + c] +=
                                (float)((double)scale * ds * (double)f16_bits_to_f32(q[(qg * H + h) * Dh + c]));
                            dv[(kg * H + h) * Dh + c] +=
                                (float)(pj * (double)f16_bits_to_f32(go[(qg * H + h) * Dh + c]));
                        }
                    }
                }
            }
        }
    }
}

static double dot_qk_h(const uint16_t *q,
                       const uint16_t *k,
                       uint32_t qg,
                       uint32_t kg,
                       uint32_t hq,
                       uint32_t Hq,
                       uint32_t hkv,
                       uint32_t Hkv,
                       uint32_t Dh)
{
    double s = 0.0;
    uint32_t c;
    for (c = 0U; c < Dh; ++c) {
        s += (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]) *
             (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]);
    }
    return s;
}

static void ref_sparse_fwd_gqa(const uint16_t *q,
                               const uint16_t *k,
                               const uint16_t *v,
                               const int32_t *cu,
                               const int32_t *topk,
                               uint32_t B,
                               uint32_t Hq,
                               uint32_t Hkv,
                               uint32_t Dh,
                               uint32_t block_size,
                               uint32_t topk_count,
                               float scale,
                               float *out)
{
    uint32_t N = (uint32_t)cu[B];
    uint32_t group = Hq / Hkv;
    memset(out, 0, (size_t)N * Hq * Dh * sizeof(out[0]));
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / group;
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                double m = -HUGE_VAL;
                double l = 0.0;
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(hkv * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double s = (double)scale * dot_qk_h(q, k, qg, start + j, hq, Hq, hkv, Hkv, Dh);
                        if (s > m) { m = s; }
                    }
                }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(hkv * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double e = exp((double)scale * dot_qk_h(q, k, qg, start + j, hq, Hq, hkv, Hkv, Dh) - m);
                        l += e;
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            out[(qg * Hq + hq) * Dh + c] +=
                                (float)(e * (double)f16_bits_to_f32(v[((start + j) * Hkv + hkv) * Dh + c]));
                        }
                    }
                }
                if (l > 0.0) {
                    for (uint32_t c = 0U; c < Dh; ++c) {
                        out[(qg * Hq + hq) * Dh + c] = (float)((double)out[(qg * Hq + hq) * Dh + c] / l);
                    }
                }
            }
        }
    }
}

static void ref_sparse_bwd_gqa(const uint16_t *go,
                               const uint16_t *q,
                               const uint16_t *k,
                               const uint16_t *v,
                               const int32_t *cu,
                               const int32_t *topk,
                               uint32_t B,
                               uint32_t Hq,
                               uint32_t Hkv,
                               uint32_t Dh,
                               uint32_t block_size,
                               uint32_t topk_count,
                               float scale,
                               float *dq,
                               float *dk,
                               float *dv)
{
    uint32_t N = (uint32_t)cu[B];
    uint32_t group = Hq / Hkv;
    float m[128];
    float l[128];
    float D[128];
    CHECK(N * Hq <= 128U, "ref GQA stats capacity");
    memset(dq, 0, (size_t)N * Hq * Dh * sizeof(dq[0]));
    memset(dk, 0, (size_t)N * Hkv * Dh * sizeof(dk[0]));
    memset(dv, 0, (size_t)N * Hkv * Dh * sizeof(dv[0]));
    for (uint32_t row = 0U; row < N * Hq; ++row) {
        m[row] = -INFINITY;
        l[row] = 0.0f;
        D[row] = 0.0f;
    }
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / group;
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * Hq + hq;
                double mm = -HUGE_VAL;
                double ll = 0.0;
                double raw = 0.0;
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(hkv * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        double s = (double)scale * dot_qk_h(q, k, qg, start + j, hq, Hq, hkv, Hkv, Dh);
                        if (s > mm) { mm = s; }
                    }
                }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(hkv * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        uint32_t kg = start + j;
                        double dp = 0.0;
                        double e = exp((double)scale * dot_qk_h(q, k, qg, kg, hq, Hq, hkv, Hkv, Dh) - mm);
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dp += (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[(kg * Hkv + hkv) * Dh + c]);
                        }
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
    for (uint32_t b = 0U; b < B; ++b) {
        uint32_t start = (uint32_t)cu[b];
        uint32_t T = (uint32_t)cu[b + 1U] - start;
        for (uint32_t hq = 0U; hq < Hq; ++hq) {
            uint32_t hkv = hq / group;
            for (uint32_t i = 0U; i < T; ++i) {
                uint32_t qg = start + i;
                uint32_t row = qg * Hq + hq;
                if (l[row] <= 0.0f) { continue; }
                for (uint32_t t = 0U; t < topk_count; ++t) {
                    int32_t blk_i32 = topk[(hkv * N + qg) * topk_count + t];
                    if (blk_i32 < 0) { continue; }
                    uint32_t blk = (uint32_t)blk_i32;
                    uint32_t k0 = blk * block_size;
                    uint32_t k1 = k0 + block_size < T ? k0 + block_size : T;
                    for (uint32_t j = k0; j < k1 && j <= i; ++j) {
                        uint32_t kg = start + j;
                        double dp = 0.0;
                        double pj = exp((double)scale * dot_qk_h(q, k, qg, kg, hq, Hq, hkv, Hkv, Dh) -
                                        (double)m[row]) / (double)l[row];
                        double ds;
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dp += (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]) *
                                  (double)f16_bits_to_f32(v[(kg * Hkv + hkv) * Dh + c]);
                        }
                        ds = pj * (dp - (double)D[row]);
                        for (uint32_t c = 0U; c < Dh; ++c) {
                            dq[(qg * Hq + hq) * Dh + c] +=
                                (float)((double)scale * ds * (double)f16_bits_to_f32(k[(kg * Hkv + hkv) * Dh + c]));
                            dk[(kg * Hkv + hkv) * Dh + c] +=
                                (float)((double)scale * ds * (double)f16_bits_to_f32(q[(qg * Hq + hq) * Dh + c]));
                            dv[(kg * Hkv + hkv) * Dh + c] +=
                                (float)(pj * (double)f16_bits_to_f32(go[(qg * Hq + hq) * Dh + c]));
                        }
                    }
                }
            }
        }
    }
}

static void test_sparse_gqa(gd_context *ctx)
{
    enum { N = 5, Hq = 4, Hkv = 2, Dh = 4, B = 1, TOPK = 8, BLOCK = 2 };
    const int64_t q_shape[3] = {N, Hq, Dh};
    const int64_t kv_shape[3] = {N, Hkv, Dh};
    const int64_t cu_shape[1] = {B + 1};
    const int64_t topk_shape[3] = {Hkv, N, TOPK};
    uint16_t q_data[N * Hq * Dh];
    uint16_t k_data[N * Hkv * Dh];
    uint16_t v_data[N * Hkv * Dh];
    uint16_t go_data[N * Hq * Dh];
    int32_t cu_data[B + 1] = {0, N};
    int32_t topk_data[Hkv * N * TOPK];
    uint16_t out_got[N * Hq * Dh];
    uint16_t dq_got[N * Hq * Dh];
    uint16_t dk_got[N * Hkv * Dh];
    uint16_t dv_got[N * Hkv * Dh];
    float out_ref[N * Hq * Dh];
    float dq_ref[N * Hq * Dh];
    float dk_ref[N * Hkv * Dh];
    float dv_ref[N * Hkv * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor go;
    gd_tensor cu;
    gd_tensor topk;
    gd_tensor out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_minimax_m3_sparse_config cfg;
    uint32_t h;
    uint32_t i;
    fill_tokens(q_data, N * Hq * Dh, -0.1875f);
    fill_tokens(k_data, N * Hkv * Dh, 0.09375f);
    fill_tokens(v_data, N * Hkv * Dh, -0.28125f);
    fill_tokens(go_data, N * Hq * Dh, 0.03125f);
    for (h = 0U; h < Hkv; ++h) {
        for (i = 0U; i < N; ++i) {
            uint32_t valid_blocks = (i + BLOCK) / BLOCK;
            for (uint32_t t = 0U; t < TOPK; ++t) {
                topk_data[(h * N + i) * TOPK + t] = -1;
            }
            topk_data[(h * N + i) * TOPK + 0U] = 0;
            topk_data[(h * N + i) * TOPK + 1U] =
                valid_blocks > 1U ? (int32_t)(valid_blocks - 1U) : -1;
        }
    }
    memset(&cfg, 0, sizeof(cfg));
    cfg.scale = 0.5f;
    cfg.block_size = BLOCK;
    cfg.topk = TOPK;
    cfg.max_seqlen = N;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, kv_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, kv_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &go));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(3U, topk_shape), 256U, &topk));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &go, go_data, sizeof(go_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));
    CHECK_OK(gd_tensor_write(ctx, &topk, topk_data, sizeof(topk_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_minimax_m3_sparse_attention(ctx, &q, &k, &v, &cu, &topk, &cfg, &out));
    CHECK_OK(gd_tensor_read(ctx, &out, out_got, sizeof(out_got)));
    ref_sparse_fwd_gqa(q_data, k_data, v_data, cu_data, topk_data, B, Hq, Hkv, Dh, BLOCK, TOPK,
                       cfg.scale, out_ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(out_got[i]) - out_ref[i]) < 2.5e-3f, "GQA sparse forward mismatch");
    }
    CHECK_OK(gd_minimax_m3_sparse_attention_backward(ctx, &q, &k, &v, &cu, &topk, &go, &cfg, &dq, &dk, &dv));
    CHECK_OK(gd_tensor_read(ctx, &dq, dq_got, sizeof(dq_got)));
    CHECK_OK(gd_tensor_read(ctx, &dk, dk_got, sizeof(dk_got)));
    CHECK_OK(gd_tensor_read(ctx, &dv, dv_got, sizeof(dv_got)));
    ref_sparse_bwd_gqa(go_data, q_data, k_data, v_data, cu_data, topk_data, B, Hq, Hkv, Dh, BLOCK, TOPK,
                       cfg.scale, dq_ref, dk_ref, dv_ref);
    for (i = 0U; i < N * Hq * Dh; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(dq_got[i]) - dq_ref[i]) < 3.0e-3f, "GQA sparse dq mismatch");
    }
    for (i = 0U; i < N * Hkv * Dh; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(dk_got[i]) - dk_ref[i]) < 3.0e-3f, "GQA sparse dk mismatch");
        CHECK(abs_f32(f16_bits_to_f32(dv_got[i]) - dv_ref[i]) < 3.0e-3f, "GQA sparse dv mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}

static void test_sparse_fwd_bwd(gd_context *ctx)
{
    enum { N = 6, H = 2, Dh = 4, B = 2, TOPK = 2, BLOCK = 2 };
    const int64_t q_shape[3] = {N, H, Dh};
    const int64_t cu_shape[1] = {B + 1};
    uint16_t q_data[N * H * Dh];
    uint16_t k_data[N * H * Dh];
    uint16_t v_data[N * H * Dh];
    uint16_t go_data[N * H * Dh];
    int32_t cu_data[B + 1] = {0, 3, 6};
    int32_t topk_got[H * N * TOPK];
    int32_t topk_ref[H * N * TOPK];
    uint16_t out_got[N * H * Dh];
    uint16_t dq_got[N * H * Dh];
    uint16_t dk_got[N * H * Dh];
    uint16_t dv_got[N * H * Dh];
    float out_ref[N * H * Dh];
    float dq_ref[N * H * Dh];
    float dk_ref[N * H * Dh];
    float dv_ref[N * H * Dh];
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor go;
    gd_tensor cu;
    gd_tensor topk;
    gd_tensor out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    gd_minimax_m3_sparse_config cfg;
    uint32_t i;
    fill_tokens(q_data, N * H * Dh, -0.25f);
    fill_tokens(k_data, N * H * Dh, 0.125f);
    fill_tokens(v_data, N * H * Dh, -0.375f);
    fill_tokens(go_data, N * H * Dh, 0.0625f);
    memset(&cfg, 0, sizeof(cfg));
    cfg.scale = 0.5f;
    cfg.block_size = BLOCK;
    cfg.topk = TOPK;
    cfg.init_blocks = 1;
    cfg.local_blocks = 1;
    cfg.max_seqlen = 3;

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &q));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &k));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, q_shape), 256U, &go));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_I32, gd_shape_make(1U, cu_shape), 256U, &cu));
    CHECK_OK(gd_tensor_write(ctx, &q, q_data, sizeof(q_data)));
    CHECK_OK(gd_tensor_write(ctx, &k, k_data, sizeof(k_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &go, go_data, sizeof(go_data)));
    CHECK_OK(gd_tensor_write(ctx, &cu, cu_data, sizeof(cu_data)));

    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_minimax_m3_index_topk(ctx, &q, &k, &cu, &cfg, &topk));
    CHECK_OK(gd_tensor_read(ctx, &topk, topk_got, sizeof(topk_got)));
    ref_topk(q_data, k_data, cu_data, B, H, Dh, BLOCK, TOPK, 1U, 1U, topk_ref);
    for (i = 0U; i < H * N * TOPK; ++i) {
        CHECK(topk_got[i] == topk_ref[i], "topk mismatch");
    }

    CHECK_OK(gd_minimax_m3_sparse_attention(ctx, &q, &k, &v, &cu, &topk, &cfg, &out));
    CHECK_OK(gd_tensor_read(ctx, &out, out_got, sizeof(out_got)));
    ref_sparse_fwd(q_data, k_data, v_data, cu_data, topk_ref, B, H, Dh, BLOCK, TOPK, cfg.scale, out_ref);
    for (i = 0U; i < N * H * Dh; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(out_got[i]) - out_ref[i]) < 2.5e-3f, "sparse forward mismatch");
    }

    CHECK_OK(gd_minimax_m3_sparse_attention_backward(ctx, &q, &k, &v, &cu, &topk, &go, &cfg, &dq, &dk, &dv));
    CHECK_OK(gd_tensor_read(ctx, &dq, dq_got, sizeof(dq_got)));
    CHECK_OK(gd_tensor_read(ctx, &dk, dk_got, sizeof(dk_got)));
    CHECK_OK(gd_tensor_read(ctx, &dv, dv_got, sizeof(dv_got)));
    ref_sparse_bwd(go_data, q_data, k_data, v_data, cu_data, topk_ref, B, H, Dh, BLOCK, TOPK, cfg.scale,
                   dq_ref, dk_ref, dv_ref);
    for (i = 0U; i < N * H * Dh; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(dq_got[i]) - dq_ref[i]) < 3.0e-3f, "sparse dq mismatch");
        CHECK(abs_f32(f16_bits_to_f32(dk_got[i]) - dk_ref[i]) < 3.0e-3f, "sparse dk mismatch");
        CHECK(abs_f32(f16_bits_to_f32(dv_got[i]) - dv_ref[i]) < 3.0e-3f, "sparse dv mismatch");
    }
    CHECK_OK(gd_end_step(ctx));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = test_config();
    gd_status st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_minimax_m3_sparse_attention: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);
    test_sparse_fwd_bwd(ctx);
    test_sparse_gqa(ctx);
    gd_context_destroy(ctx);
    printf("test_minimax_m3_sparse_attention: ok\n");
    return 0;
}
