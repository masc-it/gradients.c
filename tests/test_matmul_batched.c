#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_matmul_batched failed: %s (%s:%d)\n", (msg), \
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

static gd_memory_config test_config(void)
{
    gd_memory_config cfg = gd_memory_config_default();
    cfg.params_bytes = 4U * 1024U * 1024U;
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = 8U * 1024U * 1024U;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
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

static void fill_pattern(uint16_t *data, size_t count, float scale, float bias)
{
    size_t i;
    for (i = 0U; i < count; ++i) {
        int32_t centered = (int32_t)(i % 17U) - 8;
        data[i] = f32_to_f16_bits(bias + scale * (float)centered);
    }
}

static float hval(const uint16_t *data, size_t index)
{
    return f16_bits_to_f32(data[index]);
}

static void test_broadcast_forward_backward(gd_context *ctx)
{
    enum { B0 = 2, B1 = 5, M = 3, K = 4, N = 6 };
    const int64_t x_shape[4] = {B0, 1, M, K};
    const int64_t w_shape[4] = {1, B1, K, N};
    const size_t x_count = (size_t)B0 * M * K;
    const size_t w_count = (size_t)B1 * K * N;
    const size_t y_count = (size_t)B0 * B1 * M * N;
    uint16_t x_data[B0 * 1 * M * K];
    uint16_t w_data[1 * B1 * K * N];
    uint16_t g_data[B0 * B1 * M * N];
    uint16_t y_got[B0 * B1 * M * N];
    uint16_t dx_got[B0 * 1 * M * K];
    uint16_t dw_got[1 * B1 * K * N];
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    uint32_t b0;
    uint32_t b1;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    fill_pattern(x_data, x_count, 0.03125f, 0.125f);
    fill_pattern(w_data, w_count, -0.0234375f, 0.0625f);
    fill_pattern(g_data, y_count, 0.015625f, -0.03125f);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, w_shape), 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16,
                             gd_shape_make(4U, (const int64_t[]){B0, B1, M, N}), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_matmul(ctx, &x, &w, &y));
    CHECK(y.rank == 4U && y.shape[0] == B0 && y.shape[1] == B1 && y.shape[2] == M && y.shape[3] == N,
          "broadcast matmul output shape");
    CHECK_OK(gd_tensor_read(ctx, &y, y_got, sizeof(y_got)));
    for (b0 = 0U; b0 < B0; ++b0) {
        for (b1 = 0U; b1 < B1; ++b1) {
            for (m = 0U; m < M; ++m) {
                for (n = 0U; n < N; ++n) {
                    float want = 0.0f;
                    float have;
                    for (k = 0U; k < K; ++k) {
                        size_t xi = (((size_t)b0 * M + m) * K) + k;
                        size_t wi = (((size_t)b1 * K + k) * N) + n;
                        want += hval(x_data, xi) * hval(w_data, wi);
                    }
                    have = hval(y_got, ((((size_t)b0 * B1 + b1) * M + m) * N) + n);
                    CHECK(abs_f32(want - have) <= 0.025f, "broadcast matmul forward close");
                }
            }
        }
    }
    CHECK_OK(gd_matmul_backward(ctx, &x, &w, &g, &dx, &dw));
    CHECK(dx.rank == 4U && dx.shape[0] == B0 && dx.shape[1] == 1 && dx.shape[2] == M && dx.shape[3] == K,
          "broadcast matmul dx shape");
    CHECK(dw.rank == 4U && dw.shape[0] == 1 && dw.shape[1] == B1 && dw.shape[2] == K && dw.shape[3] == N,
          "broadcast matmul dw shape");
    CHECK_OK(gd_tensor_read(ctx, &dx, dx_got, sizeof(dx_got)));
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_got, sizeof(dw_got)));
    for (b0 = 0U; b0 < B0; ++b0) {
        for (m = 0U; m < M; ++m) {
            for (k = 0U; k < K; ++k) {
                float want = 0.0f;
                float have;
                for (b1 = 0U; b1 < B1; ++b1) {
                    for (n = 0U; n < N; ++n) {
                        size_t gi = ((((size_t)b0 * B1 + b1) * M + m) * N) + n;
                        size_t wi = (((size_t)b1 * K + k) * N) + n;
                        want += hval(g_data, gi) * hval(w_data, wi);
                    }
                }
                have = hval(dx_got, (((size_t)b0 * M + m) * K) + k);
                CHECK(abs_f32(want - have) <= 0.05f, "broadcast matmul dx close");
            }
        }
    }
    for (b1 = 0U; b1 < B1; ++b1) {
        for (k = 0U; k < K; ++k) {
            for (n = 0U; n < N; ++n) {
                float want = 0.0f;
                float have;
                for (b0 = 0U; b0 < B0; ++b0) {
                    for (m = 0U; m < M; ++m) {
                        size_t xi = (((size_t)b0 * M + m) * K) + k;
                        size_t gi = ((((size_t)b0 * B1 + b1) * M + m) * N) + n;
                        want += hval(x_data, xi) * hval(g_data, gi);
                    }
                }
                have = hval(dw_got, (((size_t)b1 * K + k) * N) + n);
                CHECK(abs_f32(want - have) <= 0.05f, "broadcast matmul dw close");
            }
        }
    }
    CHECK_OK(gd_end_step(ctx));
}

static void test_projection_weight_broadcast(gd_context *ctx)
{
    enum { B = 3, T = 7, K = 8, N = 11 };
    const int64_t x_shape[3] = {B, T, K};
    const int64_t w_shape[2] = {K, N};
    const int64_t g_shape[3] = {B, T, N};
    uint16_t x_data[B * T * K];
    uint16_t w_data[K * N];
    uint16_t g_data[B * T * N];
    uint16_t y_got[B * T * N];
    uint16_t dw_got[K * N];
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor y;
    gd_tensor dw;
    uint32_t b;
    uint32_t t;
    uint32_t k;
    uint32_t n;
    fill_pattern(x_data, (size_t)B * T * K, 0.02f, -0.05f);
    fill_pattern(w_data, (size_t)K * N, 0.017f, 0.03f);
    fill_pattern(g_data, (size_t)B * T * N, -0.011f, 0.02f);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(3U, g_shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_matmul(ctx, &x, &w, &y));
    CHECK(y.rank == 3U && y.shape[0] == B && y.shape[1] == T && y.shape[2] == N,
          "projection matmul output shape");
    CHECK_OK(gd_tensor_read(ctx, &y, y_got, sizeof(y_got)));
    for (b = 0U; b < B; ++b) {
        for (t = 0U; t < T; ++t) {
            for (n = 0U; n < N; ++n) {
                float want = 0.0f;
                float have;
                for (k = 0U; k < K; ++k) {
                    want += hval(x_data, (((size_t)b * T + t) * K) + k) * hval(w_data, (size_t)k * N + n);
                }
                have = hval(y_got, (((size_t)b * T + t) * N) + n);
                CHECK(abs_f32(want - have) <= 0.025f, "projection matmul forward close");
            }
        }
    }
    CHECK_OK(gd_matmul_backward(ctx, &x, &w, &g, NULL, &dw));
    CHECK(dw.rank == 2U && dw.shape[0] == K && dw.shape[1] == N, "projection matmul dw shape");
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_got, sizeof(dw_got)));
    for (k = 0U; k < K; ++k) {
        for (n = 0U; n < N; ++n) {
            float want = 0.0f;
            float have;
            for (b = 0U; b < B; ++b) {
                for (t = 0U; t < T; ++t) {
                    want += hval(x_data, (((size_t)b * T + t) * K) + k) *
                            hval(g_data, (((size_t)b * T + t) * N) + n);
                }
            }
            have = hval(dw_got, (size_t)k * N + n);
            CHECK(abs_f32(want - have) <= 0.06f, "projection matmul dw close");
        }
    }
    CHECK_OK(gd_end_step(ctx));
}

static void test_attention_backward_skinny(gd_context *ctx)
{
    enum { B = 1, H = 2, T = 128, D = 64 };
    const int64_t x_shape[4] = {B, H, T, D};
    const int64_t w_shape[4] = {B, H, D, T};
    const int64_t g_shape[4] = {B, H, T, T};
    uint16_t x_data[B * H * T * D];
    uint16_t w_data[B * H * D * T];
    uint16_t g_data[B * H * T * T];
    uint16_t dx_got[B * H * T * D];
    uint16_t dw_got[B * H * D * T];
    gd_tensor x;
    gd_tensor w;
    gd_tensor g;
    gd_tensor dx;
    gd_tensor dw;
    uint32_t h;
    uint32_t t;
    uint32_t d;
    uint32_t s;
    fill_pattern(x_data, (size_t)B * H * T * D, 0.003f, -0.01f);
    fill_pattern(w_data, (size_t)B * H * D * T, -0.0025f, 0.015f);
    fill_pattern(g_data, (size_t)B * H * T * T, 0.002f, 0.005f);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, x_shape), 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, w_shape), 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(4U, g_shape), 256U, &g));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    CHECK_OK(gd_tensor_write(ctx, &g, g_data, sizeof(g_data)));
    CHECK_OK(gd_context_seal_params(ctx));
    CHECK_OK(gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CHECK_OK(gd_matmul_backward(ctx, &x, &w, &g, &dx, &dw));
    CHECK(dx.rank == 4U && dx.shape[0] == B && dx.shape[1] == H && dx.shape[2] == T && dx.shape[3] == D,
          "attention dx shape");
    CHECK(dw.rank == 4U && dw.shape[0] == B && dw.shape[1] == H && dw.shape[2] == D && dw.shape[3] == T,
          "attention dw shape");
    CHECK_OK(gd_tensor_read(ctx, &dx, dx_got, sizeof(dx_got)));
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_got, sizeof(dw_got)));
    for (h = 0U; h < H; ++h) {
        for (t = 0U; t < T; ++t) {
            for (d = 0U; d < D; ++d) {
                float want = 0.0f;
                float have;
                for (s = 0U; s < T; ++s) {
                    size_t gi = ((((size_t)h * T + t) * T) + s);
                    size_t wi = ((((size_t)h * D + d) * T) + s);
                    want += hval(g_data, gi) * hval(w_data, wi);
                }
                have = hval(dx_got, ((((size_t)h * T + t) * D) + d));
                CHECK(abs_f32(want - have) <= 0.02f, "attention dx close");
            }
        }
    }
    for (h = 0U; h < H; ++h) {
        for (d = 0U; d < D; ++d) {
            for (s = 0U; s < T; ++s) {
                float want = 0.0f;
                float have;
                for (t = 0U; t < T; ++t) {
                    size_t xi = ((((size_t)h * T + t) * D) + d);
                    size_t gi = ((((size_t)h * T + t) * T) + s);
                    want += hval(x_data, xi) * hval(g_data, gi);
                }
                have = hval(dw_got, ((((size_t)h * D + d) * T) + s));
                CHECK(abs_f32(want - have) <= 0.02f, "attention dw close");
            }
        }
    }
    CHECK_OK(gd_end_step(ctx));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = test_config();
    CHECK_OK(gd_context_create(&cfg, &ctx));
    test_broadcast_forward_backward(ctx);
    gd_context_destroy(ctx);
    ctx = NULL;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    test_projection_weight_broadcast(ctx);
    gd_context_destroy(ctx);
    ctx = NULL;
    CHECK_OK(gd_context_create(&cfg, &ctx));
    test_attention_backward_skinny(ctx);
    gd_context_destroy(ctx);
    printf("test_matmul_batched: ok\n");
    return 0;
}
