#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_ops failed: %s (%s:%d)\n", (msg),           \
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

static gd_memory_config ops_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 16384U;
    cfg.state_bytes = 8192U;
    cfg.scratch_slot_bytes = 16384U;
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

static void fill_x(uint16_t *x, uint32_t m, uint32_t k)
{
    uint32_t i;
    uint32_t j;
    for (i = 0U; i < m; ++i) {
        for (j = 0U; j < k; ++j) {
            x[i * k + j] = f32_to_f16_bits(0.25f + 0.125f * (float)i - 0.0625f * (float)j);
        }
    }
}

static void fill_w(uint16_t *w, uint32_t k, uint32_t n)
{
    uint32_t i;
    uint32_t j;
    for (i = 0U; i < k; ++i) {
        for (j = 0U; j < n; ++j) {
            w[i * n + j] = f32_to_f16_bits(-0.5f + 0.125f * (float)j + 0.03125f * (float)i);
        }
    }
}

static void fill_b(uint16_t *b, uint32_t n)
{
    uint32_t j;
    for (j = 0U; j < n; ++j) {
        b[j] = f32_to_f16_bits(-0.25f + 0.125f * (float)j);
    }
}

static float ref_matmul_value(const uint16_t *x,
                              const uint16_t *w,
                              uint32_t row,
                              uint32_t col,
                              uint32_t k,
                              uint32_t n)
{
    uint32_t p;
    float sum = 0.0f;
    for (p = 0U; p < k; ++p) {
        sum += f16_bits_to_f32(x[row * k + p]) * f16_bits_to_f32(w[p * n + col]);
    }
    return f16_bits_to_f32(f32_to_f16_bits(sum));
}

static float ref_matmul_backward_x_value(const uint16_t *grad_out,
                                         const uint16_t *w,
                                         uint32_t row,
                                         uint32_t col,
                                         uint32_t n)
{
    uint32_t p;
    float sum = 0.0f;
    for (p = 0U; p < n; ++p) {
        sum += f16_bits_to_f32(grad_out[row * n + p]) * f16_bits_to_f32(w[col * n + p]);
    }
    return f16_bits_to_f32(f32_to_f16_bits(sum));
}

static float ref_matmul_backward_w_value(const uint16_t *x,
                                         const uint16_t *grad_out,
                                         uint32_t row,
                                         uint32_t col,
                                         uint32_t m,
                                         uint32_t k,
                                         uint32_t n)
{
    uint32_t p;
    float sum = 0.0f;
    for (p = 0U; p < m; ++p) {
        sum += f16_bits_to_f32(x[p * k + row]) * f16_bits_to_f32(grad_out[p * n + col]);
    }
    return f16_bits_to_f32(f32_to_f16_bits(sum));
}

static void test_matmul_linear(gd_context *ctx)
{
    enum { M = 4, K = 7, N = 6 };
    const int64_t x_shape[2] = {M, K};
    const int64_t w_shape[2] = {K, N};
    const int64_t b_shape[1] = {N};
    uint16_t x_data[M * K];
    uint16_t w_data[K * N];
    uint16_t b_data[N];
    uint16_t got[M * N];
    uint16_t y_got[M * N];
    uint16_t dx_got[M * K];
    uint16_t dw_got[K * N];
    gd_tensor x;
    gd_tensor w;
    gd_tensor b;
    gd_tensor y;
    gd_tensor lin;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
    uint32_t i;
    uint32_t j;

    fill_x(x_data, M, K);
    fill_w(w_data, K, N);
    fill_b(b_data, N);
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, x_shape, 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, w_shape, 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 1U, b_shape, 256U, &b));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    CHECK_OK(gd_tensor_write(ctx, &b, b_data, sizeof(b_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_matmul(ctx, &x, &w, &y));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &y, got, sizeof(got)));
    memcpy(y_got, got, sizeof(y_got));
    for (i = 0U; i < M; ++i) {
        for (j = 0U; j < N; ++j) {
            float want = ref_matmul_value(x_data, w_data, i, j, K, N);
            float have = f16_bits_to_f32(got[i * N + j]);
            CHECK(abs_f32(want - have) <= 0.02f, "matmul output close");
        }
    }

    CHECK_OK(gd_linear(ctx, &x, &w, &b, &lin));
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &lin, got, sizeof(got)));
    for (i = 0U; i < M; ++i) {
        for (j = 0U; j < N; ++j) {
            float want = ref_matmul_value(x_data, w_data, i, j, K, N) + f16_bits_to_f32(b_data[j]);
            float have = f16_bits_to_f32(got[i * N + j]);
            CHECK(abs_f32(want - have) <= 0.02f, "linear output close");
        }
    }

    CHECK_OK(gd_matmul_backward(ctx, &x, &w, &y, &dx, &dw));
    CHECK(dx.shape[0] == M && dx.shape[1] == K && dw.shape[0] == K && dw.shape[1] == N,
          "matmul bwd gradient shapes");
    memset(dx_got, 0, sizeof(dx_got));
    memset(dw_got, 0, sizeof(dw_got));
    CHECK_OK(gd_tensor_read(ctx, &dx, dx_got, sizeof(dx_got)));
    CHECK_OK(gd_tensor_read(ctx, &dw, dw_got, sizeof(dw_got)));
    for (i = 0U; i < M; ++i) {
        for (j = 0U; j < K; ++j) {
            float want = ref_matmul_backward_x_value(y_got, w_data, i, j, N);
            float have = f16_bits_to_f32(dx_got[i * K + j]);
            CHECK(abs_f32(want - have) <= 0.03f, "matmul backward dx close");
        }
    }
    for (i = 0U; i < K; ++i) {
        for (j = 0U; j < N; ++j) {
            float want = ref_matmul_backward_w_value(x_data, y_got, i, j, M, K, N);
            float have = f16_bits_to_f32(dw_got[i * N + j]);
            CHECK(abs_f32(want - have) <= 0.03f, "matmul backward dw close");
        }
    }
    CHECK_STATUS(gd_linear_backward(ctx, &x, &w, &b, &lin, &dx, &dw, &db), GD_ERR_NOT_IMPLEMENTED);
    CHECK(dx.storage.nbytes == 0U && dw.storage.nbytes == 0U && db.storage.nbytes == 0U,
          "linear bwd does not publish grads");
    gd_context_clear_error(ctx);
    CHECK_OK(gd_end(ctx));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = ops_config();
    {
        gd_status st = gd_context_create(&cfg, &ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("test_ops: skipped (no supported GPU backend)\n");
            return 0;
        }
        CHECK_OK(st);
    }
    test_matmul_linear(ctx);
    gd_context_destroy(ctx);
    printf("test_ops: ok\n");
    return 0;
}
