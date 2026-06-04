#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond, msg)                                                        \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "test_autograd failed: %s (%s:%d)\n", (msg),     \
                    __FILE__, __LINE__);                                       \
            exit(1);                                                           \
        }                                                                      \
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

static void fill_seq(uint16_t *dst, uint32_t count, float scale, float bias)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        float v = bias + scale * (float)((int32_t)(i % 17U) - 8);
        dst[i] = f32_to_f16_bits(v);
    }
}

static void matmul_nn(const uint16_t *a,
                      const uint16_t *b,
                      uint32_t m,
                      uint32_t k,
                      uint32_t n,
                      uint16_t *out)
{
    uint32_t i;
    uint32_t j;
    uint32_t p;
    for (i = 0U; i < m; ++i) {
        for (j = 0U; j < n; ++j) {
            float sum = 0.0f;
            for (p = 0U; p < k; ++p) {
                sum += f16_bits_to_f32(a[i * k + p]) * f16_bits_to_f32(b[p * n + j]);
            }
            out[i * n + j] = f32_to_f16_bits(sum);
        }
    }
}

static void matmul_nt(const uint16_t *a,
                      const uint16_t *b,
                      uint32_t m,
                      uint32_t k,
                      uint32_t n,
                      uint16_t *out)
{
    uint32_t i;
    uint32_t j;
    uint32_t p;
    for (i = 0U; i < m; ++i) {
        for (j = 0U; j < n; ++j) {
            float sum = 0.0f;
            for (p = 0U; p < k; ++p) {
                sum += f16_bits_to_f32(a[i * k + p]) * f16_bits_to_f32(b[j * k + p]);
            }
            out[i * n + j] = f32_to_f16_bits(sum);
        }
    }
}

static void matmul_tn(const uint16_t *a,
                      const uint16_t *b,
                      uint32_t m,
                      uint32_t k,
                      uint32_t n,
                      uint16_t *out)
{
    uint32_t i;
    uint32_t j;
    uint32_t p;
    for (i = 0U; i < k; ++i) {
        for (j = 0U; j < n; ++j) {
            float sum = 0.0f;
            for (p = 0U; p < m; ++p) {
                sum += f16_bits_to_f32(a[p * k + i]) * f16_bits_to_f32(b[p * n + j]);
            }
            out[i * n + j] = f32_to_f16_bits(sum);
        }
    }
}

static void add_f16(const uint16_t *a, const uint16_t *b, uint32_t count, uint16_t *out)
{
    uint32_t i;
    for (i = 0U; i < count; ++i) {
        out[i] = f32_to_f16_bits(f16_bits_to_f32(a[i]) + f16_bits_to_f32(b[i]));
    }
}

static void reduce_rows(const uint16_t *x, uint32_t rows, uint32_t cols, uint16_t *out)
{
    uint32_t i;
    uint32_t j;
    for (j = 0U; j < cols; ++j) {
        float sum = 0.0f;
        for (i = 0U; i < rows; ++i) {
            sum += f16_bits_to_f32(x[i * cols + j]);
        }
        out[j] = f32_to_f16_bits(sum);
    }
}

static gd_memory_config autograd_config(void)
{
    gd_memory_config cfg;
    cfg.params_bytes = 65536U;
    cfg.state_bytes = 8192U;
    cfg.scratch_slot_bytes = 262144U;
    cfg.data_slot_bytes = 4096U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static void test_fanout_many_loss(gd_context *ctx)
{
    enum { M = 4, K = 7, N = 6, P = 5, Q = 3 };
    const int64_t x_shape[2] = {M, K};
    const int64_t w_shape[2] = {K, N};
    const int64_t v_shape[2] = {N, P};
    const int64_t b_shape[1] = {P};
    const int64_t u_shape[2] = {N, Q};
    const int64_t g1_shape[2] = {M, P};
    const int64_t g2_shape[2] = {M, Q};
    uint16_t x_data[M * K];
    uint16_t w_data[K * N];
    uint16_t v_data[N * P];
    uint16_t b_data[P];
    uint16_t u_data[N * Q];
    uint16_t g1_data[M * P];
    uint16_t g2_data[M * Q];
    uint16_t got[M * K > K * N ? M * K : K * N];
    uint16_t y_ref[M * N];
    uint16_t dy1[M * N];
    uint16_t dy2[M * N];
    uint16_t dy[M * N];
    uint16_t dx_ref[M * K];
    uint16_t dw_ref[K * N];
    uint16_t dv_ref[N * P];
    uint16_t db_ref[P];
    uint16_t du_ref[N * Q];
    gd_tensor x;
    gd_tensor w;
    gd_tensor v;
    gd_tensor b;
    gd_tensor u;
    gd_tensor g1;
    gd_tensor g2;
    gd_tensor y;
    gd_tensor z1;
    gd_tensor z2;
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor dv;
    gd_tensor db;
    gd_tensor du;
    const gd_tensor *outputs[2];
    const gd_tensor *grad_outputs[2];
    uint32_t i;

    fill_seq(x_data, M * K, 0.0125f, 0.02f);
    fill_seq(w_data, K * N, -0.009f, 0.01f);
    fill_seq(v_data, N * P, 0.007f, -0.015f);
    fill_seq(b_data, P, 0.02f, 0.0f);
    fill_seq(u_data, N * Q, -0.011f, 0.018f);
    fill_seq(g1_data, M * P, 0.013f, -0.004f);
    fill_seq(g2_data, M * Q, -0.017f, 0.006f);

    matmul_nn(x_data, w_data, M, K, N, y_ref);
    matmul_nt(g1_data, v_data, M, P, N, dy1);
    matmul_nt(g2_data, u_data, M, Q, N, dy2);
    add_f16(dy2, dy1, M * N, dy);
    matmul_nt(dy, w_data, M, N, K, dx_ref);
    matmul_tn(x_data, dy, M, K, N, dw_ref);
    matmul_tn(y_ref, g1_data, M, N, P, dv_ref);
    reduce_rows(g1_data, M, P, db_ref);
    matmul_tn(y_ref, g2_data, M, N, Q, du_ref);

    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, x_shape, 256U, &x));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, w_shape, 256U, &w));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, v_shape, 256U, &v));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 1U, b_shape, 256U, &b));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, u_shape, 256U, &u));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, g1_shape, 256U, &g1));
    CHECK_OK(gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, 2U, g2_shape, 256U, &g2));
    CHECK_OK(gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    CHECK_OK(gd_tensor_write(ctx, &v, v_data, sizeof(v_data)));
    CHECK_OK(gd_tensor_write(ctx, &b, b_data, sizeof(b_data)));
    CHECK_OK(gd_tensor_write(ctx, &u, u_data, sizeof(u_data)));
    CHECK_OK(gd_tensor_write(ctx, &g1, g1_data, sizeof(g1_data)));
    CHECK_OK(gd_tensor_write(ctx, &g2, g2_data, sizeof(g2_data)));
    CHECK_OK(gd_context_seal_params(ctx));

    x.requires_grad = true;
    w.requires_grad = true;
    v.requires_grad = true;
    b.requires_grad = true;
    u.requires_grad = true;
    CHECK_OK(gd_begin(ctx, GD_SCOPE_TRAIN));
    CHECK_OK(gd_matmul(ctx, &x, &w, &y));
    CHECK_OK(gd_linear(ctx, &y, &v, &b, &z1));
    CHECK_OK(gd_matmul(ctx, &y, &u, &z2));
    outputs[0] = &z1;
    outputs[1] = &z2;
    grad_outputs[0] = &g1;
    grad_outputs[1] = &g2;
    CHECK_OK(gd_backward_many(ctx, 2U, outputs, grad_outputs));
    CHECK_OK(gd_tensor_grad(ctx, &x, &dx));
    CHECK_OK(gd_tensor_grad(ctx, &w, &dw));
    CHECK_OK(gd_tensor_grad(ctx, &v, &dv));
    CHECK_OK(gd_tensor_grad(ctx, &b, &db));
    CHECK_OK(gd_tensor_grad(ctx, &u, &du));
    CHECK_OK(gd_end(ctx));

    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dx, got, sizeof(dx_ref)));
    for (i = 0U; i < M * K; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(got[i]) - f16_bits_to_f32(dx_ref[i])) <= 0.03f,
              "autograd dx close");
    }
    memset(got, 0, sizeof(got));
    CHECK_OK(gd_tensor_read(ctx, &dw, got, sizeof(dw_ref)));
    for (i = 0U; i < K * N; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(got[i]) - f16_bits_to_f32(dw_ref[i])) <= 0.03f,
              "autograd dw close");
    }
    CHECK_OK(gd_tensor_read(ctx, &dv, got, sizeof(dv_ref)));
    for (i = 0U; i < N * P; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(got[i]) - f16_bits_to_f32(dv_ref[i])) <= 0.03f,
              "autograd dv close");
    }
    CHECK_OK(gd_tensor_read(ctx, &db, got, sizeof(db_ref)));
    for (i = 0U; i < P; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(got[i]) - f16_bits_to_f32(db_ref[i])) <= 0.03f,
              "autograd db close");
    }
    CHECK_OK(gd_tensor_read(ctx, &du, got, sizeof(du_ref)));
    for (i = 0U; i < N * Q; ++i) {
        CHECK(abs_f32(f16_bits_to_f32(got[i]) - f16_bits_to_f32(du_ref[i])) <= 0.03f,
              "autograd du close");
    }
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_memory_config cfg = autograd_config();
    gd_status st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("test_autograd: skipped (no supported GPU backend)\n");
        return 0;
    }
    CHECK_OK(st);
    test_fanout_many_loss(ctx);
    gd_context_destroy(ctx);
    printf("test_autograd: ok\n");
    return 0;
}
