#include "gradients/gradients.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_STATUS(expr, expected)                                               \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != (expected)) {                                               \
            fprintf(stderr, "%s got %s expected %s; last_error=%s\n",             \
                    #expr,                                                        \
                    gd_status_name(status_),                                      \
                    gd_status_name(expected),                                     \
                    gd_last_error());                                             \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static gd_status make_tensor(gd_context *ctx,
                             gd_dtype dtype,
                             int ndim,
                             const int64_t *sizes,
                             gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(dtype, cpu, ndim, sizes, &desc);

    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_empty(ctx, &desc, out);
}

static int check_shape(gd_tensor *t, int ndim, const int64_t *sizes)
{
    int i = 0;

    if (gd_tensor_ndim(t) != ndim) {
        fprintf(stderr, "ndim mismatch: got %d want %d\n", gd_tensor_ndim(t), ndim);
        return 1;
    }
    for (i = 0; i < ndim; ++i) {
        if (gd_tensor_size(t, i) != sizes[i]) {
            fprintf(stderr, "dim %d mismatch: got %lld want %lld\n", i,
                    (long long)gd_tensor_size(t, i), (long long)sizes[i]);
            return 1;
        }
    }
    return 0;
}

static int test_elementwise(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    int64_t s4[1] = {4};
    int64_t expect[2] = {2, 3};
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *b_bad = NULL;
    gd_tensor *bi = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s23, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s3, &b));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s4, &b_bad));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 1, s3, &bi));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    CHECK_OK(gd_add(ctx, a, b, &out));
    CHECK_TRUE(check_shape(out, 2, expect) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_F32);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_mul(ctx, a, b, &out));
    CHECK_TRUE(check_shape(out, 2, expect) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_STATUS(gd_add(ctx, a, b_bad, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);
    CHECK_STATUS(gd_add(ctx, a, bi, &out), GD_ERR_DTYPE);
    CHECK_TRUE(out == NULL);

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(a);
    gd_tensor_release(b);
    gd_tensor_release(b_bad);
    gd_tensor_release(bi);
    return 0;
}

static int test_powlu_schema(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s24[2] = {2, 4};
    float x1d[6] = {1.0F, -0.5F, 0.25F, 2.0F, -1.5F, 0.75F};
    float x2d[6] = {-20.0F, -1.0e-4F, 0.0F, 1.0e-8F, 1.0e-4F, 20.0F};
    float got[6];
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *x1 = NULL;
    gd_tensor *x2 = NULL;
    gd_tensor *bad_shape = NULL;
    gd_tensor *xi = NULL;
    gd_tensor *out = NULL;
    gd_tensor *bad_out = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s23, &x1));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s23, &x2));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s24, &bad_shape));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, s23, &xi));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x1, x1d, sizeof(x1d)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x2, x2d, sizeof(x2d)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    CHECK_OK(gd_powlu(ctx, x1, x2, 3.0F, &out));
    CHECK_TRUE(check_shape(out, 2, s23) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_F32);

    CHECK_STATUS(gd_powlu(ctx, x1, x2, 0.0F, &bad_out), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(bad_out == NULL);
    CHECK_STATUS(gd_powlu(ctx, x1, x2, 10.0F, &bad_out), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(bad_out == NULL);
    CHECK_STATUS(gd_powlu(ctx, x1, bad_shape, 3.0F, &bad_out), GD_ERR_SHAPE);
    CHECK_TRUE(bad_out == NULL);
    CHECK_STATUS(gd_powlu(ctx, x1, xi, 3.0F, &bad_out), GD_ERR_DTYPE);
    CHECK_TRUE(bad_out == NULL);

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, out, got, sizeof(got)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(isfinite(got[i]));
    }
    gd_tensor_release(out);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x1);
    gd_tensor_release(x2);
    gd_tensor_release(bad_shape);
    gd_tensor_release(xi);
    return 0;
}

static int test_matmul(gd_context *ctx)
{
    int64_t a2[2] = {2, 3};
    int64_t b2[2] = {3, 4};
    int64_t a3[3] = {5, 2, 3};
    int64_t b3[3] = {5, 3, 4};
    int64_t bt[2] = {4, 3};
    int64_t e2[2] = {2, 4};
    int64_t e3[3] = {5, 2, 4};
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *ba = NULL;
    gd_tensor *bb = NULL;
    gd_tensor *bbt = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;
    gd_matmul_desc tdesc = {false, true, {GD_DTYPE_F32, GD_DTYPE_F32}};

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, a2, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, b2, &b));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, a3, &ba));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, b3, &bb));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, bt, &bbt));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    CHECK_OK(gd_matmul(ctx, a, b, &out));
    CHECK_TRUE(check_shape(out, 2, e2) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_matmul(ctx, ba, bb, &out));
    CHECK_TRUE(check_shape(out, 3, e3) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_matmul(ctx, ba, b, &out)); /* broadcast batch */
    CHECK_TRUE(check_shape(out, 3, e3) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_matmul_ex(ctx, &tdesc, a, bbt, &out)); /* trans_b */
    CHECK_TRUE(check_shape(out, 2, e2) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_STATUS(gd_matmul(ctx, a, a, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(a);
    gd_tensor_release(b);
    gd_tensor_release(ba);
    gd_tensor_release(bb);
    gd_tensor_release(bbt);
    return 0;
}

static int test_sdpa_prefix_schema(gd_context *ctx)
{
    int64_t xs[4] = {1, 5, 1, 1};
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;
    gd_sdpa_config cfg = {0};

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, xs, &q));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, xs, &k));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, xs, &v));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    cfg.prefix_len = 2;
    cfg.causal = false;
    CHECK_STATUS(gd_sdpa(ctx, q, k, v, NULL, &cfg, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(out == NULL);

    cfg.causal = true;
    cfg.sliding_window = 3;
    CHECK_STATUS(gd_sdpa(ctx, q, k, v, NULL, &cfg, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(out == NULL);

    cfg.sliding_window = 0;
    cfg.prefix_len = 6;
    CHECK_STATUS(gd_sdpa(ctx, q, k, v, NULL, &cfg, &out), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(out == NULL);

    cfg.prefix_len = 5;
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &cfg, &out));
    CHECK_TRUE(check_shape(out, 4, xs) == 0);
    gd_tensor_release(out);

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(q);
    gd_tensor_release(k);
    gd_tensor_release(v);
    return 0;
}

static int test_linear(gd_context *ctx)
{
    int64_t xs[2] = {2, 3};
    int64_t ws[2] = {3, 4};
    int64_t wts[2] = {4, 3};
    int64_t bs[1] = {4};
    int64_t bs_bad[1] = {5};
    int64_t e[2] = {2, 4};
    gd_tensor *x = NULL;
    gd_tensor *w = NULL;
    gd_tensor *wt = NULL;
    gd_tensor *bias = NULL;
    gd_tensor *bias_bad = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;
    gd_linear_desc ldesc = {true, {GD_DTYPE_F32, GD_DTYPE_F32}};

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, xs, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, ws, &w));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, wts, &wt));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, bs, &bias));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, bs_bad, &bias_bad));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    CHECK_OK(gd_linear(ctx, x, w, NULL, &out));
    CHECK_TRUE(check_shape(out, 2, e) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_linear(ctx, x, w, bias, &out));
    CHECK_TRUE(check_shape(out, 2, e) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_linear_ex(ctx, &ldesc, x, wt, bias, &out)); /* trans_w */
    CHECK_TRUE(check_shape(out, 2, e) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_STATUS(gd_linear(ctx, x, w, bias_bad, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);
    CHECK_STATUS(gd_linear(ctx, x, wt, NULL, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    gd_tensor_release(w);
    gd_tensor_release(wt);
    gd_tensor_release(bias);
    gd_tensor_release(bias_bad);
    return 0;
}

static int test_reduce_softmax_ce_cast(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s2[1] = {2};
    int64_t s21[2] = {2, 1};
    int64_t sw43[2] = {4, 3};
    int64_t sw44[2] = {4, 4};
    gd_tensor *x = NULL;
    gd_tensor *w_lm = NULL;
    gd_tensor *w_bad = NULL;
    gd_tensor *xi = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *targets_f = NULL;
    gd_tensor *targets_bad = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *wh = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s23, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, sw43, &w_lm));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, sw44, &w_bad));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, s23, &xi));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 1, s2, &targets));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s2, &targets_f));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, s23, &targets_bad));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));

    CHECK_OK(gd_sum(ctx, x, -1, false, &out));
    CHECK_TRUE(check_shape(out, 1, s2) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_mean(ctx, x, 1, true, &out));
    CHECK_TRUE(check_shape(out, 2, s21) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_softmax(ctx, x, -1, &out));
    CHECK_TRUE(check_shape(out, 2, s23) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_cross_entropy(ctx, x, targets, 1, &out));
    CHECK_TRUE(gd_tensor_ndim(out) == 0);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_lm_cross_entropy(ctx, x, w_lm, targets, &out));
    CHECK_TRUE(gd_tensor_ndim(out) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_F32);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cast(ctx, w_lm, GD_DTYPE_F16, &wh));
    CHECK_OK(gd_cross_entropy(ctx, xh, targets, 1, &out));
    CHECK_TRUE(gd_tensor_ndim(out) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_F32);
    gd_tensor_release(out);
    out = NULL;
    CHECK_OK(gd_lm_cross_entropy(ctx, xh, wh, targets, &out));
    CHECK_TRUE(gd_tensor_ndim(out) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_F32);
    gd_tensor_release(out);
    out = NULL;
    gd_tensor_release(xh);
    xh = NULL;
    gd_tensor_release(wh);
    wh = NULL;

    CHECK_STATUS(gd_lm_cross_entropy(ctx, x, w_bad, targets, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);
    CHECK_STATUS(gd_lm_cross_entropy(ctx, x, w_lm, targets_f, &out), GD_ERR_DTYPE);
    CHECK_TRUE(out == NULL);

    CHECK_STATUS(gd_cross_entropy(ctx, x, targets_f, 1, &out), GD_ERR_DTYPE);
    CHECK_TRUE(out == NULL);
    CHECK_STATUS(gd_cross_entropy(ctx, x, targets_bad, 1, &out), GD_ERR_SHAPE);
    CHECK_TRUE(out == NULL);
    CHECK_STATUS(gd_sum(ctx, xi, 0, false, &out), GD_ERR_DTYPE);
    CHECK_TRUE(out == NULL);

    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_I32, &out));
    CHECK_TRUE(check_shape(out, 2, s23) == 0);
    CHECK_TRUE(gd_tensor_dtype(out) == GD_DTYPE_I32);
    gd_tensor_release(out);
    out = NULL;

    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    gd_tensor_release(w_lm);
    gd_tensor_release(w_bad);
    gd_tensor_release(xi);
    gd_tensor_release(targets);
    gd_tensor_release(targets_f);
    gd_tensor_release(targets_bad);
    gd_tensor_release(xh);
    gd_tensor_release(wh);
    return 0;
}

static int test_f16_cast_cpu(gd_context *ctx)
{
    int64_t s6[1] = {6};
    float data[6] = {0.0F, 1.0F, -2.5F, 1.0F / 3.0F, 65504.0F, 1.0e-8F};
    float roundtrip[6] = {0};
    uint16_t half_bits[6] = {0};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc f16_desc;
    gd_tensor *x = NULL;
    gd_tensor *h = NULL;
    gd_tensor *y = NULL;
    gd_graph *g = NULL;
    size_t nbytes = 0U;
    size_t alignment = 0U;
    int i = 0;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F16, cpu, 1, s6, &f16_desc));
    CHECK_OK(gd_tensor_desc_nbytes(&f16_desc, &nbytes, &alignment));
    CHECK_TRUE(nbytes == sizeof(half_bits));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s6, &x));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, data, sizeof(data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &h));
    CHECK_TRUE(gd_tensor_dtype(h) == GD_DTYPE_F16);
    CHECK_OK(gd_cast(ctx, h, GD_DTYPE_F32, &y));
    CHECK_TRUE(gd_tensor_dtype(y) == GD_DTYPE_F32);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, h, half_bits, sizeof(half_bits)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, roundtrip, sizeof(roundtrip)));
    CHECK_TRUE(half_bits[1] == 0x3c00U);
    CHECK_TRUE(half_bits[2] == 0xc100U);
    for (i = 0; i < 6; ++i) {
        float tol = 1.0e-3F * (fabsf(data[i]) > 1.0F ? fabsf(data[i]) : 1.0F);
        CHECK_TRUE(fabsf(roundtrip[i] - data[i]) <= tol);
    }
    gd_tensor_release(y);
    gd_tensor_release(h);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

static float test_sigmoid(float x)
{
    if (x >= 0.0F) {
        float e = expf(-x);
        return 1.0F / (1.0F + e);
    }
    {
        float e = expf(x);
        return e / (1.0F + e);
    }
}

static float test_powlu(float x1, float x2)
{
    float s = test_sigmoid(x2);
    if (x2 <= 0.0F) {
        return x1 * x2 * s;
    }
    return x1 * powf(x2, 3.0F / (sqrtf(x2) + 1.0F)) * s;
}

static float test_gelu_tanh(float x)
{
    const float c = 0.7978845608028654F;
    float u = c * (x + 0.044715F * x * x * x);
    return 0.5F * x * (1.0F + tanhf(u));
}

static int test_f16_elementwise_cpu(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float xdata[4] = {-2.0F, -0.5F, 0.5F, 2.0F};
    float ydata[4] = {0.25F, -1.0F, 3.0F, -0.5F};
    float got_add[4] = {0};
    float got_mul[4] = {0};
    float got_scale[4] = {0};
    float got_relu[4] = {0};
    float got_silu[4] = {0};
    float got_gelu[4] = {0};
    float got_powlu[4] = {0};
    float got_softmax[4] = {0};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *x = NULL, *y = NULL, *xh = NULL, *yh = NULL;
    gd_tensor *addh = NULL, *mulh = NULL, *scaleh = NULL, *reluh = NULL;
    gd_tensor *siluh = NULL, *geluh = NULL, *powluh = NULL, *softmaxh = NULL;
    gd_tensor *add = NULL, *mul = NULL, *scale = NULL, *relu = NULL;
    gd_tensor *silu = NULL, *gelu = NULL, *powlu = NULL, *softmax = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s4, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, s4, &y));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, xdata, sizeof(xdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, y, ydata, sizeof(ydata)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cast(ctx, y, GD_DTYPE_F16, &yh));
    CHECK_OK(gd_add(ctx, xh, yh, &addh));
    CHECK_OK(gd_mul(ctx, xh, yh, &mulh));
    CHECK_OK(gd_scale(ctx, xh, 0.5F, &scaleh));
    CHECK_OK(gd_relu(ctx, xh, &reluh));
    CHECK_OK(gd_silu(ctx, xh, &siluh));
    CHECK_OK(gd_gelu(ctx, xh, true, &geluh));
    CHECK_OK(gd_powlu(ctx, xh, yh, 3.0F, &powluh));
    CHECK_OK(gd_softmax(ctx, xh, 0, &softmaxh));
    CHECK_OK(gd_cast(ctx, addh, GD_DTYPE_F32, &add));
    CHECK_OK(gd_cast(ctx, mulh, GD_DTYPE_F32, &mul));
    CHECK_OK(gd_cast(ctx, scaleh, GD_DTYPE_F32, &scale));
    CHECK_OK(gd_cast(ctx, reluh, GD_DTYPE_F32, &relu));
    CHECK_OK(gd_cast(ctx, siluh, GD_DTYPE_F32, &silu));
    CHECK_OK(gd_cast(ctx, geluh, GD_DTYPE_F32, &gelu));
    CHECK_OK(gd_cast(ctx, powluh, GD_DTYPE_F32, &powlu));
    CHECK_OK(gd_cast(ctx, softmaxh, GD_DTYPE_F32, &softmax));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, add, got_add, sizeof(got_add)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, mul, got_mul, sizeof(got_mul)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scale, got_scale, sizeof(got_scale)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, relu, got_relu, sizeof(got_relu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, silu, got_silu, sizeof(got_silu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gelu, got_gelu, sizeof(got_gelu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, powlu, got_powlu, sizeof(got_powlu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, softmax, got_softmax, sizeof(got_softmax)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got_add[i] - (xdata[i] + ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_mul[i] - (xdata[i] * ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_scale[i] - (0.5F * xdata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_relu[i] - (xdata[i] > 0.0F ? xdata[i] : 0.0F)) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_silu[i] - (xdata[i] * test_sigmoid(xdata[i]))) <= 2.0e-2F);
        float sm_sum = expf(xdata[0]) + expf(xdata[1]) + expf(xdata[2]) + expf(xdata[3]);
        CHECK_TRUE(fabsf(got_gelu[i] - test_gelu_tanh(xdata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_powlu[i] - test_powlu(xdata[i], ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_softmax[i] - expf(xdata[i]) / sm_sum) <= 2.0e-2F);
    }
    gd_tensor_release(softmax); gd_tensor_release(powlu); gd_tensor_release(gelu); gd_tensor_release(silu);
    gd_tensor_release(relu); gd_tensor_release(scale); gd_tensor_release(mul); gd_tensor_release(add);
    gd_tensor_release(softmaxh); gd_tensor_release(powluh); gd_tensor_release(geluh); gd_tensor_release(siluh);
    gd_tensor_release(reluh); gd_tensor_release(scaleh); gd_tensor_release(mulh); gd_tensor_release(addh);
    gd_tensor_release(yh); gd_tensor_release(xh);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(y); gd_tensor_release(x);
    return 0;
}

static int test_f16_embedding_transpose_cpu(gd_context *ctx)
{
    int64_t table_shape[2] = {3, 2};
    int64_t ids_shape[1] = {3};
    int64_t matrix_shape[2] = {2, 3};
    float table_data[6] = {1.0F, 1.5F, -2.0F, 0.25F, 3.0F, -4.0F};
    int32_t ids_data[3] = {2, 0, 1};
    float matrix_data[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float weight_data[3] = {1.0F, 0.5F, -1.0F};
    float got_emb[6] = {0};
    float got_tr[6] = {0};
    float got_rms[6] = {0};
    float expect_emb[6] = {3.0F, -4.0F, 1.0F, 1.5F, -2.0F, 0.25F};
    float expect_tr[6] = {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *table = NULL, *ids = NULL, *matrix = NULL, *weight = NULL;
    gd_tensor *table_h = NULL, *matrix_h = NULL, *weight_h = NULL;
    gd_tensor *emb_h = NULL, *tr_h = NULL, *rms_h = NULL;
    gd_tensor *emb = NULL, *tr = NULL, *rms = NULL;
    gd_graph *g = NULL;
    int perm[2] = {1, 0};
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, table_shape, &table));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 1, ids_shape, &ids));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, matrix_shape, &matrix));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 1, ids_shape, &weight));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, table, table_data, sizeof(table_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, ids, ids_data, sizeof(ids_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, matrix, matrix_data, sizeof(matrix_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, weight, weight_data, sizeof(weight_data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, table, GD_DTYPE_F16, &table_h));
    CHECK_OK(gd_cast(ctx, matrix, GD_DTYPE_F16, &matrix_h));
    CHECK_OK(gd_cast(ctx, weight, GD_DTYPE_F16, &weight_h));
    CHECK_OK(gd_embedding(ctx, table_h, ids, &emb_h));
    CHECK_OK(gd_transpose(ctx, matrix_h, perm, 2, &tr_h));
    CHECK_OK(gd_rms_norm(ctx, matrix_h, weight_h, 1.0e-5F, &rms_h));
    CHECK_OK(gd_cast(ctx, emb_h, GD_DTYPE_F32, &emb));
    CHECK_OK(gd_cast(ctx, tr_h, GD_DTYPE_F32, &tr));
    CHECK_OK(gd_cast(ctx, rms_h, GD_DTYPE_F32, &rms));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, emb, got_emb, sizeof(got_emb)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tr, got_tr, sizeof(got_tr)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, rms, got_rms, sizeof(got_rms)));
    for (i = 0; i < 6; ++i) {
        int row = i / 3;
        int col = i % 3;
        float a = matrix_data[row * 3 + 0];
        float b = matrix_data[row * 3 + 1];
        float c = matrix_data[row * 3 + 2];
        float inv = 1.0F / sqrtf((a * a + b * b + c * c) / 3.0F + 1.0e-5F);
        float expect_rms = matrix_data[i] * inv * weight_data[col];
        CHECK_TRUE(fabsf(got_emb[i] - expect_emb[i]) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_tr[i] - expect_tr[i]) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_rms[i] - expect_rms) <= 2.0e-2F);
    }
    gd_tensor_release(rms); gd_tensor_release(tr); gd_tensor_release(emb);
    gd_tensor_release(rms_h); gd_tensor_release(tr_h); gd_tensor_release(emb_h);
    gd_tensor_release(weight_h); gd_tensor_release(matrix_h); gd_tensor_release(table_h);
    CHECK_OK(gd_graph_reset(g)); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(weight); gd_tensor_release(matrix); gd_tensor_release(ids); gd_tensor_release(table);
    return 0;
}

static int test_f16_cross_entropy_cpu(gd_context *ctx)
{
    int64_t xshape[2] = {2, 3};
    int64_t tshape[1] = {2};
    float logits_data[6] = {1.0F, 2.0F, 0.0F, -1.0F, 0.5F, 3.0F};
    int32_t targets_data[2] = {1, 2};
    float got = 0.0F;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *x = NULL, *targets = NULL, *xh = NULL, *loss = NULL;
    gd_graph *g = NULL;
    double expect = 0.0;
    int r = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, xshape, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 1, tshape, &targets));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, logits_data, sizeof(logits_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, targets, targets_data, sizeof(targets_data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cross_entropy(ctx, xh, targets, 1, &loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &got, sizeof(got)));
    for (r = 0; r < 2; ++r) {
        float *row = logits_data + r * 3;
        float maxv = fmaxf(row[0], fmaxf(row[1], row[2]));
        double sum = exp((double)row[0] - (double)maxv) +
                     exp((double)row[1] - (double)maxv) +
                     exp((double)row[2] - (double)maxv);
        expect += -((double)row[targets_data[r]] - (double)maxv - log(sum));
    }
    expect *= 0.5;
    CHECK_TRUE(fabsf(got - (float)expect) <= 2.0e-2F);
    gd_tensor_release(loss); gd_tensor_release(xh);
    CHECK_OK(gd_graph_reset(g)); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(targets); gd_tensor_release(x);
    return 0;
}

static int test_f16_lm_cross_entropy_cpu(gd_context *ctx)
{
    int64_t hshape[2] = {2, 3};
    int64_t wshape[2] = {4, 3};
    int64_t tshape[1] = {2};
    float hdata[6] = {0.5F, -1.0F, 2.0F, -0.25F, 0.75F, 1.5F};
    float wdata[12] = {0.1F, -0.2F, 0.3F, 0.4F, 0.5F, -0.6F,
                       -0.7F, 0.8F, 0.9F, -1.0F, 1.1F, 0.2F};
    int32_t targets_data[2] = {0, 3};
    float got = 0.0F;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *h = NULL, *w = NULL, *targets = NULL, *hh = NULL, *wh = NULL, *loss = NULL;
    gd_graph *g = NULL;
    double expect = 0.0;
    int n = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, hshape, &h));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, wshape, &w));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 1, tshape, &targets));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, h, hdata, sizeof(hdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, w, wdata, sizeof(wdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, targets, targets_data, sizeof(targets_data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, h, GD_DTYPE_F16, &hh));
    CHECK_OK(gd_cast(ctx, w, GD_DTYPE_F16, &wh));
    CHECK_OK(gd_lm_cross_entropy(ctx, hh, wh, targets, &loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &got, sizeof(got)));
    for (n = 0; n < 2; ++n) {
        double logits[4];
        double maxv = -1.0e30;
        double sum = 0.0;
        int v = 0;
        for (v = 0; v < 4; ++v) {
            logits[v] = (double)hdata[n * 3 + 0] * (double)wdata[v * 3 + 0] +
                        (double)hdata[n * 3 + 1] * (double)wdata[v * 3 + 1] +
                        (double)hdata[n * 3 + 2] * (double)wdata[v * 3 + 2];
            if (logits[v] > maxv) {
                maxv = logits[v];
            }
        }
        for (v = 0; v < 4; ++v) {
            sum += exp(logits[v] - maxv);
        }
        expect += -(logits[targets_data[n]] - maxv - log(sum));
    }
    expect *= 0.5;
    CHECK_TRUE(fabsf(got - (float)expect) <= 3.0e-2F);
    gd_tensor_release(loss); gd_tensor_release(wh); gd_tensor_release(hh);
    CHECK_OK(gd_graph_reset(g)); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(targets); gd_tensor_release(w); gd_tensor_release(h);
    return 0;
}

static int test_f16_sdpa_cpu(gd_context *ctx)
{
    int64_t shape[4] = {1, 3, 1, 4};
    float qdata[12] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F,
                       0.7F, -0.8F, 0.9F, 1.0F, -1.1F, 1.2F};
    float kdata[12] = {0.2F, 0.1F, -0.3F, 0.5F, 0.4F, -0.6F,
                       0.8F, -0.7F, 0.9F, -1.0F, 1.1F, -1.2F};
    float vdata[12] = {-0.1F, 0.2F, -0.4F, 0.6F, 0.3F, -0.5F,
                       0.7F, 0.9F, -0.8F, 1.0F, -1.1F, 1.2F};
    float f32_out[12] = {0};
    float f16_out[12] = {0};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_sdpa_config cfg = {0.0F, true, 2, 0};
    gd_tensor *q = NULL, *k = NULL, *v = NULL, *qh = NULL, *kh = NULL, *vh = NULL;
    gd_tensor *y32 = NULL, *y16h = NULL, *y16 = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, shape, &q));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, shape, &k));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, shape, &v));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, q, qdata, sizeof(qdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, k, kdata, sizeof(kdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, v, vdata, sizeof(vdata)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &cfg, &y32));
    CHECK_OK(gd_cast(ctx, q, GD_DTYPE_F16, &qh));
    CHECK_OK(gd_cast(ctx, k, GD_DTYPE_F16, &kh));
    CHECK_OK(gd_cast(ctx, v, GD_DTYPE_F16, &vh));
    CHECK_OK(gd_sdpa(ctx, qh, kh, vh, NULL, &cfg, &y16h));
    CHECK_OK(gd_cast(ctx, y16h, GD_DTYPE_F32, &y16));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y32, f32_out, sizeof(f32_out)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y16, f16_out, sizeof(f16_out)));
    for (i = 0; i < 12; ++i) {
        CHECK_TRUE(fabsf(f32_out[i] - f16_out[i]) <= 3.0e-2F);
    }
    gd_tensor_release(y16); gd_tensor_release(y16h); gd_tensor_release(y32);
    gd_tensor_release(vh); gd_tensor_release(kh); gd_tensor_release(qh);
    CHECK_OK(gd_graph_reset(g)); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(v); gd_tensor_release(k); gd_tensor_release(q);
    return 0;
}

static int test_f16_rope_cpu(gd_context *ctx)
{
    int64_t xshape[4] = {1, 2, 1, 4};
    int64_t pshape[2] = {1, 2};
    float xdata[8] = {1.0F, 2.0F, 3.0F, 4.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    int32_t pos_data[2] = {0, 1};
    float got[8] = {0};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_rope_config cfg = {10000.0F, 4, false};
    gd_tensor *x = NULL, *pos = NULL, *xh = NULL, *rh = NULL, *r = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 4, xshape, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, pshape, &pos));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, xdata, sizeof(xdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, pos, pos_data, sizeof(pos_data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_rope(ctx, xh, pos, &cfg, &rh));
    CHECK_OK(gd_cast(ctx, rh, GD_DTYPE_F32, &r));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, r, got, sizeof(got)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got[i] - xdata[i]) <= 2.0e-2F);
    }
    {
        float c0 = cosf(1.0F);
        float s0 = sinf(1.0F);
        float c1 = cosf(0.01F);
        float s1 = sinf(0.01F);
        float expect[4] = {c0, -s1, s0, c1};
        for (i = 0; i < 4; ++i) {
            CHECK_TRUE(fabsf(got[4 + i] - expect[i]) <= 2.0e-2F);
        }
    }
    gd_tensor_release(r); gd_tensor_release(rh); gd_tensor_release(xh);
    CHECK_OK(gd_graph_reset(g)); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(pos); gd_tensor_release(x);
    return 0;
}

static int test_f16_matmul_linear_cpu(gd_context *ctx)
{
    int64_t xshape[2] = {2, 3};
    int64_t wshape[2] = {3, 2};
    float xdata[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F};
    float wdata[6] = {0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F};
    float expect[4] = {3.0F, 4.0F, -1.5F, 7.125F};
    float got_mm[4] = {0};
    float got_lin[4] = {0};
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor *x = NULL;
    gd_tensor *w = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *wh = NULL;
    gd_tensor *mmh = NULL;
    gd_tensor *linh = NULL;
    gd_tensor *mm = NULL;
    gd_tensor *lin = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, xshape, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, wshape, &w));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, xdata, sizeof(xdata)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, w, wdata, sizeof(wdata)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cast(ctx, w, GD_DTYPE_F16, &wh));
    CHECK_OK(gd_matmul(ctx, xh, wh, &mmh));
    CHECK_OK(gd_cast(ctx, mmh, GD_DTYPE_F32, &mm));
    CHECK_OK(gd_linear(ctx, xh, wh, NULL, &linh));
    CHECK_OK(gd_cast(ctx, linh, GD_DTYPE_F32, &lin));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, mm, got_mm, sizeof(got_mm)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, lin, got_lin, sizeof(got_lin)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got_mm[i] - expect[i]) <= 1.0e-2F);
        CHECK_TRUE(fabsf(got_lin[i] - expect[i]) <= 1.0e-2F);
    }
    gd_tensor_release(lin);
    gd_tensor_release(mm);
    gd_tensor_release(linh);
    gd_tensor_release(mmh);
    gd_tensor_release(wh);
    gd_tensor_release(xh);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(w);
    gd_tensor_release(x);
    return 0;
}

static int test_chain_dump(gd_context *ctx)
{
    int64_t xs[2] = {2, 3};
    int64_t ws[2] = {3, 4};
    gd_tensor *x = NULL;
    gd_tensor *w = NULL;
    gd_tensor *h = NULL;
    gd_tensor *y = NULL;
    gd_graph *g = NULL;
    FILE *file = NULL;
    char buf[256];
    int found_nodes = 0;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, xs, &x));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, ws, &w));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_linear(ctx, x, w, NULL, &h));
    CHECK_OK(gd_relu(ctx, h, &y));
    CHECK_OK(gd_graph_end(ctx));

    CHECK_OK(gd_graph_dump(g, GD_DUMP_TEXT, "build/test_ops_dump.txt"));
    file = fopen("build/test_ops_dump.txt", "r");
    CHECK_TRUE(file != NULL);
    while (fgets(buf, sizeof(buf), file) != NULL) {
        if (strstr(buf, "nodes=2") != NULL) {
            found_nodes = 1;
        }
    }
    CHECK_TRUE(fclose(file) == 0);
    CHECK_TRUE(found_nodes != 0);

    gd_tensor_release(y);
    gd_tensor_release(h);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    gd_tensor_release(w);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_elementwise(ctx) != 0 || test_powlu_schema(ctx) != 0 ||
        test_matmul(ctx) != 0 || test_sdpa_prefix_schema(ctx) != 0 ||
        test_linear(ctx) != 0 || test_reduce_softmax_ce_cast(ctx) != 0 ||
        test_f16_cast_cpu(ctx) != 0 || test_f16_elementwise_cpu(ctx) != 0 ||
        test_f16_embedding_transpose_cpu(ctx) != 0 ||
        test_f16_cross_entropy_cpu(ctx) != 0 || test_f16_lm_cross_entropy_cpu(ctx) != 0 ||
        test_f16_sdpa_cpu(ctx) != 0 || test_f16_rope_cpu(ctx) != 0 ||
        test_f16_matmul_linear_cpu(ctx) != 0 || test_chain_dump(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
