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
        test_f16_cast_cpu(ctx) != 0 || test_chain_dump(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
