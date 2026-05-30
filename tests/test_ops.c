#include "gradients/gradients.h"

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
    gd_tensor *x = NULL;
    gd_tensor *xi = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *targets_f = NULL;
    gd_tensor *targets_bad = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 2, s23, &x));
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
    gd_tensor_release(xi);
    gd_tensor_release(targets);
    gd_tensor_release(targets_f);
    gd_tensor_release(targets_bad);
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
    if (test_elementwise(ctx) != 0 || test_matmul(ctx) != 0 || test_linear(ctx) != 0 ||
        test_reduce_softmax_ce_cast(ctx) != 0 || test_chain_dump(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
