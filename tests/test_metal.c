#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());           \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                                \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static const gd_device CPU = {GD_DEVICE_CPU, 0};
static const gd_device METAL = {GD_DEVICE_METAL, 0};

static int close_to(float a, float b)
{
    float diff = a < b ? b - a : a - b;
    return diff <= 1e-4F * (1.0F + (b < 0.0F ? -b : b));
}

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes, const float *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, CPU, ndim, sizes, &desc);
    int64_t numel = 1;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        numel *= sizes[i];
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)numel * sizeof(float));
}

/* Direct Metal run: c = a + b, read back, verify exact values. */
static int test_metal_add_direct(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {10, 20, 30, 40, 50, 60};
    float out[6];
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tc = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &tc));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tc, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] + b[i]));
    }

    gd_tensor_release(tc);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

/* CPU<->Metal parity via the P8 harness, plain + broadcast adds. */
static int test_metal_add_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    float a[6] = {1, -2, 3, -4, 5, -6};
    float b[6] = {0.5F, 1.5F, -2.5F, 3.5F, -4.5F, 5.5F};
    float r[3] = {100, 200, 300};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tr = NULL;
    gd_tensor *ab = NULL;
    gd_tensor *bc = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(make_f32(ctx, 1, s3, r, &tr));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &ab));        /* elementwise */
    CHECK_OK(gd_add(ctx, ab, tr, &bc));        /* broadcast [2,3] + [3] */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ab);
    gd_tensor_release(bc);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tr);
    return 0;
}

/* Elementwise + unary parity: mul (plain + broadcast), scale, relu, silu. */
static int test_metal_unary_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    float a[6] = {1, -2, 3, -4, 5, -6};
    float b[6] = {0.5F, 1.5F, -2.5F, 3.5F, -4.5F, 5.5F};
    float r[3] = {2, -3, 4};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tr = NULL;
    gd_tensor *prod = NULL;
    gd_tensor *bcast = NULL;
    gd_tensor *scaled = NULL;
    gd_tensor *act = NULL;
    gd_tensor *gate = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(make_f32(ctx, 1, s3, r, &tr));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_mul(ctx, ta, tb, &prod));      /* elementwise */
    CHECK_OK(gd_mul(ctx, prod, tr, &bcast));   /* broadcast [2,3]*[3] */
    CHECK_OK(gd_scale(ctx, bcast, 0.25F, &scaled));
    CHECK_OK(gd_relu(ctx, scaled, &act));
    CHECK_OK(gd_silu(ctx, ta, &gate));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(prod);
    gd_tensor_release(bcast);
    gd_tensor_release(scaled);
    gd_tensor_release(act);
    gd_tensor_release(gate);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tr);
    return 0;
}

/* Cast parity (byte-exact for integer outputs): F32->I32 and F32->F32. */
static int test_metal_cast_parity(gd_context *ctx)
{
    int64_t s5[1] = {5};
    float a[5] = {1.9F, -2.4F, 3.5F, -4.6F, 0.0F};
    gd_tensor *ta = NULL;
    gd_tensor *ci = NULL;
    gd_tensor *cf = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {0.0, 0.0, false};

    CHECK_OK(make_f32(ctx, 1, s5, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, ta, GD_DTYPE_I32, &ci));   /* truncation toward zero */
    CHECK_OK(gd_cast(ctx, ta, GD_DTYPE_F32, &cf));   /* identity */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ci);
    gd_tensor_release(cf);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

static gd_status make_i32(gd_context *ctx, int ndim, const int64_t *sizes, const int32_t *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_I32, CPU, ndim, sizes, &desc);
    int64_t numel = 1;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        numel *= sizes[i];
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)numel * sizeof(int32_t));
}

/* matmul parity: plain 2D, trans_b, and batched. */
static int test_metal_matmul_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s34[2] = {3, 4};
    int64_t s43[2] = {4, 3};
    int64_t a3[3] = {2, 2, 3};
    int64_t b3[3] = {2, 3, 4};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float bt[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    float ab[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float bb[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    gd_tensor *ta = NULL, *tb = NULL, *tbt = NULL, *tab = NULL, *tbb = NULL;
    gd_tensor *mm = NULL, *mmt = NULL, *mmb = NULL;
    gd_graph *g = NULL;
    gd_matmul_desc td = {false, true, {GD_DTYPE_F32, GD_DTYPE_F32}};
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s34, b, &tb));
    CHECK_OK(make_f32(ctx, 2, s43, bt, &tbt));
    CHECK_OK(make_f32(ctx, 3, a3, ab, &tab));
    CHECK_OK(make_f32(ctx, 3, b3, bb, &tbb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_matmul(ctx, ta, tb, &mm));         /* [2,3]@[3,4] */
    CHECK_OK(gd_matmul_ex(ctx, &td, ta, tbt, &mmt)); /* [2,3]@[4,3]^T */
    CHECK_OK(gd_matmul(ctx, tab, tbb, &mmb));       /* [2,2,3]@[2,3,4] */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(mm);
    gd_tensor_release(mmt);
    gd_tensor_release(mmb);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tbt);
    gd_tensor_release(tab);
    gd_tensor_release(tbb);
    return 0;
}

/* linear parity: with and without bias. */
static int test_metal_linear_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s34[2] = {3, 4};
    int64_t s4[1] = {4};
    float x[6] = {1, -2, 3, 4, -5, 6};
    float w[12] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F,
                   0.7F, 0.8F, 0.9F, 1.0F, 1.1F, 1.2F};
    float bias[4] = {0.5F, -0.5F, 1.0F, -1.0F};
    gd_tensor *tx = NULL, *tw = NULL, *tbias = NULL;
    gd_tensor *ln = NULL, *lnb = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, x, &tx));
    CHECK_OK(make_f32(ctx, 2, s34, w, &tw));
    CHECK_OK(make_f32(ctx, 1, s4, bias, &tbias));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_linear(ctx, tx, tw, tbias, &lnb));  /* with bias */
    CHECK_OK(gd_linear(ctx, tx, tw, NULL, &ln));    /* no bias */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ln);
    gd_tensor_release(lnb);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tx);
    gd_tensor_release(tw);
    gd_tensor_release(tbias);
    return 0;
}

/* reductions + softmax + rms_norm parity. */
static int test_metal_reduce_parity(gd_context *ctx)
{
    int64_t s234[3] = {2, 3, 4};
    int64_t s24[2] = {2, 4};
    int64_t s4[1] = {4};
    float x[24];
    float wt[4] = {1.0F, 0.5F, 2.0F, 1.5F};
    float x2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    gd_tensor *tx = NULL, *tx2 = NULL, *tw = NULL;
    gd_tensor *sm = NULL, *mn = NULL, *soft = NULL, *rms = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    int i = 0;

    for (i = 0; i < 24; ++i) {
        x[i] = (float)(i - 12) * 0.3F;
    }
    CHECK_OK(make_f32(ctx, 3, s234, x, &tx));
    CHECK_OK(make_f32(ctx, 2, s24, x2, &tx2));
    CHECK_OK(make_f32(ctx, 1, s4, wt, &tw));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sum(ctx, tx, 1, false, &sm));     /* reduce middle dim */
    CHECK_OK(gd_mean(ctx, tx, 2, true, &mn));     /* reduce last, keepdim */
    CHECK_OK(gd_softmax(ctx, tx, 2, &soft));      /* softmax last dim */
    CHECK_OK(gd_rms_norm(ctx, tx2, tw, 1e-5F, &rms));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(sm);
    gd_tensor_release(mn);
    gd_tensor_release(soft);
    gd_tensor_release(rms);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tx);
    gd_tensor_release(tx2);
    gd_tensor_release(tw);
    return 0;
}

/* cross_entropy parity: scalar mean loss, I32 targets. */
static int test_metal_cross_entropy_parity(gd_context *ctx)
{
    int64_t s35[2] = {3, 5};
    int64_t s3[1] = {3};
    float logits[15] = {2.0F, 1.0F, 0.1F, -1.0F, 0.5F,
                        0.2F, 3.0F, -0.5F, 1.5F, 0.0F,
                        -1.0F, 0.3F, 2.2F, 0.7F, 1.1F};
    int32_t targets[3] = {0, 1, 2};
    gd_tensor *tl = NULL, *tt = NULL, *ce = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s35, logits, &tl));
    CHECK_OK(make_i32(ctx, 1, s3, targets, &tt));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cross_entropy(ctx, tl, tt, 1, &ce));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ce);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tl);
    gd_tensor_release(tt);
    return 0;
}

/* Fallback policy at dispatch: assert_finite has no Metal kernel (debug op). */
static int test_metal_fallback(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float a[4] = {-1, 2, -3, 4};
    float out[4];
    gd_tensor *ta = NULL;
    gd_tensor *act = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 1, s4, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_relu(ctx, ta, &act));
    CHECK_OK(gd_assert_finite(ctx, act)); /* no Metal kernel -> unsupported */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(act);

    /* NONE: unsupported op on Metal must fail loud and name the op. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    CHECK_TRUE(gd_graph_compile(g, METAL) == GD_ERR_UNSUPPORTED);
    CHECK_TRUE(strstr(gd_last_error(), "assert_finite") != NULL);

    /* CPU_REF: whole-graph fallback runs correctly on CPU. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    (void)out;
    (void)i;

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    int rc = 0;

    if (gd_context_create(&ctx) != GD_OK) {
        fprintf(stderr, "context create failed: %s\n", gd_last_error());
        return 1;
    }

    /* Skip gracefully where no Metal backend is registered (no GPU / no lib). */
    if (gd_synchronize(ctx, METAL) != GD_OK) {
        printf("test_metal: skipped (no Metal backend)\n");
        gd_context_destroy(ctx);
        return 0;
    }

    rc |= test_metal_add_direct(ctx);
    rc |= test_metal_add_parity(ctx);
    rc |= test_metal_unary_parity(ctx);
    rc |= test_metal_cast_parity(ctx);
    rc |= test_metal_matmul_parity(ctx);
    rc |= test_metal_linear_parity(ctx);
    rc |= test_metal_reduce_parity(ctx);
    rc |= test_metal_cross_entropy_parity(ctx);
    rc |= test_metal_fallback(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal: ok\n");
    }
    return rc;
}
