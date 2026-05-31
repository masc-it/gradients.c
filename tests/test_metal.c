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

    /* Reusing a compiled Metal graph must restage CPU-backed leaves when their
     * host storage changes. This catches stale shadow-buffer bugs in dirty
     * staging. */
    for (i = 0; i < 6; ++i) {
        a[i] = (float)(100 + i);
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, ta, a, sizeof(a)));
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

static int64_t tensor_numel(gd_tensor *t)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < gd_tensor_ndim(t); ++i) {
        n *= gd_tensor_size(t, i);
    }
    return n;
}

/* A backward-graph builder: creates its own (deterministic) input tensors,
 * records the forward ops, and returns the scalar loss. Inputs that require
 * grad are returned in `inputs_out`. */
typedef int (*build_loss_fn)(gd_context *ctx, gd_tensor **inputs_out, int *n_out,
                            gd_tensor **loss_out);

/* Runs build->backward on `dev`, capturing each input's gradient into gbuf. */
static int run_grads(gd_context *ctx, gd_device dev, build_loss_fn build,
                     float gbuf[][64], int64_t numel[], int *n_inputs)
{
    gd_tensor *inputs[4] = {0};
    gd_tensor *loss = NULL;
    gd_graph *g = NULL;
    int n = 0;
    int i = 0;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    if (build(ctx, inputs, &n, &loss) != 0) {
        return 1;
    }
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, dev));
    CHECK_OK(gd_graph_run(g));

    for (i = 0; i < n; ++i) {
        gd_tensor *grad = NULL;
        numel[i] = tensor_numel(inputs[i]);
        CHECK_OK(gd_tensor_grad(inputs[i], &grad));
        CHECK_TRUE(grad != NULL);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, gbuf[i], (size_t)numel[i] * sizeof(float)));
    }
    *n_inputs = n;

    gd_tensor_release(loss);
    CHECK_OK(gd_graph_destroy(g));
    for (i = 0; i < n; ++i) {
        gd_tensor_release(inputs[i]);
    }
    return 0;
}

/* Compares CPU vs Metal gradients for a backward graph builder. */
static int backward_parity(gd_context *ctx, build_loss_fn build, const char *name)
{
    float gcpu[4][64];
    float gmtl[4][64];
    int64_t ncpu[4];
    int64_t nmtl[4];
    int n_cpu = 0;
    int n_mtl = 0;
    int i = 0;
    int64_t j = 0;

    if (run_grads(ctx, CPU, build, gcpu, ncpu, &n_cpu) != 0) {
        fprintf(stderr, "%s: CPU run failed\n", name);
        return 1;
    }
    if (run_grads(ctx, METAL, build, gmtl, nmtl, &n_mtl) != 0) {
        fprintf(stderr, "%s: METAL run failed\n", name);
        return 1;
    }
    CHECK_TRUE(n_cpu == n_mtl);
    for (i = 0; i < n_cpu; ++i) {
        CHECK_TRUE(ncpu[i] == nmtl[i]);
        for (j = 0; j < ncpu[i]; ++j) {
            if (!close_to(gcpu[i][j], gmtl[i][j])) {
                fprintf(stderr, "%s: grad[%d][%lld] cpu=%g metal=%g\n",
                        name, i, (long long)j, (double)gcpu[i][j], (double)gmtl[i][j]);
                return 1;
            }
        }
    }
    return 0;
}

/* matmul -> relu -> sum -> mean: relu_bwd, sum_bwd, mean_bwd, matmul backward. */
static int build_mlp(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[2] = {2, 3};
    int64_t sw[2] = {3, 4};
    float xd[6] = {1, -2, 3, -4, 5, -6};
    float wd[12] = {0.1F, -0.2F, 0.3F, -0.4F, 0.5F, -0.6F,
                    0.7F, -0.8F, 0.9F, -1.0F, 1.1F, -1.2F};
    gd_tensor *x = NULL, *w = NULL, *h = NULL, *a = NULL, *s = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sx, xd, &x));
    CHECK_OK(make_f32(ctx, 2, sw, wd, &w));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_tensor_set_requires_grad(w, true));
    CHECK_OK(gd_matmul(ctx, x, w, &h));
    CHECK_OK(gd_relu(ctx, h, &a));
    CHECK_OK(gd_sum(ctx, a, 1, false, &s));
    CHECK_OK(gd_mean(ctx, s, 0, false, &loss));
    gd_tensor_release(h);
    gd_tensor_release(a);
    gd_tensor_release(s);
    inputs[0] = x;
    inputs[1] = w;
    *n = 2;
    *loss_out = loss;
    return 0;
}

/* softmax -> mul(broadcast) -> sum -> mean: softmax_bwd, reduce_to, sum/mean_bwd. */
static int build_softmax(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[2] = {2, 4};
    int64_t sw[1] = {4};
    float xd[8] = {0.5F, -1.0F, 2.0F, 0.3F, -0.7F, 1.2F, 0.0F, -0.4F};
    float wd[4] = {1.0F, -2.0F, 0.5F, 3.0F};
    gd_tensor *x = NULL, *wt = NULL, *y = NULL, *m = NULL, *s = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sx, xd, &x));
    CHECK_OK(make_f32(ctx, 1, sw, wd, &wt));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_tensor_set_requires_grad(wt, true));
    CHECK_OK(gd_softmax(ctx, x, 1, &y));
    CHECK_OK(gd_mul(ctx, y, wt, &m));
    CHECK_OK(gd_sum(ctx, m, 1, false, &s));
    CHECK_OK(gd_mean(ctx, s, 0, false, &loss));
    gd_tensor_release(y);
    gd_tensor_release(m);
    gd_tensor_release(s);
    inputs[0] = x;
    inputs[1] = wt;
    *n = 2;
    *loss_out = loss;
    return 0;
}

/* silu -> mean: silu_bwd, mean_bwd. */
static int build_silu(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[1] = {6};
    float xd[6] = {1.0F, -2.0F, 0.5F, -0.3F, 2.0F, -1.5F};
    gd_tensor *x = NULL, *a = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 1, sx, xd, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_silu(ctx, x, &a));
    CHECK_OK(gd_mean(ctx, a, 0, false, &loss));
    gd_tensor_release(a);
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

/* cross_entropy -> backward: cross_entropy_bwd. */
static int build_ce(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sl[2] = {3, 5};
    int64_t st[1] = {3};
    float ld[15] = {2.0F, 1.0F, 0.1F, -1.0F, 0.5F,
                    0.2F, 3.0F, -0.5F, 1.5F, 0.0F,
                    -1.0F, 0.3F, 2.2F, 0.7F, 1.1F};
    int32_t td[3] = {0, 1, 2};
    gd_tensor *logits = NULL, *targets = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sl, ld, &logits));
    CHECK_OK(make_i32(ctx, 1, st, td, &targets));
    CHECK_OK(gd_tensor_set_requires_grad(logits, true));
    CHECK_OK(gd_cross_entropy(ctx, logits, targets, 1, &loss));
    gd_tensor_release(targets); /* graph retains it */
    inputs[0] = logits;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int test_metal_backward_parity(gd_context *ctx)
{
    if (backward_parity(ctx, build_mlp, "mlp") != 0) return 1;
    if (backward_parity(ctx, build_softmax, "softmax") != 0) return 1;
    if (backward_parity(ctx, build_silu, "silu") != 0) return 1;
    if (backward_parity(ctx, build_ce, "cross_entropy") != 0) return 1;
    return 0;
}

/* AdamW reference (double precision), matching the CPU_REF kernel. */
static void ref_adamw(float *p, float *m, float *v, const float *g, int n, int t,
                      float lr, float b1, float b2, float eps, float wd)
{
    double bc1 = 1.0 - pow((double)b1, (double)t);
    double bc2 = 1.0 - pow((double)b2, (double)t);
    int i = 0;
    for (i = 0; i < n; ++i) {
        double mi = (double)b1 * (double)m[i] + (1.0 - (double)b1) * (double)g[i];
        double vi = (double)b2 * (double)v[i] + (1.0 - (double)b2) * (double)g[i] * (double)g[i];
        double mhat = mi / bc1;
        double vhat = vi / bc2;
        double pp = (double)p[i];
        m[i] = (float)mi;
        v[i] = (float)vi;
        pp -= (double)lr * (double)wd * pp;
        pp -= (double)lr * mhat / (sqrt(vhat) + (double)eps);
        p[i] = (float)pp;
    }
}

/* AdamW on Metal: in-place param/m/v/step updates + write-back over 3 steps. */
static int test_metal_adamw(gd_context *ctx)
{
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.5F, -1.0F, 0.25F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float ref_p[3], ref_m[3] = {0, 0, 0}, ref_v[3] = {0, 0, 0};
    float got[3];
    int step = 0, i = 0;
    int64_t s3[1] = {3};

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.01F;
    memcpy(ref_p, pinit, sizeof(pinit));

    CHECK_OK(make_f32(ctx, 1, s3, pinit, &param));
    CHECK_OK(gd_tensor_set_requires_grad(param, true));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, METAL));

    for (step = 1; step <= 3; ++step) {
        CHECK_OK(gd_graph_run(graph));
        ref_adamw(ref_p, ref_m, ref_v, g, 3, step, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps,
                  cfg.weight_decay);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
        for (i = 0; i < 3; ++i) {
            CHECK_TRUE(close_to(got[i], ref_p[i]));
        }
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
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
    rc |= test_metal_backward_parity(ctx);
    rc |= test_metal_adamw(ctx);
    rc |= test_metal_fallback(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal: ok\n");
    }
    return rc;
}
