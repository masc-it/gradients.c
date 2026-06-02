/* G0 (GPT primitives) CPU<->Metal parity: gelu, transpose, embedding. */
#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

/* White-box hook from the Metal backend: count of fusions applied at compile.
 * Lets this test assert a fusion actually engaged (parity alone can't, since a
 * silent fallback to the unfused path would still pass). */
#if defined(GD_ENABLE_METAL) && GD_ENABLE_METAL
extern unsigned long _gd_metal_fusions_applied(void);
#else
static unsigned long _gd_metal_fusions_applied(void) { return 0UL; }
#endif

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

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes,
                          const float *data, gd_tensor **out)
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

static gd_status make_i32(gd_context *ctx, int ndim, const int64_t *sizes,
                          const int32_t *data, gd_tensor **out)
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

/* Forward parity for gelu (both modes), transpose, embedding in one graph. */
static int test_forward_parity(gd_context *ctx)
{
    int64_t vs[2] = {5, 4};
    int64_t is[1] = {3};
    int64_t xs[2] = {2, 3};
    float table[20];
    int32_t ids[3] = {0, 4, 2};
    float xd[6] = {-2.0F, -0.5F, 0.0F, 0.5F, 1.5F, 3.0F};
    int perm[2] = {1, 0};
    gd_tensor *tab = NULL, *tid = NULL, *tx = NULL;
    gd_tensor *emb = NULL, *ge = NULL, *gt = NULL, *tr = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    int i = 0;

    for (i = 0; i < 20; ++i) {
        table[i] = (float)(i - 10) * 0.25F;
    }
    CHECK_OK(make_f32(ctx, 2, vs, table, &tab));
    CHECK_OK(make_i32(ctx, 1, is, ids, &tid));
    CHECK_OK(make_f32(ctx, 2, xs, xd, &tx));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_embedding(ctx, tab, tid, &emb));     /* [3,4] */
    CHECK_OK(gd_gelu(ctx, tx, false, &ge));          /* exact erf */
    CHECK_OK(gd_gelu(ctx, tx, true, &gt));           /* tanh approx */
    CHECK_OK(gd_transpose(ctx, tx, perm, 2, &tr));   /* [3,2] */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(emb);
    gd_tensor_release(ge);
    gd_tensor_release(gt);
    gd_tensor_release(tr);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tab);
    gd_tensor_release(tid);
    gd_tensor_release(tx);
    return 0;
}

/* RoPE forward parity on a [B,T,H,Dh] tensor with explicit positions. */
static int test_rope_forward(gd_context *ctx)
{
    int64_t xs[4] = {1, 2, 2, 4};
    int64_t ps[2] = {1, 2};
    float xd[16];
    int32_t pid[2] = {0, 1};
    gd_tensor *x = NULL, *pos = NULL, *y = NULL, *yi = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    gd_rope_config interleaved = {10000.0F, 4, true};
    int i = 0;

    for (i = 0; i < 16; ++i) {
        xd[i] = (float)(i - 8) * 0.2F;
    }
    CHECK_OK(make_f32(ctx, 4, xs, xd, &x));
    CHECK_OK(make_i32(ctx, 2, ps, pid, &pos));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_rope(ctx, x, pos, NULL, &y));            /* NeoX half-split */
    CHECK_OK(gd_rope(ctx, x, pos, &interleaved, &yi));   /* GPT-J interleaved */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(y);
    gd_tensor_release(yi);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    gd_tensor_release(pos);
    return 0;
}

/* SDPA + additive-bias parity (forward + q/k/v gradients). */
static int build_sdpa_bias(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t qs[4] = {1, 3, 4, 4};
    int64_t ks[4] = {1, 3, 2, 4};
    int64_t bs[4] = {1, 1, 3, 3};
    int64_t flat[1] = {48};
    float qd[48];
    float kd[24];
    float vd[24];
    float bd[9] = {0.0F, -1e9F, -1e9F, 0.1F, 0.0F, -1e9F, 0.2F, -0.1F, 0.3F};
    gd_tensor *q = NULL, *k = NULL, *v = NULL, *bias = NULL;
    gd_tensor *y = NULL, *gel = NULL, *fl = NULL, *loss = NULL;
    int i = 0;

    for (i = 0; i < 48; ++i) {
        qd[i] = 0.1F * (float)((i % 9) - 4);
    }
    for (i = 0; i < 24; ++i) {
        kd[i] = 0.15F * (float)((i % 7) - 3);
        vd[i] = 0.2F * (float)((i % 5) - 2);
    }
    CHECK_OK(make_f32(ctx, 4, qs, qd, &q));
    CHECK_OK(make_f32(ctx, 4, ks, kd, &k));
    CHECK_OK(make_f32(ctx, 4, ks, vd, &v));
    CHECK_OK(make_f32(ctx, 4, bs, bd, &bias));
    CHECK_OK(gd_tensor_set_requires_grad(q, true));
    CHECK_OK(gd_tensor_set_requires_grad(k, true));
    CHECK_OK(gd_tensor_set_requires_grad(v, true));
    CHECK_OK(gd_sdpa(ctx, q, k, v, bias, NULL, &y));
    CHECK_OK(gd_gelu(ctx, y, false, &gel));
    CHECK_OK(gd_tensor_reshape(gel, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    gd_tensor_release(y);
    gd_tensor_release(gel);
    gd_tensor_release(fl);
    gd_tensor_release(bias); /* graph retains it */
    inputs[0] = q;
    inputs[1] = k;
    inputs[2] = v;
    *n = 3;
    *loss_out = loss;
    return 0;
}

/* SDPA forward parity (dense + causal, grouped-query). */
static int test_sdpa_forward(gd_context *ctx)
{
    int64_t qs[4] = {1, 3, 4, 4};
    int64_t ks[4] = {1, 3, 2, 4};
    float qd[48];
    float kd[24];
    float vd[24];
    gd_tensor *q = NULL, *k = NULL, *v = NULL, *yd = NULL, *yc = NULL, *yp = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    gd_sdpa_config causal = {0.0F, true, 0, 0};
    gd_sdpa_config prefix = {0.0F, true, 0, 2};
    int i = 0;

    for (i = 0; i < 48; ++i) {
        qd[i] = 0.1F * (float)((i % 9) - 4);
    }
    for (i = 0; i < 24; ++i) {
        kd[i] = 0.15F * (float)((i % 7) - 3);
        vd[i] = 0.2F * (float)((i % 5) - 2);
    }
    CHECK_OK(make_f32(ctx, 4, qs, qd, &q));
    CHECK_OK(make_f32(ctx, 4, ks, kd, &k));
    CHECK_OK(make_f32(ctx, 4, ks, vd, &v));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, NULL, &yd));      /* dense */
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &causal, &yc));   /* causal */
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &prefix, &yp));   /* prefix-causal */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(yd);
    gd_tensor_release(yc);
    gd_tensor_release(yp);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(q);
    gd_tensor_release(k);
    gd_tensor_release(v);
    return 0;
}

/* --- gradient parity harness (mirrors test_metal.c) --- */
typedef int (*build_loss_fn)(gd_context *ctx, gd_tensor **inputs_out, int *n_out,
                            gd_tensor **loss_out);

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
        int64_t k = 1;
        int d = 0;
        for (d = 0; d < gd_tensor_ndim(inputs[i]); ++d) {
            k *= gd_tensor_size(inputs[i], d);
        }
        numel[i] = k;
        CHECK_OK(gd_tensor_grad(inputs[i], &grad));
        CHECK_TRUE(grad != NULL);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, gbuf[i], (size_t)k * sizeof(float)));
    }
    *n_inputs = n;

    gd_tensor_release(loss);
    CHECK_OK(gd_graph_destroy(g));
    for (i = 0; i < n; ++i) {
        gd_tensor_release(inputs[i]);
    }
    return 0;
}

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

    if (run_grads(ctx, CPU, build, gcpu, ncpu, &n_cpu) != 0 ||
        run_grads(ctx, METAL, build, gmtl, nmtl, &n_mtl) != 0) {
        fprintf(stderr, "%s: run failed\n", name);
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

static int build_gelu(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t s[1] = {6};
    float xd[6] = {-2.0F, -0.5F, 0.3F, 1.0F, 2.0F, -1.5F};
    gd_tensor *x = NULL, *a = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 1, s, xd, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_gelu(ctx, x, false, &a));
    CHECK_OK(gd_mean(ctx, a, 0, false, &loss));
    gd_tensor_release(a);
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int build_gelu_tanh(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t s[1] = {6};
    float xd[6] = {-2.0F, -0.5F, 0.3F, 1.0F, 2.0F, -1.5F};
    gd_tensor *x = NULL, *a = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 1, s, xd, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_gelu(ctx, x, true, &a));
    CHECK_OK(gd_mean(ctx, a, 0, false, &loss));
    gd_tensor_release(a);
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int build_transpose(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t s[2] = {2, 3};
    float xd[6] = {1, -2, 3, -4, 5, -6};
    int perm[2] = {1, 0};
    gd_tensor *x = NULL, *t = NULL, *sq = NULL, *red = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, s, xd, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_transpose(ctx, x, perm, 2, &t));   /* [3,2] */
    CHECK_OK(gd_gelu(ctx, t, false, &sq));          /* nonlinearity so grad != const */
    CHECK_OK(gd_sum(ctx, sq, 1, false, &red));      /* [3] */
    CHECK_OK(gd_mean(ctx, red, 0, false, &loss));
    gd_tensor_release(t);
    gd_tensor_release(sq);
    gd_tensor_release(red);
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int build_embedding(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t vs[2] = {5, 4};
    int64_t is[1] = {6};
    float table[20];
    int32_t ids[6] = {0, 2, 2, 4, 1, 2}; /* repeats exercise scatter-add accumulation */
    gd_tensor *tab = NULL, *tid = NULL, *emb = NULL, *red = NULL, *loss = NULL;
    int i = 0;

    for (i = 0; i < 20; ++i) {
        table[i] = (float)(i - 10) * 0.1F;
    }
    CHECK_OK(make_f32(ctx, 2, vs, table, &tab));
    CHECK_OK(make_i32(ctx, 1, is, ids, &tid));
    CHECK_OK(gd_tensor_set_requires_grad(tab, true));
    CHECK_OK(gd_embedding(ctx, tab, tid, &emb));   /* [6,4] */
    CHECK_OK(gd_sum(ctx, emb, 1, false, &red));     /* [6] */
    CHECK_OK(gd_mean(ctx, red, 0, false, &loss));
    gd_tensor_release(emb);
    gd_tensor_release(red);
    gd_tensor_release(tid); /* graph retains it; table is the grad input */
    inputs[0] = tab;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int build_rope(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t xs[4] = {1, 2, 2, 4};
    int64_t ps[2] = {1, 2};
    int64_t flat[1] = {16};
    float xd[16];
    int32_t pid[2] = {0, 1};
    gd_tensor *x = NULL, *pos = NULL, *y = NULL, *gel = NULL, *fl = NULL, *loss = NULL;
    int i = 0;

    for (i = 0; i < 16; ++i) {
        xd[i] = (float)(i - 8) * 0.2F;
    }
    CHECK_OK(make_f32(ctx, 4, xs, xd, &x));
    CHECK_OK(make_i32(ctx, 2, ps, pid, &pos));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_rope(ctx, x, pos, NULL, &y));
    CHECK_OK(gd_gelu(ctx, y, false, &gel));    /* nonlinearity: varied upstream grad */
    CHECK_OK(gd_tensor_reshape(gel, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    gd_tensor_release(y);
    gd_tensor_release(gel);
    gd_tensor_release(fl);
    gd_tensor_release(pos); /* graph retains it */
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

static int build_sdpa_impl(gd_context *ctx, gd_tensor **inputs, int *n,
                           gd_tensor **loss_out, bool causal, int prefix_len)
{
    int64_t qs[4] = {1, 3, 4, 4};
    int64_t ks[4] = {1, 3, 2, 4};
    int64_t flat[1] = {48};
    float qd[48];
    float kd[24];
    float vd[24];
    gd_sdpa_config cfg = {0};
    gd_tensor *q = NULL, *k = NULL, *v = NULL, *y = NULL, *gel = NULL, *fl = NULL, *loss = NULL;
    int i = 0;

    for (i = 0; i < 48; ++i) {
        qd[i] = 0.1F * (float)((i % 9) - 4);
    }
    for (i = 0; i < 24; ++i) {
        kd[i] = 0.15F * (float)((i % 7) - 3);
        vd[i] = 0.2F * (float)((i % 5) - 2);
    }
    cfg.causal = causal;
    cfg.prefix_len = prefix_len;
    CHECK_OK(make_f32(ctx, 4, qs, qd, &q));
    CHECK_OK(make_f32(ctx, 4, ks, kd, &k));
    CHECK_OK(make_f32(ctx, 4, ks, vd, &v));
    CHECK_OK(gd_tensor_set_requires_grad(q, true));
    CHECK_OK(gd_tensor_set_requires_grad(k, true));
    CHECK_OK(gd_tensor_set_requires_grad(v, true));
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &cfg, &y));
    CHECK_OK(gd_gelu(ctx, y, false, &gel));  /* varied upstream grad into sdpa */
    CHECK_OK(gd_tensor_reshape(gel, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    gd_tensor_release(y);
    gd_tensor_release(gel);
    gd_tensor_release(fl);
    inputs[0] = q;
    inputs[1] = k;
    inputs[2] = v;
    *n = 3;
    *loss_out = loss;
    return 0;
}

static int build_sdpa(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    return build_sdpa_impl(ctx, inputs, n, loss_out, false, 0);
}

static int build_sdpa_causal(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    return build_sdpa_impl(ctx, inputs, n, loss_out, true, 0);
}

static int build_sdpa_prefix(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    return build_sdpa_impl(ctx, inputs, n, loss_out, true, 2);
}

/* F1: fused SwiGLU silu+mul. Checks CPU<->Metal parity of the whole fwd+bwd
 * graph AND that the fusion engaged (so a silent fallback can't hide a missed
 * fusion or a broken fused kernel). */
static int test_swiglu_fusion(gd_context *ctx)
{
    enum { SG_N = 4 * 8 };
    float gate_d[SG_N];
    float up_d[SG_N];
    int64_t s2[2] = {4, 8};
    int64_t flat[1] = {SG_N};
    gd_compare_options opts = {1e-4, 1e-4, false};
    gd_tensor *gate = NULL, *up = NULL, *act = NULL, *hh = NULL, *fl = NULL, *loss = NULL;
    gd_graph *g = NULL;
    unsigned long n0 = 0, n1 = 0;
    int i = 0;

    for (i = 0; i < SG_N; ++i) {
        gate_d[i] = 0.1F * (float)((i % 11) - 5);
        up_d[i] = 0.2F * (float)((i % 7) - 3);
    }
    CHECK_OK(make_f32(ctx, 2, s2, gate_d, &gate));
    CHECK_OK(make_f32(ctx, 2, s2, up_d, &up));
    CHECK_OK(gd_tensor_set_requires_grad(gate, true));
    CHECK_OK(gd_tensor_set_requires_grad(up, true));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_silu(ctx, gate, &act));
    CHECK_OK(gd_mul(ctx, act, up, &hh));
    CHECK_OK(gd_tensor_reshape(hh, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(act);
    gd_tensor_release(hh);
    gd_tensor_release(fl);
    gd_tensor_release(loss);

    n0 = _gd_metal_fusions_applied();
    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));
    n1 = _gd_metal_fusions_applied();
    CHECK_TRUE(n1 > n0); /* the silu+mul fusion must have engaged */

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(gate);
    gd_tensor_release(up);
    return 0;
}

/* F4: add -> rms_norm forward plus rms_norm_bwd -> gradient add backward. The
 * extra z=sum+y consumer gives the residual stream an outside grad, forcing the
 * backward accumulation ADD that the fused bwd kernel absorbs. */
static int test_add_rms_norm_fusion(gd_context *ctx)
{
    enum { AR_ROWS = 4, AR_LAST = 320, AR_N = AR_ROWS * AR_LAST };
    float a_d[AR_N];
    float b_d[AR_N];
    float w_d[AR_LAST];
    int64_t s2[2] = {AR_ROWS, AR_LAST};
    int64_t ws[1] = {AR_LAST};
    int64_t flat[1] = {AR_N};
    gd_compare_options opts = {1e-4, 1e-4, false};
    gd_tensor *a = NULL, *b = NULL, *w = NULL, *sum = NULL, *y = NULL;
    gd_tensor *z = NULL, *fl = NULL, *loss = NULL;
    gd_graph *g = NULL;
    unsigned long n0 = 0, n1 = 0;
    int i = 0;

    for (i = 0; i < AR_N; ++i) {
        a_d[i] = 0.03F * (float)((i % 17) - 8);
        b_d[i] = 0.02F * (float)((i % 13) - 6);
    }
    for (i = 0; i < AR_LAST; ++i) {
        w_d[i] = 0.8F + 0.001F * (float)(i % 23);
    }
    CHECK_OK(make_f32(ctx, 2, s2, a_d, &a));
    CHECK_OK(make_f32(ctx, 2, s2, b_d, &b));
    CHECK_OK(make_f32(ctx, 1, ws, w_d, &w));
    CHECK_OK(gd_tensor_set_requires_grad(a, true));
    CHECK_OK(gd_tensor_set_requires_grad(b, true));
    CHECK_OK(gd_tensor_set_requires_grad(w, true));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, a, b, &sum));
    CHECK_OK(gd_rms_norm(ctx, sum, w, 1e-5F, &y));
    CHECK_OK(gd_add(ctx, sum, y, &z));
    CHECK_OK(gd_tensor_reshape(z, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(sum);
    gd_tensor_release(y);
    gd_tensor_release(z);
    gd_tensor_release(fl);
    gd_tensor_release(loss);

    n0 = _gd_metal_fusions_applied();
    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));
    n1 = _gd_metal_fusions_applied();
    CHECK_TRUE(n1 >= n0 + 2); /* fwd add+rms_norm and bwd rms_norm_bwd+add */

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(a);
    gd_tensor_release(b);
    gd_tensor_release(w);
    return 0;
}

/* Multi-block SDPA parity (forward + backward via gd_graph_compare). The tiny-T
 * builders above fit in a single tile (BQ=64, BK=16) and run through the 64-float
 * gradient harness. These cases use real [B,T,H,Dh] layout with T crossing query
 * blocks, key blocks, and split-K boundaries. Window and prefix cases check that
 * the per-element predicate stays exact alongside block skip. */
static int test_sdpa_multiblock(gd_context *ctx, int T, int window, int prefix_len)
{
    /* MB_TMAX sizes the buffers; T (<= MB_TMAX) selects the actual sequence so a
     * single test covers both the single-pass+skip path (T small => 1 split) and
     * the split-K path (T >= 2*GD_METAL_SDPA_SPLIT_MIN => multiple splits). */
    enum { MB_B = 2, MB_H = 2, MB_TMAX = 300, MB_DH = 32, MB_NMAX = MB_B * MB_H * MB_TMAX * MB_DH };
    static float qd[MB_NMAX];
    static float kd[MB_NMAX];
    static float vd[MB_NMAX];
    int64_t qs[4] = {MB_B, T, MB_H, MB_DH};
    int64_t flat[1] = {(int64_t)MB_B * T * MB_H * MB_DH};
    int MB_N = MB_B * MB_H * T * MB_DH;
    gd_compare_options opts = {1e-4, 1e-4, false};
    gd_sdpa_config cfg = {0};
    gd_tensor *q = NULL, *k = NULL, *v = NULL, *y = NULL, *gel = NULL, *fl = NULL, *loss = NULL;
    gd_graph *g = NULL;
    int i = 0;

    for (i = 0; i < MB_N; ++i) {
        qd[i] = 0.1F * (float)((i % 9) - 4);
        kd[i] = 0.15F * (float)((i % 7) - 3);
        vd[i] = 0.2F * (float)((i % 5) - 2);
    }
    cfg.causal = true;
    cfg.sliding_window = window;
    cfg.prefix_len = prefix_len;
    CHECK_OK(make_f32(ctx, 4, qs, qd, &q));
    CHECK_OK(make_f32(ctx, 4, qs, kd, &k));
    CHECK_OK(make_f32(ctx, 4, qs, vd, &v));
    CHECK_OK(gd_tensor_set_requires_grad(q, true));
    CHECK_OK(gd_tensor_set_requires_grad(k, true));
    CHECK_OK(gd_tensor_set_requires_grad(v, true));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &cfg, &y));
    CHECK_OK(gd_gelu(ctx, y, false, &gel)); /* varied upstream grad into sdpa */
    CHECK_OK(gd_tensor_reshape(gel, 1, flat, &fl));
    CHECK_OK(gd_mean(ctx, fl, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(y);
    gd_tensor_release(gel);
    gd_tensor_release(fl);
    gd_tensor_release(loss);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(q);
    gd_tensor_release(k);
    gd_tensor_release(v);
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
    if (gd_synchronize(ctx, METAL) != GD_OK) {
        printf("test_metal_gpt: skipped (no Metal backend)\n");
        gd_context_destroy(ctx);
        return 0;
    }

    rc |= test_forward_parity(ctx);
    rc |= test_rope_forward(ctx);
    rc |= test_sdpa_forward(ctx);
    rc |= backward_parity(ctx, build_gelu, "gelu");
    rc |= backward_parity(ctx, build_rope, "rope");
    rc |= backward_parity(ctx, build_sdpa, "sdpa");
    rc |= backward_parity(ctx, build_sdpa_causal, "sdpa_causal");
    rc |= backward_parity(ctx, build_sdpa_prefix, "sdpa_prefix");
    rc |= test_sdpa_multiblock(ctx, 130, 0, 0);    /* causal, single-pass + block-skip */
    rc |= test_sdpa_multiblock(ctx, 130, 48, 0);   /* causal + sliding window */
    rc |= test_sdpa_multiblock(ctx, 300, 0, 0);    /* causal, split-K */
    rc |= test_sdpa_multiblock(ctx, 300, 48, 0);   /* causal + window, split-K */
    rc |= test_sdpa_multiblock(ctx, 130, 0, 33);   /* prefix-causal, boundary block */
    rc |= test_sdpa_multiblock(ctx, 130, 48, 33);  /* prefix + suffix sliding window */
    rc |= test_sdpa_multiblock(ctx, 300, 0, 129);  /* prefix-causal, split-K */
    rc |= test_sdpa_multiblock(ctx, 300, 48, 129); /* prefix + window, split-K */
    rc |= test_swiglu_fusion(ctx);            /* F1 fused silu+mul + fired check */
    rc |= test_add_rms_norm_fusion(ctx);      /* F4 residual add+rms_norm fwd/bwd */
    rc |= backward_parity(ctx, build_sdpa_bias, "sdpa_bias");
    rc |= backward_parity(ctx, build_gelu_tanh, "gelu_tanh");
    rc |= backward_parity(ctx, build_transpose, "transpose");
    rc |= backward_parity(ctx, build_embedding, "embedding");

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal_gpt: ok\n");
    }
    return rc;
}
