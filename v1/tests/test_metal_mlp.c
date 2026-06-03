/*
 * M4: full MLP training parity, CPU_REF vs Metal.
 *
 * Model: x[N,2] -> linear(2,H) -> relu -> linear(H,2) -> cross_entropy.
 * Three checks:
 *   1. forward activations match (gd_graph_compare on a side-effect-free graph),
 *   2. leaf gradients match after one backward (separate per-device runs),
 *   3. a few AdamW steps keep parameters in agreement.
 */

#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MLP_H 4
#define MLP_N 4
#define MLP_C 2

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

static int close_tol(float a, float b, float tol)
{
    float diff = a < b ? b - a : a - b;
    return diff <= tol * (1.0F + (b < 0.0F ? -b : b));
}

/* Deterministic parameter init so both devices start from identical weights. */
static float init_value(int i)
{
    return 0.2F * sinf((float)i * 1.3F + 0.5F);
}

typedef struct {
    gd_tensor *x;
    gd_tensor *targets;
    gd_tensor *w1; /* [2,H] */
    gd_tensor *b1; /* [H]   */
    gd_tensor *w2; /* [H,C] */
    gd_tensor *b2; /* [C]   */
} mlp_t;

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

static gd_status make_param(gd_context *ctx, int ndim, const int64_t *sizes,
                            int base, gd_tensor **out)
{
    float buf[MLP_H * MLP_C];
    int64_t numel = 1;
    int i = 0;

    for (i = 0; i < ndim; ++i) {
        numel *= sizes[i];
    }
    for (i = 0; i < (int)numel; ++i) {
        buf[i] = init_value(base + i);
    }
    if (make_f32(ctx, ndim, sizes, buf, out) != GD_OK) {
        return GD_ERR_INTERNAL;
    }
    return gd_tensor_set_requires_grad(*out, true);
}

static int mlp_create(gd_context *ctx, mlp_t *m)
{
    int64_t xs[2] = {MLP_N, 2};
    int64_t ts[1] = {MLP_N};
    int64_t w1s[2] = {2, MLP_H};
    int64_t b1s[1] = {MLP_H};
    int64_t w2s[2] = {MLP_H, MLP_C};
    int64_t b2s[1] = {MLP_C};
    float xdata[MLP_N * 2] = {0, 0, 0, 1, 1, 0, 1, 1};
    int32_t tdata[MLP_N] = {0, 1, 1, 0};
    gd_tensor_desc tdesc;

    memset(m, 0, sizeof(*m));
    CHECK_OK(make_f32(ctx, 2, xs, xdata, &m->x));
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, CPU, 1, ts, &tdesc));
    CHECK_OK(gd_tensor_empty(ctx, &tdesc, &m->targets));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, m->targets, tdata, sizeof(tdata)));
    CHECK_OK(make_param(ctx, 2, w1s, 0, &m->w1));
    CHECK_OK(make_param(ctx, 1, b1s, 100, &m->b1));
    CHECK_OK(make_param(ctx, 2, w2s, 200, &m->w2));
    CHECK_OK(make_param(ctx, 1, b2s, 300, &m->b2));
    return 0;
}

static void mlp_free(mlp_t *m)
{
    gd_tensor_release(m->x);
    gd_tensor_release(m->targets);
    gd_tensor_release(m->w1);
    gd_tensor_release(m->b1);
    gd_tensor_release(m->w2);
    gd_tensor_release(m->b2);
}

static gd_status build_forward(gd_context *ctx, mlp_t *m, gd_tensor **logits_out)
{
    gd_tensor *h = NULL;
    gd_tensor *a = NULL;
    gd_status status = gd_linear(ctx, m->x, m->w1, m->b1, &h);

    if (status != GD_OK) {
        return status;
    }
    status = gd_relu(ctx, h, &a);
    gd_tensor_release(h);
    if (status != GD_OK) {
        return status;
    }
    status = gd_linear(ctx, a, m->w2, m->b2, logits_out);
    gd_tensor_release(a);
    return status;
}

/* (1) Forward activations match via the parity harness. */
static int test_forward_parity(gd_context *ctx)
{
    mlp_t m;
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    if (mlp_create(ctx, &m) != 0) {
        return 1;
    }
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(build_forward(ctx, &m, &logits));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(logits);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    mlp_free(&m);
    return 0;
}

/* (2) Leaf gradients match after one forward+backward. */
static int grads_for_device(gd_context *ctx, gd_device dev, float grads[4][32], int64_t numel[4])
{
    mlp_t m;
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *params[4];
    int i = 0;

    if (mlp_create(ctx, &m) != 0) {
        return 1;
    }
    params[0] = m.w1;
    params[1] = m.b1;
    params[2] = m.w2;
    params[3] = m.b2;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(build_forward(ctx, &m, &logits));
    CHECK_OK(gd_cross_entropy(ctx, logits, m.targets, 1, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_compile(g, dev));
    CHECK_OK(gd_graph_run(g));

    for (i = 0; i < 4; ++i) {
        gd_tensor *grad = NULL;
        int64_t n = 1;
        int d = 0;
        for (d = 0; d < gd_tensor_ndim(params[i]); ++d) {
            n *= gd_tensor_size(params[i], d);
        }
        numel[i] = n;
        CHECK_OK(gd_tensor_grad(params[i], &grad));
        CHECK_TRUE(grad != NULL);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, grads[i], (size_t)n * sizeof(float)));
    }

    CHECK_OK(gd_graph_destroy(g));
    mlp_free(&m);
    return 0;
}

static int test_grad_parity(gd_context *ctx)
{
    float gcpu[4][32];
    float gmtl[4][32];
    int64_t ncpu[4];
    int64_t nmtl[4];
    int i = 0;
    int64_t j = 0;

    if (grads_for_device(ctx, CPU, gcpu, ncpu) != 0) {
        return 1;
    }
    if (grads_for_device(ctx, METAL, gmtl, nmtl) != 0) {
        return 1;
    }
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(ncpu[i] == nmtl[i]);
        for (j = 0; j < ncpu[i]; ++j) {
            if (!close_tol(gcpu[i][j], gmtl[i][j], 1e-4F)) {
                fprintf(stderr, "grad param %d elem %lld: cpu=%g metal=%g\n",
                        i, (long long)j, (double)gcpu[i][j], (double)gmtl[i][j]);
                return 1;
            }
        }
    }
    return 0;
}

/* (3) A few AdamW steps keep parameters in agreement. Returns final params. */
static int train_for_device(gd_context *ctx, gd_device dev, int n_steps,
                            float w1[8], float w2[8], float *final_loss)
{
    mlp_t m;
    gd_optimizer *opt = NULL;
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *params[4];
    gd_adamw_config cfg = {0};
    int step = 0;

    if (mlp_create(ctx, &m) != 0) {
        return 1;
    }
    params[0] = m.w1;
    params[1] = m.b1;
    params[2] = m.w2;
    params[3] = m.b2;

    cfg.lr = 0.05F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;
    CHECK_OK(gd_adamw_create(ctx, params, 4, &cfg, &opt));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(build_forward(ctx, &m, &logits));
    CHECK_OK(gd_cross_entropy(ctx, logits, m.targets, 1, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, dev));

    for (step = 0; step < n_steps; ++step) {
        CHECK_OK(gd_graph_run(g));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, final_loss, sizeof(*final_loss)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, m.w1, w1, 8 * sizeof(float)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, m.w2, w2, 8 * sizeof(float)));

    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_optimizer_destroy(opt);
    mlp_free(&m);
    return 0;
}

static int test_train_parity(gd_context *ctx)
{
    float w1_cpu[8], w2_cpu[8], w1_mtl[8], w2_mtl[8];
    float loss_cpu = 0.0F, loss_mtl = 0.0F;
    const int steps = 8;
    int i = 0;

    if (train_for_device(ctx, CPU, steps, w1_cpu, w2_cpu, &loss_cpu) != 0) {
        return 1;
    }
    if (train_for_device(ctx, METAL, steps, w1_mtl, w2_mtl, &loss_mtl) != 0) {
        return 1;
    }
    /* Drift accumulates across steps (float GPU accum vs double CPU accum), so a
     * looser tolerance than single-op parity. */
    if (!close_tol(loss_cpu, loss_mtl, 1e-3F)) {
        fprintf(stderr, "train loss: cpu=%g metal=%g\n", (double)loss_cpu, (double)loss_mtl);
        return 1;
    }
    for (i = 0; i < 8; ++i) {
        if (!close_tol(w1_cpu[i], w1_mtl[i], 1e-3F) || !close_tol(w2_cpu[i], w2_mtl[i], 1e-3F)) {
            fprintf(stderr, "train param %d: w1 cpu=%g metal=%g  w2 cpu=%g metal=%g\n",
                    i, (double)w1_cpu[i], (double)w1_mtl[i],
                    (double)w2_cpu[i], (double)w2_mtl[i]);
            return 1;
        }
    }
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
        printf("test_metal_mlp: skipped (no Metal backend)\n");
        gd_context_destroy(ctx);
        return 0;
    }

    rc |= test_forward_parity(ctx);
    rc |= test_grad_parity(ctx);
    rc |= test_train_parity(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal_mlp: ok\n");
    }
    return rc;
}
