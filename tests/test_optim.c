#include "gradients/gradients.h"

#include <math.h>
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

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int close_to(float a, float b)
{
    float diff = fabsf(a - b);
    return diff <= 1e-5F * (1.0F + fabsf(b));
}

static gd_status make_param(gd_context *ctx, int64_t n, const float *data, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 1, &n, &desc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float));
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_set_requires_grad(*out, true);
}

/* Reference AdamW (matches the CPU_REF kernel). */
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

static int test_adamw_steps(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.5F, -1.0F, 0.25F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float ref_p[3];
    float ref_m[3] = {0, 0, 0};
    float ref_v[3] = {0, 0, 0};
    float got[3];
    int step = 0;
    int i = 0;

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.01F;

    memcpy(ref_p, pinit, sizeof(pinit));
    CHECK_OK(make_param(ctx, 3, pinit, &param));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));

    /* Pre-fill the gradient slot (no backward in this graph). */
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_TRUE(grad != NULL);
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));

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

static int test_tied_single_update(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    float pinit[2] = {1.0F, 1.0F};
    float g[2] = {1.0F, 1.0F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *params[2];
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float ref_p[2];
    float ref_m[2] = {0, 0};
    float ref_v[2] = {0, 0};
    float got[2];
    int i = 0;

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;

    memcpy(ref_p, pinit, sizeof(pinit));
    CHECK_OK(make_param(ctx, 2, pinit, &param));
    params[0] = param;
    params[1] = param; /* same tensor passed twice -> must dedup */
    CHECK_OK(gd_adamw_create(ctx, params, 2, &cfg, &opt));

    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));

    ref_adamw(ref_p, ref_m, ref_v, g, 2, 1, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps,
              cfg.weight_decay);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
    for (i = 0; i < 2; ++i) {
        /* single update (deduped); a double update would diverge */
        CHECK_TRUE(close_to(got[i], ref_p[i]));
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_adamw_steps(ctx) != 0 || test_tied_single_update(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    printf("optim ok\n");
    return 0;
}
