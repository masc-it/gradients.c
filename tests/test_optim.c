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

static gd_status make_scalar(gd_context *ctx, float value, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 0, NULL, &desc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_copy_from_cpu(ctx, *out, &value, sizeof(value));
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

static int test_lr_scheduler_values(void)
{
    gd_lr_scheduler_config cfg;
    float lr = -1.0F;
    const float pi = 3.14159265358979323846F;
    float expect_mid;

    cfg.max_lr = 0.1F;
    cfg.min_lr = 0.01F;
    cfg.warmup_steps = 10;
    cfg.total_steps = 110;

    CHECK_OK(gd_lr_scheduler_value(&cfg, 0, &lr));
    CHECK_TRUE(close_to(lr, 0.0F));
    CHECK_OK(gd_lr_scheduler_value(&cfg, 5, &lr));
    CHECK_TRUE(close_to(lr, 0.05F));
    CHECK_OK(gd_lr_scheduler_value(&cfg, 10, &lr));
    CHECK_TRUE(close_to(lr, 0.1F));
    CHECK_OK(gd_lr_scheduler_value(&cfg, 60, &lr));
    expect_mid = cfg.min_lr + 0.5F * (cfg.max_lr - cfg.min_lr) *
                              (1.0F + cosf(pi * 0.5F));
    CHECK_TRUE(close_to(lr, expect_mid));
    CHECK_OK(gd_lr_scheduler_value(&cfg, 110, &lr));
    CHECK_TRUE(close_to(lr, cfg.min_lr));
    CHECK_OK(gd_lr_scheduler_value(&cfg, 999, &lr));
    CHECK_TRUE(close_to(lr, cfg.min_lr));
    return 0;
}

static int test_adamw_lr_tensor(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.5F, -1.0F, 0.25F};
    float lrs[2] = {0.1F, 0.01F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *lr = NULL;
    gd_graph *graph = NULL;
    float ref_p[3];
    float ref_m[3] = {0, 0, 0};
    float ref_v[3] = {0, 0, 0};
    float got[3];
    int step = 0;
    int i = 0;

    cfg.lr = 0.0F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.01F;

    memcpy(ref_p, pinit, sizeof(pinit));
    CHECK_OK(make_param(ctx, 3, pinit, &param));
    CHECK_OK(make_scalar(ctx, lrs[0], &lr));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_lr(ctx, opt, lr));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));

    for (step = 1; step <= 2; ++step) {
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, lr, &lrs[step - 1], sizeof(float)));
        CHECK_OK(gd_graph_run(graph));
        ref_adamw(ref_p, ref_m, ref_v, g, 3, step, lrs[step - 1], cfg.beta1,
                  cfg.beta2, cfg.eps, cfg.weight_decay);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
        for (i = 0; i < 3; ++i) {
            CHECK_TRUE(close_to(got[i], ref_p[i]));
        }
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(lr);
    gd_tensor_release(param);
    return 0;
}

static int test_lr_scheduler_write(gd_context *ctx)
{
    gd_lr_scheduler_config cfg;
    gd_tensor *lr_tensor = NULL;
    float lr = -1.0F;
    float got = -1.0F;

    cfg.max_lr = 0.2F;
    cfg.min_lr = 0.02F;
    cfg.warmup_steps = 4;
    cfg.total_steps = 20;
    CHECK_OK(make_scalar(ctx, 0.0F, &lr_tensor));
    CHECK_OK(gd_lr_scheduler_write(ctx, &cfg, 2, lr_tensor, &lr));
    CHECK_TRUE(close_to(lr, 0.1F));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, lr_tensor, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, lr));
    gd_tensor_release(lr_tensor);
    return 0;
}

static int test_clip_grad_norm(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    float p1_init[2] = {1.0F, 2.0F};
    float p2_init[3] = {3.0F, 4.0F, 5.0F};
    float p3_init[1] = {6.0F};
    float g1[2] = {3.0F, 4.0F};
    float g2[3] = {0.0F, 12.0F, -5.0F};
    float small1[2] = {1.0F, 2.0F};
    float small2[3] = {0.0F, 0.0F, 0.0F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *params_for_opt[2];
    gd_tensor *params_for_clip[3];
    gd_tensor *p1 = NULL;
    gd_tensor *p2 = NULL;
    gd_tensor *p3 = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *norm = NULL;
    gd_graph *graph = NULL;
    float got1[2];
    float got2[3];
    float got3[1];
    float got_norm = 0.0F;
    float expect_norm = sqrtf(194.0F);
    float scale = 7.0F / (expect_norm + 1e-6F);
    int i = 0;

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;

    CHECK_OK(make_param(ctx, 2, p1_init, &p1));
    CHECK_OK(make_param(ctx, 3, p2_init, &p2));
    CHECK_OK(make_param(ctx, 1, p3_init, &p3));
    params_for_opt[0] = p1;
    params_for_opt[1] = p2;
    CHECK_OK(gd_adamw_create(ctx, params_for_opt, 2, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g1, sizeof(g1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g2, sizeof(g2)));

    params_for_clip[0] = p1;
    params_for_clip[1] = p2;
    params_for_clip[2] = p3; /* missing grad -> zero grad slot */
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_clip_grad_norm(ctx, params_for_clip, 3, 7.0F, &norm));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, norm, &got_norm, sizeof(got_norm)));
    CHECK_TRUE(close_to(got_norm, expect_norm));
    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got1, sizeof(got1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got2, sizeof(got2)));
    CHECK_OK(gd_tensor_grad(p3, &grad));
    CHECK_TRUE(grad != NULL);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got3, sizeof(got3)));
    for (i = 0; i < 2; ++i) {
        CHECK_TRUE(close_to(got1[i], g1[i] * scale));
    }
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(close_to(got2[i], g2[i] * scale));
    }
    CHECK_TRUE(close_to(got3[0], 0.0F));

    gd_tensor_release(norm);
    norm = NULL;
    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    graph = NULL;

    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, small1, sizeof(small1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, small2, sizeof(small2)));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_clip_grad_norm(ctx, params_for_clip, 3, 10.0F, &norm));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, norm, &got_norm, sizeof(got_norm)));
    CHECK_TRUE(close_to(got_norm, sqrtf(5.0F)));
    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got1, sizeof(got1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got2, sizeof(got2)));
    for (i = 0; i < 2; ++i) {
        CHECK_TRUE(close_to(got1[i], small1[i]));
    }
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(close_to(got2[i], small2[i]));
    }
    gd_tensor_release(norm);
    norm = NULL;
    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(p3);
    gd_tensor_release(p2);
    gd_tensor_release(p1);
    return 0;
}

static int test_clip_before_adamw(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    float pinit[1] = {1.0F};
    float g[1] = {10.0F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *params[1];
    gd_graph *graph = NULL;
    float got = 0.0F;

    cfg.lr = 0.1F;
    cfg.beta1 = 0.1F;
    cfg.beta2 = 0.1F;
    cfg.eps = 1.0F;
    cfg.weight_decay = 0.0F;

    CHECK_OK(make_param(ctx, 1, pinit, &param));
    params[0] = param;
    CHECK_OK(gd_adamw_create(ctx, params, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_clip_grad_norm(ctx, params, 1, 2.0F, NULL));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.93333334F));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 2.0F));

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
    if (test_lr_scheduler_values() != 0 ||
        test_adamw_steps(ctx) != 0 ||
        test_adamw_lr_tensor(ctx) != 0 ||
        test_lr_scheduler_write(ctx) != 0 ||
        test_clip_grad_norm(ctx) != 0 ||
        test_clip_before_adamw(ctx) != 0 ||
        test_tied_single_update(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    printf("optim ok\n");
    return 0;
}
