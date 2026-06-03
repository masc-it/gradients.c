#include "gradients/gradients.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

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
    return fabsf(a - b) <= 1e-5F * (1.0F + fabsf(b));
}

static gd_status make_param_vec(gd_context *ctx, const float *values, int64_t n, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 1, &n, &desc);
    if (status != GD_OK) { return status; }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) { return status; }
    status = gd_tensor_copy_from_cpu(ctx, *out, values, (size_t)n * sizeof(float));
    if (status != GD_OK) { return status; }
    return gd_tensor_set_requires_grad(*out, true);
}

static gd_status make_param(gd_context *ctx, float value, gd_tensor **out)
{
    return make_param_vec(ctx, &value, 1, out);
}

static gd_status make_f16_param(gd_context *ctx, uint16_t value, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t n = 1;
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F16, cpu, 1, &n, &desc);
    if (status != GD_OK) { return status; }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) { return status; }
    status = gd_tensor_copy_from_cpu(ctx, *out, &value, sizeof(value));
    if (status != GD_OK) { return status; }
    return gd_tensor_set_requires_grad(*out, true);
}

static gd_status make_scalar(gd_context *ctx, float value, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 0, NULL, &desc);
    if (status != GD_OK) { return status; }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) { return status; }
    return gd_tensor_copy_from_cpu(ctx, *out, &value, sizeof(value));
}

static void adamw_config(gd_adamw_config *cfg)
{
    cfg->lr = 0.1F;
    cfg->beta1 = 0.9F;
    cfg->beta2 = 0.999F;
    cfg->eps = 1e-8F;
    cfg->weight_decay = 0.0F;
}

static void scaler_config(gd_amp_scaler_config *cfg)
{
    cfg->init_scale = 8.0F;
    cfg->growth_factor = 2.0F;
    cfg->backoff_factor = 0.5F;
    cfg->growth_interval = 1;
    cfg->min_scale = 1.0F;
    cfg->max_scale = 1024.0F;
}

static int test_amp_scaled_backward(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *scaled = NULL;
    gd_graph *graph = NULL;
    float got = 0.0F;
    bool stepped = false;

    adamw_config(&ocfg);
    scaler_config(&scfg);
    CHECK_OK(make_param(ctx, 1.0F, &param));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_mean(ctx, param, 0, false, &loss));
    CHECK_OK(gd_amp_scaler_scale_loss(ctx, scaler, loss, &scaled));
    CHECK_OK(gd_backward(ctx, scaled));
    CHECK_OK(gd_optimizer_step_amp(ctx, opt, scaler));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_TRUE(close_to(gd_amp_scaler_scale(scaler), 16.0F));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.9F));

    gd_tensor_release(scaled);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

static int test_amp_clip_unscales_before_clipping(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *norm = NULL;
    gd_graph *graph = NULL;
    float init[2] = {1.0F, 2.0F};
    float scaled_grad[2] = {48.0F, 64.0F};
    float got[2] = {0};

    adamw_config(&ocfg);
    scaler_config(&scfg);
    ocfg.lr = 0.0F;
    CHECK_OK(make_param_vec(ctx, init, 2, &param));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, scaled_grad, sizeof(scaled_grad)));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_amp_clip(ctx, opt, scaler, 5.0F, &norm));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got, sizeof(got)));
    CHECK_TRUE(close_to(got[0], 3.0F));
    CHECK_TRUE(close_to(got[1], 4.0F));

    gd_tensor_release(norm);
    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

static int test_amp_skip_preserves_step(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float inf_grad = INFINITY;
    float scaled_grad = 0.0F;
    float got = 0.0F;
    bool stepped = true;

    adamw_config(&ocfg);
    scaler_config(&scfg);
    CHECK_OK(make_param(ctx, 1.0F, &param));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &inf_grad, sizeof(inf_grad)));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_amp(ctx, opt, scaler));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(!stepped);
    CHECK_TRUE(close_to(gd_amp_scaler_scale(scaler), 4.0F));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 1.0F));

    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    scaled_grad = gd_amp_scaler_scale(scaler);
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &scaled_grad, sizeof(scaled_grad)));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.9F));

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

static int test_amp_lr_tensor_changes_without_rebuild(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *lr = NULL;
    gd_graph *graph = NULL;
    float grad_value = 1.0F;
    float lr_value = 0.1F;
    float got = 0.0F;
    bool stepped = false;

    adamw_config(&ocfg);
    scaler_config(&scfg);
    ocfg.lr = 0.0F;
    scfg.init_scale = 1.0F;
    scfg.growth_interval = 1000;
    CHECK_OK(make_param(ctx, 1.0F, &param));
    CHECK_OK(make_scalar(ctx, lr_value, &lr));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_amp_lr(ctx, opt, scaler, lr));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));

    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &grad_value, sizeof(grad_value)));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.9F));

    lr_value = 0.01F;
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, lr, &lr_value, sizeof(lr_value)));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &grad_value, sizeof(grad_value)));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.89F));

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(lr);
    gd_tensor_release(param);
    return 0;
}

static int test_amp_lr_tensor_f16_master_refresh(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *lr = NULL;
    gd_graph *graph = NULL;
    float grad_value = 1.0F;
    float lr_value = 0.0F;
    uint16_t got = 0U;
    bool stepped = false;

    adamw_config(&ocfg);
    scaler_config(&scfg);
    ocfg.lr = 0.25F;
    scfg.init_scale = 1.0F;
    CHECK_OK(make_f16_param(ctx, 0x3c00U, &param));
    CHECK_OK(make_scalar(ctx, lr_value, &lr));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_amp_lr(ctx, opt, scaler, lr));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &grad_value, sizeof(grad_value)));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(got == 0x3c00U);

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(lr);
    gd_tensor_release(param);
    return 0;
}

static int test_amp_master_always_refreshes_f32_param(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_adamw_config ocfg = {0};
    gd_amp_scaler_config scfg = {0};
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float grad_value = 1.0F;
    float got = 0.0F;
    bool stepped = false;

    adamw_config(&ocfg);
    scaler_config(&scfg);
    ocfg.master_param_policy = GD_MASTER_PARAM_ALWAYS;
    scfg.init_scale = 1.0F;
    CHECK_OK(make_param(ctx, 1.0F, &param));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &ocfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_amp(ctx, opt, scaler));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, cpu));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, &grad_value, sizeof(grad_value)));
    CHECK_OK(gd_graph_run(graph));
    CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
    CHECK_TRUE(stepped);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, &got, sizeof(got)));
    CHECK_TRUE(close_to(got, 0.9F));

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_amp_scaled_backward(ctx) != 0 ||
        test_amp_clip_unscales_before_clipping(ctx) != 0 ||
        test_amp_skip_preserves_step(ctx) != 0 ||
        test_amp_lr_tensor_changes_without_rebuild(ctx) != 0 ||
        test_amp_lr_tensor_f16_master_refresh(ctx) != 0 ||
        test_amp_master_always_refreshes_f32_param(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    printf("amp ok\n");
    return 0;
}
