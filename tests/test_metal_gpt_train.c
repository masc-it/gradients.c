/* G5: tiny GPT assembly — forward parity, training parity (CPU<->Metal), and
 * overfit sanity. */
#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

static int env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0;
}

static const gd_device CPU = {GD_DEVICE_CPU, 0};
static const gd_device METAL = {GD_DEVICE_METAL, 0};

#define GPT_B 1
#define GPT_T 6
#define GPT_V 16
#define GPT_SEED 0xC0FFEEULL
#define PARAM_CAP 16384

static gd_gpt_config tiny_config(void)
{
    gd_gpt_config c = {0};
    c.vocab_size = GPT_V;
    c.d_model = 16;
    c.n_layers = 2;
    c.n_heads = 4;
    c.n_kv_heads = 2; /* grouped-query */
    c.head_dim = 4;   /* 4*4 == 16 */
    c.d_ff = 32;
    c.max_seq_len = GPT_T;
    c.rope_theta = 10000.0f;
    c.norm_eps = 1e-5f;
    c.mlp_kind = GD_GPT_MLP_POWLU;
    c.powlu_m = 3.0f;
    c.tie_embeddings = true;
    return c;
}

static gd_status make_inputs(gd_context *ctx, gd_tensor **tokens, gd_tensor **pos,
                             gd_tensor **targets)
{
    gd_tensor_desc d;
    int32_t tok[GPT_B * GPT_T] = {1, 5, 2, 9, 4, 7};
    int32_t tgt[GPT_B * GPT_T] = {5, 2, 9, 4, 7, 3}; /* next-token */
    int32_t ps[GPT_B * GPT_T] = {0, 1, 2, 3, 4, 5};
    int64_t s2[2] = {GPT_B, GPT_T};
    gd_status status = GD_OK;

    status = gd_tensor_desc_contiguous(GD_DTYPE_I32, CPU, 2, s2, &d);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &d, tokens);
    if (status != GD_OK) {
        return status;
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, *tokens, tok, sizeof(tok)));
    status = gd_tensor_empty(ctx, &d, pos);
    if (status != GD_OK) {
        return status;
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, *pos, ps, sizeof(ps)));
    status = gd_tensor_empty(ctx, &d, targets);
    if (status != GD_OK) {
        return status;
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, *targets, tgt, sizeof(tgt)));
    return GD_OK;
}

static int test_forward_parity(gd_context *ctx, gd_tensor *tokens, gd_tensor *pos)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_compare_options opts = {2e-4, 2e-4, false};

    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward(ctx, gpt, tokens, pos, &logits));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(logits);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_gpt_destroy(gpt);
    return 0;
}

static int test_forward_loss_parity(gd_context *ctx, gd_tensor *tokens,
                                    gd_tensor *pos, gd_tensor *targets)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_graph *g = NULL;
    gd_tensor *loss = NULL;
    gd_compare_options opts = {2e-4, 2e-4, false};

    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward_loss(ctx, gpt, tokens, pos, targets, &loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(loss);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_gpt_destroy(gpt);
    return 0;
}

static int collect_f16_forward_loss(gd_context *ctx, gd_device dev,
                                    gd_tensor *tokens, gd_tensor *pos,
                                    gd_tensor *targets, float *loss_out)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_tensor **params = NULL;
    int n_params = 0;
    gd_graph *g = NULL;
    gd_tensor *loss = NULL;
    float loss_value = 0.0f;
    int i = 0;

    cfg.param_dtype = GD_DTYPE_F16;
    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_gpt_parameters(gpt, &params, &n_params));
    for (i = 0; i < n_params; ++i) {
        CHECK_TRUE(gd_tensor_dtype(params[i]) == GD_DTYPE_F16);
    }
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward_loss(ctx, gpt, tokens, pos, targets, &loss));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, dev));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, dev));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &loss_value, sizeof(loss_value)));
    CHECK_TRUE(isfinite(loss_value));
    *loss_out = loss_value;
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_destroy(g));
    gd_gpt_destroy(gpt);
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

/* Trains a fresh same-seed model on `dev` for `steps`; returns final loss and a
 * flat snapshot of all parameters. */
static int train_collect(gd_context *ctx, gd_device dev, int steps,
                         gd_tensor *tokens, gd_tensor *pos, gd_tensor *targets,
                         float *params_out, int *nfloats_out, float *loss_out)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_optimizer *opt = NULL;
    gd_tensor **params = NULL;
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_adamw_config acfg = {0};
    int n_params = 0;
    int step = 0;
    int i = 0;
    int off = 0;

    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_gpt_parameters(gpt, &params, &n_params));
    acfg.lr = 0.02f;
    acfg.beta1 = 0.9f;
    acfg.beta2 = 0.999f;
    acfg.eps = 1e-8f;
    acfg.weight_decay = 0.0f;
    CHECK_OK(gd_adamw_create(ctx, params, n_params, &acfg, &opt));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward(ctx, gpt, tokens, pos, &logits));
    CHECK_OK(gd_cross_entropy(ctx, logits, targets, 2, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, dev));

    for (step = 0; step < steps; ++step) {
        CHECK_OK(gd_graph_run(g));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, loss_out, sizeof(float)));

    for (i = 0; i < n_params; ++i) {
        int64_t k = tensor_numel(params[i]);
        CHECK_TRUE(off + (int)k <= PARAM_CAP);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, params[i], params_out + off,
                                       (size_t)k * sizeof(float)));
        off += (int)k;
    }
    *nfloats_out = off;

    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_optimizer_destroy(opt);
    gd_gpt_destroy(gpt);
    return 0;
}

static int test_f16_amp_train_metal(gd_context *ctx, gd_tensor *tokens, gd_tensor *pos,
                                    gd_tensor *targets)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor **params = NULL;
    gd_graph *g = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *scaled = NULL;
    gd_tensor *clip_norm = NULL;
    gd_tensor *grad = NULL;
    gd_adamw_config acfg = {0};
    gd_amp_scaler_config scfg = {0};
    int n_params = 0;
    int step = 0;
    float loss0 = 0.0f;
    float lossN = 0.0f;
    bool stepped = false;

    cfg.param_dtype = GD_DTYPE_F16;
    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_gpt_parameters(gpt, &params, &n_params));
    acfg.lr = 0.03f;
    acfg.beta1 = 0.9f;
    acfg.beta2 = 0.999f;
    acfg.eps = 1e-8f;
    acfg.weight_decay = 0.0f;
    acfg.master_param_policy = GD_MASTER_PARAM_AUTO;
    scfg.init_scale = 8.0f;
    scfg.growth_factor = 2.0f;
    scfg.backoff_factor = 0.5f;
    scfg.growth_interval = 4;
    scfg.min_scale = 1.0f;
    scfg.max_scale = 1024.0f;
    CHECK_OK(gd_adamw_create(ctx, params, n_params, &acfg, &opt));
    CHECK_OK(gd_amp_scaler_create(ctx, &scfg, &scaler));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward_loss(ctx, gpt, tokens, pos, targets, &loss));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_amp_scaler_scale_loss(ctx, scaler, loss, &scaled));
    CHECK_OK(gd_backward(ctx, scaled));
    CHECK_OK(gd_optimizer_step_amp_clip(ctx, opt, scaler, 1.0f, &clip_norm));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));

    for (step = 0; step < 16; ++step) {
        CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
        CHECK_OK(gd_graph_run(g));
        CHECK_OK(gd_synchronize(ctx, METAL));
        CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
        CHECK_TRUE(stepped);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &lossN, sizeof(lossN)));
        CHECK_TRUE(isfinite(lossN));
        if (step == 0) {
            loss0 = lossN;
            CHECK_OK(gd_tensor_grad(params[0], &grad));
            CHECK_TRUE(grad != NULL && gd_tensor_dtype(grad) == GD_DTYPE_F32);
        }
    }
    fprintf(stderr, "[gpt] f16 amp metal loss %.4f -> %.4f\n", (double)loss0, (double)lossN);
    CHECK_TRUE(lossN < loss0);

    gd_tensor_release(clip_norm);
    gd_tensor_release(scaled);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_gpt_destroy(gpt);
    return 0;
}

static int test_train_parity(gd_context *ctx, gd_tensor *tokens, gd_tensor *pos,
                             gd_tensor *targets)
{
    static float pc[PARAM_CAP];
    static float pm[PARAM_CAP];
    float loss_cpu = 0.0f, loss_mtl = 0.0f;
    int nc = 0, nm = 0, i = 0;
    const int steps = 5;

    if (train_collect(ctx, CPU, steps, tokens, pos, targets, pc, &nc, &loss_cpu) != 0) {
        return 1;
    }
    if (train_collect(ctx, METAL, steps, tokens, pos, targets, pm, &nm, &loss_mtl) != 0) {
        return 1;
    }
    CHECK_TRUE(nc == nm);
    /* Drift accumulates over steps (float GPU accum vs double CPU accum). */
    if (fabsf(loss_cpu - loss_mtl) > 1e-3f * (1.0f + fabsf(loss_cpu))) {
        fprintf(stderr, "gpt train loss cpu=%g metal=%g\n", (double)loss_cpu, (double)loss_mtl);
        return 1;
    }
    for (i = 0; i < nc; ++i) {
        float a = pc[i], b = pm[i];
        float diff = a < b ? b - a : a - b;
        if (diff > 1e-3f * (1.0f + (b < 0.0f ? -b : b))) {
            fprintf(stderr, "gpt param[%d] cpu=%g metal=%g\n", i, (double)a, (double)b);
            return 1;
        }
    }
    return 0;
}

/* Overfit a tiny sequence on CPU: loss must drop substantially. */
static int test_overfit(gd_context *ctx, gd_tensor *tokens, gd_tensor *pos,
                        gd_tensor *targets)
{
    gd_gpt *gpt = NULL;
    gd_gpt_config cfg = tiny_config();
    gd_optimizer *opt = NULL;
    gd_tensor **params = NULL;
    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_adamw_config acfg = {0};
    int n_params = 0;
    int step = 0;
    float loss0 = 0.0f, lossN = 0.0f;

    CHECK_OK(gd_gpt_create(ctx, &cfg, GPT_SEED, &gpt));
    CHECK_OK(gd_gpt_parameters(gpt, &params, &n_params));
    acfg.lr = 0.05f;
    acfg.beta1 = 0.9f;
    acfg.beta2 = 0.999f;
    acfg.eps = 1e-8f;
    CHECK_OK(gd_adamw_create(ctx, params, n_params, &acfg, &opt));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward(ctx, gpt, tokens, pos, &logits));
    CHECK_OK(gd_cross_entropy(ctx, logits, targets, 2, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));

    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &loss0, sizeof(loss0)));
    for (step = 0; step < 400; ++step) {
        CHECK_OK(gd_graph_run(g));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &lossN, sizeof(lossN)));
    fprintf(stderr, "[gpt] overfit loss %.4f -> %.4f\n", (double)loss0, (double)lossN);
    CHECK_TRUE(lossN < 0.05f);
    CHECK_TRUE(lossN < loss0);

    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_optimizer_destroy(opt);
    gd_gpt_destroy(gpt);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_tensor *tokens = NULL;
    gd_tensor *pos = NULL;
    gd_tensor *targets = NULL;
    int rc = 0;
    int metal = 0;
    int mps_enabled = env_flag_enabled("GD_METAL_MPS");

    if (gd_context_create(&ctx) != GD_OK) {
        fprintf(stderr, "context create failed: %s\n", gd_last_error());
        return 1;
    }
    metal = (gd_synchronize(ctx, METAL) == GD_OK);
    if (make_inputs(ctx, &tokens, &pos, &targets) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }

    rc |= test_overfit(ctx, tokens, pos, targets); /* CPU-only, always runs */
    {
        float loss_cpu = 0.0f;
        rc |= collect_f16_forward_loss(ctx, CPU, tokens, pos, targets, &loss_cpu);
        if (metal && mps_enabled && rc == 0) {
            float loss_mtl = 0.0f;
            rc |= collect_f16_forward_loss(ctx, METAL, tokens, pos, targets, &loss_mtl);
            CHECK_TRUE(fabsf(loss_cpu - loss_mtl) <= 5.0e-2f);
        }
    }
    if (metal) {
        rc |= test_forward_parity(ctx, tokens, pos);
        if (mps_enabled) {
            rc |= test_forward_loss_parity(ctx, tokens, pos, targets);
            rc |= test_f16_amp_train_metal(ctx, tokens, pos, targets);
        }
        rc |= test_train_parity(ctx, tokens, pos, targets);
    } else {
        printf("test_metal_gpt_train: metal unavailable, ran CPU overfit only\n");
    }

    gd_tensor_release(tokens);
    gd_tensor_release(pos);
    gd_tensor_release(targets);
    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal_gpt_train: ok\n");
    }
    return rc;
}
