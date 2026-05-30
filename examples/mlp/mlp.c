/*
 * MLP trained on the XOR task with gradients.c.
 *
 * Demonstrates the graph-only foundation end to end:
 *   - materialized parameters collected through a gd_module
 *   - a single compiled train-step graph (forward + backward + AdamW)
 *   - graph reuse across steps (no per-step allocation)
 *   - a separate forward-only eval graph for accuracy
 *
 * Model: x[4,2] -> linear(2,H) -> relu -> linear(H,2) -> cross_entropy
 */

#include "gradients/gradients.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H 8
#define N_SAMPLES 4
#define N_CLASSES 2
#define N_STEPS 3000

#define CHECK(expr)                                                              \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());         \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static float frand(void)
{
    return ((float)rand() / (float)RAND_MAX - 0.5F); /* [-0.5, 0.5] */
}

static gd_status make_param(gd_context *ctx, int ndim, const int64_t *sizes,
                            int init_random, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    int64_t numel = 1;
    int i = 0;
    float buf[H * N_CLASSES];
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, ndim, sizes, &desc);

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
    for (i = 0; i < (int)numel; ++i) {
        buf[i] = init_random ? frand() : 0.0F;
    }
    status = gd_tensor_copy_from_cpu(ctx, *out, buf, (size_t)numel * sizeof(float));
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_set_requires_grad(*out, true);
}

static gd_status build_forward(gd_context *ctx,
                               gd_tensor *x,
                               gd_tensor *w1,
                               gd_tensor *b1,
                               gd_tensor *w2,
                               gd_tensor *b2,
                               gd_tensor **logits_out)
{
    gd_tensor *h = NULL;
    gd_tensor *a = NULL;
    gd_status status = gd_linear(ctx, x, w1, b1, &h);

    if (status != GD_OK) {
        return status;
    }
    status = gd_relu(ctx, h, &a);
    gd_tensor_release(h);
    if (status != GD_OK) {
        return status;
    }
    status = gd_linear(ctx, a, w2, b2, logits_out);
    gd_tensor_release(a);
    return status;
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_module *model = NULL;
    gd_optimizer *opt = NULL;
    gd_tensor *x = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *w1 = NULL;
    gd_tensor *b1 = NULL;
    gd_tensor *w2 = NULL;
    gd_tensor *b2 = NULL;
    gd_tensor **params = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_graph *train = NULL;
    gd_graph *eval = NULL;
    gd_adamw_config cfg = {0};
    int n_params = 0;
    int step = 0;
    int correct = 0;
    int i = 0;

    int64_t xs[2] = {N_SAMPLES, 2};
    int64_t ts[1] = {N_SAMPLES};
    int64_t w1s[2] = {2, H};
    int64_t b1s[1] = {H};
    int64_t w2s[2] = {H, N_CLASSES};
    int64_t b2s[1] = {N_CLASSES};
    float xdata[N_SAMPLES * 2] = {0, 0, 0, 1, 1, 0, 1, 1};
    int32_t tdata[N_SAMPLES] = {0, 1, 1, 0};
    gd_tensor_desc tdesc;
    float logit_buf[N_SAMPLES * N_CLASSES];
    float loss_val = 0.0F;

    /* Device selection: GD_DEVICE=metal trains on the GPU when available,
     * otherwise falls back to CPU_REF. Defaults to CPU. */
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");

    srand(1234U);
    CHECK(gd_context_create(&ctx));
    if (dev_env != NULL && strcmp(dev_env, "metal") == 0) {
        gd_device metal = {GD_DEVICE_METAL, 0};
        if (gd_synchronize(ctx, metal) == GD_OK) {
            target = metal;
            printf("device: metal\n");
        } else {
            printf("device: cpu (metal unavailable)\n");
        }
    } else {
        printf("device: cpu\n");
    }
    CHECK(gd_context_set_default_device(ctx, cpu));

    /* Data. */
    CHECK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, xs, &tdesc));
    CHECK(gd_tensor_empty(ctx, &tdesc, &x));
    CHECK(gd_tensor_copy_from_cpu(ctx, x, xdata, sizeof(xdata)));
    CHECK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, ts, &tdesc));
    CHECK(gd_tensor_empty(ctx, &tdesc, &targets));
    CHECK(gd_tensor_copy_from_cpu(ctx, targets, tdata, sizeof(tdata)));

    /* Parameters. */
    CHECK(make_param(ctx, 2, w1s, 1, &w1));
    CHECK(make_param(ctx, 1, b1s, 0, &b1));
    CHECK(make_param(ctx, 2, w2s, 1, &w2));
    CHECK(make_param(ctx, 1, b2s, 0, &b2));

    CHECK(gd_module_create(ctx, "mlp", &model));
    CHECK(gd_module_param(model, "fc1.weight", w1));
    CHECK(gd_module_param(model, "fc1.bias", b1));
    CHECK(gd_module_param(model, "fc2.weight", w2));
    CHECK(gd_module_param(model, "fc2.bias", b2));
    CHECK(gd_module_parameters(model, &params, &n_params));

    cfg.lr = 0.05F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;
    CHECK(gd_adamw_create(ctx, params, n_params, &cfg, &opt));

    /* Build the train-step graph once. */
    CHECK(gd_graph_create(ctx, &train));
    CHECK(gd_graph_begin(ctx, train));
    CHECK(build_forward(ctx, x, w1, b1, w2, b2, &logits));
    CHECK(gd_cross_entropy(ctx, logits, targets, 1, &loss));
    CHECK(gd_backward(ctx, loss));
    CHECK(gd_optimizer_step(ctx, opt));
    CHECK(gd_graph_end(ctx));
    CHECK(gd_graph_compile(train, target));

    for (step = 1; step <= N_STEPS; ++step) {
        CHECK(gd_graph_run(train));
        if (step % 500 == 0 || step == 1) {
            CHECK(gd_tensor_copy_to_cpu(ctx, loss, &loss_val, sizeof(loss_val)));
            printf("step %4d  loss %.6f\n", step, (double)loss_val);
        }
    }

    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK(gd_graph_reset(train));
    CHECK(gd_graph_destroy(train));

    /* Eval: forward only with final parameters. */
    CHECK(gd_graph_create(ctx, &eval));
    CHECK(gd_graph_begin(ctx, eval));
    CHECK(build_forward(ctx, x, w1, b1, w2, b2, &logits));
    CHECK(gd_graph_end(ctx));
    CHECK(gd_graph_compile(eval, target));
    CHECK(gd_graph_run(eval));
    CHECK(gd_tensor_copy_to_cpu(ctx, logits, logit_buf, sizeof(logit_buf)));
    CHECK(gd_debug_print_tensor(ctx, logits, N_SAMPLES * N_CLASSES));

    for (i = 0; i < N_SAMPLES; ++i) {
        int pred = logit_buf[i * N_CLASSES + 1] > logit_buf[i * N_CLASSES + 0] ? 1 : 0;
        printf("xor(%d,%d) -> pred %d (target %d)\n",
               (int)xdata[i * 2 + 0], (int)xdata[i * 2 + 1], pred, tdata[i]);
        if (pred == tdata[i]) {
            correct += 1;
        }
    }
    printf("accuracy %d/%d\n", correct, N_SAMPLES);

    gd_tensor_release(logits);
    CHECK(gd_graph_reset(eval));
    CHECK(gd_graph_destroy(eval));

    gd_optimizer_destroy(opt);
    gd_module_destroy(model);
    gd_tensor_release(x);
    gd_tensor_release(targets);
    gd_tensor_release(w1);
    gd_tensor_release(b1);
    gd_tensor_release(w2);
    gd_tensor_release(b2);
    gd_context_destroy(ctx);

    return correct == N_SAMPLES ? 0 : 1;
}
