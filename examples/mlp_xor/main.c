#include <gradients/gradients.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct xor_mlp {
    gd_module mod;
    gd_linear_layer fc1;
    gd_linear_layer fc2;
} xor_mlp;

static void fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "mlp_xor failed at line %d: %s -> %s (%s)\n",
            line,
            expr,
            gd_status_string(st),
            ctx != NULL ? gd_context_error(ctx) : "no context");
    exit(1);
}

#define TRY(ctx, expr)                                                        \
    do {                                                                      \
        gd_status _st = (expr);                                               \
        if (_st != GD_OK) {                                                   \
            fail_status((ctx), _st, #expr, __LINE__);                         \
        }                                                                     \
    } while (0)

static gd_memory_config xor_memory_config(void)
{
    return (gd_memory_config){
        .params_bytes = 64U * 1024U,
        .state_bytes = 128U * 1024U,
        .scratch_slot_bytes = 256U * 1024U,
        .data_slot_bytes = 16U * 1024U,
        .scratch_slots = 3U,
        .data_slots = 3U,
        .default_alignment = 256U,
    };
}

static gd_adamw_config xor_adamw_config(void)
{
    gd_adamw_config cfg = gd_adamw_config_default();
    cfg.lr = 5.0e-2f;
    cfg.weight_decay = 0.0f;
    return cfg;
}

static gd_amp_config xor_amp_config(void)
{
    gd_amp_config cfg = gd_amp_config_default();
    cfg.init_scale = 64.0f;
    cfg.growth_interval = 32U;
    cfg.max_scale = 4096.0f;
    return cfg;
}

static void xor_mlp_init(gd_context *ctx, xor_mlp *model)
{
    enum { INIT_SEED = 42 };
    const gd_linear_layer_config fc1_cfg = gd_linear_layer_config_build(2, 4, GD_DTYPE_F16);
    const gd_linear_layer_config fc2_cfg = gd_linear_layer_config_build(4, 1, GD_DTYPE_F16);
    memset(model, 0, sizeof(*model));
    TRY(ctx, gd_module_init(ctx, &model->mod, "xor_mlp"));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc1", &model->fc1, &fc1_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc2", &model->fc2, &fc2_cfg));
    TRY(ctx, gd_module_init_params_uniform(ctx,
                                           &model->mod,
                                           "xor_mlp.*.weight",
                                           -1.0f,
                                           1.0f,
                                           INIT_SEED));
    TRY(ctx, gd_module_init_params_zero(ctx, &model->mod, "xor_mlp.*.bias"));
}

static void xor_mlp_deinit(xor_mlp *model)
{
    gd_linear_layer_deinit(&model->fc2);
    gd_linear_layer_deinit(&model->fc1);
    gd_module_deinit(&model->mod);
}

static gd_status xor_mlp_forward(gd_context *ctx,
                                 xor_mlp *model,
                                 const gd_tensor *x,
                                 gd_tensor *out)
{
    gd_tensor h;
    gd_tensor a;
    gd_status st;
    st = gd_linear_layer_forward(ctx, &model->fc1, x, &h);
    if (st != GD_OK) {
        return st;
    }
    st = gd_relu(ctx, &h, &a);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_layer_forward(ctx, &model->fc2, &a, out);
}

static float mean_squared_error(const float *pred, const float *target, size_t count)
{
    float loss = 0.0f;
    size_t i;
    for (i = 0U; i < count; ++i) {
        float d = pred[i] - target[i];
        loss += d * d;
    }
    return loss / (float)count;
}

static void print_param_set(const gd_param_set *params)
{
    uint32_t i;
    printf("parameters:\n");
    for (i = 0U; i < params->count; ++i) {
        printf("  %-22s lr_mult=%.2f weight_decay=%.4f trainable=%s\n",
               params->items[i].path,
               (double)params->items[i].lr_mult,
               (double)params->items[i].weight_decay,
               params->items[i].trainable ? "yes" : "no");
    }
}

int main(void)
{
    enum { BATCH = 4, IN = 2, OUT = 1, STEPS = 240 };
    static const float x_f32[BATCH * IN] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
    };
    static const float y_f32[BATCH * OUT] = {0.0f, 1.0f, 1.0f, 0.0f};
    const gd_memory_config mem = xor_memory_config();
    gd_context *ctx = NULL;
    gd_status st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("mlp_xor: skipped (no supported gradients.c backend)\n");
        return 0;
    }
    if (st != GD_OK) {
        fail_status(ctx, st, "gd_context_create", __LINE__);
    }

    xor_mlp model = {0};
    xor_mlp_init(ctx, &model);

    gd_dataset *dataset = NULL;
    gd_dataloader *loader = NULL;
    gd_batch_field_desc batch_fields[2];
    int n_batch_fields = 0;
    TRY(ctx, gd_dataset_open_gdds_split("data", "xor", &dataset));
    TRY(ctx, gd_gdds_init_batch_fields(dataset,
                                       BATCH,
                                       batch_fields,
                                       (int)GD_ARRAY_LEN(batch_fields),
                                       &n_batch_fields));
    {
        const gd_dataloader_config dl_cfg = {
            .batch_size = BATCH,
            .seed = 0U,
            .sampler = GD_SAMPLER_SEQUENTIAL,
            .expected_dataset_fingerprint = gd_dataset_fingerprint(dataset),
            .num_workers = 1,
            .prefetch_factor = 2,
        };
        TRY(ctx, gd_dataloader_create(ctx, dataset, &dl_cfg, batch_fields, n_batch_fields,
                                      gd_collate_gdds, NULL, &loader));
    }
    TRY(ctx, gd_dataloader_prefetch(loader));

    /* Showcase optimizer groups: split hidden-layer and head parameters by module path. */
    const gd_param_group groups[] = {
        {
            .name = "hidden",
            .match = "xor_mlp.fc1.*",
            .lr_mult = 1.0f,
            .weight_decay = 0.0f,
            .trainable = true,
        },
        {
            .name = "head",
            .match = "xor_mlp.fc2.*",
            .lr_mult = 1.0f,
            .weight_decay = 0.0f,
            .trainable = true,
        },
    };
    gd_param_set params = {0};
    TRY(ctx, gd_module_collect_params(ctx, &model.mod, groups, GD_ARRAY_LEN(groups), &params));
    print_param_set(&params);

    const gd_adamw_config adam = xor_adamw_config();
    const gd_amp_config amp = xor_amp_config();
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    TRY(ctx, gd_adamw_create(ctx, &params, &adam, &optimizer));
    TRY(ctx, gd_amp_scaler_create(&amp, &scaler));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, true);

    for (int step = 0; step < STEPS; ++step) {
        gd_batch *batch = NULL;
        gd_tensor *x;
        gd_tensor *target;
        gd_tensor pred_tensor;
        gd_tensor loss;
        const int report = step == 0 || (step + 1) % 40 == 0;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, batch));
        x = gd_batch_tensor(batch, "x");
        target = gd_batch_tensor(batch, "target");
        TRY(ctx, xor_mlp_forward(ctx, &model, x, &pred_tensor));
        TRY(ctx, gd_huber(ctx, &pred_tensor, target, &loss));
        TRY(ctx, gd_backward_scaled(ctx, &loss, NULL, gd_amp_scaler_scale(scaler)));
        TRY(ctx, gd_optimizer_step_amp(ctx, optimizer, scaler));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_dataloader_release(loader, batch));
        TRY(ctx, gd_dataloader_prefetch(loader));

        if (report) {
            float loss_value = 0.0f;
            TRY(ctx, gd_tensor_item(ctx, &loss, &loss_value));
            printf("step=%3d loss=%.6f scale=%.1f overflow=%s\n",
                   step + 1,
                   (double)loss_value,
                   (double)gd_amp_scaler_scale(scaler),
                   gd_amp_scaler_last_found_inf(scaler) ? "yes" : "no");
        }
    }

    gd_module_set_training(&model.mod, false);
    float pred[BATCH * OUT];
    {
        gd_batch *batch = NULL;
        gd_tensor *x;
        gd_tensor pred_tensor;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
        x = gd_batch_tensor(batch, "x");
        TRY(ctx, xor_mlp_forward(ctx, &model, x, &pred_tensor));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_tensor_read_f32(ctx, &pred_tensor, pred, GD_ARRAY_LEN(pred)));
        TRY(ctx, gd_dataloader_release(loader, batch));
    }

    const float final_loss = mean_squared_error(pred, y_f32, BATCH * OUT);
    int correct = 0;
    printf("\nfinal predictions:\n");
    for (int sample = 0; sample < BATCH; ++sample) {
        int label = pred[sample] >= 0.5f ? 1 : 0;
        int target_label = y_f32[sample] >= 0.5f ? 1 : 0;
        if (label == target_label) {
            correct += 1;
        }
        printf("  [%.0f %.0f] -> %.4f class=%d target=%d\n",
               (double)x_f32[sample * IN + 0],
               (double)x_f32[sample * IN + 1],
               (double)pred[sample],
               label,
               target_label);
    }
    printf("final_loss=%.6f accuracy=%d/%d optimizer_steps=%llu\n",
           (double)final_loss,
           correct,
           BATCH,
           (unsigned long long)gd_optimizer_step_count(optimizer));

    if (correct != BATCH || final_loss > 0.05f) {
        fprintf(stderr, "mlp_xor: training did not solve XOR\n");
        gd_dataloader_destroy(loader);
        gd_dataset_destroy(dataset);
        gd_amp_scaler_destroy(scaler);
        gd_optimizer_destroy(optimizer);
        gd_param_set_free(&params);
        xor_mlp_deinit(&model);
        gd_context_destroy(ctx);
        return 1;
    }

    gd_dataloader_destroy(loader);
    gd_dataset_destroy(dataset);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    xor_mlp_deinit(&model);
    gd_context_destroy(ctx);
    printf("mlp_xor: ok\n");
    return 0;
}
