#include <gradients/gradients.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define MNIST_INPUT_DIM 784
#define MNIST_HIDDEN_DIM 128
#define MNIST_CLASSES 10
#define MNIST_TRAIN_BATCH 128
#define MNIST_EVAL_BATCH 100

#define MNIST_DEFAULT_EPOCHS 2
#define MNIST_DEFAULT_REPORT_EVERY 100
#define MNIST_DEFAULT_MIN_ACCURACY 0.85f

typedef struct mnist_mlp {
    gd_module mod;
    gd_linear_layer fc1;
    gd_linear_layer fc2;
} mnist_mlp;

static void fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "mlp_mnist failed at line %d: %s -> %s (%s)\n",
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

static int env_int(const char *name, int fallback, int min_value, int max_value)
{
    const char *text = getenv(name);
    char *end = NULL;
    long parsed;
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < (long)min_value ||
        parsed > (long)max_value) {
        fprintf(stderr,
                "mlp_mnist: ignoring invalid %s=%s; using %d\n",
                name,
                text,
                fallback);
        return fallback;
    }
    return (int)parsed;
}

static float env_float(const char *name, float fallback, float min_value, float max_value)
{
    const char *text = getenv(name);
    char *end = NULL;
    float parsed;
    if (text == NULL || text[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        fprintf(stderr,
                "mlp_mnist: ignoring invalid %s=%s; using %.3f\n",
                name,
                text,
                (double)fallback);
        return fallback;
    }
    return parsed;
}

static double wall_seconds(void)
{
    struct timeval tv;
    if (gettimeofday(&tv, NULL) != 0) {
        return 0.0;
    }
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static gd_memory_config mnist_memory_config(void)
{
    return (gd_memory_config){
        .params_bytes = 4U * 1024U * 1024U,
        .state_bytes = 16U * 1024U * 1024U,
        .scratch_slot_bytes = 32U * 1024U * 1024U,
        .data_slot_bytes = 1U * 1024U * 1024U,
        .scratch_slots = 3U,
        .data_slots = 4U,
        .default_alignment = 256U,
    };
}

static gd_adamw_config mnist_adamw_config(void)
{
    gd_adamw_config cfg = gd_adamw_config_default();
    cfg.lr = 1.0e-3f;
    cfg.weight_decay = 1.0e-4f;
    return cfg;
}

static gd_amp_config mnist_amp_config(void)
{
    gd_amp_config cfg = gd_amp_config_default();
    cfg.init_scale = 128.0f;
    cfg.growth_interval = 64U;
    cfg.max_scale = 4096.0f;
    return cfg;
}

static gd_linear_layer_config mnist_linear_config(int64_t in_features,
                                                  int64_t out_features,
                                                  uint64_t seed,
                                                  float limit)
{
    gd_linear_layer_config cfg = gd_linear_layer_config_make(in_features,
                                                             out_features,
                                                             GD_DTYPE_F16,
                                                             seed);
    cfg.weight_low = -limit;
    cfg.weight_high = limit;
    return cfg;
}

static void mnist_mlp_init(gd_context *ctx, mnist_mlp *model)
{
    const gd_linear_layer_config fc1_cfg = mnist_linear_config(MNIST_INPUT_DIM,
                                                               MNIST_HIDDEN_DIM,
                                                               101U,
                                                               0.088f);
    const gd_linear_layer_config fc2_cfg = mnist_linear_config(MNIST_HIDDEN_DIM,
                                                               MNIST_CLASSES,
                                                               202U,
                                                               0.217f);
    memset(model, 0, sizeof(*model));
    TRY(ctx, gd_module_init(ctx, &model->mod, "mnist_mlp"));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc1", &model->fc1, &fc1_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &model->mod, "fc2", &model->fc2, &fc2_cfg));
}

static void mnist_mlp_deinit(mnist_mlp *model)
{
    gd_linear_layer_deinit(&model->fc2);
    gd_linear_layer_deinit(&model->fc1);
    gd_module_deinit(&model->mod);
}

static gd_status mnist_mlp_forward(gd_context *ctx,
                                   mnist_mlp *model,
                                   const gd_tensor *x,
                                   gd_tensor *out)
{
    gd_tensor hidden;
    gd_tensor activated;
    gd_status st;
    st = gd_linear_layer_forward(ctx, &model->fc1, x, &hidden);
    if (st != GD_OK) {
        return st;
    }
    st = gd_relu(ctx, &hidden, &activated);
    if (st != GD_OK) {
        return st;
    }
    return gd_linear_layer_forward(ctx, &model->fc2, &activated, out);
}

static void print_param_set(const gd_param_set *params)
{
    uint32_t i;
    printf("parameters:\n");
    for (i = 0U; i < params->count; ++i) {
        printf("  %-24s lr_mult=%.2f weight_decay=%.4f trainable=%s\n",
               params->items[i].path,
               (double)params->items[i].lr_mult,
               (double)params->items[i].weight_decay,
               params->items[i].trainable ? "yes" : "no");
    }
}

static gd_status create_gdds_loader(gd_context *ctx,
                                    gd_dataset *dataset,
                                    gd_sampler *sampler,
                                    int batch_size,
                                    gd_dataloader **out)
{
    gd_dataloader_config cfg;
    gd_status st;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    cfg = gd_dataloader_config_build(dataset, batch_size);
    cfg.num_workers = 1;
    cfg.prefetch_factor = 2;
    st = gd_dataloader_create(ctx, dataset, sampler, &cfg, out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_dataloader_prefetch(*out);
    if (st != GD_OK) {
        gd_dataloader_destroy(*out);
        *out = NULL;
        return st;
    }
    return GD_OK;
}

static gd_tensor *required_batch_tensor(gd_context *ctx, gd_batch *batch, const char *name, int line)
{
    gd_tensor *tensor = gd_batch_tensor(batch, name);
    if (tensor == NULL) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, name, line);
    }
    return tensor;
}

static int argmax10(const float *values)
{
    int best = 0;
    int i;
    for (i = 1; i < MNIST_CLASSES; ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

static void evaluate_accuracy(gd_context *ctx,
                              mnist_mlp *model,
                              gd_dataloader *loader,
                              uint64_t n_samples,
                              int *correct_out,
                              uint64_t *total_out)
{
    float logits[MNIST_EVAL_BATCH * MNIST_CLASSES];
    int32_t labels[MNIST_EVAL_BATCH];
    uint64_t seen = 0U;
    int correct = 0;
    while (seen < n_samples) {
        gd_batch *batch = NULL;
        gd_tensor *image;
        gd_tensor *target;
        gd_tensor pred;
        uint64_t remaining = n_samples - seen;
        int rows = remaining < (uint64_t)MNIST_EVAL_BATCH ? (int)remaining : MNIST_EVAL_BATCH;
        int row;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_EVAL, batch));
        image = required_batch_tensor(ctx, batch, "image", __LINE__);
        target = required_batch_tensor(ctx, batch, "target", __LINE__);
        TRY(ctx, mnist_mlp_forward(ctx, model, image, &pred));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_tensor_read_f32(ctx, &pred, logits, GD_ARRAY_LEN(logits)));
        TRY(ctx, gd_tensor_read(ctx, target, labels, sizeof(labels)));
        TRY(ctx, gd_dataloader_release(loader, batch));
        TRY(ctx, gd_dataloader_prefetch(loader));
        for (row = 0; row < rows; ++row) {
            int predicted = argmax10(&logits[row * MNIST_CLASSES]);
            if (predicted == labels[row]) {
                correct += 1;
            }
        }
        seen += (uint64_t)rows;
    }
    *correct_out = correct;
    *total_out = n_samples;
}

int main(void)
{
    const int train_epochs = env_int("GD_MNIST_EPOCHS", MNIST_DEFAULT_EPOCHS, 1, 1000000);
    const int report_every = env_int("GD_MNIST_REPORT_EVERY", MNIST_DEFAULT_REPORT_EVERY, 0, 1000000);
    const float min_accuracy = env_float("GD_MNIST_MIN_ACCURACY",
                                         MNIST_DEFAULT_MIN_ACCURACY,
                                         0.0f,
                                         1.0f);
    const char *data_dir_env = getenv("GD_MNIST_DATA_DIR");
    const char *data_dir = (data_dir_env != NULL && data_dir_env[0] != '\0') ? data_dir_env : "data";
    const gd_memory_config mem = mnist_memory_config();
    gd_context *ctx = NULL;
    gd_status st = gd_context_create(&mem, &ctx);
    gd_dataset *train_dataset = NULL;
    gd_dataset *test_dataset = NULL;
    gd_sampler *train_sampler = NULL;
    gd_dataloader *train_loader = NULL;
    gd_dataloader *test_loader = NULL;
    mnist_mlp model = {0};
    gd_param_set params = {0};
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    int correct = 0;
    uint64_t total = 0U;
    uint64_t steps_per_epoch = 0U;
    uint64_t train_steps = 0U;
    float accuracy;

    if (st == GD_ERR_UNSUPPORTED) {
        printf("mlp_mnist: skipped (no supported gradients.c backend)\n");
        return 0;
    }
    if (st != GD_OK) {
        fail_status(ctx, st, "gd_context_create", __LINE__);
    }

    TRY(ctx, gd_dataset_open_gdds_split(data_dir, "train", &train_dataset));
    TRY(ctx, gd_dataset_open_gdds_split(data_dir, "test", &test_dataset));

    mnist_mlp_init(ctx, &model);
    {
        const gd_param_group groups[] = {
            {
                .name = "encoder",
                .match = "mnist_mlp.fc1.*",
                .lr_mult = 1.0f,
                .weight_decay = 1.0e-4f,
                .trainable = true,
            },
            {
                .name = "classifier",
                .match = "mnist_mlp.fc2.*",
                .lr_mult = 1.0f,
                .weight_decay = 1.0e-4f,
                .trainable = true,
            },
        };
        TRY(ctx, gd_module_collect_params(ctx, &model.mod, groups, GD_ARRAY_LEN(groups), &params));
    }
    print_param_set(&params);

    {
        const gd_adamw_config adam = mnist_adamw_config();
        const gd_amp_config amp = mnist_amp_config();
        TRY(ctx, gd_adamw_create(ctx, &params, &adam, &optimizer));
        TRY(ctx, gd_amp_scaler_create(&amp, &scaler));
    }
    TRY(ctx, gd_context_seal_params(ctx));
    TRY(ctx, gd_sampler_create_random(train_dataset, 1234U, &train_sampler));
    TRY(ctx, create_gdds_loader(ctx,
                                train_dataset,
                                train_sampler,
                                MNIST_TRAIN_BATCH,
                                &train_loader));
    steps_per_epoch = gd_dataloader_steps_per_epoch(train_loader);
    if (steps_per_epoch == 0U || steps_per_epoch > UINT64_MAX / (uint64_t)train_epochs) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "GD_MNIST_EPOCHS", __LINE__);
    }
    train_steps = (uint64_t)train_epochs * steps_per_epoch;
    printf("dataset: dir=%s train=%llu test=%llu batch=%d epochs=%d steps_per_epoch=%llu samples_per_epoch=%llu total_steps=%llu\n",
           data_dir,
           (unsigned long long)gd_dataset_num_samples(train_dataset),
           (unsigned long long)gd_dataset_num_samples(test_dataset),
           MNIST_TRAIN_BATCH,
           train_epochs,
           (unsigned long long)steps_per_epoch,
           (unsigned long long)gd_dataloader_samples_per_epoch(train_loader),
           (unsigned long long)train_steps);

    gd_module_set_training(&model.mod, true);
    double last_report_time = wall_seconds();
    uint64_t last_report_step = 0U;
    for (uint64_t step = 0U; step < train_steps; ++step) {
        gd_batch *batch = NULL;
        gd_tensor *image;
        gd_tensor *target;
        gd_tensor logits;
        gd_tensor loss;
        const uint64_t current_step = step + 1U;
        const uint64_t epoch = step / steps_per_epoch + 1U;
        const uint64_t epoch_step = step % steps_per_epoch + 1U;
        const int report = report_every > 0 &&
                           (step == 0U || current_step % (uint64_t)report_every == 0U ||
                            current_step == train_steps);
        TRY(ctx, gd_dataloader_next(train_loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, batch));
        image = required_batch_tensor(ctx, batch, "image", __LINE__);
        target = required_batch_tensor(ctx, batch, "target", __LINE__);
        TRY(ctx, mnist_mlp_forward(ctx, &model, image, &logits));
        TRY(ctx, gd_cross_entropy(ctx, &logits, target, &loss));
        TRY(ctx, gd_backward_scaled(ctx, &loss, NULL, gd_amp_scaler_scale(scaler)));
        TRY(ctx, gd_optimizer_step_amp(ctx, optimizer, scaler));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_dataloader_release(train_loader, batch));
        TRY(ctx, gd_dataloader_prefetch(train_loader));

        if (report) {
            const double now = wall_seconds();
            const double elapsed = now - last_report_time;
            const uint64_t batches = current_step - last_report_step;
            const double batches_per_sec = elapsed > 0.0 ? (double)batches / elapsed : 0.0;
            float loss_value = 0.0f;
            TRY(ctx, gd_tensor_item(ctx, &loss, &loss_value));
            printf("epoch=%llu/%d batch=%llu/%llu step=%llu loss=%.6f batch/s=%.2f\n",
                   (unsigned long long)epoch,
                   train_epochs,
                   (unsigned long long)epoch_step,
                   (unsigned long long)steps_per_epoch,
                   (unsigned long long)current_step,
                   (double)loss_value,
                   batches_per_sec);
            last_report_time = now;
            last_report_step = current_step;
        }
    }

    gd_dataloader_destroy(train_loader);
    train_loader = NULL;

    gd_module_set_training(&model.mod, false);
    TRY(ctx, create_gdds_loader(ctx,
                                test_dataset,
                                NULL,
                                MNIST_EVAL_BATCH,
                                &test_loader));
    evaluate_accuracy(ctx, &model, test_loader, gd_dataloader_samples_per_epoch(test_loader), &correct, &total);
    accuracy = total > 0U ? (float)correct / (float)total : 0.0f;
    printf("test_accuracy=%.4f correct=%d/%llu optimizer_steps=%llu\n",
           (double)accuracy,
           correct,
           (unsigned long long)total,
           (unsigned long long)gd_optimizer_step_count(optimizer));

    if (accuracy < min_accuracy) {
        fprintf(stderr,
                "mlp_mnist: accuracy %.4f below threshold %.4f\n",
                (double)accuracy,
                (double)min_accuracy);
        gd_dataloader_destroy(test_loader);
        gd_sampler_destroy(train_sampler);
        gd_dataset_destroy(test_dataset);
        gd_dataset_destroy(train_dataset);
        gd_amp_scaler_destroy(scaler);
        gd_optimizer_destroy(optimizer);
        gd_param_set_free(&params);
        mnist_mlp_deinit(&model);
        gd_context_destroy(ctx);
        return 1;
    }

    gd_dataloader_destroy(test_loader);
    gd_sampler_destroy(train_sampler);
    gd_dataset_destroy(test_dataset);
    gd_dataset_destroy(train_dataset);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    mnist_mlp_deinit(&model);
    gd_context_destroy(ctx);
    printf("mlp_mnist: ok\n");
    return 0;
}
