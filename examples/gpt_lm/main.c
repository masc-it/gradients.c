#include "gpt_lm_shared.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

static int parse_i64_arg(const char *text, int64_t min_value, int64_t max_value, int64_t *out)
{
    char *end = NULL;
    long long parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) {
        return 0;
    }
    *out = (int64_t)parsed;
    return 1;
}

static int parse_u64_arg(const char *text, uint64_t max_value, uint64_t *out)
{
    char *end = NULL;
    unsigned long long parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || (uint64_t)parsed > max_value) {
        return 0;
    }
    *out = (uint64_t)parsed;
    return 1;
}

static int parse_float_arg(const char *text, float min_value, float max_value, float *out)
{
    char *end = NULL;
    float parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) || parsed < min_value ||
        parsed > max_value) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static const char *arg_value(int argc, char **argv, int *index, const char *name)
{
    const size_t name_len = strlen(name);
    const char *arg = argv[*index];
    if (strncmp(arg, name, name_len) == 0 && arg[name_len] == '=') {
        return arg + name_len + 1U;
    }
    if (strcmp(arg, name) == 0) {
        if (*index + 1 >= argc) {
            fprintf(stderr, "gpt_lm: missing value for %s\n", name);
            exit(2);
        }
        *index += 1;
        return argv[*index];
    }
    return NULL;
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --data-dir PATH             GDDS directory (default: examples/gpt_lm/data)\n");
    printf("  --tokenizer-path PATH       tokenizer JSON for generation (default: DATA/tokenizer-v2048.json)\n");
    printf("  --generate TEXT             run KV-cache generation from TEXT; skips training unless --epochs is set\n");
    printf("  --generate-every-n-steps N  during training, generate batched a/e/i/o/u samples every N steps (default: 0)\n");
    printf("  --checkpoint-path PATH      best-val checkpoint path (default: checkpoints/gpt_lm_best.gdckpt)\n");
    printf("  --load-checkpoint PATH      load model weights before training/generation\n");
    printf("  --val-split NAME            validation split name for best checkpointing (default: val)\n");
    printf("  --no-save-best              disable best validation checkpoint writes\n");
    printf("  --max-new-tokens N          generated tokens for --generate / periodic generation (default: 64)\n");
    printf("  --temperature T             sampling temperature; 0 means greedy (default: 0)\n");
    printf("  --epochs N                  training epochs (default: %d; 0 allowed with --generate)\n", GPT_DEFAULT_EPOCHS);
    printf("  --batch-size N              batch size in 512-token sequences (default: %d)\n", GPT_DEFAULT_BATCH_SIZE);
    printf("  --layers N                  decoder blocks (default: %d)\n", GPT_DEFAULT_LAYERS);
    printf("  --dropout P                 dropout probability (default: %.2f)\n", (double)GPT_DEFAULT_DROPOUT_P);
    printf("  --lr-max LR                 cosine schedule max LR (default: %.6g)\n", (double)GPT_DEFAULT_LR_MAX);
    printf("  --lr-min LR                 cosine schedule min LR (default: %.6g)\n", (double)GPT_DEFAULT_LR_MIN);
    printf("  --warmup-steps N            LR warmup steps; -1 means 10%% of total steps (default: -1)\n");
    printf("  --weight-decay WD           AdamW weight decay for non-norm weights (default: %.3g)\n",
           (double)GPT_DEFAULT_WEIGHT_DECAY);
    printf("  --grad-clip-norm N          global grad norm clip; 0 disables (default: %.3g)\n",
           (double)GPT_DEFAULT_GRAD_CLIP_NORM);
    printf("  --report-every N            report every N steps; 0 disables periodic reports (default: %d)\n",
           GPT_DEFAULT_REPORT_EVERY);
    printf("  --overfit-num-samples N     repeatedly train on the first N samples; 0 disables (default: 0)\n");
    printf("  --seed N                    base seed (default: %llu)\n", (unsigned long long)GPT_DEFAULT_SEED);
    printf("  --help                      show this help\n");
}

static gpt_config gpt_config_default(void)
{
    gpt_config config;
    memset(&config, 0, sizeof(config));
    config.data_dir = "examples/gpt_lm/data";
    config.tokenizer_path = NULL;
    config.generate_prompt = NULL;
    config.checkpoint_path = "checkpoints/gpt_lm_best.gdckpt";
    config.load_checkpoint_path = NULL;
    config.val_split = "val";
    config.epochs = GPT_DEFAULT_EPOCHS;
    config.batch_size = GPT_DEFAULT_BATCH_SIZE;
    config.n_layers = GPT_DEFAULT_LAYERS;
    config.report_every = GPT_DEFAULT_REPORT_EVERY;
    config.lr_warmup_steps = -1;
    config.max_new_tokens = 64;
    config.generate_every_n_steps = 0;
    config.epochs_set = false;
    config.save_best = true;
    config.overfit_num_samples = 0U;
    config.seed = GPT_DEFAULT_SEED;
    config.dropout_p = GPT_DEFAULT_DROPOUT_P;
    config.lr_max = GPT_DEFAULT_LR_MAX;
    config.lr_min = GPT_DEFAULT_LR_MIN;
    config.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
    config.grad_clip_norm = GPT_DEFAULT_GRAD_CLIP_NORM;
    config.temperature = 0.0f;
    return config;
}

static gpt_config parse_args(int argc, char **argv)
{
    gpt_config config = gpt_config_default();
    int i;
    for (i = 1; i < argc; ++i) {
        const char *value;
        int64_t parsed_i64 = 0;
        uint64_t parsed_u64 = 0U;
        float parsed_f32 = 0.0f;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        value = arg_value(argc, argv, &i, "--data-dir");
        if (value != NULL) {
            config.data_dir = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--tokenizer-path");
        if (value != NULL) {
            config.tokenizer_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--generate");
        if (value != NULL) {
            config.generate_prompt = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--checkpoint-path");
        if (value != NULL) {
            config.checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--load-checkpoint");
        if (value != NULL) {
            config.load_checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--val-split");
        if (value != NULL) {
            config.val_split = value;
            continue;
        }
        if (strcmp(argv[i], "--no-save-best") == 0) {
            config.save_best = false;
            continue;
        }
        value = arg_value(argc, argv, &i, "--generate-every-n-steps");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --generate-every-n-steps %s\n", value);
                exit(2);
            }
            config.generate_every_n_steps = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--max-new-tokens");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, GPT_CONTEXT_LENGTH - 1, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --max-new-tokens %s\n", value);
                exit(2);
            }
            config.max_new_tokens = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--temperature");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --temperature %s\n", value);
                exit(2);
            }
            config.temperature = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--epochs");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --epochs %s\n", value);
                exit(2);
            }
            config.epochs = (int)parsed_i64;
            config.epochs_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--batch-size");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 1024, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --batch-size %s\n", value);
                exit(2);
            }
            config.batch_size = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--layers");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 96, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --layers %s\n", value);
                exit(2);
            }
            config.n_layers = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--dropout");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 0.95f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --dropout %s\n", value);
                exit(2);
            }
            config.dropout_p = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--lr-max");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --lr-max %s\n", value);
                exit(2);
            }
            config.lr_max = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--lr-min");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --lr-min %s\n", value);
                exit(2);
            }
            config.lr_min = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--warmup-steps");
        if (value != NULL) {
            if (!parse_i64_arg(value, -1, 1000000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --warmup-steps %s\n", value);
                exit(2);
            }
            config.lr_warmup_steps = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--weight-decay");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --weight-decay %s\n", value);
                exit(2);
            }
            config.weight_decay = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--grad-clip-norm");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --grad-clip-norm %s\n", value);
                exit(2);
            }
            config.grad_clip_norm = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--report-every");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --report-every %s\n", value);
                exit(2);
            }
            config.report_every = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--overfit-num-samples");
        if (value != NULL) {
            if (!parse_u64_arg(value, UINT64_MAX, &parsed_u64)) {
                fprintf(stderr, "gpt_lm: invalid --overfit-num-samples %s\n", value);
                exit(2);
            }
            config.overfit_num_samples = parsed_u64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--seed");
        if (value != NULL) {
            if (!parse_u64_arg(value, UINT64_MAX, &parsed_u64)) {
                fprintf(stderr, "gpt_lm: invalid --seed %s\n", value);
                exit(2);
            }
            config.seed = parsed_u64;
            continue;
        }
        fprintf(stderr, "gpt_lm: unknown argument %s\n", argv[i]);
        print_usage(argv[0]);
        exit(2);
    }
    if (config.generate_prompt != NULL && !config.epochs_set) {
        config.epochs = 0;
    }
    if (config.epochs == 0 && config.generate_prompt == NULL) {
        fprintf(stderr, "gpt_lm: --epochs 0 requires --generate\n");
        exit(2);
    }
    if (config.lr_min > config.lr_max) {
        fprintf(stderr, "gpt_lm: --lr-min must be <= --lr-max\n");
        exit(2);
    }
    return config;
}


static gd_status param_set_stats(const gd_param_set *params,
                                  uint64_t *total_out,
                                  uint64_t *trainable_out,
                                  uint64_t *bytes_out)
{
    uint64_t total = 0U;
    uint64_t trainable = 0U;
    uint64_t bytes = 0U;
    uint32_t i;
    if (params == NULL || total_out == NULL || trainable_out == NULL || bytes_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < params->count; ++i) {
        const gd_param_ref *ref = &params->items[i];
        int64_t numel_i64 = 0;
        uint64_t numel;
        uint64_t nbytes;
        gd_status st;
        if (ref->tensor == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        st = gd_tensor_numel(ref->tensor, &numel_i64);
        if (st != GD_OK) {
            return st;
        }
        if (numel_i64 < 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        numel = (uint64_t)numel_i64;
        if (numel != 0U && gd_dtype_size(ref->tensor->dtype) > SIZE_MAX / (size_t)numel) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        nbytes = (uint64_t)((size_t)numel * gd_dtype_size(ref->tensor->dtype));
        if (numel > UINT64_MAX - total || nbytes > UINT64_MAX - bytes) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        total += numel;
        bytes += nbytes;
        if (ref->trainable) {
            if (numel > UINT64_MAX - trainable) {
                return GD_ERR_OUT_OF_MEMORY;
            }
            trainable += numel;
        }
    }
    *total_out = total;
    *trainable_out = trainable;
    *bytes_out = bytes;
    return GD_OK;
}

static void print_param_set(const gd_param_set *params)
{
    uint32_t i;
    printf("parameter_tensors: count=%u\n", params != NULL ? params->count : 0U);
    if (params == NULL) {
        return;
    }
    for (i = 0U; i < params->count; ++i) {
        printf("  %-48s lr_mult=%.2f weight_decay=%.4f trainable=%s\n",
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
    cfg = gd_dataloader_config_default(batch_size);
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
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, name, line);
    }
    return tensor;
}

static gd_adamw_config gpt_adamw_config(float lr, float weight_decay)
{
    gd_adamw_config config = gd_adamw_config_default();
    config.lr = lr;
    config.weight_decay = weight_decay;
    config.beta1 = 0.9f;
    config.beta2 = 0.95f;
    config.eps = 1.0e-8f;
    config.bias_correction = true;
    return config;
}

static gd_amp_config gpt_amp_config(void)
{
    gd_amp_config config = gd_amp_config_default();
    config.defer_found_inf = true;
    config.init_scale = 8192.0f;
    config.growth_interval = 1000U;
    return config;
}

static size_t effective_steps_per_epoch(uint64_t dataset_samples,
                                        const gpt_config *config,
                                        size_t *samples_per_epoch_out)
{
    const uint64_t batch = (uint64_t)config->batch_size;
    uint64_t full_batches = dataset_samples / batch;
    uint64_t requested_samples;
    uint64_t requested_batches;
    if (samples_per_epoch_out != NULL) {
        *samples_per_epoch_out = 0U;
    }
    if (full_batches == 0U) {
        return 0U;
    }
    if (config->overfit_num_samples == 0U) {
        if (samples_per_epoch_out != NULL) {
            *samples_per_epoch_out = (size_t)(full_batches * batch);
        }
        return (size_t)full_batches;
    }
    requested_samples = config->overfit_num_samples < dataset_samples ?
                            config->overfit_num_samples :
                            dataset_samples;
    requested_batches = requested_samples / batch;
    if (requested_batches == 0U) {
        requested_batches = 1U;
    }
    if (requested_batches > full_batches) {
        requested_batches = full_batches;
    }
    if (samples_per_epoch_out != NULL) {
        *samples_per_epoch_out = (size_t)(requested_batches * batch);
    }
    return (size_t)requested_batches;
}

static void train_one_batch(gd_context *ctx,
                            gpt_lm *model,
                            gd_dataloader *loader,
                            gd_optimizer *optimizer,
                            gd_amp_scaler *scaler,
                            const gd_lr_scheduler_config *lr_config,
                            const gpt_config *config,
                            size_t global_step,
                            size_t total_steps,
                            size_t epoch,
                            size_t epoch_step,
                            size_t steps_per_epoch,
                            double *last_report_time,
                            size_t *last_report_step)
{
    gd_batch *batch = NULL;
    gd_tensor *input_ids;
    gd_tensor *target_ids;
    gd_tensor *positions;
    gd_tensor *cu_seqlens;
    gd_tensor loss;
    float lr = 0.0f;
    const size_t current_step = global_step + 1U;
    const int report = config->report_every > 0 &&
                       (global_step == 0U || current_step % (size_t)config->report_every == 0U ||
                        current_step == total_steps);

    TRY(ctx, gd_lr_scheduler_value(lr_config, (uint64_t)global_step, &lr));
    TRY(ctx, gd_dataloader_next(loader, &batch));
    TRY(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, batch));
    input_ids = required_batch_tensor(ctx, batch, "input_ids", __LINE__);
    target_ids = required_batch_tensor(ctx, batch, "target_ids", __LINE__);
    positions = required_batch_tensor(ctx, batch, "positions", __LINE__);
    cu_seqlens = required_batch_tensor(ctx, batch, "cu_seqlens", __LINE__);
    TRY(ctx, gpt_lm_forward(ctx,
                            model,
                            input_ids,
                            target_ids,
                            positions,
                            cu_seqlens,
                            (uint64_t)current_step,
                            &loss));
    TRY(ctx, gd_backward_scaled(ctx, &loss, NULL, gd_amp_scaler_scale(scaler)));
    if (config->grad_clip_norm > 0.0f) {
        TRY(ctx, gd_optimizer_step_amp_clip_lr(ctx, optimizer, scaler, lr, config->grad_clip_norm));
    } else {
        TRY(ctx, gd_optimizer_step_amp_lr(ctx, optimizer, scaler, lr));
    }
    TRY(ctx, gd_end_step(ctx));
    TRY(ctx, gd_dataloader_release(loader, batch));
    TRY(ctx, gd_dataloader_prefetch(loader));

    if (report) {
        const double now = gpt_wall_seconds();
        const double elapsed = now - *last_report_time;
        const size_t batches = current_step - *last_report_step;
        const size_t tokens = batches * (size_t)config->batch_size * (size_t)GPT_CONTEXT_LENGTH;
        const double batches_per_sec = elapsed > 0.0 ? (double)batches / elapsed : 0.0;
        const double tokens_per_sec = elapsed > 0.0 ? (double)tokens / elapsed : 0.0;
        float loss_value = 0.0f;
        TRY(ctx, gd_tensor_item(ctx, &loss, &loss_value));
        printf("epoch=%zu/%d batch=%zu/%zu step=%zu/%zu loss=%.6f lr=%.6g batch/s=%.2f tok/s=%.0f amp_scale=%.1f%s\n",
               epoch,
               config->epochs,
               epoch_step,
               steps_per_epoch,
               current_step,
               total_steps,
               (double)loss_value,
               (double)lr,
               batches_per_sec,
               tokens_per_sec,
               (double)gd_amp_scaler_scale(scaler),
               gd_amp_scaler_last_found_inf(scaler) ? " found_inf" : "");
        *last_report_time = now;
        *last_report_step = current_step;
    }

    if (config->generate_every_n_steps > 0 &&
        (current_step % (size_t)config->generate_every_n_steps) == 0U) {
        gpt_generate_vowels(ctx, model, config, current_step);
    }
}

static void ensure_checkpoint_parent_dir(gd_context *ctx, const char *path)
{
    char *copy;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "checkpoint path", __LINE__);
    }
    copy = (char *)malloc(strlen(path) + 1U);
    if (copy == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "checkpoint parent allocation", __LINE__);
    }
    (void)strcpy(copy, path);
    for (i = 1U; copy[i] != '\0'; ++i) {
        if (copy[i] == '/') {
            copy[i] = '\0';
            if (copy[0] != '\0' && mkdir(copy, 0775) != 0 && errno != EEXIST) {
                free(copy);
                gpt_fail_status(ctx, GD_ERR_IO, "checkpoint mkdir", __LINE__);
            }
            copy[i] = '/';
        }
    }
    free(copy);
}

static char *gpt_checkpoint_metadata(const gpt_config *config,
                                     size_t epoch,
                                     size_t global_step,
                                     float val_loss)
{
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    char *metadata;
    int n;
    if (config == NULL) {
        return NULL;
    }
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        return NULL;
    }
    n = snprintf(NULL,
                 0U,
                 "model=gpt_lm\n"
                 "vocab_size=%d\n"
                 "context_length=%d\n"
                 "d_model=%d\n"
                 "n_layers=%d\n"
                 "n_heads=%d\n"
                 "head_dim=%d\n"
                 "mlp_hidden=%d\n"
                 "sdpa_window=%d\n"
                 "dropout=%.9g\n"
                 "grad_clip_norm=%.9g\n"
                 "epoch=%zu\n"
                 "global_step=%zu\n"
                 "val_loss=%.9g\n"
                 "tokenizer_path=%s\n",
                 GPT_VOCAB_SIZE,
                 GPT_CONTEXT_LENGTH,
                 GPT_D_MODEL,
                 config->n_layers,
                 GPT_N_HEADS,
                 GPT_HEAD_DIM,
                 GPT_MLP_HIDDEN,
                 GPT_SDPA_WINDOW,
                 (double)config->dropout_p,
                 (double)config->grad_clip_norm,
                 epoch,
                 global_step,
                 (double)val_loss,
                 tokenizer_path);
    if (n < 0) {
        free(default_tok_path);
        return NULL;
    }
    metadata = (char *)malloc((size_t)n + 1U);
    if (metadata == NULL) {
        free(default_tok_path);
        return NULL;
    }
    (void)snprintf(metadata,
                   (size_t)n + 1U,
                   "model=gpt_lm\n"
                   "vocab_size=%d\n"
                   "context_length=%d\n"
                   "d_model=%d\n"
                   "n_layers=%d\n"
                   "n_heads=%d\n"
                   "head_dim=%d\n"
                   "mlp_hidden=%d\n"
                   "sdpa_window=%d\n"
                   "dropout=%.9g\n"
                   "grad_clip_norm=%.9g\n"
                   "epoch=%zu\n"
                   "global_step=%zu\n"
                   "val_loss=%.9g\n"
                   "tokenizer_path=%s\n",
                   GPT_VOCAB_SIZE,
                   GPT_CONTEXT_LENGTH,
                   GPT_D_MODEL,
                   config->n_layers,
                   GPT_N_HEADS,
                   GPT_HEAD_DIM,
                   GPT_MLP_HIDDEN,
                   GPT_SDPA_WINDOW,
                   (double)config->dropout_p,
                   (double)config->grad_clip_norm,
                   epoch,
                   global_step,
                   (double)val_loss,
                   tokenizer_path);
    free(default_tok_path);
    return metadata;
}

static float evaluate_gpt_loss(gd_context *ctx,
                               gpt_lm *model,
                               gd_dataset *dataset,
                               const gpt_config *config)
{
    gd_dataloader *loader = NULL;
    const bool was_training = model->mod.training;
    const uint64_t samples = gd_dataset_num_samples(dataset);
    int eval_batch_size;
    size_t steps;
    size_t step;
    double total_loss = 0.0;
    if (dataset == NULL || samples == 0U) {
        return NAN;
    }
    eval_batch_size = config->batch_size;
    if ((uint64_t)eval_batch_size > samples) {
        eval_batch_size = (int)samples;
    }
    if (eval_batch_size <= 0) {
        return NAN;
    }
    steps = (size_t)(samples / (uint64_t)eval_batch_size);
    if (steps == 0U) {
        return NAN;
    }
    gd_module_set_training(&model->mod, false);
    TRY(ctx, create_gdds_loader(ctx, dataset, NULL, eval_batch_size, &loader));
    for (step = 0U; step < steps; ++step) {
        gd_batch *batch = NULL;
        gd_tensor *input_ids;
        gd_tensor *target_ids;
        gd_tensor *positions;
        gd_tensor *cu_seqlens;
        gd_tensor loss;
        float loss_value = 0.0f;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, batch));
        input_ids = required_batch_tensor(ctx, batch, "input_ids", __LINE__);
        target_ids = required_batch_tensor(ctx, batch, "target_ids", __LINE__);
        positions = required_batch_tensor(ctx, batch, "positions", __LINE__);
        cu_seqlens = required_batch_tensor(ctx, batch, "cu_seqlens", __LINE__);
        TRY(ctx, gpt_lm_forward(ctx, model, input_ids, target_ids, positions, cu_seqlens, 0U, &loss));
        TRY(ctx, gd_tensor_item(ctx, &loss, &loss_value));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_dataloader_release(loader, batch));
        if (step + 1U < steps) {
            TRY(ctx, gd_dataloader_prefetch(loader));
        }
        total_loss += (double)loss_value;
    }
    gd_dataloader_destroy(loader);
    gd_module_set_training(&model->mod, was_training);
    return (float)(total_loss / (double)steps);
}

static void validate_and_maybe_checkpoint(gd_context *ctx,
                                          gpt_lm *model,
                                          gd_dataset *val_dataset,
                                          const gpt_config *config,
                                          size_t epoch,
                                          size_t global_step,
                                          float *best_val_loss)
{
    float val_loss;
    if (val_dataset == NULL) {
        return;
    }
    val_loss = evaluate_gpt_loss(ctx, model, val_dataset, config);
    printf("validation: epoch=%zu step=%zu val_loss=%.6f", epoch, global_step, (double)val_loss);
    if (isfinite(val_loss) && best_val_loss != NULL && val_loss < *best_val_loss) {
        printf(" improved=%.6f", (double)*best_val_loss);
        *best_val_loss = val_loss;
        if (config->save_best) {
            char *metadata = gpt_checkpoint_metadata(config, epoch, global_step, val_loss);
            gd_module_save_options save_options;
            if (metadata == NULL) {
                gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "checkpoint metadata", __LINE__);
            }
            ensure_checkpoint_parent_dir(ctx, config->checkpoint_path);
            save_options.metadata = metadata;
            save_options.metadata_len = strlen(metadata);
            save_options.include_buffers = true;
            TRY(ctx, gd_module_save_state(ctx, &model->mod, config->checkpoint_path, &save_options));
            printf(" saved=%s", config->checkpoint_path);
            free(metadata);
        }
    }
    printf("\n");
}

static void train_gpt(gd_context *ctx,
                      gpt_lm *model,
                      gd_dataset *dataset,
                      gd_dataset *val_dataset,
                      gd_optimizer *optimizer,
                      gd_amp_scaler *scaler,
                      const gd_lr_scheduler_config *lr_config,
                      const gpt_config *config,
                      size_t steps_per_epoch,
                      size_t total_steps)
{
    double last_report_time = gpt_wall_seconds();
    size_t last_report_step = 0U;
    size_t global_step = 0U;
    size_t epoch;
    float best_val_loss = INFINITY;

    for (epoch = 1U; epoch <= (size_t)config->epochs; ++epoch) {
        gd_sampler *sampler = NULL;
        gd_dataloader *loader = NULL;
        size_t epoch_step;
        gd_module_set_training(&model->mod, true);
        if (config->overfit_num_samples == 0U) {
            TRY(ctx, gd_sampler_create_random(dataset,
                                              config->seed ^ UINT64_C(0x51504c) ^ (uint64_t)epoch,
                                              &sampler));
            TRY(ctx, create_gdds_loader(ctx, dataset, sampler, config->batch_size, &loader));
        } else {
            TRY(ctx, create_gdds_loader(ctx, dataset, NULL, config->batch_size, &loader));
        }
        for (epoch_step = 1U; epoch_step <= steps_per_epoch; ++epoch_step) {
            train_one_batch(ctx,
                            model,
                            loader,
                            optimizer,
                            scaler,
                            lr_config,
                            config,
                            global_step,
                            total_steps,
                            epoch,
                            epoch_step,
                            steps_per_epoch,
                            &last_report_time,
                            &last_report_step);
            global_step += 1U;
        }
        gd_dataloader_destroy(loader);
        gd_sampler_destroy(sampler);
        validate_and_maybe_checkpoint(ctx,
                                      model,
                                      val_dataset,
                                      config,
                                      epoch,
                                      global_step,
                                      &best_val_loss);
    }
}


int main(int argc, char **argv)
{
    const gpt_config config = parse_args(argc, argv);
    const gd_memory_config mem = gpt_memory_config(&config);
    gd_context *ctx = NULL;
    gd_status st = gd_context_create(&mem, &ctx);
    gd_dataset *dataset = NULL;
    gd_dataset *val_dataset = NULL;
    gpt_lm model;
    gd_param_set params;
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_lr_scheduler_config lr_config;
    size_t dataset_samples = 0U;
    size_t val_samples = 0U;
    size_t samples_per_epoch = 0U;
    size_t steps_per_epoch = 0U;
    size_t total_steps = 0U;
    size_t optimizer_steps = 0U;
    uint64_t total_params = 0U;
    uint64_t trainable_params = 0U;
    uint64_t param_bytes = 0U;
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    memset(&params, 0, sizeof(params));
    memset(&lr_config, 0, sizeof(lr_config));

    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm: skipped (no supported gradients.c backend)\n");
        return 0;
    }
    if (st != GD_OK) {
        gpt_fail_status(ctx, st, "gd_context_create", __LINE__);
    }
    if (GPT_D_MODEL != GPT_N_HEADS * GPT_HEAD_DIM) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "invalid GPT head config", __LINE__);
    }

    if (config.epochs > 0) {
        TRY(ctx, gd_dataset_open_gdds_split(config.data_dir, "train", &dataset));
        dataset_samples = (size_t)gd_dataset_num_samples(dataset);
        steps_per_epoch = effective_steps_per_epoch((uint64_t)dataset_samples, &config, &samples_per_epoch);
        if (steps_per_epoch == 0U || (size_t)config.epochs > SIZE_MAX / steps_per_epoch) {
            gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "dataset too small for requested batch size", __LINE__);
        }
        total_steps = (size_t)config.epochs * steps_per_epoch;
        if (config.save_best) {
            TRY(ctx, gd_dataset_open_gdds_split(config.data_dir, config.val_split, &val_dataset));
            val_samples = (size_t)gd_dataset_num_samples(val_dataset);
            if (val_samples == 0U) {
                gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty validation split", __LINE__);
            }
        }
    }

    gpt_lm_init(ctx, &model, &config);
    if (config.load_checkpoint_path != NULL) {
        gd_module_load_options load_options;
        load_options.strict = true;
        load_options.load_buffers = true;
        TRY(ctx, gd_module_load_state(ctx, &model.mod, config.load_checkpoint_path, &load_options));
        printf("loaded_checkpoint: %s\n", config.load_checkpoint_path);
    }
    {
        const gd_param_group groups[] = {
            gd_param_group_build("no_decay_norm", "gpt_lm.*norm_w", 1.0f, 0.0f, true),
            gd_param_group_build("decay", "gpt_lm.*", 1.0f, config.weight_decay, true),
        };
        TRY(ctx, gd_module_collect_params(ctx, &model.mod, groups, GD_ARRAY_LEN(groups), &params));
    }
    TRY(ctx, param_set_stats(&params, &total_params, &trainable_params, &param_bytes));
    printf("model_params: total=%llu (%.3fM) trainable=%llu (%.3fM) param_bytes=%.2fMB target≈8M\n",
           (unsigned long long)total_params,
           (double)total_params / 1000000.0,
           (unsigned long long)trainable_params,
           (double)trainable_params / 1000000.0,
           (double)param_bytes / (1024.0 * 1024.0));
    print_param_set(&params);
    if (config.epochs > 0) {
        const gd_adamw_config adam = gpt_adamw_config(config.lr_max, config.weight_decay);
        const gd_amp_config amp = gpt_amp_config();
        TRY(ctx, gd_adamw_create(ctx, &params, &adam, &optimizer));
        TRY(ctx, gd_amp_scaler_create(&amp, &scaler));
    }
    TRY(ctx, gd_context_seal_params(ctx));

    lr_config = gd_lr_scheduler_config_default();
    lr_config.max_lr = config.lr_max;
    lr_config.min_lr = config.lr_min;
    lr_config.total_steps = (uint64_t)total_steps;
    if (config.lr_warmup_steps >= 0) {
        lr_config.warmup_steps = (uint64_t)config.lr_warmup_steps;
    } else {
        lr_config.warmup_steps = (uint64_t)(total_steps / 10U);
    }

    printf("dataset: dir=%s samples=%zu val_samples=%zu batch=%d context=%d steps_per_epoch=%zu samples_per_epoch=%zu epochs=%d total_steps=%zu%s\n",
           config.data_dir,
           dataset_samples,
           val_samples,
           config.batch_size,
           GPT_CONTEXT_LENGTH,
           steps_per_epoch,
           samples_per_epoch,
           config.epochs,
           total_steps,
           config.overfit_num_samples > 0U ? " overfit" : "");
    if (config.overfit_num_samples > 0U && samples_per_epoch != (size_t)config.overfit_num_samples) {
        printf("overfit: requested=%llu using=%zu full-batch samples\n",
               (unsigned long long)config.overfit_num_samples,
               samples_per_epoch);
    }
    printf("model: vocab=%d d_model=%d layers=%d heads=%d head_dim=%d mlp_hidden=%d sdpa_window=%d dropout=%.3f\n",
           GPT_VOCAB_SIZE,
           GPT_D_MODEL,
           config.n_layers,
           GPT_N_HEADS,
           GPT_HEAD_DIM,
           GPT_MLP_HIDDEN,
           GPT_SDPA_WINDOW,
           (double)config.dropout_p);
    if (config.generate_prompt != NULL || config.generate_every_n_steps > 0) {
        printf("generation: max_new_tokens=%d temperature=%.3f every_n_steps=%d batched_vowels=%s\n",
               config.max_new_tokens,
               (double)config.temperature,
               config.generate_every_n_steps,
               config.generate_every_n_steps > 0 ? "yes" : "no");
    }
    if (config.epochs > 0) {
        printf("checkpoint: save_best=%s path=%s val_split=%s\n",
               config.save_best ? "yes" : "no",
               config.checkpoint_path,
               config.val_split);
        printf("optim: adamw lr_max=%.6g lr_min=%.6g warmup=%llu total=%llu weight_decay=%.4g grad_clip=%.4g amp_scale=%.1f\n",
               (double)lr_config.max_lr,
               (double)lr_config.min_lr,
               (unsigned long long)lr_config.warmup_steps,
               (unsigned long long)lr_config.total_steps,
               (double)config.weight_decay,
               (double)config.grad_clip_norm,
               (double)gd_amp_scaler_scale(scaler));
    } else {
        printf("optim: skipped (generation-only run)\n");
    }
    printf("memory: params=%zuMB state=%zuMB scratch_slot=%zuMB data_slot=%zuMB\n",
           mem.params_bytes / (1024U * 1024U),
           mem.state_bytes / (1024U * 1024U),
           mem.scratch_slot_bytes / (1024U * 1024U),
           mem.data_slot_bytes / (1024U * 1024U));

    if (config.epochs > 0) {
        train_gpt(ctx,
                  &model,
                  dataset,
                  val_dataset,
                  optimizer,
                  scaler,
                  &lr_config,
                  &config,
                  steps_per_epoch,
                  total_steps);
        optimizer_steps = (size_t)gd_optimizer_step_count(optimizer);
    }
    if (config.generate_prompt != NULL) {
        gpt_generate(ctx, &model, &config);
    }
    {
        gd_memory_stats stats;
        TRY(ctx, gd_memory_stats_query(ctx, &stats));
        printf("memory_watermark: params=%zuMB state=%zuMB scratch_max_slot=%zuMB data_max_slot=%zuMB backend_waits=%llu\n",
               stats.params.watermark / (1024U * 1024U),
               stats.state.watermark / (1024U * 1024U),
               stats.scratch.max_slot_watermark / (1024U * 1024U),
               stats.data.max_slot_watermark / (1024U * 1024U),
               (unsigned long long)stats.backend_waits);
    }
    printf("gpt_lm: ok optimizer_steps=%zu\n", optimizer_steps);
    exit_code = 0;

    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    gpt_lm_deinit(&model);
    gd_dataset_destroy(val_dataset);
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    return exit_code;
}
