#include "gpt_lm_shared.h"
#include "gd_progress.h"

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
    printf("  --latest-checkpoint-path PATH full-resume checkpoint path saved every epoch (default: checkpoints/gpt_lm_latest.gdckpt)\n");
    printf("  --load-checkpoint PATH      load model weights only before training/generation\n");
    printf("  --resume-checkpoint PATH    resume model + optimizer + scaler/trainer sidecars\n");
    printf("  --val-split NAME            validation split name for validation/checkpointing (default: val)\n");
    printf("  --metrics-dir PATH          metrics JSONL root directory (default: data/metrics)\n");
    printf("  --metrics-project NAME      metrics project directory (default: gpt_lm)\n");
    printf("  --metrics-run-id ID         metrics run id/file stem (default: timestamp-pid)\n");
    printf("  --no-metrics                disable metrics JSONL logging\n");
    printf("  --no-save-best              disable best validation checkpoint writes\n");
    printf("  --no-save-latest            disable latest full-resume checkpoint writes\n");
    printf("  --early-stopping-patience N validation epochs without improvement before stopping; 0 disables (default: %d)\n",
           GPT_DEFAULT_EARLY_STOPPING_PATIENCE);
    printf("  --max-new-tokens N          generated tokens for --generate / periodic generation (default: 64)\n");
    printf("  --temperature T             sampling temperature; 0 means greedy (default: 0)\n");
    printf("  --min-p P                   min-p sampling cutoff relative to top token; 0 disables (default: 0)\n");
    printf("  --repetition-penalty P      repetition penalty; 1 disables (default: 1)\n");
    printf("  --logits-softcap C          final logits softcap; 0 disables (default: 30)\n");
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
    printf("  --eval-every-n-epochs N    run validation every N epochs; final epoch is always evaluated (default: %d)\n",
           GPT_DEFAULT_EVAL_EVERY_N_EPOCHS);
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
    config.latest_checkpoint_path = "checkpoints/gpt_lm_latest.gdckpt";
    config.load_checkpoint_path = NULL;
    config.resume_checkpoint_path = NULL;
    config.val_split = "val";
    config.metrics_dir = "data/metrics";
    config.metrics_project = "gpt_lm";
    config.metrics_run_id = NULL;
    config.epochs = GPT_DEFAULT_EPOCHS;
    config.batch_size = GPT_DEFAULT_BATCH_SIZE;
    config.n_layers = GPT_DEFAULT_LAYERS;
    config.report_every = GPT_DEFAULT_REPORT_EVERY;
    config.eval_every_n_epochs = GPT_DEFAULT_EVAL_EVERY_N_EPOCHS;
    config.early_stopping_patience = GPT_DEFAULT_EARLY_STOPPING_PATIENCE;
    config.lr_warmup_steps = -1;
    config.max_new_tokens = 64;
    config.generate_every_n_steps = 0;
    config.epochs_set = false;
    config.save_best = true;
    config.save_latest = true;
    config.metrics_enabled = true;
    config.overfit_num_samples = 0U;
    config.seed = GPT_DEFAULT_SEED;
    config.dropout_p = GPT_DEFAULT_DROPOUT_P;
    config.lr_max = GPT_DEFAULT_LR_MAX;
    config.lr_min = GPT_DEFAULT_LR_MIN;
    config.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
    config.grad_clip_norm = GPT_DEFAULT_GRAD_CLIP_NORM;
    config.temperature = 0.0f;
    config.min_p = GPT_DEFAULT_MIN_P;
    config.repetition_penalty = GPT_DEFAULT_REPETITION_PENALTY;
    config.logits_softcap = 30.0f;
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
        value = arg_value(argc, argv, &i, "--latest-checkpoint-path");
        if (value != NULL) {
            config.latest_checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--load-checkpoint");
        if (value != NULL) {
            config.load_checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--resume-checkpoint");
        if (value != NULL) {
            config.resume_checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--val-split");
        if (value != NULL) {
            config.val_split = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--metrics-dir");
        if (value != NULL) {
            config.metrics_dir = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--metrics-project");
        if (value != NULL) {
            config.metrics_project = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--metrics-run-id");
        if (value != NULL) {
            config.metrics_run_id = value;
            continue;
        }
        if (strcmp(argv[i], "--no-metrics") == 0) {
            config.metrics_enabled = false;
            continue;
        }
        if (strcmp(argv[i], "--no-save-best") == 0) {
            config.save_best = false;
            continue;
        }
        if (strcmp(argv[i], "--no-save-latest") == 0) {
            config.save_latest = false;
            continue;
        }
        value = arg_value(argc, argv, &i, "--early-stopping-patience");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --early-stopping-patience %s\n", value);
                exit(2);
            }
            config.early_stopping_patience = (int)parsed_i64;
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
            if (!parse_i64_arg(value, 1, GPT_CONTEXT_LENGTH, &parsed_i64)) {
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
        value = arg_value(argc, argv, &i, "--min-p");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --min-p %s\n", value);
                exit(2);
            }
            config.min_p = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--repetition-penalty");
        if (value != NULL) {
            if (!parse_float_arg(value, 1.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --repetition-penalty %s\n", value);
                exit(2);
            }
            config.repetition_penalty = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm: invalid --logits-softcap %s\n", value);
                exit(2);
            }
            config.logits_softcap = parsed_f32;
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
        value = arg_value(argc, argv, &i, "--eval-every-n-epochs");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 1000000, &parsed_i64)) {
                fprintf(stderr, "gpt_lm: invalid --eval-every-n-epochs %s\n", value);
                exit(2);
            }
            config.eval_every_n_epochs = (int)parsed_i64;
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
    if (config.load_checkpoint_path != NULL && config.resume_checkpoint_path != NULL) {
        fprintf(stderr, "gpt_lm: use either --load-checkpoint or --resume-checkpoint, not both\n");
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
    cfg.num_workers = 2;
    cfg.prefetch_factor = 4;
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

static gd_status gpt_lm_forward_batch(gd_context *ctx,
                                      gpt_lm *model,
                                      gd_batch *batch,
                                      uint64_t step,
                                      gd_tensor *loss_out)
{
    gd_tensor *input_ids;
    gd_tensor *target_ids;
    gd_tensor *positions;
    gd_tensor *cu_seqlens;
    if (ctx == NULL || model == NULL || batch == NULL || loss_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    input_ids = gd_batch_tensor(batch, "input_ids");
    target_ids = gd_batch_tensor(batch, "target_ids");
    positions = gd_batch_tensor(batch, "positions");
    cu_seqlens = gd_batch_tensor(batch, "cu_seqlens");
    if (input_ids == NULL || target_ids == NULL || positions == NULL || cu_seqlens == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gpt_lm_forward(ctx,
                          model,
                          input_ids,
                          target_ids,
                          positions,
                          cu_seqlens,
                          step,
                          loss_out);
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
    config.init_scale = 65536.0f;
    config.growth_factor = 2.0f;
    config.backoff_factor = 0.5f;
    config.growth_interval = 2000U;
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

typedef struct gpt_batch_user {
    gpt_lm *model;
} gpt_batch_user;

static gd_status gpt_train_loss_fn(const gd_train_batch_step *step,
                                   void *user_data,
                                   gd_tensor *loss_out)
{
    const gpt_batch_user *run = (const gpt_batch_user *)user_data;
    if (step == NULL || run == NULL || run->model == NULL || loss_out == NULL ||
        step->ctx == NULL || step->batch == NULL || step->global_step == UINT64_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gpt_lm_forward_batch(step->ctx,
                                run->model,
                                step->batch,
                                step->global_step + UINT64_C(1),
                                loss_out);
}

static gd_status gpt_eval_loss_fn(const gd_eval_batch_step *step,
                                  void *user_data,
                                  gd_tensor *loss_out)
{
    const gpt_batch_user *run = (const gpt_batch_user *)user_data;
    if (step == NULL || run == NULL || run->model == NULL || loss_out == NULL ||
        step->ctx == NULL || step->batch == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gpt_lm_forward_batch(step->ctx, run->model, step->batch, 0U, loss_out);
}

typedef struct gpt_report_state {
    gd_progress progress;
    float last_train_loss;
    float last_eval_loss;
    size_t best_epoch;
    bool has_train_loss;
    bool has_eval_loss;
    bool has_best_epoch;
} gpt_report_state;

static void gpt_report_summary(gpt_report_state *report)
{
    char train_loss[32];
    char eval_loss[32];
    char best_epoch[32];
    if (report == NULL) {
        return;
    }
    if (report->has_train_loss) {
        (void)snprintf(train_loss, sizeof(train_loss), "%.6f", (double)report->last_train_loss);
    } else {
        (void)snprintf(train_loss, sizeof(train_loss), "n/a");
    }
    if (report->has_eval_loss) {
        (void)snprintf(eval_loss, sizeof(eval_loss), "%.6f", (double)report->last_eval_loss);
    } else {
        (void)snprintf(eval_loss, sizeof(eval_loss), "n/a");
    }
    if (report->has_best_epoch) {
        (void)snprintf(best_epoch, sizeof(best_epoch), "%zu", report->best_epoch);
    } else {
        (void)snprintf(best_epoch, sizeof(best_epoch), "n/a");
    }
    gd_progress_rowf(&report->progress,
                     4U,
                     "last_train_loss=%s last_eval_loss=%s last_best_epoch=%s",
                     train_loss,
                     eval_loss,
                     best_epoch);
}

static bool gpt_should_eval_epoch(const gpt_config *config, size_t epoch)
{
    if (config == NULL || config->eval_every_n_epochs <= 0) {
        return false;
    }
    if (epoch >= (size_t)config->epochs) {
        return true;
    }
    return (epoch % (size_t)config->eval_every_n_epochs) == 0U;
}

static size_t gpt_next_eval_epoch(const gpt_config *config, size_t epoch)
{
    size_t next;
    const size_t every = config != NULL && config->eval_every_n_epochs > 0 ?
                             (size_t)config->eval_every_n_epochs :
                             1U;
    if (config == NULL || epoch >= (size_t)config->epochs) {
        return epoch;
    }
    next = ((epoch / every) + 1U) * every;
    if (next > (size_t)config->epochs) {
        next = (size_t)config->epochs;
    }
    return next;
}

static void train_one_batch(gd_context *ctx,
                            gpt_lm *model,
                            gd_dataloader *loader,
                            gd_optimizer *optimizer,
                            gd_amp_scaler *scaler,
                            const gd_lr_scheduler_config *lr_config,
                            const gpt_config *config,
                            const gpt_generation_tokenizer *generation_tokenizer,
                            gpt_report_state *report,
                            gd_metrics_logger *metrics,
                            size_t global_step,
                            size_t total_steps,
                            size_t epoch,
                            size_t epoch_step,
                            size_t steps_per_epoch,
                            double *last_report_time,
                            size_t *last_report_step)
{
    const size_t current_step = global_step + 1U;
    const int epoch_complete = epoch_step == steps_per_epoch;
    const int should_report = config->report_every > 0 &&
                              (global_step == 0U || epoch_complete ||
                               current_step % (size_t)config->report_every == 0U ||
                               current_step == total_steps);
    const gd_train_batch_config step_config = {
        .loader = loader,
        .optimizer = optimizer,
        .scaler = scaler,
        .lr_schedule = lr_config,
        .global_step = (uint64_t)global_step,
        .grad_clip_norm = config->grad_clip_norm,
        .prefetch_next = true,
        .read_loss_value = should_report != 0,
        .read_grad_norm = should_report != 0 && config->grad_clip_norm > 0.0f,
        .sync_scaler = should_report != 0,
    };
    gpt_batch_user step_user = {
        .model = model,
    };
    gd_train_batch_result step_result;
    TRY(ctx, gd_train_batch(ctx, &step_config, gpt_train_loss_fn, &step_user, &step_result));

    if (should_report) {
        float grad_norm = 0.0f;
        char grad_norm_text[64];
        double now;
        double elapsed;
        size_t batches;
        size_t tokens;
        double batches_per_sec;
        double tokens_per_sec;
        const float train_loss = step_result.has_loss_value ? step_result.loss_value : 0.0f;
        if (config->grad_clip_norm > 0.0f) {
            grad_norm = step_result.has_grad_norm ? step_result.grad_norm : 0.0f;
            (void)snprintf(grad_norm_text,
                           sizeof(grad_norm_text),
                           " grad_norm=%.4g",
                           (double)grad_norm);
        } else {
            (void)snprintf(grad_norm_text, sizeof(grad_norm_text), " grad_norm=off");
        }
        now = gpt_wall_seconds();
        elapsed = now - *last_report_time;
        batches = current_step - *last_report_step;
        tokens = batches * (size_t)config->batch_size * (size_t)GPT_CONTEXT_LENGTH;
        batches_per_sec = elapsed > 0.0 ? (double)batches / elapsed : 0.0;
        tokens_per_sec = elapsed > 0.0 ? (double)tokens / elapsed : 0.0;
        if (report != NULL) {
            if (epoch_complete) {
                report->last_train_loss = train_loss;
                report->has_train_loss = true;
            }
            gd_progress_rowf(&report->progress, 0U, "");
            gd_progress_row_append_bar(&report->progress,
                                       0U,
                                       "train",
                                       (uint64_t)current_step,
                                       (uint64_t)total_steps);
            gd_progress_row_appendf(&report->progress,
                                    0U,
                                    " epoch=%zu/%d batch=%zu/%zu",
                                    epoch,
                                    config->epochs,
                                    epoch_step,
                                    steps_per_epoch);
            gd_progress_rowf(&report->progress,
                             1U,
                             "loss=%.6f lr=%.6g%s batch/s=%.2f tok/s=%.0f amp_scale=%.1f%s",
                             (double)train_loss,
                             (double)(step_result.has_lr ? step_result.lr : 0.0f),
                             grad_norm_text,
                             batches_per_sec,
                             tokens_per_sec,
                             (double)(step_result.has_amp_state ?
                                          step_result.amp_scale :
                                          gd_amp_scaler_scale(scaler)),
                             step_result.has_amp_state && step_result.found_inf ? " found_inf" : "");
            gpt_report_summary(report);
            gd_progress_render(&report->progress);
        }
        if (metrics != NULL) {
            const gd_metrics_field fields[] = {
                gd_metrics_u64("epoch", (uint64_t)epoch),
                gd_metrics_u64("epoch_step", (uint64_t)epoch_step),
                gd_metrics_u64("steps_per_epoch", (uint64_t)steps_per_epoch),
                gd_metrics_u64("step", (uint64_t)current_step),
                gd_metrics_u64("total_steps", (uint64_t)total_steps),
                gd_metrics_f64("loss", (double)train_loss),
                gd_metrics_f64("lr", (double)(step_result.has_lr ? step_result.lr : 0.0f)),
                gd_metrics_bool("grad_clip_enabled", config->grad_clip_norm > 0.0f),
                gd_metrics_f64("grad_norm", (double)grad_norm),
                gd_metrics_f64("grad_clip_norm", (double)config->grad_clip_norm),
                gd_metrics_f64("batch_s", batches_per_sec),
                gd_metrics_f64("tok_s", tokens_per_sec),
                gd_metrics_f64("amp_scale",
                               (double)(step_result.has_amp_state ?
                                            step_result.amp_scale :
                                            gd_amp_scaler_scale(scaler))),
                gd_metrics_bool("found_inf", step_result.has_amp_state && step_result.found_inf),
            };
            (void)gd_metrics_logger_log_event(metrics, "train", fields, GD_ARRAY_LEN(fields));
        }
        *last_report_time = now;
        *last_report_step = current_step;
    }

    if (config->generate_every_n_steps > 0 &&
        (current_step % (size_t)config->generate_every_n_steps) == 0U) {
        if (report != NULL) {
            gd_progress_finish(&report->progress);
        }
        if (generation_tokenizer != NULL) {
            gpt_generate_vowels_with_tokenizer(ctx, model, config, current_step, generation_tokenizer);
        } else {
            gpt_generate_vowels(ctx, model, config, current_step);
        }
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

typedef struct gpt_training_state {
    bool loaded;
    size_t epoch;
    size_t global_step;
    size_t epochs_without_improvement;
    float best_val_loss;
} gpt_training_state;

static bool gpt_has_suffix(const char *text, const char *suffix)
{
    const size_t text_len = text != NULL ? strlen(text) : 0U;
    const size_t suffix_len = suffix != NULL ? strlen(suffix) : 0U;
    return text_len >= suffix_len && suffix_len > 0U &&
           memcmp(text + text_len - suffix_len, suffix, suffix_len) == 0;
}

static char *gpt_checkpoint_sidecar_path(const char *checkpoint_path, const char *suffix)
{
    const char *ckpt_suffix = ".gdckpt";
    const size_t path_len = checkpoint_path != NULL ? strlen(checkpoint_path) : 0U;
    const size_t suffix_len = suffix != NULL ? strlen(suffix) : 0U;
    size_t stem_len;
    char *path;
    if (path_len == 0U || suffix_len == 0U) {
        return NULL;
    }
    stem_len = gpt_has_suffix(checkpoint_path, ckpt_suffix) ? path_len - strlen(ckpt_suffix) : path_len;
    if (stem_len > SIZE_MAX - suffix_len - 1U) {
        return NULL;
    }
    path = (char *)malloc(stem_len + suffix_len + 1U);
    if (path == NULL) {
        return NULL;
    }
    memcpy(path, checkpoint_path, stem_len);
    memcpy(path + stem_len, suffix, suffix_len);
    path[stem_len + suffix_len] = '\0';
    return path;
}

static char *gpt_checkpoint_metadata(const gpt_config *config,
                                     size_t epoch,
                                     size_t global_step,
                                     float val_loss,
                                     float best_val_loss,
                                     const char *optimizer_path,
                                     const char *train_state_path)
{
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    const char *optimizer_text = optimizer_path != NULL ? optimizer_path : "";
    const char *train_state_text = train_state_path != NULL ? train_state_path : "";
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
                 "ffn_activation=swiglu\n"
                 "sdpa_window=%d\n"
                 "dropout=%.9g\n"
                 "logits_softcap=%.9g\n"
                 "grad_clip_norm=%.9g\n"
                 "eval_every_n_epochs=%d\n"
                 "early_stopping_patience=%d\n"
                 "epoch=%zu\n"
                 "global_step=%zu\n"
                 "val_loss=%.9g\n"
                 "best_val_loss=%.9g\n"
                 "tokenizer_path=%s\n"
                 "optimizer_path=%s\n"
                 "train_state_path=%s\n",
                 GPT_VOCAB_SIZE,
                 GPT_CONTEXT_LENGTH,
                 GPT_D_MODEL,
                 config->n_layers,
                 GPT_N_HEADS,
                 GPT_HEAD_DIM,
                 GPT_MLP_HIDDEN,
                 GPT_SDPA_WINDOW,
                 (double)config->dropout_p,
                 (double)config->logits_softcap,
                 (double)config->grad_clip_norm,
                 config->eval_every_n_epochs,
                 config->early_stopping_patience,
                 epoch,
                 global_step,
                 (double)val_loss,
                 (double)best_val_loss,
                 tokenizer_path,
                 optimizer_text,
                 train_state_text);
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
                   "ffn_activation=swiglu\n"
                   "sdpa_window=%d\n"
                   "dropout=%.9g\n"
                   "logits_softcap=%.9g\n"
                   "grad_clip_norm=%.9g\n"
                   "eval_every_n_epochs=%d\n"
                   "early_stopping_patience=%d\n"
                   "epoch=%zu\n"
                   "global_step=%zu\n"
                   "val_loss=%.9g\n"
                   "best_val_loss=%.9g\n"
                   "tokenizer_path=%s\n"
                   "optimizer_path=%s\n"
                   "train_state_path=%s\n",
                   GPT_VOCAB_SIZE,
                   GPT_CONTEXT_LENGTH,
                   GPT_D_MODEL,
                   config->n_layers,
                   GPT_N_HEADS,
                   GPT_HEAD_DIM,
                   GPT_MLP_HIDDEN,
                   GPT_SDPA_WINDOW,
                   (double)config->dropout_p,
                   (double)config->logits_softcap,
                   (double)config->grad_clip_norm,
                   config->eval_every_n_epochs,
                   config->early_stopping_patience,
                   epoch,
                   global_step,
                   (double)val_loss,
                   (double)best_val_loss,
                   tokenizer_path,
                   optimizer_text,
                   train_state_text);
    free(default_tok_path);
    return metadata;
}

static void gpt_write_training_state(gd_context *ctx,
                                     const char *path,
                                     const char *model_path,
                                     const char *optimizer_path,
                                     const gd_lr_scheduler_config *lr_config,
                                     gd_amp_scaler *scaler,
                                     size_t epoch,
                                     size_t global_step,
                                     size_t epochs_without_improvement,
                                     float val_loss,
                                     float best_val_loss)
{
    gd_amp_scaler_state amp_state;
    FILE *file;
    int write_failed;
    if (path == NULL || model_path == NULL || optimizer_path == NULL || lr_config == NULL || scaler == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "training state arguments", __LINE__);
    }
    TRY(ctx, gd_amp_scaler_get_state(ctx, scaler, &amp_state));
    ensure_checkpoint_parent_dir(ctx, path);
    file = fopen(path, "wb");
    if (file == NULL) {
        gpt_fail_status(ctx, GD_ERR_IO, "training state open", __LINE__);
    }
    fprintf(file, "format=gpt_lm_train_state_v1\n");
    fprintf(file, "model_checkpoint=%s\n", model_path);
    fprintf(file, "optimizer_checkpoint=%s\n", optimizer_path);
    fprintf(file, "epoch=%zu\n", epoch);
    fprintf(file, "global_step=%zu\n", global_step);
    fprintf(file, "epochs_without_improvement=%zu\n", epochs_without_improvement);
    fprintf(file, "val_loss=%.9g\n", (double)val_loss);
    fprintf(file, "best_val_loss=%.9g\n", (double)best_val_loss);
    fprintf(file, "lr.max_lr=%.9g\n", (double)lr_config->max_lr);
    fprintf(file, "lr.min_lr=%.9g\n", (double)lr_config->min_lr);
    fprintf(file, "lr.warmup_steps=%llu\n", (unsigned long long)lr_config->warmup_steps);
    fprintf(file, "lr.total_steps=%llu\n", (unsigned long long)lr_config->total_steps);
    fprintf(file, "amp.enabled=%u\n", amp_state.config.enabled ? 1U : 0U);
    fprintf(file, "amp.init_scale=%.9g\n", (double)amp_state.config.init_scale);
    fprintf(file, "amp.growth_factor=%.9g\n", (double)amp_state.config.growth_factor);
    fprintf(file, "amp.backoff_factor=%.9g\n", (double)amp_state.config.backoff_factor);
    fprintf(file, "amp.min_scale=%.9g\n", (double)amp_state.config.min_scale);
    fprintf(file, "amp.max_scale=%.9g\n", (double)amp_state.config.max_scale);
    fprintf(file, "amp.growth_interval=%u\n", amp_state.config.growth_interval);
    fprintf(file, "amp.scale=%.9g\n", (double)amp_state.scale);
    fprintf(file, "amp.growth_tracker=%u\n", amp_state.growth_tracker);
    fprintf(file, "amp.last_found_inf=%u\n", amp_state.last_found_inf ? 1U : 0U);
    write_failed = ferror(file);
    if (fclose(file) != 0 || write_failed) {
        gpt_fail_status(ctx, GD_ERR_IO, "training state write", __LINE__);
    }
}

static char *gpt_read_text_file(gd_context *ctx, const char *path)
{
    FILE *file;
    long size_long;
    size_t size;
    char *data;
    if (path == NULL || path[0] == '\0') {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "read text path", __LINE__);
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        gpt_fail_status(ctx, GD_ERR_IO, "read text open", __LINE__);
    }
    if (fseek(file, 0L, SEEK_END) != 0) {
        (void)fclose(file);
        gpt_fail_status(ctx, GD_ERR_IO, "read text seek", __LINE__);
    }
    size_long = ftell(file);
    if (size_long < 0) {
        (void)fclose(file);
        gpt_fail_status(ctx, GD_ERR_IO, "read text tell", __LINE__);
    }
    if (fseek(file, 0L, SEEK_SET) != 0) {
        (void)fclose(file);
        gpt_fail_status(ctx, GD_ERR_IO, "read text rewind", __LINE__);
    }
    size = (size_t)size_long;
    data = (char *)malloc(size + 1U);
    if (data == NULL) {
        (void)fclose(file);
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "read text allocation", __LINE__);
    }
    if (size != 0U && fread(data, 1U, size, file) != size) {
        free(data);
        (void)fclose(file);
        gpt_fail_status(ctx, GD_ERR_IO, "read text data", __LINE__);
    }
    if (fclose(file) != 0) {
        free(data);
        gpt_fail_status(ctx, GD_ERR_IO, "read text close", __LINE__);
    }
    data[size] = '\0';
    return data;
}

static const char *gpt_state_value(const char *text, const char *key, size_t *value_len_out)
{
    const size_t key_len = key != NULL ? strlen(key) : 0U;
    const char *line = text;
    if (text == NULL || key == NULL || value_len_out == NULL || key_len == 0U) {
        return NULL;
    }
    while (*line != '\0') {
        const char *end = strchr(line, '\n');
        const size_t line_len = end != NULL ? (size_t)(end - line) : strlen(line);
        if (line_len > key_len && memcmp(line, key, key_len) == 0 && line[key_len] == '=') {
            *value_len_out = line_len - key_len - 1U;
            return line + key_len + 1U;
        }
        if (end == NULL) {
            break;
        }
        line = end + 1;
    }
    return NULL;
}

static void gpt_state_copy_value(gd_context *ctx,
                                 const char *text,
                                 const char *key,
                                 char *buffer,
                                 size_t buffer_size)
{
    const char *value;
    size_t value_len = 0U;
    if (buffer == NULL || buffer_size == 0U) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "state value buffer", __LINE__);
    }
    value = gpt_state_value(text, key, &value_len);
    if (value == NULL || value_len >= buffer_size) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state value missing", __LINE__);
    }
    memcpy(buffer, value, value_len);
    buffer[value_len] = '\0';
}

static uint64_t gpt_state_u64(gd_context *ctx, const char *text, const char *key)
{
    char buffer[128];
    char *end = NULL;
    unsigned long long parsed;
    gpt_state_copy_value(ctx, text, key, buffer, sizeof(buffer));
    errno = 0;
    parsed = strtoull(buffer, &end, 10);
    if (errno != 0 || end == buffer || *end != '\0') {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state uint parse", __LINE__);
    }
    return (uint64_t)parsed;
}

static float gpt_state_f32(gd_context *ctx, const char *text, const char *key)
{
    char buffer[128];
    char *end = NULL;
    float parsed;
    gpt_state_copy_value(ctx, text, key, buffer, sizeof(buffer));
    errno = 0;
    parsed = strtof(buffer, &end);
    if (errno != 0 || end == buffer || *end != '\0' || parsed != parsed) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state float parse", __LINE__);
    }
    return parsed;
}

static bool gpt_state_bool(gd_context *ctx, const char *text, const char *key)
{
    const uint64_t value = gpt_state_u64(ctx, text, key);
    if (value > 1U) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state bool parse", __LINE__);
    }
    return value != 0U;
}

static size_t gpt_state_size(gd_context *ctx, const char *text, const char *key)
{
    const uint64_t value = gpt_state_u64(ctx, text, key);
    if (value > (uint64_t)SIZE_MAX) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state size overflow", __LINE__);
    }
    return (size_t)value;
}

static uint32_t gpt_state_u32(gd_context *ctx, const char *text, const char *key)
{
    const uint64_t value = gpt_state_u64(ctx, text, key);
    if (value > (uint64_t)UINT32_MAX) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "state u32 overflow", __LINE__);
    }
    return (uint32_t)value;
}

static void gpt_load_training_state(gd_context *ctx,
                                    const char *path,
                                    gd_amp_scaler *scaler,
                                    gd_lr_scheduler_config *lr_config,
                                    gpt_training_state *out)
{
    char *text;
    char format[64];
    gd_amp_scaler_state amp_state;
    gd_lr_scheduler_config saved_lr;
    if (path == NULL || scaler == NULL || lr_config == NULL || out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "load training state args", __LINE__);
    }
    text = gpt_read_text_file(ctx, path);
    gpt_state_copy_value(ctx, text, "format", format, sizeof(format));
    if (strcmp(format, "gpt_lm_train_state_v1") != 0) {
        free(text);
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "training state format", __LINE__);
    }
    memset(out, 0, sizeof(*out));
    out->loaded = true;
    out->epoch = gpt_state_size(ctx, text, "epoch");
    out->global_step = gpt_state_size(ctx, text, "global_step");
    out->epochs_without_improvement = gpt_state_size(ctx, text, "epochs_without_improvement");
    out->best_val_loss = gpt_state_f32(ctx, text, "best_val_loss");

    saved_lr.max_lr = gpt_state_f32(ctx, text, "lr.max_lr");
    saved_lr.min_lr = gpt_state_f32(ctx, text, "lr.min_lr");
    saved_lr.warmup_steps = gpt_state_u64(ctx, text, "lr.warmup_steps");
    saved_lr.total_steps = gpt_state_u64(ctx, text, "lr.total_steps");
    if (saved_lr.max_lr != lr_config->max_lr || saved_lr.min_lr != lr_config->min_lr ||
        saved_lr.warmup_steps != lr_config->warmup_steps || saved_lr.total_steps != lr_config->total_steps) {
        printf("resume: using current scheduler config; saved checkpoint schedule differs "
               "saved=(max=%.6g min=%.6g warmup=%llu total=%llu) "
               "current=(max=%.6g min=%.6g warmup=%llu total=%llu)\n",
               (double)saved_lr.max_lr,
               (double)saved_lr.min_lr,
               (unsigned long long)saved_lr.warmup_steps,
               (unsigned long long)saved_lr.total_steps,
               (double)lr_config->max_lr,
               (double)lr_config->min_lr,
               (unsigned long long)lr_config->warmup_steps,
               (unsigned long long)lr_config->total_steps);
    }

    amp_state.config.enabled = gpt_state_bool(ctx, text, "amp.enabled");
    amp_state.config.init_scale = gpt_state_f32(ctx, text, "amp.init_scale");
    amp_state.config.growth_factor = gpt_state_f32(ctx, text, "amp.growth_factor");
    amp_state.config.backoff_factor = gpt_state_f32(ctx, text, "amp.backoff_factor");
    amp_state.config.min_scale = gpt_state_f32(ctx, text, "amp.min_scale");
    amp_state.config.max_scale = gpt_state_f32(ctx, text, "amp.max_scale");
    amp_state.config.growth_interval = gpt_state_u32(ctx, text, "amp.growth_interval");
    amp_state.scale = gpt_state_f32(ctx, text, "amp.scale");
    amp_state.growth_tracker = gpt_state_u32(ctx, text, "amp.growth_tracker");
    amp_state.last_found_inf = gpt_state_bool(ctx, text, "amp.last_found_inf");
    TRY(ctx, gd_amp_scaler_set_state(ctx, scaler, &amp_state));
    free(text);
    printf("resumed_training_state: path=%s epoch=%zu global_step=%zu best_val_loss=%.6f amp_scale=%.1f\n",
           path,
           out->epoch,
           out->global_step,
           (double)out->best_val_loss,
           (double)gd_amp_scaler_scale(scaler));
}

static float evaluate_gpt_loss(gd_context *ctx,
                               gpt_lm *model,
                               gd_dataset *dataset,
                               const gpt_config *config)
{
    float mean_loss = NAN;
    gd_eval_config eval_config;
    gpt_batch_user eval_user;
    if (dataset == NULL || gd_dataset_num_samples(dataset) == 0U) {
        return NAN;
    }
    eval_config = (gd_eval_config){
        .dataset = dataset,
        .module = &model->mod,
        .batch_size = config->batch_size,
        .num_workers = 1,
        .prefetch_factor = 2,
        .scope = GD_SCOPE_EVAL,
        .prefetch_next = true,
    };
    eval_user = (gpt_batch_user){
        .model = model,
    };
    TRY(ctx, gd_eval_mean_loss(ctx, &eval_config, gpt_eval_loss_fn, &eval_user, &mean_loss));
    return mean_loss;
}

static void gpt_save_checkpoint_bundle(gd_context *ctx,
                                       gpt_lm *model,
                                       gd_optimizer *optimizer,
                                       gd_amp_scaler *scaler,
                                       const gd_lr_scheduler_config *lr_config,
                                       const gpt_config *config,
                                       const char *checkpoint_path,
                                       const char *print_key,
                                       size_t epoch,
                                       size_t global_step,
                                       size_t epochs_without_improvement,
                                       float val_loss,
                                       float best_val_loss)
{
    char *optimizer_path = NULL;
    char *train_state_path = NULL;
    char *metadata = NULL;
    gd_module_save_options save_options;
    if (model == NULL || optimizer == NULL || scaler == NULL || lr_config == NULL || config == NULL ||
        checkpoint_path == NULL || checkpoint_path[0] == '\0') {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "checkpoint bundle arguments", __LINE__);
    }
    optimizer_path = gpt_checkpoint_sidecar_path(checkpoint_path, ".optim.gdckpt");
    train_state_path = gpt_checkpoint_sidecar_path(checkpoint_path, ".train");
    if (optimizer_path == NULL || train_state_path == NULL) {
        free(optimizer_path);
        free(train_state_path);
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "checkpoint sidecar path", __LINE__);
    }
    metadata = gpt_checkpoint_metadata(config,
                                       epoch,
                                       global_step,
                                       val_loss,
                                       best_val_loss,
                                       optimizer_path,
                                       train_state_path);
    if (metadata == NULL) {
        free(optimizer_path);
        free(train_state_path);
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "checkpoint metadata", __LINE__);
    }
    ensure_checkpoint_parent_dir(ctx, checkpoint_path);
    ensure_checkpoint_parent_dir(ctx, optimizer_path);
    ensure_checkpoint_parent_dir(ctx, train_state_path);
    save_options.metadata = metadata;
    save_options.metadata_len = strlen(metadata);
    save_options.include_buffers = true;
    TRY(ctx, gd_module_save_state(ctx, &model->mod, checkpoint_path, &save_options));
    TRY(ctx, gd_optimizer_save_state(ctx, optimizer, optimizer_path));
    gpt_write_training_state(ctx,
                             train_state_path,
                             checkpoint_path,
                             optimizer_path,
                             lr_config,
                             scaler,
                             epoch,
                             global_step,
                             epochs_without_improvement,
                             val_loss,
                             best_val_loss);
    if (print_key != NULL) {
        printf(" %s=%s optim=%s state=%s", print_key, checkpoint_path, optimizer_path, train_state_path);
    }
    free(metadata);
    free(optimizer_path);
    free(train_state_path);
}

static bool validate_and_maybe_checkpoint(gd_context *ctx,
                                          gpt_lm *model,
                                          gd_optimizer *optimizer,
                                          gd_amp_scaler *scaler,
                                          const gd_lr_scheduler_config *lr_config,
                                          gd_dataset *val_dataset,
                                          const gpt_config *config,
                                          gpt_report_state *report,
                                          gd_metrics_logger *metrics,
                                          size_t epoch,
                                          size_t global_step,
                                          float *best_val_loss,
                                          float *out_val_loss,
                                          bool *out_improved)
{
    bool improved = false;
    float val_loss;
    if (out_improved != NULL) {
        *out_improved = false;
    }
    if (out_val_loss != NULL) {
        *out_val_loss = NAN;
    }
    if (val_dataset == NULL) {
        return false;
    }
    if (report != NULL) {
        gd_progress_rowf(&report->progress,
                         2U,
                         "eval epoch=%zu step=%zu running...",
                         epoch,
                         global_step);
        gd_progress_rowf(&report->progress, 3U, "checkpoint best=pending latest=pending");
        gpt_report_summary(report);
        gd_progress_render(&report->progress);
    }
    val_loss = evaluate_gpt_loss(ctx, model, val_dataset, config);
    if (out_val_loss != NULL) {
        *out_val_loss = val_loss;
    }
    if (isfinite(val_loss) && best_val_loss != NULL && val_loss < *best_val_loss) {
        *best_val_loss = val_loss;
        improved = true;
        if (config->save_best) {
            if (report != NULL) {
                gd_progress_rowf(&report->progress,
                                 3U,
                                 "checkpoint best=saving path=%s latest=pending",
                                 config->checkpoint_path);
                gpt_report_summary(report);
                gd_progress_render(&report->progress);
            }
            gpt_save_checkpoint_bundle(ctx,
                                       model,
                                       optimizer,
                                       scaler,
                                       lr_config,
                                       config,
                                       config->checkpoint_path,
                                       NULL,
                                       epoch,
                                       global_step,
                                       0U,
                                       val_loss,
                                       *best_val_loss);
            if (metrics != NULL) {
                const gd_metrics_field fields[] = {
                    gd_metrics_string("kind", "best"),
                    gd_metrics_string("path", config->checkpoint_path),
                    gd_metrics_u64("epoch", (uint64_t)epoch),
                    gd_metrics_u64("step", (uint64_t)global_step),
                    gd_metrics_f64("val_loss", (double)val_loss),
                    gd_metrics_f64("best_val_loss", (double)*best_val_loss),
                };
                (void)gd_metrics_logger_log_event(metrics, "checkpoint", fields, GD_ARRAY_LEN(fields));
            }
        }
    }
    if (out_improved != NULL) {
        *out_improved = improved;
    }
    if (metrics != NULL) {
        const gd_metrics_field fields[] = {
            gd_metrics_u64("epoch", (uint64_t)epoch),
            gd_metrics_u64("step", (uint64_t)global_step),
            gd_metrics_f64("val_loss", (double)val_loss),
            gd_metrics_f64("best_val_loss", best_val_loss != NULL ? (double)*best_val_loss : (double)val_loss),
            gd_metrics_bool("improved", improved),
            gd_metrics_bool("save_best", config->save_best),
        };
        (void)gd_metrics_logger_log_event(metrics, "eval", fields, GD_ARRAY_LEN(fields));
    }
    return true;
}

static void train_gpt(gd_context *ctx,
                      gpt_lm *model,
                      gd_dataset *dataset,
                      gd_dataset *val_dataset,
                      gd_optimizer *optimizer,
                      gd_amp_scaler *scaler,
                      const gd_lr_scheduler_config *lr_config,
                      const gpt_config *config,
                      const gpt_training_state *resume_state,
                      gd_metrics_logger *metrics,
                      size_t steps_per_epoch,
                      size_t total_steps)
{
    double last_report_time = gpt_wall_seconds();
    size_t last_report_step = resume_state != NULL && resume_state->loaded ? resume_state->global_step : 0U;
    size_t global_step = resume_state != NULL && resume_state->loaded ? resume_state->global_step : 0U;
    size_t epochs_without_improvement = resume_state != NULL && resume_state->loaded ?
                                            resume_state->epochs_without_improvement :
                                            0U;
    size_t start_epoch = resume_state != NULL && resume_state->loaded ? resume_state->epoch + 1U : 1U;
    size_t epoch;
    gpt_report_state report;
    gpt_generation_tokenizer generation_tokenizer;
    const gpt_generation_tokenizer *generation_tokenizer_ptr = NULL;
    float best_val_loss = resume_state != NULL && resume_state->loaded ? resume_state->best_val_loss : INFINITY;

    memset(&report, 0, sizeof(report));
    report.last_train_loss = NAN;
    report.last_eval_loss = NAN;
    gd_progress_init(&report.progress, stdout);
    if (!gd_progress_set_row_count(&report.progress, 5U)) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "progress rows", __LINE__);
    }
    gd_progress_rowf(&report.progress, 0U, "train pending");
    gd_progress_rowf(&report.progress, 1U, "train metrics pending");
    gd_progress_rowf(&report.progress, 2U, "eval pending");
    gd_progress_rowf(&report.progress, 3U, "checkpoint pending");
    gpt_report_summary(&report);
    memset(&generation_tokenizer, 0, sizeof(generation_tokenizer));
    if (config->generate_every_n_steps > 0) {
        gpt_generation_tokenizer_init(ctx, config, &generation_tokenizer);
        generation_tokenizer_ptr = &generation_tokenizer;
    }

    if (resume_state != NULL && resume_state->loaded) {
        printf("resume_training: next_epoch=%zu/%d global_step=%zu best_val_loss=%.6f epochs_without_improvement=%zu\n",
               start_epoch,
               config->epochs,
               global_step,
               (double)best_val_loss,
               epochs_without_improvement);
    }
    for (epoch = start_epoch; epoch <= (size_t)config->epochs; ++epoch) {
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
                            generation_tokenizer_ptr,
                            &report,
                            metrics,
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
        {
            bool validated = false;
            bool val_improved = false;
            bool should_stop = false;
            const bool should_eval = val_dataset != NULL && gpt_should_eval_epoch(config, epoch);
            float val_loss = NAN;
            if (should_eval) {
                validated = validate_and_maybe_checkpoint(ctx,
                                                          model,
                                                          optimizer,
                                                          scaler,
                                                          lr_config,
                                                          val_dataset,
                                                          config,
                                                          &report,
                                                          metrics,
                                                          epoch,
                                                          global_step,
                                                          &best_val_loss,
                                                          &val_loss,
                                                          &val_improved);
            }
            if (validated && config->early_stopping_patience > 0) {
                if (val_improved) {
                    epochs_without_improvement = 0U;
                } else {
                    epochs_without_improvement += 1U;
                    if (epochs_without_improvement >= (size_t)config->early_stopping_patience) {
                        should_stop = true;
                    }
                }
            }
            if (validated) {
                report.last_eval_loss = val_loss;
                report.has_eval_loss = true;
                if (val_improved) {
                    report.best_epoch = epoch;
                    report.has_best_epoch = true;
                }
                if (config->early_stopping_patience > 0) {
                    gd_progress_rowf(&report.progress,
                                     2U,
                                     "eval epoch=%zu step=%zu val_loss=%.6f best_val_loss=%.6f improved=%s patience=%zu/%d",
                                     epoch,
                                     global_step,
                                     (double)val_loss,
                                     (double)best_val_loss,
                                     val_improved ? "yes" : "no",
                                     epochs_without_improvement,
                                     config->early_stopping_patience);
                } else {
                    gd_progress_rowf(&report.progress,
                                     2U,
                                     "eval epoch=%zu step=%zu val_loss=%.6f best_val_loss=%.6f improved=%s patience=off",
                                     epoch,
                                     global_step,
                                     (double)val_loss,
                                     (double)best_val_loss,
                                     val_improved ? "yes" : "no");
                }
            } else if (val_dataset != NULL) {
                const size_t next_eval_epoch = gpt_next_eval_epoch(config, epoch);
                gd_progress_rowf(&report.progress,
                                 2U,
                                 "eval skipped epoch=%zu next_eval_epoch=%zu every=%d",
                                 epoch,
                                 next_eval_epoch,
                                 config->eval_every_n_epochs);
            } else {
                gd_progress_rowf(&report.progress, 2U, "eval skipped no validation split loaded");
            }
            if (config->save_latest) {
                const float checkpoint_val_loss = validated ?
                                                    val_loss :
                                                    (report.has_eval_loss ? report.last_eval_loss : NAN);
                const char *best_status = validated ?
                                              (val_improved ?
                                                   (config->save_best ? "saved" : "improved save_best=off") :
                                                   "unchanged") :
                                              (val_dataset != NULL ? "deferred" : "skipped");
                gd_progress_rowf(&report.progress,
                                 3U,
                                 "checkpoint best=%s latest=saving path=%s",
                                 best_status,
                                 config->latest_checkpoint_path);
                gpt_report_summary(&report);
                gd_progress_render(&report.progress);
                gpt_save_checkpoint_bundle(ctx,
                                           model,
                                           optimizer,
                                           scaler,
                                           lr_config,
                                           config,
                                           config->latest_checkpoint_path,
                                           NULL,
                                           epoch,
                                           global_step,
                                           epochs_without_improvement,
                                           checkpoint_val_loss,
                                           best_val_loss);
                gd_progress_rowf(&report.progress,
                                 3U,
                                 "checkpoint best=%s latest=saved path=%s",
                                 best_status,
                                 config->latest_checkpoint_path);
                gpt_report_summary(&report);
                gd_progress_render(&report.progress);
                if (metrics != NULL) {
                    const gd_metrics_field fields[] = {
                        gd_metrics_string("kind", "latest"),
                        gd_metrics_string("path", config->latest_checkpoint_path),
                        gd_metrics_u64("epoch", (uint64_t)epoch),
                        gd_metrics_u64("step", (uint64_t)global_step),
                        gd_metrics_u64("epochs_without_improvement", (uint64_t)epochs_without_improvement),
                        gd_metrics_f64("val_loss", (double)checkpoint_val_loss),
                        gd_metrics_f64("best_val_loss", (double)best_val_loss),
                    };
                    (void)gd_metrics_logger_log_event(metrics, "checkpoint", fields, GD_ARRAY_LEN(fields));
                }
            } else {
                const char *best_status = validated ?
                                              (val_improved ?
                                                   (config->save_best ? "saved" : "improved save_best=off") :
                                                   "unchanged") :
                                              (val_dataset != NULL ? "deferred" : "skipped");
                gd_progress_rowf(&report.progress, 3U, "checkpoint best=%s latest=off", best_status);
                gpt_report_summary(&report);
                gd_progress_render(&report.progress);
            }
            if (should_stop) {
                gd_progress_rowf(&report.progress,
                                 3U,
                                 "early_stopping epoch=%zu step=%zu patience=%d best_val_loss=%.6f",
                                 epoch,
                                 global_step,
                                 config->early_stopping_patience,
                                 (double)best_val_loss);
                gpt_report_summary(&report);
                gd_progress_render(&report.progress);
                if (metrics != NULL) {
                    const gd_metrics_field fields[] = {
                        gd_metrics_u64("epoch", (uint64_t)epoch),
                        gd_metrics_u64("step", (uint64_t)global_step),
                        gd_metrics_i64("patience", (int64_t)config->early_stopping_patience),
                        gd_metrics_f64("best_val_loss", (double)best_val_loss),
                    };
                    (void)gd_metrics_logger_log_event(metrics, "early_stopping", fields, GD_ARRAY_LEN(fields));
                }
                break;
            }
        }
    }
    gd_progress_finish(&report.progress);
    gpt_generation_tokenizer_deinit(&generation_tokenizer);
    gd_progress_deinit(&report.progress);
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
    gd_metrics_logger *metrics = NULL;
    gd_lr_scheduler_config lr_config;
    gpt_training_state resume_state;
    size_t dataset_samples = 0U;
    size_t val_samples = 0U;
    size_t dataset_tokens = 0U;
    size_t val_tokens = 0U;
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
    memset(&resume_state, 0, sizeof(resume_state));

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

    if (config.metrics_enabled) {
        gd_metrics_config metrics_config = gd_metrics_config_default(config.metrics_project);
        metrics_config.root_dir = config.metrics_dir;
        metrics_config.run_id = config.metrics_run_id;
        TRY(ctx, gd_metrics_logger_start(&metrics_config, &metrics));
        printf("metrics: path=%s project=%s run_id=%s\n",
               gd_metrics_logger_path(metrics),
               gd_metrics_logger_project(metrics),
               gd_metrics_logger_run_id(metrics));
    } else {
        printf("metrics: disabled\n");
    }

    if (config.epochs > 0) {
        TRY(ctx, gd_dataset_open_gdds_split(config.data_dir, "train", &dataset));
        dataset_samples = (size_t)gd_dataset_num_samples(dataset);
        if (dataset_samples > SIZE_MAX / (size_t)GPT_CONTEXT_LENGTH) {
            gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "dataset token count overflow", __LINE__);
        }
        dataset_tokens = dataset_samples * (size_t)GPT_CONTEXT_LENGTH;
        steps_per_epoch = effective_steps_per_epoch((uint64_t)dataset_samples, &config, &samples_per_epoch);
        if (steps_per_epoch == 0U || (size_t)config.epochs > SIZE_MAX / steps_per_epoch) {
            gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "dataset too small for requested batch size", __LINE__);
        }
        total_steps = (size_t)config.epochs * steps_per_epoch;
        if (config.save_best || config.early_stopping_patience > 0) {
            TRY(ctx, gd_dataset_open_gdds_split(config.data_dir, config.val_split, &val_dataset));
            val_samples = (size_t)gd_dataset_num_samples(val_dataset);
            if (val_samples > SIZE_MAX / (size_t)GPT_CONTEXT_LENGTH) {
                gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "validation token count overflow", __LINE__);
            }
            val_tokens = val_samples * (size_t)GPT_CONTEXT_LENGTH;
            if (val_samples == 0U) {
                gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty validation split", __LINE__);
            }
        }
    }

    gpt_lm_init(ctx, &model, &config);
    if (config.load_checkpoint_path != NULL || config.resume_checkpoint_path != NULL) {
        const char *load_path = config.resume_checkpoint_path != NULL ?
                                    config.resume_checkpoint_path :
                                    config.load_checkpoint_path;
        gd_module_load_options load_options;
        load_options.strict = true;
        load_options.load_buffers = true;
        TRY(ctx, gd_module_load_state(ctx, &model.mod, load_path, &load_options));
        printf("loaded_checkpoint: %s%s\n",
               load_path,
               config.resume_checkpoint_path != NULL ? " (resume)" : "");
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
        TRY(ctx, gd_amp_scaler_create(ctx, &amp, &scaler));
    }

    lr_config = gd_lr_scheduler_config_default();
    lr_config.max_lr = config.lr_max;
    lr_config.min_lr = config.lr_min;
    lr_config.total_steps = (uint64_t)total_steps;
    if (config.lr_warmup_steps >= 0) {
        lr_config.warmup_steps = (uint64_t)config.lr_warmup_steps;
    } else {
        lr_config.warmup_steps = (uint64_t)(total_steps / 10U);
    }

    if (config.resume_checkpoint_path != NULL && config.epochs > 0) {
        char *optimizer_path = gpt_checkpoint_sidecar_path(config.resume_checkpoint_path, ".optim.gdckpt");
        char *train_state_path = gpt_checkpoint_sidecar_path(config.resume_checkpoint_path, ".train");
        if (optimizer_path == NULL || train_state_path == NULL) {
            free(optimizer_path);
            free(train_state_path);
            gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "resume sidecar path", __LINE__);
        }
        TRY(ctx, gd_optimizer_load_state(ctx, optimizer, optimizer_path, true));
        printf("loaded_optimizer_state: %s\n", optimizer_path);
        gpt_load_training_state(ctx, train_state_path, scaler, &lr_config, &resume_state);
        free(optimizer_path);
        free(train_state_path);
    }

    TRY(ctx, gd_context_seal_params(ctx));

    printf("dataset: dir=%s samples=%zu train_tokens=%zu val_samples=%zu val_tokens=%zu batch=%d context=%d steps_per_epoch=%zu samples_per_epoch=%zu epochs=%d total_steps=%zu%s\n",
           config.data_dir,
           dataset_samples,
           dataset_tokens,
           val_samples,
           val_tokens,
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
    printf("model: vocab=%d d_model=%d layers=%d heads=%d head_dim=%d mlp_hidden=%d ffn=swiglu sdpa_window=%d dropout=%.3f logits_softcap=%.3f\n",
           GPT_VOCAB_SIZE,
           GPT_D_MODEL,
           config.n_layers,
           GPT_N_HEADS,
           GPT_HEAD_DIM,
           GPT_MLP_HIDDEN,
           GPT_SDPA_WINDOW,
           (double)config.dropout_p,
           (double)config.logits_softcap);
    if (config.generate_prompt != NULL || config.generate_every_n_steps > 0) {
        printf("generation: max_new_tokens=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f every_n_steps=%d batched_vowels=%s\n",
               config.max_new_tokens,
               (double)config.temperature,
               (double)config.min_p,
               (double)config.repetition_penalty,
               (double)config.logits_softcap,
               config.generate_every_n_steps,
               config.generate_every_n_steps > 0 ? "yes" : "no");
    }
    if (config.epochs > 0) {
        printf("checkpoint: save_best=%s path=%s save_latest=%s latest_path=%s val_split=%s eval_every_n_epochs=%d early_stopping_patience=%d\n",
               config.save_best ? "yes" : "no",
               config.checkpoint_path,
               config.save_latest ? "yes" : "no",
               config.latest_checkpoint_path,
               config.val_split,
               config.eval_every_n_epochs,
               config.early_stopping_patience);
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
    if (metrics != NULL) {
        const gd_metrics_field fields[] = {
            gd_metrics_string("data_dir", config.data_dir),
            gd_metrics_string("tokenizer_path", config.tokenizer_path),
            gd_metrics_string("checkpoint_path", config.checkpoint_path),
            gd_metrics_string("latest_checkpoint_path", config.latest_checkpoint_path),
            gd_metrics_string("load_checkpoint_path", config.load_checkpoint_path),
            gd_metrics_string("resume_checkpoint_path", config.resume_checkpoint_path),
            gd_metrics_string("val_split", config.val_split),
            gd_metrics_i64("epochs", (int64_t)config.epochs),
            gd_metrics_i64("batch_size", (int64_t)config.batch_size),
            gd_metrics_i64("layers", (int64_t)config.n_layers),
            gd_metrics_i64("context_length", (int64_t)GPT_CONTEXT_LENGTH),
            gd_metrics_i64("vocab_size", (int64_t)GPT_VOCAB_SIZE),
            gd_metrics_i64("d_model", (int64_t)GPT_D_MODEL),
            gd_metrics_i64("heads", (int64_t)GPT_N_HEADS),
            gd_metrics_i64("head_dim", (int64_t)GPT_HEAD_DIM),
            gd_metrics_i64("mlp_hidden", (int64_t)GPT_MLP_HIDDEN),
            gd_metrics_i64("report_every", (int64_t)config.report_every),
            gd_metrics_i64("eval_every_n_epochs", (int64_t)config.eval_every_n_epochs),
            gd_metrics_i64("early_stopping_patience", (int64_t)config.early_stopping_patience),
            gd_metrics_i64("warmup_steps", (int64_t)config.lr_warmup_steps),
            gd_metrics_u64("seed", config.seed),
            gd_metrics_f64("dropout", (double)config.dropout_p),
            gd_metrics_f64("lr_max", (double)config.lr_max),
            gd_metrics_f64("lr_min", (double)config.lr_min),
            gd_metrics_f64("weight_decay", (double)config.weight_decay),
            gd_metrics_f64("grad_clip_norm", (double)config.grad_clip_norm),
            gd_metrics_u64("dataset_samples", (uint64_t)dataset_samples),
            gd_metrics_u64("dataset_tokens", (uint64_t)dataset_tokens),
            gd_metrics_u64("val_samples", (uint64_t)val_samples),
            gd_metrics_u64("val_tokens", (uint64_t)val_tokens),
            gd_metrics_u64("steps_per_epoch", (uint64_t)steps_per_epoch),
            gd_metrics_u64("samples_per_epoch", (uint64_t)samples_per_epoch),
            gd_metrics_u64("total_steps", (uint64_t)total_steps),
            gd_metrics_u64("total_params", total_params),
            gd_metrics_u64("trainable_params", trainable_params),
            gd_metrics_u64("param_bytes", param_bytes),
            gd_metrics_u64("memory_params_bytes", (uint64_t)mem.params_bytes),
            gd_metrics_u64("memory_state_bytes", (uint64_t)mem.state_bytes),
            gd_metrics_u64("memory_scratch_slot_bytes", (uint64_t)mem.scratch_slot_bytes),
            gd_metrics_u64("memory_data_slot_bytes", (uint64_t)mem.data_slot_bytes),
            gd_metrics_bool("save_best", config.save_best),
            gd_metrics_bool("save_latest", config.save_latest),
        };
        (void)gd_metrics_logger_log_event(metrics, "run_config", fields, GD_ARRAY_LEN(fields));
    }

    if (config.epochs > 0) {
        train_gpt(ctx,
                  &model,
                  dataset,
                  val_dataset,
                  optimizer,
                  scaler,
                  &lr_config,
                  &config,
                  &resume_state,
                  metrics,
                  steps_per_epoch,
                  total_steps);
        TRY(ctx, gd_optimizer_sync_state(ctx, optimizer));
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
        if (metrics != NULL) {
            const gd_metrics_field fields[] = {
                gd_metrics_u64("params_watermark_bytes", (uint64_t)stats.params.watermark),
                gd_metrics_u64("state_watermark_bytes", (uint64_t)stats.state.watermark),
                gd_metrics_u64("scratch_max_slot_watermark_bytes",
                               (uint64_t)stats.scratch.max_slot_watermark),
                gd_metrics_u64("data_max_slot_watermark_bytes", (uint64_t)stats.data.max_slot_watermark),
                gd_metrics_u64("backend_waits", stats.backend_waits),
            };
            (void)gd_metrics_logger_log_event(metrics, "memory_watermark", fields, GD_ARRAY_LEN(fields));
        }
    }
    if (metrics != NULL) {
        const gd_metrics_field fields[] = {
            gd_metrics_u64("optimizer_steps", (uint64_t)optimizer_steps),
            gd_metrics_u64("metrics_dropped", gd_metrics_logger_dropped(metrics)),
        };
        (void)gd_metrics_logger_log_event(metrics, "run_end", fields, GD_ARRAY_LEN(fields));
    }
    printf("gpt_lm: ok optimizer_steps=%zu\n", optimizer_steps);
    exit_code = 0;

    gd_metrics_logger_stop(metrics);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(optimizer);
    gd_param_set_free(&params);
    gpt_lm_deinit(&model);
    gd_dataset_destroy(val_dataset);
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    return exit_code;
}
