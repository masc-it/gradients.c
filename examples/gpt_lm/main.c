#include "gpt_lm_shared.h"
#include "gd_progress.h"
#include "gd_example_config.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

static char *gpt_vasprintf(const char *fmt, va_list ap)
{
    va_list ap_copy;
    char *out;
    int len;
    int written;
    if (fmt == NULL) {
        return NULL;
    }
    va_copy(ap_copy, ap);
    len = vsnprintf(NULL, 0U, fmt, ap_copy);
    va_end(ap_copy);
    if (len < 0) {
        return NULL;
    }
    out = (char *)malloc((size_t)len + 1U);
    if (out == NULL) {
        return NULL;
    }
    written = vsnprintf(out, (size_t)len + 1U, fmt, ap);
    if (written < 0 || written > len) {
        free(out);
        return NULL;
    }
    return out;
}

static char *gpt_asprintf(const char *fmt, ...)
{
    va_list ap;
    char *out;
    va_start(ap, fmt);
    out = gpt_vasprintf(fmt, ap);
    va_end(ap);
    return out;
}

static const char *gpt_architecture_name(gpt_architecture architecture)
{
    return architecture == GPT_ARCH_MINIMAX_M3 ? "minimax_m3" : "gpt";
}

static int parse_architecture_arg(const char *text, gpt_architecture *out)
{
    if (text == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(text, "gpt") == 0 || strcmp(text, "dense") == 0) {
        *out = GPT_ARCH_GPT;
        return 1;
    }
    if (strcmp(text, "minimax_m3") == 0 || strcmp(text, "minimax") == 0 ||
        strcmp(text, "m3") == 0) {
        *out = GPT_ARCH_MINIMAX_M3;
        return 1;
    }
    return 0;
}

static const char *gpt_shuffle_scope_name(gpt_shuffle_scope scope)
{
    switch (scope) {
    case GPT_SHUFFLE_SHARD: return "shard";
    case GPT_SHUFFLE_NONE: return "none";
    case GPT_SHUFFLE_GLOBAL:
    default: return "global";
    }
}

static int parse_shuffle_scope_arg(const char *text, gpt_shuffle_scope *out)
{
    if (text == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(text, "global") == 0 || strcmp(text, "all") == 0) {
        *out = GPT_SHUFFLE_GLOBAL;
        return 1;
    }
    if (strcmp(text, "shard") == 0 || strcmp(text, "intra-shard") == 0 ||
        strcmp(text, "intra_shard") == 0) {
        *out = GPT_SHUFFLE_SHARD;
        return 1;
    }
    if (strcmp(text, "none") == 0 || strcmp(text, "sequential") == 0 || strcmp(text, "off") == 0) {
        *out = GPT_SHUFFLE_NONE;
        return 1;
    }
    return 0;
}

static void print_usage(const char *argv0)
{
    printf("usage: %s --config PATH\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --config, -c PATH   YAML configuration file\n");
    printf("  --help, -h          show this help\n");
}

static const char *parse_config_path(int argc, char **argv)
{
    const char *config_path = NULL;
    int i;
    for (i = 1; i < argc; ++i) {
        const char *value;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        value = arg_value(argc, argv, &i, "--config");
        if (value == NULL && strcmp(argv[i], "-c") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "gpt_lm: missing value for -c\n");
                exit(2);
            }
            ++i;
            value = argv[i];
        }
        if (value == NULL && argv[i][0] != '-') {
            value = argv[i];
        }
        if (value != NULL) {
            if (config_path != NULL) {
                fprintf(stderr, "gpt_lm: --config specified more than once\n");
                exit(2);
            }
            config_path = value;
            continue;
        }
        fprintf(stderr, "gpt_lm: unknown argument %s\n", argv[i]);
        print_usage(argv[0]);
        exit(2);
    }
    if (config_path == NULL || config_path[0] == '\0') {
        fprintf(stderr, "gpt_lm: --config PATH is required\n");
        print_usage(argv[0]);
        exit(2);
    }
    return config_path;
}

static void gpt_config_set_error(gd_example_config_error *error,
                                 unsigned line,
                                 const char *fmt,
                                 ...)
{
    va_list ap;
    if (error == NULL) {
        return;
    }
    error->line = line;
    va_start(ap, fmt);
    (void)vsnprintf(error->message, sizeof(error->message), fmt, ap);
    va_end(ap);
}

static void gpt_config_die(const char *path, const gd_example_config_error *error)
{
    if (error != NULL && error->line > 0U) {
        fprintf(stderr,
                "gpt_lm: invalid config %s:%u: %s\n",
                path != NULL ? path : "(null)",
                error->line,
                gd_example_config_error_message(error));
    } else {
        fprintf(stderr,
                "gpt_lm: invalid config %s: %s\n",
                path != NULL ? path : "(null)",
                gd_example_config_error_message(error));
    }
    exit(2);
}

static void gpt_config_deinit(gpt_config *config)
{
    if (config == NULL) {
        return;
    }
    free((void *)config->data_dir);
    free((void *)config->tokenizer_path);
    free((void *)config->generate_prompt);
    free((void *)config->checkpoint_path);
    free((void *)config->latest_checkpoint_path);
    free((void *)config->load_checkpoint_path);
    free((void *)config->resume_checkpoint_path);
    free((void *)config->val_split);
    free((void *)config->metrics_dir);
    free((void *)config->metrics_project);
    free((void *)config->metrics_run_id);
    free((void *)config->local_shard_cache_dir);
    config->data_dir = NULL;
    config->tokenizer_path = NULL;
    config->generate_prompt = NULL;
    config->checkpoint_path = NULL;
    config->latest_checkpoint_path = NULL;
    config->load_checkpoint_path = NULL;
    config->resume_checkpoint_path = NULL;
    config->val_split = NULL;
    config->metrics_dir = NULL;
    config->metrics_project = NULL;
    config->metrics_run_id = NULL;
    config->local_shard_cache_dir = NULL;
}

static void gpt_config_invalid(gpt_config *config, const char *message)
{
    fprintf(stderr,
            "gpt_lm: invalid config %s: %s\n",
            config->config_path != NULL ? config->config_path : "(null)",
            message != NULL ? message : "invalid value");
    gpt_config_deinit(config);
    exit(2);
}

static char *gpt_strdup(const char *text)
{
    const size_t len = text != NULL ? strlen(text) : 0U;
    char *out;
    if (text == NULL || len > SIZE_MAX - 1U) {
        return NULL;
    }
    out = (char *)malloc(len + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, text, len + 1U);
    return out;
}

static char *gpt_config_dup_string(gpt_config *config,
                                   const char *key,
                                   const char *value,
                                   bool empty_means_null)
{
    char *copy;
    char message[256];
    if (empty_means_null && (value == NULL || value[0] == '\0')) {
        return NULL;
    }
    copy = gpt_strdup(value);
    if (copy == NULL) {
        (void)snprintf(message,
                       sizeof(message),
                       "out of memory while storing string key '%s'",
                       key != NULL ? key : "(unknown)");
        gpt_config_invalid(config, message);
    }
    return copy;
}

static bool gpt_config_string_empty(const char *text)
{
    return text == NULL || text[0] == '\0';
}

static void gpt_config_finalize(gpt_config *config)
{
    const size_t dataloader_slots = (size_t)config->dataloader_workers *
                                    (size_t)config->dataloader_prefetch_factor;
    if (gpt_config_string_empty(config->data_dir)) {
        gpt_config_invalid(config, "data_dir must not be empty");
    }
    if (gpt_config_string_empty(config->checkpoint_path)) {
        gpt_config_invalid(config, "checkpoint_path must not be empty");
    }
    if (gpt_config_string_empty(config->latest_checkpoint_path)) {
        gpt_config_invalid(config, "latest_checkpoint_path must not be empty");
    }
    if (gpt_config_string_empty(config->val_split)) {
        gpt_config_invalid(config, "val_split must not be empty");
    }
    if (gpt_config_string_empty(config->metrics_dir)) {
        gpt_config_invalid(config, "metrics_dir must not be empty");
    }
    if (gpt_config_string_empty(config->metrics_project)) {
        gpt_config_invalid(config, "metrics_project must not be empty");
    }
    if (config->epochs == 0 && config->generate_prompt == NULL) {
        gpt_config_invalid(config, "epochs: 0 requires a non-empty generate_prompt");
    }
    if (dataloader_slots > (size_t)GPT_MAX_DATALOADER_SLOTS) {
        gpt_config_invalid(config,
                           "dataloader_workers * dataloader_prefetch_factor exceeds the data slot limit");
    }
    if (config->lr_min > config->lr_max) {
        gpt_config_invalid(config, "lr_min must be <= lr_max");
    }
    if (config->local_shard_cache_dir != NULL && config->shuffle_scope != GPT_SHUFFLE_SHARD) {
        gpt_config_invalid(config, "local_shard_cache_dir requires shuffle_scope: shard");
    }
    if (config->load_checkpoint_path != NULL && config->resume_checkpoint_path != NULL) {
        gpt_config_invalid(config,
                           "use either load_checkpoint_path or resume_checkpoint_path, not both");
    }
}

static int gpt_config_require_architecture(const gd_example_config_doc *doc,
                                           gpt_architecture *out,
                                           gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    const char *text = NULL;
    if (!gd_example_config_require_string(doc, "architecture", &text, error)) {
        return 0;
    }
    if (parse_architecture_arg(text, out)) {
        return 1;
    }
    entry = gd_example_config_find(doc, "architecture");
    gpt_config_set_error(error,
                         entry != NULL ? entry->line : 0U,
                         "key 'architecture' must be 'gpt' or 'minimax_m3'; got '%s'",
                         text);
    return 0;
}

static int gpt_config_require_shuffle_scope(const gd_example_config_doc *doc,
                                            gpt_shuffle_scope *out,
                                            gd_example_config_error *error)
{
    const gd_example_config_entry *entry;
    const char *text = NULL;
    if (!gd_example_config_require_string(doc, "shuffle_scope", &text, error)) {
        return 0;
    }
    if (parse_shuffle_scope_arg(text, out)) {
        return 1;
    }
    entry = gd_example_config_find(doc, "shuffle_scope");
    gpt_config_set_error(error,
                         entry != NULL ? entry->line : 0U,
                         "key 'shuffle_scope' must be 'global', 'shard', or 'none'; got '%s'",
                         text);
    return 0;
}

static gpt_config gpt_config_from_yaml(const char *path)
{
    static const char *const known_keys[] = {
        "data_dir",
        "tokenizer_path",
        "generate_prompt",
        "checkpoint_path",
        "latest_checkpoint_path",
        "load_checkpoint_path",
        "resume_checkpoint_path",
        "val_split",
        "metrics_dir",
        "metrics_project",
        "metrics_run_id",
        "local_shard_cache_dir",
        "epochs",
        "batch_size",
        "dataloader_workers",
        "dataloader_prefetch_factor",
        "layers",
        "architecture",
        "shuffle_scope",
        "minimax_m3_topk_blocks",
        "minimax_m3_init_blocks",
        "minimax_m3_local_blocks",
        "report_every",
        "eval_every_n_epochs",
        "early_stopping_patience",
        "warmup_steps",
        "max_new_tokens",
        "latest_every_n_steps",
        "generate_every_n_steps",
        "save_best",
        "save_latest",
        "metrics_enabled",
        "keep_shard_cache",
        "overfit_num_samples",
        "seed",
        "dropout_p",
        "lr_max",
        "lr_min",
        "weight_decay",
        "grad_clip_norm",
        "temperature",
        "min_p",
        "repetition_penalty",
        "logits_softcap",
    };
    gd_example_config_doc doc;
    gd_example_config_error error;
    gpt_config config;
    uint64_t parsed_u64 = 0U;
    const char *data_dir = NULL;
    const char *tokenizer_path = NULL;
    const char *generate_prompt = NULL;
    const char *checkpoint_path = NULL;
    const char *latest_checkpoint_path = NULL;
    const char *load_checkpoint_path = NULL;
    const char *resume_checkpoint_path = NULL;
    const char *val_split = NULL;
    const char *metrics_dir = NULL;
    const char *metrics_project = NULL;
    const char *metrics_run_id = NULL;
    const char *local_shard_cache_dir = NULL;
    int ok;

    memset(&config, 0, sizeof(config));
    config.config_path = path;
    config.epochs_set = true;
    config.pad_token_id = -1;

    ok = gd_example_config_load_yaml_file(path, &doc, &error) &&
         gd_example_config_validate_keys(&doc, known_keys, GD_ARRAY_LEN(known_keys), &error) &&
         gd_example_config_require_string(&doc, "data_dir", &data_dir, &error) &&
         gd_example_config_require_string_allow_empty(&doc, "tokenizer_path", &tokenizer_path, &error) &&
         gd_example_config_require_string_allow_empty(&doc, "generate_prompt", &generate_prompt, &error) &&
         gd_example_config_require_string(&doc, "checkpoint_path", &checkpoint_path, &error) &&
         gd_example_config_require_string(&doc,
                                          "latest_checkpoint_path",
                                          &latest_checkpoint_path,
                                          &error) &&
         gd_example_config_require_string_allow_empty(&doc,
                                                      "load_checkpoint_path",
                                                      &load_checkpoint_path,
                                                      &error) &&
         gd_example_config_require_string_allow_empty(&doc,
                                                      "resume_checkpoint_path",
                                                      &resume_checkpoint_path,
                                                      &error) &&
         gd_example_config_require_string(&doc, "val_split", &val_split, &error) &&
         gd_example_config_require_string(&doc, "metrics_dir", &metrics_dir, &error) &&
         gd_example_config_require_string(&doc, "metrics_project", &metrics_project, &error) &&
         gd_example_config_require_string_allow_empty(&doc, "metrics_run_id", &metrics_run_id, &error) &&
         gd_example_config_require_string_allow_empty(&doc,
                                                      "local_shard_cache_dir",
                                                      &local_shard_cache_dir,
                                                      &error) &&
         gd_example_config_require_int(&doc, "epochs", 0, 1000000, &config.epochs, &error) &&
         gd_example_config_require_int(&doc, "batch_size", 1, 1024, &config.batch_size, &error) &&
         gd_example_config_require_int(&doc,
                                       "dataloader_workers",
                                       1,
                                       64,
                                       &config.dataloader_workers,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "dataloader_prefetch_factor",
                                       1,
                                       16,
                                       &config.dataloader_prefetch_factor,
                                       &error) &&
         gd_example_config_require_int(&doc, "layers", 1, 96, &config.n_layers, &error) &&
         gpt_config_require_architecture(&doc, &config.architecture, &error) &&
         gpt_config_require_shuffle_scope(&doc, &config.shuffle_scope, &error) &&
         gd_example_config_require_int(&doc,
                                       "minimax_m3_topk_blocks",
                                       1,
                                       16,
                                       &config.minimax_m3_topk_blocks,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "minimax_m3_init_blocks",
                                       0,
                                       16,
                                       &config.minimax_m3_init_blocks,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "minimax_m3_local_blocks",
                                       0,
                                       16,
                                       &config.minimax_m3_local_blocks,
                                       &error) &&
         gd_example_config_require_int(&doc, "report_every", 0, 1000000000, &config.report_every, &error) &&
         gd_example_config_require_int(&doc,
                                       "eval_every_n_epochs",
                                       1,
                                       1000000,
                                       &config.eval_every_n_epochs,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "early_stopping_patience",
                                       0,
                                       1000000,
                                       &config.early_stopping_patience,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "warmup_steps",
                                       -1,
                                       1000000000,
                                       &config.lr_warmup_steps,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "max_new_tokens",
                                       1,
                                       GPT_CONTEXT_LENGTH,
                                       &config.max_new_tokens,
                                       &error) &&
         gd_example_config_require_u64(&doc,
                                       "latest_every_n_steps",
                                       (uint64_t)SIZE_MAX,
                                       &parsed_u64,
                                       &error) &&
         gd_example_config_require_int(&doc,
                                       "generate_every_n_steps",
                                       0,
                                       1000000000,
                                       &config.generate_every_n_steps,
                                       &error) &&
         gd_example_config_require_bool(&doc, "save_best", &config.save_best, &error) &&
         gd_example_config_require_bool(&doc, "save_latest", &config.save_latest, &error) &&
         gd_example_config_require_bool(&doc, "metrics_enabled", &config.metrics_enabled, &error) &&
         gd_example_config_require_bool(&doc, "keep_shard_cache", &config.keep_shard_cache, &error) &&
         gd_example_config_require_u64(&doc,
                                       "overfit_num_samples",
                                       UINT64_MAX,
                                       &config.overfit_num_samples,
                                       &error) &&
         gd_example_config_require_u64(&doc, "seed", UINT64_MAX, &config.seed, &error) &&
         gd_example_config_require_f32(&doc, "dropout_p", 0.0f, 0.95f, &config.dropout_p, &error) &&
         gd_example_config_require_f32(&doc, "lr_max", 0.0f, 10.0f, &config.lr_max, &error) &&
         gd_example_config_require_f32(&doc, "lr_min", 0.0f, 10.0f, &config.lr_min, &error) &&
         gd_example_config_require_f32(&doc,
                                       "weight_decay",
                                       0.0f,
                                       10.0f,
                                       &config.weight_decay,
                                       &error) &&
         gd_example_config_require_f32(&doc,
                                       "grad_clip_norm",
                                       0.0f,
                                       1000000.0f,
                                       &config.grad_clip_norm,
                                       &error) &&
         gd_example_config_require_f32(&doc,
                                       "temperature",
                                       0.0f,
                                       10.0f,
                                       &config.temperature,
                                       &error) &&
         gd_example_config_require_f32(&doc, "min_p", 0.0f, 1.0f, &config.min_p, &error) &&
         gd_example_config_require_f32(&doc,
                                       "repetition_penalty",
                                       1.0f,
                                       10.0f,
                                       &config.repetition_penalty,
                                       &error) &&
         gd_example_config_require_f32(&doc,
                                       "logits_softcap",
                                       0.0f,
                                       1000000.0f,
                                       &config.logits_softcap,
                                       &error);
    if (!ok) {
        gd_example_config_doc_free(&doc);
        gpt_config_die(path, &error);
    }

    config.latest_every_n_steps = (size_t)parsed_u64;
    config.data_dir = gpt_config_dup_string(&config, "data_dir", data_dir, false);
    config.tokenizer_path = gpt_config_dup_string(&config, "tokenizer_path", tokenizer_path, true);
    config.generate_prompt = gpt_config_dup_string(&config, "generate_prompt", generate_prompt, true);
    config.checkpoint_path = gpt_config_dup_string(&config, "checkpoint_path", checkpoint_path, false);
    config.latest_checkpoint_path = gpt_config_dup_string(&config,
                                                          "latest_checkpoint_path",
                                                          latest_checkpoint_path,
                                                          false);
    config.load_checkpoint_path = gpt_config_dup_string(&config,
                                                        "load_checkpoint_path",
                                                        load_checkpoint_path,
                                                        true);
    config.resume_checkpoint_path = gpt_config_dup_string(&config,
                                                          "resume_checkpoint_path",
                                                          resume_checkpoint_path,
                                                          true);
    config.val_split = gpt_config_dup_string(&config, "val_split", val_split, false);
    config.metrics_dir = gpt_config_dup_string(&config, "metrics_dir", metrics_dir, false);
    config.metrics_project = gpt_config_dup_string(&config, "metrics_project", metrics_project, false);
    config.metrics_run_id = gpt_config_dup_string(&config, "metrics_run_id", metrics_run_id, true);
    config.local_shard_cache_dir = gpt_config_dup_string(&config,
                                                         "local_shard_cache_dir",
                                                         local_shard_cache_dir,
                                                         true);
    gd_example_config_doc_free(&doc);
    gpt_config_finalize(&config);
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
        printf("  %-48s lr_mult=%.9g weight_decay=%.9g trainable=%s\n",
               params->items[i].path,
               (double)params->items[i].lr_mult,
               (double)params->items[i].weight_decay,
               params->items[i].trainable ? "yes" : "no");
    }
}

#define GPT_TRAIN_STATE_EPOCH_END SIZE_MAX
#define GPT_TRAIN_SHUFFLE_SEED_BASE UINT64_C(0x51504c)
#define GPT_TRAIN_SHARD_SEED_STRIDE UINT64_C(0x9e3779b97f4a7c15)

typedef enum gpt_compact_source_field {
    GPT_COMPACT_SRC_TOKENS = 0,
    GPT_COMPACT_SRC_SEGMENT_LENGTHS = 1,
    GPT_COMPACT_SRC_FIELD_COUNT = 2,
} gpt_compact_source_field;

typedef enum gpt_runtime_field {
    GPT_FIELD_INPUT_IDS = 0,
    GPT_FIELD_POSITIONS = 1,
    GPT_FIELD_TARGET_IDS = 2,
    GPT_FIELD_SEGMENT_LENGTHS = 3,
    GPT_FIELD_CU_SEQLENS = 4,
    GPT_FIELD_COUNT = 5,
} gpt_runtime_field;

typedef enum gpt_progress_row {
    GPT_PROGRESS_TRAIN = 0,
    GPT_PROGRESS_TRAIN_METRICS = 1,
    GPT_PROGRESS_EVAL = 2,
    GPT_PROGRESS_CHECKPOINT = 3,
    GPT_PROGRESS_SUMMARY = 4,
    GPT_PROGRESS_COUNT = 5,
} gpt_progress_row;

typedef enum gpt_checkpoint_kind {
    GPT_CHECKPOINT_BEST = 0,
    GPT_CHECKPOINT_LATEST = 1,
    GPT_CHECKPOINT_LATEST_STEP = 2,
} gpt_checkpoint_kind;

typedef struct gpt_lm_special_ids {
    int32_t pad_id;
    int32_t im_start_id;
    int32_t im_end_id;
} gpt_lm_special_ids;

typedef struct gpt_lm_compact_transform {
    int32_t context_length;
    int32_t record_length;
    int32_t vocab_size;
    int32_t pad_id;
    int32_t im_start_id;
    int32_t im_end_id;
} gpt_lm_compact_transform;

static const gd_dataset_field_spec gpt_lm_compact_runtime_fields[GPT_FIELD_COUNT] = {
    [GPT_FIELD_INPUT_IDS] = {
        "input_ids",
        GD_DTYPE_I32,
        1,
        {-1, 0, 0, 0, 0, 0, 0, 0},
        GD_GDDS_COLLATE_PACKED_SEQUENCE,
        GD_GDDS_GENERATED_NONE,
        0,
        -1,
        0U,
    },
    [GPT_FIELD_POSITIONS] = {
        "positions",
        GD_DTYPE_I32,
        1,
        {-1, 0, 0, 0, 0, 0, 0, 0},
        GD_GDDS_COLLATE_PACKED_SEQUENCE,
        GD_GDDS_GENERATED_NONE,
        0,
        -1,
        0U,
    },
    [GPT_FIELD_TARGET_IDS] = {
        "target_ids",
        GD_DTYPE_I32,
        1,
        {-1, 0, 0, 0, 0, 0, 0, 0},
        GD_GDDS_COLLATE_PACKED_SEQUENCE,
        GD_GDDS_GENERATED_NONE,
        0,
        -1,
        0U,
    },
    [GPT_FIELD_SEGMENT_LENGTHS] = {
        "segment_lengths",
        GD_DTYPE_I32,
        1,
        {-1, 0, 0, 0, 0, 0, 0, 0},
        GD_GDDS_COLLATE_PACKED_SEQUENCE,
        GD_GDDS_GENERATED_NONE,
        0,
        -1,
        0U,
    },
    [GPT_FIELD_CU_SEQLENS] = {
        "cu_seqlens",
        GD_DTYPE_I32,
        1,
        {-1, 0, 0, 0, 0, 0, 0, 0},
        GD_GDDS_COLLATE_GENERATED,
        GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS,
        -1,
        GPT_FIELD_SEGMENT_LENGTHS,
        0U,
    },
};

static gd_status gpt_lm_resolve_special_ids(gd_context *ctx,
                                             const gpt_config *config,
                                             gpt_lm_special_ids *out)
{
    gd_tokenizer_config tok_cfg;
    gd_tokenizer *tok = NULL;
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    gd_status st;
    if (ctx == NULL || config == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out->pad_id = -1;
    out->im_start_id = -1;
    out->im_end_id = -1;
    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_bpe_tokenizer_load(tokenizer_path, &tok_cfg, &tok);
    if (st != GD_OK) {
        fprintf(stderr,
                "gpt_lm: failed to load tokenizer '%s' while resolving special ids; "
                "ensure config data_dir points at the rebuilt dataset and rebuild with "
                "GPT_LM_VOCAB_SIZE matching manifest.json (or set tokenizer_path).\n",
                tokenizer_path);
        free(default_tok_path);
        (void)ctx;
        return st;
    }
    free(default_tok_path);
    (void)gd_tokenizer_id(tok, "<|pad|>", &out->pad_id);
    (void)gd_tokenizer_id(tok, "<|im_start|>", &out->im_start_id);
    (void)gd_tokenizer_id(tok, "<|im_end|>", &out->im_end_id);
    gd_tokenizer_destroy(tok);
    return GD_OK;
}

static int gpt_lm_dataset_is_compact(gd_dataset *dataset)
{
    gd_gdds_field_info tokens;
    gd_gdds_field_info segments;
    int token_index;
    int segment_index;
    if (dataset == NULL) {
        return 0;
    }
    token_index = gd_gdds_dataset_field_index(dataset, "tokens");
    segment_index = gd_gdds_dataset_field_index(dataset, "segment_lengths");
    if (token_index != GPT_COMPACT_SRC_TOKENS || segment_index != GPT_COMPACT_SRC_SEGMENT_LENGTHS ||
        gd_gdds_dataset_field_info(dataset, token_index, &tokens) != GD_OK ||
        gd_gdds_dataset_field_info(dataset, segment_index, &segments) != GD_OK) {
        return 0;
    }
    return tokens.dtype == GD_DTYPE_U16 && tokens.rank == 1 &&
           tokens.collate == GD_GDDS_COLLATE_STACK &&
           tokens.shape[0] == (int64_t)GPT_CONTEXT_LENGTH + 1 &&
           segments.dtype == GD_DTYPE_I32 && segments.rank == 1 &&
           segments.collate == GD_GDDS_COLLATE_PACKED_SEQUENCE &&
           segments.shape[0] == -1;
}

static uint16_t gpt_lm_load_le_u16(const void *src)
{
    const uint8_t *p = (const uint8_t *)src;
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8U));
}

static int32_t gpt_lm_load_le_i32(const void *src)
{
    const uint8_t *p = (const uint8_t *)src;
    uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8U) |
                 ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
    int32_t out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

static gd_status gpt_lm_compact_transform_fn(const gd_sample *src,
                                             gd_sample *dst,
                                             void *user_data)
{
    const gpt_lm_compact_transform *cfg = (const gpt_lm_compact_transform *)user_data;
    const uint8_t *token_bytes;
    const uint8_t *segment_bytes;
    int32_t *input_ids;
    int32_t *positions;
    int32_t *target_ids;
    int32_t *dst_segments;
    int64_t context_shape[1];
    int64_t segments_shape[1];
    int64_t n_segments_i64;
    int32_t offset = 0;
    int64_t s;
    int32_t i;
    gd_status st;
    if (cfg == NULL || src == NULL || dst == NULL || cfg->context_length <= 0 ||
        cfg->record_length != cfg->context_length + 1 || cfg->vocab_size <= 0 ||
        cfg->pad_id < 0 || cfg->pad_id >= cfg->vocab_size || cfg->im_start_id < 0 ||
        cfg->im_end_id < 0 || gd_sample_field_count(src) < GPT_COMPACT_SRC_FIELD_COUNT ||
        gd_sample_field_count(dst) < GPT_FIELD_COUNT) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (gd_sample_field_dtype(src, GPT_COMPACT_SRC_TOKENS) != GD_DTYPE_U16 ||
        gd_sample_field_rank(src, GPT_COMPACT_SRC_TOKENS) != 1 ||
        gd_sample_field_dim(src, GPT_COMPACT_SRC_TOKENS, 0) != cfg->record_length ||
        gd_sample_field_nbytes(src, GPT_COMPACT_SRC_TOKENS) !=
            (size_t)cfg->record_length * sizeof(uint16_t) ||
        gd_sample_field_dtype(src, GPT_COMPACT_SRC_SEGMENT_LENGTHS) != GD_DTYPE_I32 ||
        gd_sample_field_rank(src, GPT_COMPACT_SRC_SEGMENT_LENGTHS) != 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    token_bytes = (const uint8_t *)gd_sample_field_data(src, GPT_COMPACT_SRC_TOKENS);
    segment_bytes = (const uint8_t *)gd_sample_field_data(src, GPT_COMPACT_SRC_SEGMENT_LENGTHS);
    n_segments_i64 = gd_sample_field_dim(src, GPT_COMPACT_SRC_SEGMENT_LENGTHS, 0);
    if (token_bytes == NULL || segment_bytes == NULL || n_segments_i64 <= 0 ||
        n_segments_i64 > (int64_t)cfg->context_length ||
        gd_sample_field_nbytes(src, GPT_COMPACT_SRC_SEGMENT_LENGTHS) !=
            (size_t)n_segments_i64 * sizeof(int32_t)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    context_shape[0] = cfg->context_length;
    segments_shape[0] = n_segments_i64;
    st = gd_sample_resize_field(dst, GPT_FIELD_INPUT_IDS, GD_DTYPE_I32, 1, context_shape);
    if (st != GD_OK) { return st; }
    st = gd_sample_resize_field(dst, GPT_FIELD_POSITIONS, GD_DTYPE_I32, 1, context_shape);
    if (st != GD_OK) { return st; }
    st = gd_sample_resize_field(dst, GPT_FIELD_TARGET_IDS, GD_DTYPE_I32, 1, context_shape);
    if (st != GD_OK) { return st; }
    st = gd_sample_resize_field(dst, GPT_FIELD_SEGMENT_LENGTHS, GD_DTYPE_I32, 1, segments_shape);
    if (st != GD_OK) { return st; }
    input_ids = (int32_t *)gd_sample_mutable_field_data(dst, GPT_FIELD_INPUT_IDS);
    positions = (int32_t *)gd_sample_mutable_field_data(dst, GPT_FIELD_POSITIONS);
    target_ids = (int32_t *)gd_sample_mutable_field_data(dst, GPT_FIELD_TARGET_IDS);
    dst_segments = (int32_t *)gd_sample_mutable_field_data(dst, GPT_FIELD_SEGMENT_LENGTHS);
    if (input_ids == NULL || positions == NULL || target_ids == NULL || dst_segments == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < cfg->context_length; ++i) {
        const int32_t token = (int32_t)gpt_lm_load_le_u16(token_bytes + (size_t)i * sizeof(uint16_t));
        const int32_t next = (int32_t)gpt_lm_load_le_u16(token_bytes + (size_t)(i + 1) * sizeof(uint16_t));
        if (token >= cfg->vocab_size || next >= cfg->vocab_size) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        input_ids[i] = token;
        target_ids[i] = next;
    }
    for (s = 0; s < n_segments_i64; ++s) {
        const int32_t len = gpt_lm_load_le_i32(segment_bytes + (size_t)s * sizeof(int32_t));
        int32_t j;
        if (len <= 0 || len > cfg->context_length - offset) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        dst_segments[s] = len;
        for (j = 0; j < len; ++j) {
            positions[offset + j] = j;
        }
        offset += len;
        if (offset < cfg->context_length) {
            target_ids[offset - 1] = cfg->pad_id;
        }
    }
    if (offset != cfg->context_length) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (input_ids[cfg->context_length - 1] == cfg->im_end_id &&
        target_ids[cfg->context_length - 1] == cfg->im_start_id) {
        target_ids[cfg->context_length - 1] = cfg->pad_id;
    }
    return GD_OK;
}

typedef enum gpt_gdds_source_kind {
    GPT_GDDS_SOURCE_SPLIT = 0,
    GPT_GDDS_SOURCE_FILE = 1,
} gpt_gdds_source_kind;

static gd_status gpt_lm_init_compact_transform(const gpt_lm_special_ids *special_ids,
                                               gpt_lm_compact_transform *transform)
{
    if (special_ids == NULL || transform == NULL || special_ids->pad_id < 0 ||
        special_ids->im_start_id < 0 || special_ids->im_end_id < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    transform->context_length = GPT_CONTEXT_LENGTH;
    transform->record_length = GPT_CONTEXT_LENGTH + 1;
    transform->vocab_size = GPT_VOCAB_SIZE;
    transform->pad_id = special_ids->pad_id;
    transform->im_start_id = special_ids->im_start_id;
    transform->im_end_id = special_ids->im_end_id;
    return GD_OK;
}

static gd_status gpt_lm_open_gdds_source(gd_context *ctx,
                                         const gpt_config *config,
                                         const char *name,
                                         gpt_gdds_source_kind kind,
                                         const gpt_lm_special_ids *special_ids,
                                         gpt_lm_compact_transform *transform,
                                         gd_dataset **out)
{
    gd_dataset *raw = NULL;
    gd_status st;
    if (ctx == NULL || name == NULL || transform == NULL || out == NULL ||
        (kind == GPT_GDDS_SOURCE_SPLIT && config == NULL)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (kind == GPT_GDDS_SOURCE_SPLIT) {
        st = gd_dataset_open_gdds_split(config->data_dir, name, &raw);
    } else {
        st = gd_dataset_open_gdds_file(name, &raw);
    }
    if (st != GD_OK) {
        return st;
    }
    if (gpt_lm_dataset_is_compact(raw) == 0) {
        *out = raw;
        return GD_OK;
    }
    gd_dataset_destroy(raw);
    st = gpt_lm_init_compact_transform(special_ids, transform);
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_dataset_transform_config transform_cfg = {
            .transform = gpt_lm_compact_transform_fn,
            .user_data = transform,
            .output_fields = gpt_lm_compact_runtime_fields,
            .n_output_fields = (int)GD_ARRAY_LEN(gpt_lm_compact_runtime_fields),
        };
        if (kind == GPT_GDDS_SOURCE_SPLIT) {
            st = gd_dataset_open_gdds_split_with_transform(config->data_dir, name, &transform_cfg, out);
        } else {
            st = gd_dataset_open_gdds_file_with_transform(name, &transform_cfg, out);
        }
    }
    return st;
}

static gd_status gpt_lm_open_gdds_split(gd_context *ctx,
                                        const gpt_config *config,
                                        const char *split,
                                        const gpt_lm_special_ids *special_ids,
                                        gpt_lm_compact_transform *transform,
                                        gd_dataset **out)
{
    return gpt_lm_open_gdds_source(ctx,
                                   config,
                                   split,
                                   GPT_GDDS_SOURCE_SPLIT,
                                   special_ids,
                                   transform,
                                   out);
}

static gd_status gpt_lm_open_gdds_file(gd_context *ctx,
                                       const char *path,
                                       const gpt_lm_special_ids *special_ids,
                                       gpt_lm_compact_transform *transform,
                                       gd_dataset **out)
{
    return gpt_lm_open_gdds_source(ctx,
                                   NULL,
                                   path,
                                   GPT_GDDS_SOURCE_FILE,
                                   special_ids,
                                   transform,
                                   out);
}

static gd_status create_gdds_loader(gd_context *ctx,
                                    gd_dataset *dataset,
                                    gd_sampler *sampler,
                                    const gpt_config *config,
                                    gd_dataloader **out)
{
    gd_dataloader_config cfg;
    gd_status st;
    if (config == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    cfg = gd_dataloader_config_default(config->batch_size);
    cfg.num_workers = config->dataloader_workers;
    cfg.prefetch_factor = config->dataloader_prefetch_factor;
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
                     GPT_PROGRESS_SUMMARY,
                     "last_train_loss=%s last_eval_loss=%s last_best_epoch=%s",
                     train_loss,
                     eval_loss,
                     best_epoch);
}

static void gpt_report_rowf(gd_context *ctx,
                            gpt_report_state *report,
                            gpt_progress_row row,
                            const char *fmt,
                            ...)
{
    va_list ap;
    char *message;
    if (report == NULL) {
        return;
    }
    va_start(ap, fmt);
    message = gpt_vasprintf(fmt, ap);
    va_end(ap);
    if (message == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "progress row format", __LINE__);
    }
    gd_progress_rowf(&report->progress, (unsigned)row, "%s", message);
    free(message);
    gpt_report_summary(report);
    gd_progress_render(&report->progress);
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

static bool gpt_should_generate_step(const gpt_config *config, size_t step)
{
    return config != NULL && config->generate_every_n_steps > 0 &&
           (step % (size_t)config->generate_every_n_steps) == 0U;
}

static bool gpt_should_save_step_latest_checkpoint(const gpt_config *config,
                                                   size_t epoch_step,
                                                   size_t steps_per_epoch,
                                                   size_t global_step)
{
    return config != NULL && config->save_latest && config->latest_every_n_steps > 0U &&
           global_step > 0U && (global_step % config->latest_every_n_steps) == 0U &&
           epoch_step != steps_per_epoch;
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
                            bool prefetch_next,
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
        .prefetch_next = prefetch_next,
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
            gd_progress_rowf(&report->progress, GPT_PROGRESS_TRAIN, "");
            gd_progress_row_append_bar(&report->progress,
                                       GPT_PROGRESS_TRAIN,
                                       "train",
                                       (uint64_t)current_step,
                                       (uint64_t)total_steps);
            gd_progress_row_appendf(&report->progress,
                                    GPT_PROGRESS_TRAIN,
                                    " epoch=%zu/%d batch=%zu/%zu",
                                    epoch,
                                    config->epochs,
                                    epoch_step,
                                    steps_per_epoch);
            gd_progress_rowf(&report->progress,
                             GPT_PROGRESS_TRAIN_METRICS,
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

    if (gpt_should_generate_step(config, current_step)) {
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

static void gpt_mkdirs(gd_context *ctx, const char *path, bool include_leaf, const char *what)
{
    char *copy;
    size_t len;
    size_t i;
    bool create_leaf = include_leaf;
    if (path == NULL || path[0] == '\0') {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, what, __LINE__);
    }
    copy = (char *)malloc(strlen(path) + 1U);
    if (copy == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, what, __LINE__);
    }
    (void)strcpy(copy, path);
    len = strlen(copy);
    while (len > 1U && copy[len - 1U] == '/') {
        create_leaf = true;
        copy[len - 1U] = '\0';
        len -= 1U;
    }
    for (i = 1U; copy[i] != '\0'; ++i) {
        if (copy[i] == '/') {
            copy[i] = '\0';
            if (copy[0] != '\0' && mkdir(copy, 0775) != 0 && errno != EEXIST) {
                free(copy);
                gpt_fail_status(ctx, GD_ERR_IO, what, __LINE__);
            }
            copy[i] = '/';
        }
    }
    if (create_leaf && mkdir(copy, 0775) != 0 && errno != EEXIST) {
        free(copy);
        gpt_fail_status(ctx, GD_ERR_IO, what, __LINE__);
    }
    free(copy);
}

static void ensure_checkpoint_parent_dir(gd_context *ctx, const char *path)
{
    gpt_mkdirs(ctx, path, false, "checkpoint mkdir");
}

typedef struct gpt_training_state {
    bool loaded;
    size_t epoch;
    size_t epoch_step;
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

typedef struct gpt_gdds_shard_ref {
    char *path;
    uint64_t samples;
    uint64_t bytes;
} gpt_gdds_shard_ref;

typedef struct gpt_gdds_shard_list {
    gpt_gdds_shard_ref *items;
    size_t count;
} gpt_gdds_shard_list;

static char *gpt_strdup_or_fail(gd_context *ctx, const char *text, const char *what)
{
    char *out;
    size_t n;
    if (text == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, what, __LINE__);
    }
    n = strlen(text);
    out = (char *)malloc(n + 1U);
    if (out == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, what, __LINE__);
    }
    memcpy(out, text, n + 1U);
    return out;
}

static void ensure_directory_path(gd_context *ctx, const char *path)
{
    gpt_mkdirs(ctx, path, true, "mkdir cache dir");
}

static const char *gpt_basename_ptr(const char *path)
{
    const char *slash;
    if (path == NULL) {
        return "";
    }
    slash = strrchr(path, '/');
    return slash != NULL ? slash + 1 : path;
}

static char *gpt_join_dir_file(gd_context *ctx, const char *dir, const char *file)
{
    const size_t dir_len = dir != NULL ? strlen(dir) : 0U;
    const size_t file_len = file != NULL ? strlen(file) : 0U;
    const bool need_slash = dir_len > 0U && dir[dir_len - 1U] != '/';
    char *out;
    if (dir_len == 0U || file_len == 0U || dir_len > SIZE_MAX - file_len - (need_slash ? 2U : 1U)) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "join cache path", __LINE__);
    }
    out = (char *)malloc(dir_len + (need_slash ? 1U : 0U) + file_len + 1U);
    if (out == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "join cache path", __LINE__);
    }
    memcpy(out, dir, dir_len);
    if (need_slash) {
        out[dir_len] = '/';
    }
    memcpy(out + dir_len + (need_slash ? 1U : 0U), file, file_len + 1U);
    return out;
}

static int gpt_file_size(const char *path, uint64_t *out_size)
{
    struct stat st;
    if (out_size != NULL) {
        *out_size = 0U;
    }
    if (path == NULL || stat(path, &st) != 0 || st.st_size < 0) {
        return 0;
    }
    if (out_size != NULL) {
        *out_size = (uint64_t)st.st_size;
    }
    return 1;
}

static char *gpt_copy_shard_to_local_cache(gd_context *ctx,
                                           const char *src_path,
                                           uint64_t expected_bytes,
                                           const gpt_config *config)
{
    enum { COPY_BUFFER_BYTES = 8 * 1024 * 1024 };
#define GPT_SHARD_CACHE_FAIL(status_value, message_value) \
    do {                                                  \
        fail_status = (status_value);                     \
        fail_message = (message_value);                   \
        goto fail;                                        \
    } while (0)

    const char *base;
    char *dst_path = NULL;
    char *tmp_path = NULL;
    FILE *src = NULL;
    FILE *dst = NULL;
    unsigned char *buffer = NULL;
    uint64_t existing_bytes = 0U;
    uint64_t copied = 0U;
    uint64_t next_report = UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);
    gd_status fail_status = GD_OK;
    const char *fail_message = "shard cache copy";

    if (config == NULL || config->local_shard_cache_dir == NULL || src_path == NULL) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_INVALID_ARGUMENT, "local shard cache arguments");
    }
    ensure_directory_path(ctx, config->local_shard_cache_dir);
    base = gpt_basename_ptr(src_path);
    dst_path = gpt_join_dir_file(ctx, config->local_shard_cache_dir, base);
    if (gpt_file_size(dst_path, &existing_bytes) != 0 &&
        (expected_bytes == 0U || existing_bytes == expected_bytes)) {
        printf("shard_cache: reuse %s bytes=%llu\n", dst_path, (unsigned long long)existing_bytes);
        return dst_path;
    }
    tmp_path = gpt_asprintf("%s.tmp.%ld", dst_path, (long)getpid());
    if (tmp_path == NULL) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_OUT_OF_MEMORY, "cache temp path");
    }
    (void)remove(tmp_path);
    src = fopen(src_path, "rb");
    if (src == NULL) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "open source shard cache");
    }
    dst = fopen(tmp_path, "wb");
    if (dst == NULL) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "open temp shard cache");
    }
    buffer = (unsigned char *)malloc(COPY_BUFFER_BYTES);
    if (buffer == NULL) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_OUT_OF_MEMORY, "shard cache copy buffer");
    }
    printf("shard_cache: copy %s -> %s bytes=%llu\n",
           src_path,
           dst_path,
           (unsigned long long)expected_bytes);
    fflush(stdout);
    for (;;) {
        const size_t n = fread(buffer, 1U, COPY_BUFFER_BYTES, src);
        if (n > 0U) {
            if (fwrite(buffer, 1U, n, dst) != n) {
                GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "write shard cache");
            }
            copied += (uint64_t)n;
            if (copied >= next_report) {
                printf("shard_cache: copied %.2f/%.2f GB\n",
                       (double)copied / (1024.0 * 1024.0 * 1024.0),
                       (double)expected_bytes / (1024.0 * 1024.0 * 1024.0));
                fflush(stdout);
                next_report += UINT64_C(1024) * UINT64_C(1024) * UINT64_C(1024);
            }
        }
        if (n < COPY_BUFFER_BYTES) {
            if (ferror(src)) {
                GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "read shard cache");
            }
            break;
        }
    }
    free(buffer);
    buffer = NULL;
    if (fclose(dst) != 0) {
        dst = NULL;
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "close temp shard cache");
    }
    dst = NULL;
    if (fclose(src) != 0) {
        src = NULL;
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "close source shard cache");
    }
    src = NULL;
    if (expected_bytes != 0U && copied != expected_bytes) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "shard cache size mismatch");
    }
    if (rename(tmp_path, dst_path) != 0) {
        GPT_SHARD_CACHE_FAIL(GD_ERR_IO, "rename shard cache");
    }
    free(tmp_path);
    return dst_path;

fail:
    free(buffer);
    if (dst != NULL) {
        (void)fclose(dst);
    }
    if (src != NULL) {
        (void)fclose(src);
    }
    if (tmp_path != NULL) {
        (void)remove(tmp_path);
    }
    free(tmp_path);
    free(dst_path);
#undef GPT_SHARD_CACHE_FAIL
    gpt_fail_status(ctx, fail_status, fail_message, __LINE__);
    return NULL;
}

static void gpt_gdds_shard_list_deinit(gpt_gdds_shard_list *list)
{
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i].path);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static void gpt_gdds_shard_list_from_dataset(gd_context *ctx,
                                             const gd_dataset *dataset,
                                             gpt_gdds_shard_list *out)
{
    const int shard_count = gd_gdds_dataset_shard_count(dataset);
    int i;
    if (out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "shard list output", __LINE__);
    }
    memset(out, 0, sizeof(*out));
    if (shard_count <= 0) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty GDDS shard list", __LINE__);
    }
    out->items = (gpt_gdds_shard_ref *)calloc((size_t)shard_count, sizeof(out->items[0]));
    if (out->items == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "allocate shard list", __LINE__);
    }
    out->count = (size_t)shard_count;
    for (i = 0; i < shard_count; ++i) {
        gd_gdds_shard_info info;
        TRY(ctx, gd_gdds_dataset_shard_info(dataset, i, &info));
        if (info.path == NULL || info.samples == 0U) {
            gpt_gdds_shard_list_deinit(out);
            gpt_fail_status(ctx, GD_ERR_BAD_STATE, "invalid GDDS shard info", __LINE__);
        }
        out->items[i].path = gpt_strdup_or_fail(ctx, info.path, "copy shard path");
        out->items[i].samples = info.samples;
        out->items[i].bytes = info.bytes;
    }
}

static size_t gpt_shard_list_steps_per_epoch(const gpt_gdds_shard_list *list,
                                             const gpt_config *config,
                                             size_t *samples_per_epoch_out)
{
    size_t steps = 0U;
    size_t samples = 0U;
    size_t i;
    if (samples_per_epoch_out != NULL) {
        *samples_per_epoch_out = 0U;
    }
    if (list == NULL || config == NULL || config->batch_size <= 0) {
        return 0U;
    }
    for (i = 0U; i < list->count; ++i) {
        const size_t shard_steps = (size_t)(list->items[i].samples / (uint64_t)config->batch_size);
        if (shard_steps > SIZE_MAX - steps) {
            return 0U;
        }
        steps += shard_steps;
        if (shard_steps > 0U) {
            const size_t shard_samples = shard_steps * (size_t)config->batch_size;
            if (shard_samples > SIZE_MAX - samples) {
                return 0U;
            }
            samples += shard_samples;
        }
    }
    if (samples_per_epoch_out != NULL) {
        *samples_per_epoch_out = samples;
    }
    return steps;
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
                                     size_t epoch_step,
                                     size_t global_step,
                                     float val_loss,
                                     float best_val_loss,
                                     const char *optimizer_path,
                                     const char *train_state_path)
{
    static const char metadata_format[] =
        "model=gpt_lm\n"
        "vocab_size=%d\n"
        "context_length=%d\n"
        "d_model=%d\n"
        "n_layers=%d\n"
        "n_heads=%d\n"
        "head_dim=%d\n"
        "mlp_hidden=%d\n"
        "ffn_activation=swiglu\n"
        "architecture=%s\n"
        "minimax_m3_block_size=%d\n"
        "minimax_m3_topk_blocks=%d\n"
        "minimax_m3_init_blocks=%d\n"
        "minimax_m3_local_blocks=%d\n"
        "sdpa_window=%d\n"
        "dropout=%.9g\n"
        "logits_softcap=%.9g\n"
        "grad_clip_norm=%.9g\n"
        "eval_every_n_epochs=%d\n"
        "early_stopping_patience=%d\n"
        "epoch=%zu\n"
        "epoch_step=%zu\n"
        "global_step=%zu\n"
        "val_loss=%.9g\n"
        "best_val_loss=%.9g\n"
        "tokenizer_path=%s\n"
        "optimizer_path=%s\n"
        "train_state_path=%s\n";
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    const char *optimizer_text = optimizer_path != NULL ? optimizer_path : "";
    const char *train_state_text = train_state_path != NULL ? train_state_path : "";
    char *metadata;
    if (config == NULL) {
        return NULL;
    }
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        free(default_tok_path);
        return NULL;
    }
    metadata = gpt_asprintf(metadata_format,
                            GPT_VOCAB_SIZE,
                            GPT_CONTEXT_LENGTH,
                            GPT_D_MODEL,
                            config->n_layers,
                            GPT_N_HEADS,
                            GPT_HEAD_DIM,
                            GPT_MLP_HIDDEN,
                            gpt_architecture_name(config->architecture),
                            GPT_MINIMAX_M3_BLOCK_SIZE,
                            config->minimax_m3_topk_blocks,
                            config->minimax_m3_init_blocks,
                            config->minimax_m3_local_blocks,
                            GPT_SDPA_WINDOW,
                            (double)config->dropout_p,
                            (double)config->logits_softcap,
                            (double)config->grad_clip_norm,
                            config->eval_every_n_epochs,
                            config->early_stopping_patience,
                            epoch,
                            epoch_step,
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
                                     size_t epoch_step,
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
    fprintf(file, "format=gpt_lm_train_state_v2\n");
    fprintf(file, "model_checkpoint=%s\n", model_path);
    fprintf(file, "optimizer_checkpoint=%s\n", optimizer_path);
    fprintf(file, "epoch=%zu\n", epoch);
    fprintf(file, "epoch_step=%zu\n", epoch_step);
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
    if (strcmp(format, "gpt_lm_train_state_v1") != 0 &&
        strcmp(format, "gpt_lm_train_state_v2") != 0) {
        free(text);
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "training state format", __LINE__);
    }
    memset(out, 0, sizeof(*out));
    out->loaded = true;
    out->epoch = gpt_state_size(ctx, text, "epoch");
    out->epoch_step = strcmp(format, "gpt_lm_train_state_v2") == 0 ?
                          gpt_state_size(ctx, text, "epoch_step") :
                          GPT_TRAIN_STATE_EPOCH_END;
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
    printf("resumed_training_state: path=%s epoch=%zu epoch_step=%s global_step=%zu best_val_loss=%.6f amp_scale=%.1f\n",
           path,
           out->epoch,
           out->epoch_step == GPT_TRAIN_STATE_EPOCH_END ? "epoch_end" : "mid_epoch",
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
                                       size_t epoch_step,
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
                                       epoch_step,
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
                             epoch_step,
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

static const char *gpt_checkpoint_kind_name(gpt_checkpoint_kind kind)
{
    switch (kind) {
    case GPT_CHECKPOINT_BEST: return "best";
    case GPT_CHECKPOINT_LATEST_STEP: return "latest_step";
    case GPT_CHECKPOINT_LATEST:
    default: return "latest";
    }
}

static void gpt_log_checkpoint_event(gd_metrics_logger *metrics,
                                     gpt_checkpoint_kind kind,
                                     const char *path,
                                     size_t epoch,
                                     size_t epoch_step,
                                     size_t global_step,
                                     size_t epochs_without_improvement,
                                     float val_loss,
                                     float best_val_loss)
{
    if (metrics == NULL) {
        return;
    }
    switch (kind) {
    case GPT_CHECKPOINT_BEST: {
        const gd_metrics_field fields[] = {
            gd_metrics_string("kind", gpt_checkpoint_kind_name(kind)),
            gd_metrics_string("path", path),
            gd_metrics_u64("epoch", (uint64_t)epoch),
            gd_metrics_u64("step", (uint64_t)global_step),
            gd_metrics_f64("val_loss", (double)val_loss),
            gd_metrics_f64("best_val_loss", (double)best_val_loss),
        };
        (void)gd_metrics_logger_log_event(metrics, "checkpoint", fields, GD_ARRAY_LEN(fields));
        break;
    }
    case GPT_CHECKPOINT_LATEST_STEP: {
        const gd_metrics_field fields[] = {
            gd_metrics_string("kind", gpt_checkpoint_kind_name(kind)),
            gd_metrics_string("path", path),
            gd_metrics_u64("epoch", (uint64_t)epoch),
            gd_metrics_u64("epoch_step", (uint64_t)epoch_step),
            gd_metrics_u64("step", (uint64_t)global_step),
            gd_metrics_u64("epochs_without_improvement", (uint64_t)epochs_without_improvement),
            gd_metrics_f64("val_loss", (double)val_loss),
            gd_metrics_f64("best_val_loss", (double)best_val_loss),
        };
        (void)gd_metrics_logger_log_event(metrics, "checkpoint", fields, GD_ARRAY_LEN(fields));
        break;
    }
    case GPT_CHECKPOINT_LATEST:
    default: {
        const gd_metrics_field fields[] = {
            gd_metrics_string("kind", gpt_checkpoint_kind_name(kind)),
            gd_metrics_string("path", path),
            gd_metrics_u64("epoch", (uint64_t)epoch),
            gd_metrics_u64("step", (uint64_t)global_step),
            gd_metrics_u64("epochs_without_improvement", (uint64_t)epochs_without_improvement),
            gd_metrics_f64("val_loss", (double)val_loss),
            gd_metrics_f64("best_val_loss", (double)best_val_loss),
        };
        (void)gd_metrics_logger_log_event(metrics, "checkpoint", fields, GD_ARRAY_LEN(fields));
        break;
    }
    }
}

static void gpt_save_and_log_checkpoint(gd_context *ctx,
                                        gpt_lm *model,
                                        gd_optimizer *optimizer,
                                        gd_amp_scaler *scaler,
                                        const gd_lr_scheduler_config *lr_config,
                                        const gpt_config *config,
                                        gd_metrics_logger *metrics,
                                        gpt_checkpoint_kind kind,
                                        const char *checkpoint_path,
                                        size_t epoch,
                                        size_t epoch_step,
                                        size_t global_step,
                                        size_t epochs_without_improvement,
                                        float val_loss,
                                        float best_val_loss)
{
    gpt_save_checkpoint_bundle(ctx,
                               model,
                               optimizer,
                               scaler,
                               lr_config,
                               config,
                               checkpoint_path,
                               NULL,
                               epoch,
                               epoch_step,
                               global_step,
                               epochs_without_improvement,
                               val_loss,
                               best_val_loss);
    gpt_log_checkpoint_event(metrics,
                             kind,
                             checkpoint_path,
                             epoch,
                             epoch_step,
                             global_step,
                             epochs_without_improvement,
                             val_loss,
                             best_val_loss);
}

static void gpt_skip_loader_batches(gd_context *ctx, gd_dataloader *loader, size_t batches)
{
    size_t i;
    if (batches == 0U) {
        return;
    }
    for (i = 0U; i < batches; ++i) {
        gd_batch *batch = NULL;
        TRY(ctx, gd_dataloader_next(loader, &batch));
        TRY(ctx, gd_dataloader_release(loader, batch));
        TRY(ctx, gd_dataloader_prefetch(loader));
    }
}

static void maybe_save_step_latest_checkpoint(gd_context *ctx,
                                              gpt_lm *model,
                                              gd_optimizer *optimizer,
                                              gd_amp_scaler *scaler,
                                              const gd_lr_scheduler_config *lr_config,
                                              const gpt_config *config,
                                              gpt_report_state *report,
                                              gd_metrics_logger *metrics,
                                              size_t epoch,
                                              size_t epoch_step,
                                              size_t steps_per_epoch,
                                              size_t global_step,
                                              size_t epochs_without_improvement,
                                              float best_val_loss)
{
    const float checkpoint_val_loss = report != NULL && report->has_eval_loss ?
                                          report->last_eval_loss :
                                          NAN;
    if (!gpt_should_save_step_latest_checkpoint(config, epoch_step, steps_per_epoch, global_step)) {
        return;
    }
    gpt_report_rowf(ctx,
                    report,
                    GPT_PROGRESS_CHECKPOINT,
                    "checkpoint latest=step-saving step=%zu path=%s",
                    global_step,
                    config->latest_checkpoint_path);
    gpt_save_and_log_checkpoint(ctx,
                                model,
                                optimizer,
                                scaler,
                                lr_config,
                                config,
                                metrics,
                                GPT_CHECKPOINT_LATEST_STEP,
                                config->latest_checkpoint_path,
                                epoch,
                                epoch_step,
                                global_step,
                                epochs_without_improvement,
                                checkpoint_val_loss,
                                best_val_loss);
    gpt_report_rowf(ctx,
                    report,
                    GPT_PROGRESS_CHECKPOINT,
                    "checkpoint latest=step-saved step=%zu path=%s",
                    global_step,
                    config->latest_checkpoint_path);
}

typedef struct gpt_train_epoch_runner {
    gpt_lm *model;
    gd_optimizer *optimizer;
    gd_amp_scaler *scaler;
    const gd_lr_scheduler_config *lr_config;
    const gpt_config *config;
    const gpt_generation_tokenizer *generation_tokenizer;
    gpt_report_state *report;
    gd_metrics_logger *metrics;
    size_t total_steps;
    double *last_report_time;
    size_t *last_report_step;
    size_t *global_step;
} gpt_train_epoch_runner;

static void gpt_train_epoch_step(gd_context *ctx,
                                 const gpt_train_epoch_runner *runner,
                                 gd_dataloader *loader,
                                 size_t epoch,
                                 size_t *epoch_step,
                                 size_t steps_per_epoch,
                                 size_t epochs_without_improvement,
                                 float best_val_loss)
{
    size_t current_step;
    bool defer_prefetch;
    if (runner == NULL || runner->global_step == NULL || epoch_step == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "train epoch step", __LINE__);
    }
    current_step = *runner->global_step + 1U;
    defer_prefetch = gpt_should_generate_step(runner->config, current_step) ||
                     gpt_should_save_step_latest_checkpoint(runner->config,
                                                            *epoch_step,
                                                            steps_per_epoch,
                                                            current_step);
    train_one_batch(ctx,
                    runner->model,
                    loader,
                    runner->optimizer,
                    runner->scaler,
                    runner->lr_config,
                    runner->config,
                    runner->generation_tokenizer,
                    runner->report,
                    runner->metrics,
                    *runner->global_step,
                    runner->total_steps,
                    epoch,
                    *epoch_step,
                    steps_per_epoch,
                    !defer_prefetch,
                    runner->last_report_time,
                    runner->last_report_step);
    *runner->global_step = current_step;
    maybe_save_step_latest_checkpoint(ctx,
                                      runner->model,
                                      runner->optimizer,
                                      runner->scaler,
                                      runner->lr_config,
                                      runner->config,
                                      runner->report,
                                      runner->metrics,
                                      epoch,
                                      *epoch_step,
                                      steps_per_epoch,
                                      *runner->global_step,
                                      epochs_without_improvement,
                                      best_val_loss);
    if (defer_prefetch && *epoch_step < steps_per_epoch) {
        TRY(ctx, gd_dataloader_prefetch(loader));
    }
    *epoch_step += 1U;
}

static void gpt_resume_start_position(const gpt_training_state *resume_state,
                                      size_t steps_per_epoch,
                                      size_t *start_epoch,
                                      size_t *start_epoch_step)
{
    if (start_epoch == NULL || start_epoch_step == NULL) {
        return;
    }
    *start_epoch = 1U;
    *start_epoch_step = 1U;
    if (resume_state == NULL || !resume_state->loaded) {
        return;
    }
    if (resume_state->epoch_step != GPT_TRAIN_STATE_EPOCH_END &&
        resume_state->epoch_step < steps_per_epoch) {
        *start_epoch = resume_state->epoch;
        *start_epoch_step = resume_state->epoch_step + 1U;
    } else {
        *start_epoch = resume_state->epoch + 1U;
    }
}

static const char *gpt_epoch_best_checkpoint_status(bool validated,
                                                    bool val_improved,
                                                    bool has_val_dataset,
                                                    bool save_best)
{
    return !validated ? (has_val_dataset ? "deferred" : "skipped") :
           !val_improved ? "unchanged" :
           save_best ? "saved" : "improved save_best=off";
}

static bool gpt_finish_epoch(gd_context *ctx,
                             const gpt_train_epoch_runner *runner,
                             gd_dataset *val_dataset,
                             size_t epoch,
                             size_t steps_per_epoch,
                             size_t *epochs_without_improvement,
                             float *best_val_loss)
{
    const gpt_config *config;
    gpt_report_state *report;
    size_t global_step;
    bool validated = false;
    bool val_improved = false;
    bool should_stop = false;
    float val_loss = NAN;
    const char *best_status;

    if (runner == NULL || runner->model == NULL || runner->optimizer == NULL ||
        runner->scaler == NULL || runner->lr_config == NULL || runner->config == NULL ||
        runner->report == NULL || runner->global_step == NULL || epochs_without_improvement == NULL ||
        best_val_loss == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "finish epoch", __LINE__);
    }
    config = runner->config;
    report = runner->report;
    global_step = *runner->global_step;

    if (val_dataset != NULL && gpt_should_eval_epoch(config, epoch)) {
        gd_progress_rowf(&report->progress,
                         GPT_PROGRESS_EVAL,
                         "eval epoch=%zu step=%zu running...",
                         epoch,
                         global_step);
        gd_progress_rowf(&report->progress,
                         GPT_PROGRESS_CHECKPOINT,
                         "checkpoint best=pending latest=pending");
        gpt_report_summary(report);
        gd_progress_render(&report->progress);
        val_loss = evaluate_gpt_loss(ctx, runner->model, val_dataset, config);
        if (isfinite(val_loss) && val_loss < *best_val_loss) {
            *best_val_loss = val_loss;
            val_improved = true;
            if (config->save_best) {
                gpt_report_rowf(ctx,
                                report,
                                GPT_PROGRESS_CHECKPOINT,
                                "checkpoint best=saving path=%s latest=pending",
                                config->checkpoint_path);
                gpt_save_and_log_checkpoint(ctx,
                                            runner->model,
                                            runner->optimizer,
                                            runner->scaler,
                                            runner->lr_config,
                                            config,
                                            runner->metrics,
                                            GPT_CHECKPOINT_BEST,
                                            config->checkpoint_path,
                                            epoch,
                                            steps_per_epoch,
                                            global_step,
                                            0U,
                                            val_loss,
                                            *best_val_loss);
            }
        }
        if (runner->metrics != NULL) {
            const gd_metrics_field fields[] = {
                gd_metrics_u64("epoch", (uint64_t)epoch),
                gd_metrics_u64("step", (uint64_t)global_step),
                gd_metrics_f64("val_loss", (double)val_loss),
                gd_metrics_f64("best_val_loss", (double)*best_val_loss),
                gd_metrics_bool("improved", val_improved),
                gd_metrics_bool("save_best", config->save_best),
            };
            (void)gd_metrics_logger_log_event(runner->metrics, "eval", fields, GD_ARRAY_LEN(fields));
        }
        validated = true;
    }
    if (validated && config->early_stopping_patience > 0) {
        if (val_improved) {
            *epochs_without_improvement = 0U;
        } else {
            *epochs_without_improvement += 1U;
            should_stop = *epochs_without_improvement >= (size_t)config->early_stopping_patience;
        }
    }

    if (validated) {
        char patience_text[64];
        report->last_eval_loss = val_loss;
        report->has_eval_loss = true;
        if (val_improved) {
            report->best_epoch = epoch;
            report->has_best_epoch = true;
        }
        if (config->early_stopping_patience > 0) {
            (void)snprintf(patience_text,
                           sizeof(patience_text),
                           "%zu/%d",
                           *epochs_without_improvement,
                           config->early_stopping_patience);
        } else {
            (void)snprintf(patience_text, sizeof(patience_text), "off");
        }
        gd_progress_rowf(&report->progress,
                         GPT_PROGRESS_EVAL,
                         "eval epoch=%zu step=%zu val_loss=%.6f best_val_loss=%.6f improved=%s patience=%s",
                         epoch,
                         global_step,
                         (double)val_loss,
                         (double)*best_val_loss,
                         val_improved ? "yes" : "no",
                         patience_text);
    } else if (val_dataset != NULL) {
        gd_progress_rowf(&report->progress,
                         GPT_PROGRESS_EVAL,
                         "eval skipped epoch=%zu next_eval_epoch=%zu every=%d",
                         epoch,
                         gpt_next_eval_epoch(config, epoch),
                         config->eval_every_n_epochs);
    } else {
        gd_progress_rowf(&report->progress,
                         GPT_PROGRESS_EVAL,
                         "eval skipped no validation split loaded");
    }

    best_status = gpt_epoch_best_checkpoint_status(validated,
                                                   val_improved,
                                                   val_dataset != NULL,
                                                   config->save_best);
    if (config->save_latest) {
        const float checkpoint_val_loss = validated ?
                                            val_loss :
                                            (report->has_eval_loss ? report->last_eval_loss : NAN);
        gpt_report_rowf(ctx,
                        report,
                        GPT_PROGRESS_CHECKPOINT,
                        "checkpoint best=%s latest=saving path=%s",
                        best_status,
                        config->latest_checkpoint_path);
        gpt_save_and_log_checkpoint(ctx,
                                    runner->model,
                                    runner->optimizer,
                                    runner->scaler,
                                    runner->lr_config,
                                    config,
                                    runner->metrics,
                                    GPT_CHECKPOINT_LATEST,
                                    config->latest_checkpoint_path,
                                    epoch,
                                    steps_per_epoch,
                                    global_step,
                                    *epochs_without_improvement,
                                    checkpoint_val_loss,
                                    *best_val_loss);
        gpt_report_rowf(ctx,
                        report,
                        GPT_PROGRESS_CHECKPOINT,
                        "checkpoint best=%s latest=saved path=%s",
                        best_status,
                        config->latest_checkpoint_path);
    } else {
        gpt_report_rowf(ctx,
                        report,
                        GPT_PROGRESS_CHECKPOINT,
                        "checkpoint best=%s latest=off",
                        best_status);
    }

    if (should_stop) {
        gpt_report_rowf(ctx,
                        report,
                        GPT_PROGRESS_CHECKPOINT,
                        "early_stopping epoch=%zu step=%zu patience=%d best_val_loss=%.6f",
                        epoch,
                        global_step,
                        config->early_stopping_patience,
                        (double)*best_val_loss);
        if (runner->metrics != NULL) {
            const gd_metrics_field fields[] = {
                gd_metrics_u64("epoch", (uint64_t)epoch),
                gd_metrics_u64("step", (uint64_t)global_step),
                gd_metrics_i64("patience", (int64_t)config->early_stopping_patience),
                gd_metrics_f64("best_val_loss", (double)*best_val_loss),
            };
            (void)gd_metrics_logger_log_event(runner->metrics, "early_stopping", fields, GD_ARRAY_LEN(fields));
        }
    }
    return should_stop;
}

static void train_gpt(gd_context *ctx,
                      gpt_lm *model,
                      gd_dataset *dataset,
                      gd_dataset *val_dataset,
                      gd_optimizer *optimizer,
                      gd_amp_scaler *scaler,
                      const gd_lr_scheduler_config *lr_config,
                      const gpt_config *config,
                      const gpt_lm_special_ids *special_ids,
                      const gpt_gdds_shard_list *train_shards,
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
    size_t start_epoch = 1U;
    size_t start_epoch_step = 1U;
    size_t epoch;
    gpt_report_state report;
    gpt_generation_tokenizer generation_tokenizer;
    gpt_train_epoch_runner runner;
    const gpt_generation_tokenizer *generation_tokenizer_ptr = NULL;
    float best_val_loss = resume_state != NULL && resume_state->loaded ? resume_state->best_val_loss : INFINITY;

    memset(&report, 0, sizeof(report));
    report.last_train_loss = NAN;
    report.last_eval_loss = NAN;
    gd_progress_init(&report.progress, stdout);
    if (!gd_progress_set_row_count(&report.progress, (unsigned)GPT_PROGRESS_COUNT)) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "progress rows", __LINE__);
    }
    gd_progress_rowf(&report.progress, GPT_PROGRESS_TRAIN, "train pending");
    gd_progress_rowf(&report.progress, GPT_PROGRESS_TRAIN_METRICS, "train metrics pending");
    gd_progress_rowf(&report.progress, GPT_PROGRESS_EVAL, "eval pending");
    gd_progress_rowf(&report.progress, GPT_PROGRESS_CHECKPOINT, "checkpoint pending");
    gpt_report_summary(&report);
    memset(&generation_tokenizer, 0, sizeof(generation_tokenizer));
    if (config->generate_every_n_steps > 0) {
        gpt_generation_tokenizer_init(ctx, config, &generation_tokenizer);
        generation_tokenizer_ptr = &generation_tokenizer;
    }
    runner = (gpt_train_epoch_runner){
        .model = model,
        .optimizer = optimizer,
        .scaler = scaler,
        .lr_config = lr_config,
        .config = config,
        .generation_tokenizer = generation_tokenizer_ptr,
        .report = &report,
        .metrics = metrics,
        .total_steps = total_steps,
        .last_report_time = &last_report_time,
        .last_report_step = &last_report_step,
        .global_step = &global_step,
    };
    gpt_resume_start_position(resume_state, steps_per_epoch, &start_epoch, &start_epoch_step);

    if (resume_state != NULL && resume_state->loaded) {
        printf("resume_training: next_epoch=%zu/%d next_epoch_step=%zu/%zu global_step=%zu best_val_loss=%.6f epochs_without_improvement=%zu\n",
               start_epoch,
               config->epochs,
               start_epoch_step,
               steps_per_epoch,
               global_step,
               (double)best_val_loss,
               epochs_without_improvement);
    }
    for (epoch = start_epoch; epoch <= (size_t)config->epochs; ++epoch) {
        const size_t epoch_first_step = epoch == start_epoch ? start_epoch_step : 1U;
        size_t epoch_step = epoch_first_step;
        gd_module_set_training(&model->mod, true);
        if (config->local_shard_cache_dir != NULL && config->overfit_num_samples == 0U &&
            train_shards != NULL && train_shards->count > 0U) {
            size_t shard_index;
            size_t shard_epoch_step_base = 1U;
            for (shard_index = 0U; shard_index < train_shards->count; ++shard_index) {
                const gpt_gdds_shard_ref *shard = &train_shards->items[shard_index];
                const size_t shard_steps = (size_t)(shard->samples / (uint64_t)config->batch_size);
                const size_t shard_first_step = shard_epoch_step_base;
                const size_t shard_last_step = shard_steps > 0U ? shard_first_step + shard_steps - 1U : shard_first_step;
                char *cached_path;
                gd_dataset *shard_dataset = NULL;
                gd_sampler *sampler = NULL;
                gd_dataloader *loader = NULL;
                gpt_lm_compact_transform shard_transform;
                size_t local_first_step;
                size_t local_step;
                if (shard_steps == 0U) {
                    continue;
                }
                if (shard_last_step < epoch_first_step) {
                    shard_epoch_step_base += shard_steps;
                    continue;
                }
                local_first_step = epoch_first_step > shard_first_step ?
                                       epoch_first_step - shard_first_step + 1U :
                                       1U;
                memset(&shard_transform, 0, sizeof(shard_transform));
                cached_path = gpt_copy_shard_to_local_cache(ctx, shard->path, shard->bytes, config);
                TRY(ctx, gpt_lm_open_gdds_file(ctx,
                                               cached_path,
                                               special_ids,
                                               &shard_transform,
                                               &shard_dataset));
                TRY(ctx, gd_sampler_create_random(shard_dataset,
                                                  config->seed ^ GPT_TRAIN_SHUFFLE_SEED_BASE ^
                                                      (uint64_t)epoch ^
                                                      (uint64_t)(shard_index + 1U) * GPT_TRAIN_SHARD_SEED_STRIDE,
                                                  &sampler));
                TRY(ctx, create_gdds_loader(ctx, shard_dataset, sampler, config, &loader));
                if (local_first_step > 1U) {
                    gpt_skip_loader_batches(ctx, loader, local_first_step - 1U);
                }
                for (local_step = local_first_step; local_step <= shard_steps; ++local_step) {
                    gpt_train_epoch_step(ctx,
                                         &runner,
                                         loader,
                                         epoch,
                                         &epoch_step,
                                         steps_per_epoch,
                                         epochs_without_improvement,
                                         best_val_loss);
                }
                gd_dataloader_destroy(loader);
                gd_sampler_destroy(sampler);
                gd_dataset_destroy(shard_dataset);
                if (!config->keep_shard_cache && strcmp(cached_path, shard->path) != 0) {
                    (void)remove(cached_path);
                }
                free(cached_path);
                shard_epoch_step_base += shard_steps;
            }
            if (epoch_step != steps_per_epoch + 1U) {
                gpt_fail_status(ctx, GD_ERR_BAD_STATE, "cached shard epoch step mismatch", __LINE__);
            }
        } else {
            gd_sampler *sampler = NULL;
            gd_dataloader *loader = NULL;
            if (config->overfit_num_samples == 0U) {
                const uint64_t shuffle_seed = config->seed ^ GPT_TRAIN_SHUFFLE_SEED_BASE ^ (uint64_t)epoch;
                if (config->shuffle_scope == GPT_SHUFFLE_GLOBAL) {
                    TRY(ctx, gd_sampler_create_random(dataset, shuffle_seed, &sampler));
                } else if (config->shuffle_scope == GPT_SHUFFLE_SHARD) {
                    TRY(ctx, gd_sampler_create_gdds_shard_random(dataset, shuffle_seed, &sampler));
                }
                TRY(ctx, create_gdds_loader(ctx, dataset, sampler, config, &loader));
            } else {
                TRY(ctx, create_gdds_loader(ctx, dataset, NULL, config, &loader));
            }
            if (epoch_first_step > 1U) {
                gpt_skip_loader_batches(ctx, loader, epoch_first_step - 1U);
            }
            for (; epoch_step <= steps_per_epoch;) {
                gpt_train_epoch_step(ctx,
                                     &runner,
                                     loader,
                                     epoch,
                                     &epoch_step,
                                     steps_per_epoch,
                                     epochs_without_improvement,
                                     best_val_loss);
            }
            gd_dataloader_destroy(loader);
            gd_sampler_destroy(sampler);
        }
        if (gpt_finish_epoch(ctx,
                             &runner,
                             val_dataset,
                             epoch,
                             steps_per_epoch,
                             &epochs_without_improvement,
                             &best_val_loss)) {
            break;
        }
    }
    gd_progress_finish(&report.progress);
    gpt_generation_tokenizer_deinit(&generation_tokenizer);
    gd_progress_deinit(&report.progress);
}

static void gpt_start_metrics(gd_context *ctx,
                              const gpt_config *config,
                              gd_metrics_logger **out)
{
    if (config == NULL || out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "metrics init", __LINE__);
    }
    *out = NULL;
    if (config->metrics_enabled) {
        gd_metrics_config metrics_config = gd_metrics_config_default(config->metrics_project);
        metrics_config.root_dir = config->metrics_dir;
        metrics_config.run_id = config->metrics_run_id;
        TRY(ctx, gd_metrics_logger_start(&metrics_config, out));
        printf("metrics: path=%s project=%s run_id=%s\n",
               gd_metrics_logger_path(*out),
               gd_metrics_logger_project(*out),
               gd_metrics_logger_run_id(*out));
    } else {
        printf("metrics: disabled\n");
    }
}

static void gpt_open_training_data(gd_context *ctx,
                                   gpt_config *config,
                                   gpt_lm_special_ids *special_ids,
                                   gpt_lm_compact_transform *train_transform,
                                   gpt_lm_compact_transform *val_transform,
                                   gd_dataset **dataset_out,
                                   gd_dataset **val_dataset_out,
                                   gpt_gdds_shard_list *train_shards,
                                   size_t *dataset_samples_out,
                                   size_t *dataset_tokens_out,
                                   size_t *val_samples_out,
                                   size_t *val_tokens_out,
                                   size_t *samples_per_epoch_out,
                                   size_t *steps_per_epoch_out,
                                   size_t *total_steps_out)
{
    size_t dataset_samples = 0U;
    size_t dataset_tokens = 0U;
    size_t val_samples = 0U;
    size_t val_tokens = 0U;
    size_t samples_per_epoch = 0U;
    size_t steps_per_epoch = 0U;
    size_t total_steps = 0U;
    if (config == NULL || special_ids == NULL || train_transform == NULL || val_transform == NULL ||
        dataset_out == NULL || val_dataset_out == NULL || train_shards == NULL ||
        dataset_samples_out == NULL || dataset_tokens_out == NULL || val_samples_out == NULL ||
        val_tokens_out == NULL || samples_per_epoch_out == NULL || steps_per_epoch_out == NULL ||
        total_steps_out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "training data outputs", __LINE__);
    }
    if (config->epochs <= 0) {
        return;
    }
    TRY(ctx, gpt_lm_resolve_special_ids(ctx, config, special_ids));
    if (special_ids->pad_id >= 0) {
        config->pad_token_id = special_ids->pad_id;
    }
    TRY(ctx, gpt_lm_open_gdds_split(ctx, config, "train", special_ids, train_transform, dataset_out));
    dataset_samples = (size_t)gd_dataset_num_samples(*dataset_out);
    if (dataset_samples > SIZE_MAX / (size_t)GPT_CONTEXT_LENGTH) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "dataset token count overflow", __LINE__);
    }
    dataset_tokens = dataset_samples * (size_t)GPT_CONTEXT_LENGTH;
    if (config->local_shard_cache_dir != NULL && config->overfit_num_samples == 0U) {
        gpt_gdds_shard_list_from_dataset(ctx, *dataset_out, train_shards);
        steps_per_epoch = gpt_shard_list_steps_per_epoch(train_shards, config, &samples_per_epoch);
    } else {
        steps_per_epoch = effective_steps_per_epoch((uint64_t)dataset_samples,
                                                    config,
                                                    &samples_per_epoch);
    }
    if (steps_per_epoch == 0U || (size_t)config->epochs > SIZE_MAX / steps_per_epoch) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "dataset too small for requested batch size", __LINE__);
    }
    total_steps = (size_t)config->epochs * steps_per_epoch;
    if (config->save_best || config->early_stopping_patience > 0) {
        TRY(ctx, gpt_lm_open_gdds_split(ctx,
                                        config,
                                        config->val_split,
                                        special_ids,
                                        val_transform,
                                        val_dataset_out));
        val_samples = (size_t)gd_dataset_num_samples(*val_dataset_out);
        if (val_samples > SIZE_MAX / (size_t)GPT_CONTEXT_LENGTH) {
            gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "validation token count overflow", __LINE__);
        }
        val_tokens = val_samples * (size_t)GPT_CONTEXT_LENGTH;
        if (val_samples == 0U) {
            gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty validation split", __LINE__);
        }
    }
    *dataset_samples_out = dataset_samples;
    *dataset_tokens_out = dataset_tokens;
    *val_samples_out = val_samples;
    *val_tokens_out = val_tokens;
    *samples_per_epoch_out = samples_per_epoch;
    *steps_per_epoch_out = steps_per_epoch;
    *total_steps_out = total_steps;
}

static void gpt_load_requested_checkpoint(gd_context *ctx,
                                          gpt_lm *model,
                                          const gpt_config *config)
{
    if (model == NULL || config == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "checkpoint load args", __LINE__);
    }
    if (config->load_checkpoint_path != NULL || config->resume_checkpoint_path != NULL) {
        const char *load_path = config->resume_checkpoint_path != NULL ?
                                    config->resume_checkpoint_path :
                                    config->load_checkpoint_path;
        gd_module_load_options load_options;
        load_options.strict = true;
        load_options.load_buffers = true;
        TRY(ctx, gd_module_load_state(ctx, &model->mod, load_path, &load_options));
        printf("loaded_checkpoint: %s%s\n",
               load_path,
               config->resume_checkpoint_path != NULL ? " (resume)" : "");
    }
}

static void gpt_create_training_optimizer(gd_context *ctx,
                                          const gpt_config *config,
                                          const gd_param_set *params,
                                          gd_optimizer **optimizer_out,
                                          gd_amp_scaler **scaler_out)
{
    if (config == NULL || params == NULL || optimizer_out == NULL || scaler_out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "optimizer init", __LINE__);
    }
    if (config->epochs <= 0) {
        return;
    }
    {
        const gd_adamw_config adam = gpt_adamw_config(config->lr_max, config->weight_decay);
        const gd_amp_config amp = gpt_amp_config();
        TRY(ctx, gd_adamw_create(ctx, params, &adam, optimizer_out));
        TRY(ctx, gd_amp_scaler_create(ctx, &amp, scaler_out));
    }
}

static gd_lr_scheduler_config gpt_lr_scheduler_config(const gpt_config *config, size_t total_steps)
{
    gd_lr_scheduler_config lr_config = gd_lr_scheduler_config_default();
    lr_config.max_lr = config->lr_max;
    lr_config.min_lr = config->lr_min;
    lr_config.total_steps = (uint64_t)total_steps;
    if (config->lr_warmup_steps >= 0) {
        lr_config.warmup_steps = (uint64_t)config->lr_warmup_steps;
    } else {
        lr_config.warmup_steps = (uint64_t)(total_steps / 10U);
    }
    return lr_config;
}

static void gpt_resume_training_checkpoint(gd_context *ctx,
                                           const gpt_config *config,
                                           gd_optimizer *optimizer,
                                           gd_amp_scaler *scaler,
                                           gd_lr_scheduler_config *lr_config,
                                           gpt_training_state *resume_state)
{
    char *optimizer_path;
    char *train_state_path;
    if (config == NULL || lr_config == NULL || resume_state == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "resume checkpoint args", __LINE__);
    }
    if (config->resume_checkpoint_path == NULL || config->epochs <= 0) {
        return;
    }
    if (optimizer == NULL || scaler == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "resume optimizer state", __LINE__);
    }
    optimizer_path = gpt_checkpoint_sidecar_path(config->resume_checkpoint_path, ".optim.gdckpt");
    train_state_path = gpt_checkpoint_sidecar_path(config->resume_checkpoint_path, ".train");
    if (optimizer_path == NULL || train_state_path == NULL) {
        free(optimizer_path);
        free(train_state_path);
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "resume sidecar path", __LINE__);
    }
    TRY(ctx, gd_optimizer_load_state(ctx, optimizer, optimizer_path, true));
    printf("loaded_optimizer_state: %s\n", optimizer_path);
    gpt_load_training_state(ctx, train_state_path, scaler, lr_config, resume_state);
    free(optimizer_path);
    free(train_state_path);
}

static void gpt_log_run_config_metrics(gd_metrics_logger *metrics,
                                       const gpt_config *config,
                                       const gd_memory_config *mem,
                                       size_t dataset_samples,
                                       size_t dataset_tokens,
                                       size_t val_samples,
                                       size_t val_tokens,
                                       size_t steps_per_epoch,
                                       size_t samples_per_epoch,
                                       size_t total_steps,
                                       uint64_t total_params,
                                       uint64_t trainable_params,
                                       uint64_t param_bytes)
{
    if (metrics == NULL) {
        return;
    }
    {
        const gd_metrics_field fields[] = {
            gd_metrics_string("config_path", config->config_path),
            gd_metrics_string("data_dir", config->data_dir),
            gd_metrics_string("tokenizer_path", config->tokenizer_path),
            gd_metrics_string("checkpoint_path", config->checkpoint_path),
            gd_metrics_string("latest_checkpoint_path", config->latest_checkpoint_path),
            gd_metrics_string("load_checkpoint_path", config->load_checkpoint_path),
            gd_metrics_string("resume_checkpoint_path", config->resume_checkpoint_path),
            gd_metrics_string("val_split", config->val_split),
            gd_metrics_i64("epochs", (int64_t)config->epochs),
            gd_metrics_i64("batch_size", (int64_t)config->batch_size),
            gd_metrics_i64("dataloader_workers", (int64_t)config->dataloader_workers),
            gd_metrics_i64("dataloader_prefetch_factor", (int64_t)config->dataloader_prefetch_factor),
            gd_metrics_u64("dataloader_slots",
                           (uint64_t)((size_t)config->dataloader_workers *
                                      (size_t)config->dataloader_prefetch_factor)),
            gd_metrics_u64("reserved_data_slots", (uint64_t)GPT_RESERVED_DATA_SLOTS),
            gd_metrics_i64("layers", (int64_t)config->n_layers),
            gd_metrics_string("architecture", gpt_architecture_name(config->architecture)),
            gd_metrics_string("shuffle_scope", gpt_shuffle_scope_name(config->shuffle_scope)),
            gd_metrics_i64("minimax_m3_block_size", (int64_t)GPT_MINIMAX_M3_BLOCK_SIZE),
            gd_metrics_i64("minimax_m3_topk_blocks", (int64_t)config->minimax_m3_topk_blocks),
            gd_metrics_i64("minimax_m3_init_blocks", (int64_t)config->minimax_m3_init_blocks),
            gd_metrics_i64("minimax_m3_local_blocks", (int64_t)config->minimax_m3_local_blocks),
            gd_metrics_i64("context_length", (int64_t)GPT_CONTEXT_LENGTH),
            gd_metrics_i64("vocab_size", (int64_t)GPT_VOCAB_SIZE),
            gd_metrics_i64("d_model", (int64_t)GPT_D_MODEL),
            gd_metrics_i64("heads", (int64_t)GPT_N_HEADS),
            gd_metrics_i64("head_dim", (int64_t)GPT_HEAD_DIM),
            gd_metrics_i64("mlp_hidden", (int64_t)GPT_MLP_HIDDEN),
            gd_metrics_i64("report_every", (int64_t)config->report_every),
            gd_metrics_i64("eval_every_n_epochs", (int64_t)config->eval_every_n_epochs),
            gd_metrics_i64("early_stopping_patience", (int64_t)config->early_stopping_patience),
            gd_metrics_i64("warmup_steps", (int64_t)config->lr_warmup_steps),
            gd_metrics_u64("latest_every_n_steps", (uint64_t)config->latest_every_n_steps),
            gd_metrics_u64("seed", config->seed),
            gd_metrics_f64("dropout", (double)config->dropout_p),
            gd_metrics_f64("lr_max", (double)config->lr_max),
            gd_metrics_f64("lr_min", (double)config->lr_min),
            gd_metrics_f64("weight_decay", (double)config->weight_decay),
            gd_metrics_f64("grad_clip_norm", (double)config->grad_clip_norm),
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
            gd_metrics_u64("memory_params_bytes", (uint64_t)mem->params_bytes),
            gd_metrics_u64("memory_state_bytes", (uint64_t)mem->state_bytes),
            gd_metrics_u64("memory_scratch_slot_bytes", (uint64_t)mem->scratch_slot_bytes),
            gd_metrics_u64("memory_data_slot_bytes", (uint64_t)mem->data_slot_bytes),
            gd_metrics_u64("memory_scratch_slots", (uint64_t)mem->scratch_slots),
            gd_metrics_u64("memory_data_slots", (uint64_t)mem->data_slots),
            gd_metrics_bool("save_best", config->save_best),
            gd_metrics_bool("save_latest", config->save_latest),
        };
        (void)gd_metrics_logger_log_event(metrics, "run_config", fields, GD_ARRAY_LEN(fields));
    }
}

static void gpt_print_startup_summary(const gpt_config *config,
                                      const gd_memory_config *mem,
                                      const gd_lr_scheduler_config *lr_config,
                                      gd_amp_scaler *scaler,
                                      const gpt_gdds_shard_list *train_shards,
                                      size_t dataset_samples,
                                      size_t dataset_tokens,
                                      size_t val_samples,
                                      size_t val_tokens,
                                      size_t steps_per_epoch,
                                      size_t samples_per_epoch,
                                      size_t total_steps)
{
    printf("config: path=%s\n", config->config_path);
    printf("dataset: dir=%s samples=%zu train_tokens=%zu val_samples=%zu val_tokens=%zu batch=%d context=%d shuffle=%s steps_per_epoch=%zu samples_per_epoch=%zu epochs=%d total_steps=%zu%s\n",
           config->data_dir,
           dataset_samples,
           dataset_tokens,
           val_samples,
           val_tokens,
           config->batch_size,
           GPT_CONTEXT_LENGTH,
           gpt_shuffle_scope_name(config->shuffle_scope),
           steps_per_epoch,
           samples_per_epoch,
           config->epochs,
           total_steps,
           config->overfit_num_samples > 0U ? " overfit" : "");
    printf("dataloader: workers=%d prefetch_factor=%d slots=%zu reserved_data_slots=%u memory_data_slots=%u\n",
           config->dataloader_workers,
           config->dataloader_prefetch_factor,
           (size_t)config->dataloader_workers * (size_t)config->dataloader_prefetch_factor,
           (unsigned)GPT_RESERVED_DATA_SLOTS,
           mem->data_slots);
    if (config->overfit_num_samples > 0U && samples_per_epoch != (size_t)config->overfit_num_samples) {
        printf("overfit: requested=%llu using=%zu full-batch samples\n",
               (unsigned long long)config->overfit_num_samples,
               samples_per_epoch);
    }
    if (config->local_shard_cache_dir != NULL) {
        printf("shard_cache: dir=%s shards=%zu keep=%s\n",
               config->local_shard_cache_dir,
               train_shards != NULL ? train_shards->count : 0U,
               config->keep_shard_cache ? "yes" : "no");
    }
    printf("model: arch=%s vocab=%d d_model=%d layers=%d heads=%d head_dim=%d mlp_hidden=%d ffn=swiglu sdpa_window=%d minimax=(block=%d topk=%d init=%d local=%d) dropout=%.3f logits_softcap=%.3f\n",
           gpt_architecture_name(config->architecture),
           GPT_VOCAB_SIZE,
           GPT_D_MODEL,
           config->n_layers,
           GPT_N_HEADS,
           GPT_HEAD_DIM,
           GPT_MLP_HIDDEN,
           GPT_SDPA_WINDOW,
           GPT_MINIMAX_M3_BLOCK_SIZE,
           config->minimax_m3_topk_blocks,
           config->minimax_m3_init_blocks,
           config->minimax_m3_local_blocks,
           (double)config->dropout_p,
           (double)config->logits_softcap);
    if (config->generate_prompt != NULL || config->generate_every_n_steps > 0) {
        printf("generation: max_new_tokens=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f every_n_steps=%d batched_vowels=%s\n",
               config->max_new_tokens,
               (double)config->temperature,
               (double)config->min_p,
               (double)config->repetition_penalty,
               (double)config->logits_softcap,
               config->generate_every_n_steps,
               config->generate_every_n_steps > 0 ? "yes" : "no");
    }
    if (config->epochs > 0) {
        printf("checkpoint: save_best=%s path=%s save_latest=%s latest_path=%s val_split=%s eval_every_n_epochs=%d early_stopping_patience=%d\n",
               config->save_best ? "yes" : "no",
               config->checkpoint_path,
               config->save_latest ? "yes" : "no",
               config->latest_checkpoint_path,
               config->val_split,
               config->eval_every_n_epochs,
               config->early_stopping_patience);
        printf("optim: adamw lr_max=%.6g lr_min=%.6g warmup=%llu total=%llu weight_decay=%.4g grad_clip=%.4g amp_scale=%.1f\n",
               (double)lr_config->max_lr,
               (double)lr_config->min_lr,
               (unsigned long long)lr_config->warmup_steps,
               (unsigned long long)lr_config->total_steps,
               (double)config->weight_decay,
               (double)config->grad_clip_norm,
               (double)gd_amp_scaler_scale(scaler));
    } else {
        printf("optim: skipped (generation-only run)\n");
    }
    printf("memory: params=%zuMB state=%zuMB scratch_slot=%zuMBx%u data_slot=%zuMBx%u\n",
           mem->params_bytes / (1024U * 1024U),
           mem->state_bytes / (1024U * 1024U),
           mem->scratch_slot_bytes / (1024U * 1024U),
           mem->scratch_slots,
           mem->data_slot_bytes / (1024U * 1024U),
           mem->data_slots);
}


int main(int argc, char **argv)
{
    gpt_config config = gpt_config_from_yaml(parse_config_path(argc, argv));
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
    gpt_lm_special_ids special_ids;
    gpt_gdds_shard_list train_shards;
    gpt_lm_compact_transform train_transform;
    gpt_lm_compact_transform val_transform;
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
    memset(&special_ids, 0, sizeof(special_ids));
    memset(&train_shards, 0, sizeof(train_shards));
    memset(&train_transform, 0, sizeof(train_transform));
    memset(&val_transform, 0, sizeof(val_transform));

    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm: skipped (no supported gradients.c backend)\n");
        gpt_config_deinit(&config);
        return 0;
    }
    if (st != GD_OK) {
        gpt_fail_status(ctx, st, "gd_context_create", __LINE__);
    }
    if (GPT_D_MODEL != GPT_N_HEADS * GPT_HEAD_DIM) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "invalid GPT head config", __LINE__);
    }

    gpt_start_metrics(ctx, &config, &metrics);
    gpt_open_training_data(ctx,
                           &config,
                           &special_ids,
                           &train_transform,
                           &val_transform,
                           &dataset,
                           &val_dataset,
                           &train_shards,
                           &dataset_samples,
                           &dataset_tokens,
                           &val_samples,
                           &val_tokens,
                           &samples_per_epoch,
                           &steps_per_epoch,
                           &total_steps);

    gpt_lm_init(ctx, &model, &config);
    gpt_load_requested_checkpoint(ctx, &model, &config);
    {
        const gd_param_group groups[] = {
            gd_param_group_build("no_decay_norm", "gpt_lm.*norm_w", 1.0f, 0.0f, true),
            gd_param_group_build("no_decay_bias", "gpt_lm.*bias", 1.0f, 0.0f, true),
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
    gpt_create_training_optimizer(ctx, &config, &params, &optimizer, &scaler);
    lr_config = gpt_lr_scheduler_config(&config, total_steps);
    gpt_resume_training_checkpoint(ctx, &config, optimizer, scaler, &lr_config, &resume_state);

    TRY(ctx, gd_context_seal_params(ctx));

    gpt_print_startup_summary(&config,
                              &mem,
                              &lr_config,
                              scaler,
                              &train_shards,
                              dataset_samples,
                              dataset_tokens,
                              val_samples,
                              val_tokens,
                              steps_per_epoch,
                              samples_per_epoch,
                              total_steps);
    gpt_log_run_config_metrics(metrics,
                               &config,
                               &mem,
                               dataset_samples,
                               dataset_tokens,
                               val_samples,
                               val_tokens,
                               steps_per_epoch,
                               samples_per_epoch,
                               total_steps,
                               total_params,
                               trainable_params,
                               param_bytes);

    if (config.epochs > 0) {
        train_gpt(ctx,
                  &model,
                  dataset,
                  val_dataset,
                  optimizer,
                  scaler,
                  &lr_config,
                  &config,
                  &special_ids,
                  &train_shards,
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
    gpt_gdds_shard_list_deinit(&train_shards);
    gd_dataset_destroy(val_dataset);
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    gpt_config_deinit(&config);
    return exit_code;
}
