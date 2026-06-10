#include "gpt_lm_shared.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPT_GENERATE_LINE_MAX 8192U
#define GPT_GENERATE_IM_START "<|im_start|>"

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
            fprintf(stderr, "gpt_lm_generate: missing value for %s\n", name);
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
    printf("Interactive GPT LM generation. Type a prefix and press enter; :q, :quit, exit, or quit ends the session.\n");
    printf("Each prefix is encoded as <|im_start|> + your text.\n");
    printf("Generation stops early when the tokenizer emits <|im_end|>.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --checkpoint PATH      checkpoint to load (default: checkpoints/gpt_lm_best.gdckpt)\n");
    printf("  --data-dir PATH        data directory for default tokenizer (default: examples/gpt_lm/data)\n");
    printf("  --tokenizer-path PATH  tokenizer JSON; metadata value is used when available\n");
    printf("  --max-new-tokens N     generated token budget per prefix; capped by 512-token context (default: 512)\n");
    printf("  --temperature T        sampling temperature; 0 means greedy (default: 0)\n");
    printf("  --min-p P              min-p sampling cutoff relative to top token; 0 disables (default: 0)\n");
    printf("  --repetition-penalty P repetition penalty; 1 disables (default: 1)\n");
    printf("  --logits-softcap C     final logits softcap; metadata value is used when available\n");
    printf("  --layers N             model layers; metadata value is used when available\n");
    printf("  --seed N               sampling seed (default: %llu)\n", (unsigned long long)GPT_DEFAULT_SEED);
    printf("  --help                 show this help\n");
}

static gpt_config generate_config_default(void)
{
    gpt_config config;
    memset(&config, 0, sizeof(config));
    config.data_dir = "examples/gpt_lm/data";
    config.tokenizer_path = NULL;
    config.generate_prompt = "";
    config.checkpoint_path = "checkpoints/gpt_lm_best.gdckpt";
    config.load_checkpoint_path = NULL;
    config.val_split = "val";
    config.epochs = 0;
    config.batch_size = GPT_DEFAULT_BATCH_SIZE;
    config.n_layers = GPT_DEFAULT_LAYERS;
    config.report_every = GPT_DEFAULT_REPORT_EVERY;
    config.lr_warmup_steps = -1;
    config.max_new_tokens = GPT_CONTEXT_LENGTH;
    config.generate_every_n_steps = 0;
    config.epochs_set = true;
    config.save_best = false;
    config.save_latest = false;
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
    config.logits_softcap = 0.0f;
    return config;
}

static gpt_config parse_args(int argc,
                             char **argv,
                             bool *layers_set,
                             bool *tokenizer_set,
                             bool *softcap_set)
{
    gpt_config config = generate_config_default();
    int i;
    *layers_set = false;
    *tokenizer_set = false;
    *softcap_set = false;
    for (i = 1; i < argc; ++i) {
        const char *value;
        int64_t parsed_i64 = 0;
        uint64_t parsed_u64 = 0U;
        float parsed_f32 = 0.0f;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        value = arg_value(argc, argv, &i, "--checkpoint");
        if (value != NULL) {
            config.checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--checkpoint-path");
        if (value != NULL) {
            config.checkpoint_path = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--data-dir");
        if (value != NULL) {
            config.data_dir = value;
            continue;
        }
        value = arg_value(argc, argv, &i, "--tokenizer-path");
        if (value != NULL) {
            config.tokenizer_path = value;
            *tokenizer_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--max-new-tokens");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, GPT_CONTEXT_LENGTH, &parsed_i64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --max-new-tokens %s\n", value);
                exit(2);
            }
            config.max_new_tokens = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &i, "--temperature");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm_generate: invalid --temperature %s\n", value);
                exit(2);
            }
            config.temperature = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--min-p");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm_generate: invalid --min-p %s\n", value);
                exit(2);
            }
            config.min_p = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--repetition-penalty");
        if (value != NULL) {
            if (!parse_float_arg(value, 1.0f, 10.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm_generate: invalid --repetition-penalty %s\n", value);
                exit(2);
            }
            config.repetition_penalty = parsed_f32;
            continue;
        }
        value = arg_value(argc, argv, &i, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
                fprintf(stderr, "gpt_lm_generate: invalid --logits-softcap %s\n", value);
                exit(2);
            }
            config.logits_softcap = parsed_f32;
            *softcap_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--layers");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 96, &parsed_i64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --layers %s\n", value);
                exit(2);
            }
            config.n_layers = (int)parsed_i64;
            *layers_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--seed");
        if (value != NULL) {
            if (!parse_u64_arg(value, UINT64_MAX, &parsed_u64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --seed %s\n", value);
                exit(2);
            }
            config.seed = parsed_u64;
            continue;
        }
        fprintf(stderr, "gpt_lm_generate: unknown argument %s\n", argv[i]);
        print_usage(argv[0]);
        exit(2);
    }
    return config;
}

static bool metadata_value(const char *metadata,
                           size_t metadata_len,
                           const char *key,
                           char *out,
                           size_t out_size)
{
    const size_t key_len = strlen(key);
    size_t offset = 0U;
    if (metadata == NULL || key == NULL || out == NULL || out_size == 0U) {
        return false;
    }
    while (offset < metadata_len) {
        const size_t line_start = offset;
        size_t line_end = line_start;
        while (line_end < metadata_len && metadata[line_end] != '\n') {
            line_end += 1U;
        }
        if (line_end > line_start + key_len && strncmp(metadata + line_start, key, key_len) == 0 &&
            metadata[line_start + key_len] == '=') {
            const size_t value_start = line_start + key_len + 1U;
            const size_t value_len = line_end - value_start;
            if (value_len >= out_size) {
                return false;
            }
            memcpy(out, metadata + value_start, value_len);
            out[value_len] = '\0';
            return true;
        }
        offset = line_end < metadata_len ? line_end + 1U : line_end;
    }
    return false;
}

static void apply_checkpoint_metadata(gpt_config *config,
                                      const char *metadata,
                                      size_t metadata_len,
                                      bool layers_set,
                                      bool tokenizer_set,
                                      bool softcap_set,
                                      char *tokenizer_storage,
                                      size_t tokenizer_storage_size)
{
    char value[128];
    int64_t parsed_i64;
    float parsed_f32;
    if (config == NULL || metadata == NULL) {
        return;
    }
    if (!layers_set && metadata_value(metadata, metadata_len, "n_layers", value, sizeof(value)) &&
        parse_i64_arg(value, 1, 96, &parsed_i64)) {
        config->n_layers = (int)parsed_i64;
    }
    if (!softcap_set && metadata_value(metadata, metadata_len, "logits_softcap", value, sizeof(value)) &&
        parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
        config->logits_softcap = parsed_f32;
    }
    if (!tokenizer_set && metadata_value(metadata,
                                         metadata_len,
                                         "tokenizer_path",
                                         tokenizer_storage,
                                         tokenizer_storage_size)) {
        config->tokenizer_path = tokenizer_storage;
    }
}

static void trim_line(char *line)
{
    size_t len;
    if (line == NULL) {
        return;
    }
    len = strlen(line);
    while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) {
        line[len - 1U] = '\0';
        len -= 1U;
    }
}

static bool should_quit(const char *line)
{
    return line != NULL &&
           (strcmp(line, ":q") == 0 || strcmp(line, ":quit") == 0 ||
            strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0);
}

int main(int argc, char **argv)
{
    bool layers_set;
    bool tokenizer_set;
    bool softcap_set;
    gpt_config config = parse_args(argc, argv, &layers_set, &tokenizer_set, &softcap_set);
    char *metadata = NULL;
    size_t metadata_len = 0U;
    char tokenizer_from_metadata[1024];
    gd_memory_config mem;
    gd_context *ctx = NULL;
    gd_status st;
    gpt_lm model;
    gd_module_load_options load_options;
    char line[GPT_GENERATE_LINE_MAX];
    char prompt[GPT_GENERATE_LINE_MAX + sizeof(GPT_GENERATE_IM_START)];
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    tokenizer_from_metadata[0] = '\0';
    st = gd_checkpoint_read_metadata(config.checkpoint_path, &metadata, &metadata_len);
    if (st == GD_OK) {
        apply_checkpoint_metadata(&config,
                                  metadata,
                                  metadata_len,
                                  layers_set,
                                  tokenizer_set,
                                  softcap_set,
                                  tokenizer_from_metadata,
                                  sizeof(tokenizer_from_metadata));
    } else {
        fprintf(stderr, "gpt_lm_generate: warning: could not read checkpoint metadata (%s)\n",
                gd_status_string(st));
    }

    mem = gpt_memory_config(&config);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm_generate: skipped (no supported gradients.c backend)\n");
        free(metadata);
        return 0;
    }
    if (st != GD_OK) {
        gpt_fail_status(ctx, st, "gd_context_create", __LINE__);
    }
    if (GPT_D_MODEL != GPT_N_HEADS * GPT_HEAD_DIM) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "invalid GPT head config", __LINE__);
    }

    printf("interactive_generation: checkpoint=%s layers=%d max_new_tokens=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f\n",
           config.checkpoint_path,
           config.n_layers,
           config.max_new_tokens,
           (double)config.temperature,
           (double)config.min_p,
           (double)config.repetition_penalty,
           (double)config.logits_softcap);
    printf("prompt template: <|im_start|>{input}\n");
    printf("stop: <|im_end|> when present in tokenizer; commands: :q, :quit, quit, exit\n");

    gpt_lm_init(ctx, &model, &config);
    load_options.strict = true;
    load_options.load_buffers = true;
    TRY(ctx, gd_module_load_state(ctx, &model.mod, config.checkpoint_path, &load_options));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, false);

    for (;;) {
        printf("prefix> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            printf("\n");
            break;
        }
        if (strchr(line, '\n') == NULL && strlen(line) == sizeof(line) - 1U) {
            int ch;
            while ((ch = getchar()) != '\n' && ch != EOF) {
                /* discard overlong line tail */
            }
            fprintf(stderr, "gpt_lm_generate: prefix too long; max is %u bytes\n", GPT_GENERATE_LINE_MAX - 2U);
            continue;
        }
        trim_line(line);
        if (should_quit(line)) {
            break;
        }
        if (line[0] == '\0') {
            continue;
        }
        const int n = snprintf(prompt, sizeof(prompt), "%s%s", GPT_GENERATE_IM_START, line);
        if (n < 0 || (size_t)n >= sizeof(prompt)) {
            fprintf(stderr, "gpt_lm_generate: formatted prompt too long\n");
            continue;
        }
        config.generate_prompt = prompt;
        gpt_generate(ctx, &model, &config);
    }
    exit_code = 0;

    gpt_lm_deinit(&model);
    gd_context_destroy(ctx);
    free(metadata);
    return exit_code;
}
