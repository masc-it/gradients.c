#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "gpt_lm_shared.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

#define GPT_GENERATE_LINE_MAX 8192U
#define GPT_GENERATE_PROMPT_MAX (64U * 1024U)
#define GPT_GENERATE_PASTE_POLL_USEC 50000L
#define GPT_GENERATE_PASTE_END ":end"
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

static void print_usage(const char *argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("Interactive GPT LM generation. Type or paste a prefix and press enter; :q, :quit, exit, or quit ends the session.\n");
    printf("Pasted multi-line input is grouped into one prompt when it arrives together.\n");
    printf("For explicit multi-line entry, type :paste, paste text, then finish with a line containing only :end.\n");
    printf("Each prefix is used exactly as typed after escape decoding; include <|im_start|> yourself if desired.\n");
    printf("Use literal \\n in a prefix to insert a newline, e.g. Termine: mangiare\\n\\n## Definizioni.\n");
    printf("Generation stops early when the tokenizer emits <|im_end|>.\n");
    printf("\n");
    printf("Options:\n");
    printf("  --checkpoint PATH      checkpoint to load (default: checkpoints/gpt_lm_best.gdckpt)\n");
    printf("  --data-dir PATH        data directory for default tokenizer (default: examples/gpt_lm/data)\n");
    printf("  --tokenizer-path PATH  tokenizer JSON; metadata value is used when available\n");
    printf("  --max-new-tokens N     generated token budget per prefix; capped by %d-token context (default: %d)\n",
           GPT_CONTEXT_LENGTH,
           GPT_CONTEXT_LENGTH);
    printf("  --temperature T        sampling temperature; 0 means greedy (default: 0)\n");
    printf("  --min-p P              min-p sampling cutoff relative to top token; 0 disables (default: 0)\n");
    printf("  --repetition-penalty P repetition penalty; 1 disables (default: 1)\n");
    printf("  --logits-softcap C     final logits softcap; metadata value is used when available\n");
    printf("  --layers N             model layers; metadata value is used when available\n");
    printf("  --architecture NAME    gpt or minimax_m3; metadata value is used when available\n");
    printf("  --minimax-topk-blocks N MiniMax M3 sparse top-k blocks (default/metadata: %d)\n", GPT_MINIMAX_M3_TOPK_BLOCKS);
    printf("  --minimax-init-blocks N MiniMax M3 forced initial blocks (default/metadata: %d)\n", GPT_MINIMAX_M3_INIT_BLOCKS);
    printf("  --minimax-local-blocks N MiniMax M3 forced local blocks (default/metadata: %d)\n", GPT_MINIMAX_M3_LOCAL_BLOCKS);
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
    config.architecture = GPT_ARCH_GPT;
    config.minimax_m3_topk_blocks = GPT_MINIMAX_M3_TOPK_BLOCKS;
    config.minimax_m3_init_blocks = GPT_MINIMAX_M3_INIT_BLOCKS;
    config.minimax_m3_local_blocks = GPT_MINIMAX_M3_LOCAL_BLOCKS;
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
    config.pad_token_id = -1;
    return config;
}

static gpt_config parse_args(int argc,
                             char **argv,
                             bool *layers_set,
                             bool *tokenizer_set,
                             bool *softcap_set,
                             bool *architecture_set,
                             bool *minimax_topk_set,
                             bool *minimax_init_set,
                             bool *minimax_local_set)
{
    gpt_config config = generate_config_default();
    int i;
    *layers_set = false;
    *tokenizer_set = false;
    *softcap_set = false;
    *architecture_set = false;
    *minimax_topk_set = false;
    *minimax_init_set = false;
    *minimax_local_set = false;
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
        value = arg_value(argc, argv, &i, "--architecture");
        if (value == NULL) {
            value = arg_value(argc, argv, &i, "--arch");
        }
        if (value != NULL) {
            if (!parse_architecture_arg(value, &config.architecture)) {
                fprintf(stderr, "gpt_lm_generate: invalid --architecture %s (expected gpt or minimax_m3)\n", value);
                exit(2);
            }
            *architecture_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--minimax-topk-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 16, &parsed_i64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --minimax-topk-blocks %s\n", value);
                exit(2);
            }
            config.minimax_m3_topk_blocks = (int)parsed_i64;
            *minimax_topk_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--minimax-init-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 16, &parsed_i64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --minimax-init-blocks %s\n", value);
                exit(2);
            }
            config.minimax_m3_init_blocks = (int)parsed_i64;
            *minimax_init_set = true;
            continue;
        }
        value = arg_value(argc, argv, &i, "--minimax-local-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 16, &parsed_i64)) {
                fprintf(stderr, "gpt_lm_generate: invalid --minimax-local-blocks %s\n", value);
                exit(2);
            }
            config.minimax_m3_local_blocks = (int)parsed_i64;
            *minimax_local_set = true;
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
                                      bool architecture_set,
                                      bool minimax_topk_set,
                                      bool minimax_init_set,
                                      bool minimax_local_set,
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
    if (!architecture_set && metadata_value(metadata, metadata_len, "architecture", value, sizeof(value))) {
        (void)parse_architecture_arg(value, &config->architecture);
    }
    if (!minimax_topk_set && metadata_value(metadata, metadata_len, "minimax_m3_topk_blocks", value, sizeof(value)) &&
        parse_i64_arg(value, 1, 16, &parsed_i64)) {
        config->minimax_m3_topk_blocks = (int)parsed_i64;
    }
    if (!minimax_init_set && metadata_value(metadata, metadata_len, "minimax_m3_init_blocks", value, sizeof(value)) &&
        parse_i64_arg(value, 0, 16, &parsed_i64)) {
        config->minimax_m3_init_blocks = (int)parsed_i64;
    }
    if (!minimax_local_set && metadata_value(metadata, metadata_len, "minimax_m3_local_blocks", value, sizeof(value)) &&
        parse_i64_arg(value, 0, 16, &parsed_i64)) {
        config->minimax_m3_local_blocks = (int)parsed_i64;
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

static bool is_paste_command(const char *line)
{
    return line != NULL && (strcmp(line, ":paste") == 0 || strcmp(line, ":p") == 0);
}

static bool is_paste_end_line(const char *line)
{
    const size_t marker_len = strlen(GPT_GENERATE_PASTE_END);
    size_t len;
    if (line == NULL) {
        return false;
    }
    len = strlen(line);
    while (len > 0U && (line[len - 1U] == '\n' || line[len - 1U] == '\r')) {
        len -= 1U;
    }
    return len == marker_len && strncmp(line, GPT_GENERATE_PASTE_END, marker_len) == 0;
}

static bool append_prompt_text(char *prompt, size_t prompt_size, size_t *prompt_len, const char *text)
{
    const size_t text_len = text != NULL ? strlen(text) : 0U;
    if (prompt == NULL || prompt_size == 0U || prompt_len == NULL || text == NULL ||
        *prompt_len >= prompt_size || text_len >= prompt_size - *prompt_len) {
        return false;
    }
    memcpy(prompt + *prompt_len, text, text_len);
    *prompt_len += text_len;
    prompt[*prompt_len] = '\0';
    return true;
}

static bool stdin_ready_soon(void)
{
    fd_set readfds;
    struct timeval timeout;
    const int fd = fileno(stdin);
    int selected;
    if (fd < 0) {
        return false;
    }
    FD_ZERO(&readfds);
    FD_SET(fd, &readfds);
    timeout.tv_sec = 0;
    timeout.tv_usec = GPT_GENERATE_PASTE_POLL_USEC;
    selected = select(fd + 1, &readfds, NULL, NULL, &timeout);
    return selected > 0 && FD_ISSET(fd, &readfds);
}

static void drain_ready_input(void)
{
    char discard[GPT_GENERATE_LINE_MAX];
    while (stdin_ready_soon()) {
        if (fgets(discard, sizeof(discard), stdin) == NULL) {
            break;
        }
    }
}

static void discard_paste_until_end(void)
{
    char line[GPT_GENERATE_LINE_MAX];
    bool at_line_start = true;
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (at_line_start && is_paste_end_line(line)) {
            break;
        }
        at_line_start = strchr(line, '\n') != NULL;
    }
}

static int read_paste_prompt(char *prompt, size_t prompt_size)
{
    char line[GPT_GENERATE_LINE_MAX];
    size_t prompt_len = 0U;
    bool at_line_start = true;
    if (prompt == NULL || prompt_size == 0U) {
        return -1;
    }
    prompt[0] = '\0';
    printf("paste mode: paste prompt text, then finish with a line containing only %s\n", GPT_GENERATE_PASTE_END);
    fflush(stdout);
    for (;;) {
        if (fgets(line, sizeof(line), stdin) == NULL) {
            printf("\n");
            return prompt_len > 0U ? 1 : 0;
        }
        if (at_line_start && is_paste_end_line(line)) {
            break;
        }
        if (!append_prompt_text(prompt, prompt_size, &prompt_len, line)) {
            fprintf(stderr, "gpt_lm_generate: pasted prompt too long; max is %u bytes\n",
                    GPT_GENERATE_PROMPT_MAX - 1U);
            discard_paste_until_end();
            return -1;
        }
        at_line_start = strchr(line, '\n') != NULL;
    }
    trim_line(prompt);
    return 1;
}

static int collect_prompt_from_first_line(const char *first_line, char *prompt, size_t prompt_size)
{
    char line[GPT_GENERATE_LINE_MAX];
    const char *chunk = first_line;
    size_t prompt_len = 0U;
    if (first_line == NULL || prompt == NULL || prompt_size == 0U) {
        return -1;
    }
    prompt[0] = '\0';
    for (;;) {
        if (!append_prompt_text(prompt, prompt_size, &prompt_len, chunk)) {
            fprintf(stderr, "gpt_lm_generate: prompt too long; max is %u bytes\n",
                    GPT_GENERATE_PROMPT_MAX - 1U);
            drain_ready_input();
            return -1;
        }
        if (strchr(chunk, '\n') == NULL && strlen(chunk) == GPT_GENERATE_LINE_MAX - 1U) {
            if (fgets(line, sizeof(line), stdin) == NULL) {
                break;
            }
            chunk = line;
            continue;
        }
        if (!stdin_ready_soon()) {
            break;
        }
        if (fgets(line, sizeof(line), stdin) == NULL) {
            break;
        }
        chunk = line;
    }
    trim_line(prompt);
    return 1;
}

static void decode_interactive_escapes(char *line)
{
    char *read;
    char *write;
    if (line == NULL) {
        return;
    }
    read = line;
    write = line;
    while (*read != '\0') {
        if (read[0] == '\\' && read[1] != '\0') {
            if (read[1] == 'n') {
                *write++ = '\n';
                read += 2;
                continue;
            }
            if (read[1] == 'r') {
                *write++ = '\r';
                read += 2;
                continue;
            }
            if (read[1] == 't') {
                *write++ = '\t';
                read += 2;
                continue;
            }
            if (read[1] == '\\') {
                *write++ = '\\';
                read += 2;
                continue;
            }
        }
        *write++ = *read++;
    }
    *write = '\0';
}

int main(int argc, char **argv)
{
    bool layers_set;
    bool tokenizer_set;
    bool softcap_set;
    bool architecture_set;
    bool minimax_topk_set;
    bool minimax_init_set;
    bool minimax_local_set;
    gpt_config config = parse_args(argc,
                                   argv,
                                   &layers_set,
                                   &tokenizer_set,
                                   &softcap_set,
                                   &architecture_set,
                                   &minimax_topk_set,
                                   &minimax_init_set,
                                   &minimax_local_set);
    char *metadata = NULL;
    size_t metadata_len = 0U;
    char tokenizer_from_metadata[1024];
    gd_memory_config mem;
    gd_context *ctx = NULL;
    gd_status st;
    gpt_lm model;
    gpt_generation_tokenizer generation_tokenizer;
    bool generation_tokenizer_ready = false;
    gd_module_load_options load_options;
    char line[GPT_GENERATE_LINE_MAX];
    char command_line[GPT_GENERATE_LINE_MAX];
    char prompt[GPT_GENERATE_PROMPT_MAX];
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    memset(&generation_tokenizer, 0, sizeof(generation_tokenizer));
    tokenizer_from_metadata[0] = '\0';
    st = gd_checkpoint_read_metadata(config.checkpoint_path, &metadata, &metadata_len);
    if (st == GD_OK) {
        apply_checkpoint_metadata(&config,
                                  metadata,
                                  metadata_len,
                                  layers_set,
                                  tokenizer_set,
                                  softcap_set,
                                  architecture_set,
                                  minimax_topk_set,
                                  minimax_init_set,
                                  minimax_local_set,
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

    printf("interactive_generation: checkpoint=%s arch=%s layers=%d minimax=(topk=%d init=%d local=%d) max_new_tokens=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f\n",
           config.checkpoint_path,
           gpt_architecture_name(config.architecture),
           config.n_layers,
           config.minimax_m3_topk_blocks,
           config.minimax_m3_init_blocks,
           config.minimax_m3_local_blocks,
           config.max_new_tokens,
           (double)config.temperature,
           (double)config.min_p,
           (double)config.repetition_penalty,
           (double)config.logits_softcap);
    printf("prompt template: {input}\n");
    printf("note: no automatic <|im_start|> prefix is added; type it explicitly if desired\n");
    printf("multi-line: paste normally, or type :paste and finish with a line containing only :end\n");
    printf("escapes: type literal \\n for newline, \\t for tab, \\\\ for backslash\n");
    printf("stop: <|im_end|> when present in tokenizer; commands: :q, :quit, quit, exit\n");

    gpt_lm_init(ctx, &model, &config);
    load_options.strict = true;
    load_options.load_buffers = true;
    TRY(ctx, gd_module_load_state(ctx, &model.mod, config.checkpoint_path, &load_options));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, false);
    gpt_generation_tokenizer_init(ctx, &config, &generation_tokenizer);
    generation_tokenizer_ready = true;

    for (;;) {
        int prompt_status;
        printf("prefix> ");
        fflush(stdout);
        if (fgets(line, sizeof(line), stdin) == NULL) {
            printf("\n");
            break;
        }
        (void)snprintf(command_line, sizeof(command_line), "%s", line);
        trim_line(command_line);
        if (should_quit(command_line)) {
            break;
        }
        if (is_paste_command(command_line)) {
            prompt_status = read_paste_prompt(prompt, sizeof(prompt));
        } else {
            prompt_status = collect_prompt_from_first_line(line, prompt, sizeof(prompt));
        }
        if (prompt_status == 0) {
            break;
        }
        if (prompt_status < 0) {
            continue;
        }
        decode_interactive_escapes(prompt);
        if (prompt[0] == '\0') {
            continue;
        }
        /* No automatic <|im_start|> prepend: send the decoded prompt verbatim. */
        config.generate_prompt = prompt;
        {
            const double prompt_start = gpt_wall_seconds();
            const int generated = gpt_generate_with_tokenizer(ctx, &model, &config, &generation_tokenizer);
            const double prompt_elapsed = gpt_wall_seconds() - prompt_start;
            printf("interactive:user: generated=%d wall_elapsed=%.3fs wall_tok/s=%.1f\n",
                   generated,
                   prompt_elapsed,
                   prompt_elapsed > 0.0 ? (double)generated / prompt_elapsed : 0.0);
        }
    }
    exit_code = 0;

    if (generation_tokenizer_ready) {
        gpt_generation_tokenizer_deinit(&generation_tokenizer);
    }
    gpt_lm_deinit(&model);
    gd_context_destroy(ctx);
    free(metadata);
    return exit_code;
}
