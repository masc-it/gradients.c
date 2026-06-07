#include <gradients/gradients.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define GPT_VOCAB_SIZE 2048
#define GPT_CONTEXT_LENGTH 512
#define GPT_D_MODEL 256
#define GPT_N_HEADS 4
#define GPT_HEAD_DIM 64
#define GPT_SDPA_WINDOW 256
#define GPT_MLP_HIDDEN (4 * GPT_D_MODEL)

#define GPT_DEFAULT_LAYERS 7
#define GPT_DEFAULT_EPOCHS 2
#define GPT_DEFAULT_BATCH_SIZE 2
#define GPT_DEFAULT_REPORT_EVERY 10
#define GPT_DEFAULT_DROPOUT_P 0.10f
#define GPT_DEFAULT_LR_MAX 3.0e-4f
#define GPT_DEFAULT_LR_MIN 3.0e-5f
#define GPT_DEFAULT_WEIGHT_DECAY 0.10f
#define GPT_DEFAULT_RMS_EPS 1.0e-5f
#define GPT_DEFAULT_POWLU_M 2.0f
#define GPT_DEFAULT_SEED UINT64_C(0x6750746c6d5eed00)

#define GPT_ALIGNMENT 256U
#define GPT_MIN_PARAMS_BYTES (64ULL * 1024ULL * 1024ULL)
#define GPT_MIN_STATE_BYTES (128ULL * 1024ULL * 1024ULL)
#define GPT_MIN_SCRATCH_SLOT_BYTES (512ULL * 1024ULL * 1024ULL)
#define GPT_MIN_DATA_SLOT_BYTES (128ULL * 1024ULL * 1024ULL)
#define GPT_SCRATCH_RESERVE_BYTES (512ULL * 1024ULL * 1024ULL)

#define GPT_DROPOUT_EMBED UINT64_C(0x9e3779b97f4a7c15)
#define GPT_DROPOUT_ATTN UINT64_C(0xbf58476d1ce4e5b9)
#define GPT_DROPOUT_MLP UINT64_C(0x94d049bb133111eb)

typedef struct gpt_block {
    gd_module mod;
    gd_tensor attn_norm_w;
    gd_tensor mlp_norm_w;
    gd_linear_layer qkv_proj;
    gd_linear_layer attn_proj;
    gd_linear_layer up_gate;
    gd_linear_layer down_proj;
} gpt_block;

typedef struct gpt_lm {
    gd_module mod;
    int n_layers;
    int vocab_size;
    int context_length;
    int d_model;
    int n_heads;
    int head_dim;
    int mlp_hidden;
    int sdpa_window;
    float dropout_p;
    float rms_eps;
    float powlu_m;
    uint64_t dropout_seed;
    gd_tensor token_embedding;
    gd_tensor final_norm_w;
    gd_module_list blocks;
    gpt_block *block_items;
} gpt_lm;

typedef struct gpt_kv_cache {
    int batch_size;
    int max_seq;
    int n_layers;
    int n_heads;
    int head_dim;
    int32_t *pos;
    gd_tensor *k;
    gd_tensor *v;
} gpt_kv_cache;

typedef struct gpt_config {
    const char *data_dir;
    const char *tokenizer_path; /* NULL => data_dir/tokenizer-v2048.json. */
    const char *generate_prompt; /* NULL => training only. */
    int epochs;
    int batch_size;
    int n_layers;
    int report_every;
    int lr_warmup_steps; /* -1 => auto. */
    int max_new_tokens;
    int generate_every_n_steps;
    bool epochs_set;
    uint64_t overfit_num_samples; /* 0 => full shuffled dataset. */
    uint64_t seed;
    float dropout_p;
    float lr_max;
    float lr_min;
    float weight_decay;
    float temperature; /* <= 0 => greedy. */
} gpt_config;

static void fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "gpt_lm failed at line %d: %s -> %s (%s)\n",
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

static double wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static uint64_t splitmix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

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
    config.epochs = GPT_DEFAULT_EPOCHS;
    config.batch_size = GPT_DEFAULT_BATCH_SIZE;
    config.n_layers = GPT_DEFAULT_LAYERS;
    config.report_every = GPT_DEFAULT_REPORT_EVERY;
    config.lr_warmup_steps = -1;
    config.max_new_tokens = 64;
    config.generate_every_n_steps = 0;
    config.epochs_set = false;
    config.overfit_num_samples = 0U;
    config.seed = GPT_DEFAULT_SEED;
    config.dropout_p = GPT_DEFAULT_DROPOUT_P;
    config.lr_max = GPT_DEFAULT_LR_MAX;
    config.lr_min = GPT_DEFAULT_LR_MIN;
    config.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
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

static size_t size_max2(size_t a, size_t b)
{
    return a > b ? a : b;
}

static size_t checked_mul_size(size_t a, size_t b, const char *what)
{
    if (a != 0U && b > SIZE_MAX / a) {
        fprintf(stderr, "gpt_lm: size overflow while computing %s\n", what);
        exit(2);
    }
    return a * b;
}

static size_t checked_add_size(size_t a, size_t b, const char *what)
{
    if (b > SIZE_MAX - a) {
        fprintf(stderr, "gpt_lm: size overflow while computing %s\n", what);
        exit(2);
    }
    return a + b;
}

static size_t gpt_param_count_for_layers(int n_layers)
{
    size_t per_block = 0U;
    size_t total = 0U;
    total += checked_mul_size((size_t)GPT_VOCAB_SIZE, (size_t)GPT_D_MODEL, "embedding params");
    total += (size_t)GPT_D_MODEL;
    per_block += (size_t)(2 * GPT_D_MODEL);
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)(3 * GPT_D_MODEL), "qkv params");
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)GPT_D_MODEL, "attn out params");
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)(2 * GPT_MLP_HIDDEN), "up/gate params");
    per_block += checked_mul_size((size_t)GPT_MLP_HIDDEN, (size_t)GPT_D_MODEL, "down params");
    total += checked_mul_size(per_block, (size_t)n_layers, "block params");
    return total;
}

static gd_memory_config gpt_memory_config(const gpt_config *config)
{
    const size_t param_count = gpt_param_count_for_layers(config->n_layers);
    const size_t param_bytes = checked_mul_size(param_count, gd_dtype_size(GD_DTYPE_F16), "param bytes");
    const size_t adam_bytes = checked_mul_size(param_count, 3U * gd_dtype_size(GD_DTYPE_F32), "adam state bytes");
    const size_t tokens_per_batch = checked_mul_size((size_t)config->batch_size,
                                                     (size_t)GPT_CONTEXT_LENGTH,
                                                     "batch tokens");
    const size_t token_field_bytes = checked_mul_size(tokens_per_batch,
                                                      gd_dtype_size(GD_DTYPE_I32),
                                                      "packed token field bytes");
    const size_t hidden_bytes = checked_mul_size(tokens_per_batch,
                                                 (size_t)GPT_D_MODEL * gd_dtype_size(GD_DTYPE_F16),
                                                 "hidden activation bytes");
    const size_t logits_bytes = checked_mul_size(tokens_per_batch,
                                                 (size_t)GPT_VOCAB_SIZE * gd_dtype_size(GD_DTYPE_F16),
                                                 "logit activation bytes");
    const size_t activation_scratch = checked_mul_size(hidden_bytes,
                                                       (size_t)(96 + 20 * config->n_layers),
                                                       "activation scratch estimate");
    const size_t logits_scratch = checked_mul_size(logits_bytes,
                                                   8U,
                                                   "logits scratch estimate");
    gd_memory_config mem = gd_memory_config_default();
    mem.params_bytes = size_max2((size_t)GPT_MIN_PARAMS_BYTES,
                                 checked_add_size(param_bytes,
                                                  32U * 1024U * 1024U,
                                                  "param arena bytes"));
    mem.state_bytes = size_max2((size_t)GPT_MIN_STATE_BYTES,
                                checked_add_size(adam_bytes,
                                                 32U * 1024U * 1024U,
                                                 "state arena bytes"));
    mem.scratch_slot_bytes = size_max2(
        (size_t)GPT_MIN_SCRATCH_SLOT_BYTES,
        checked_add_size(checked_add_size(activation_scratch,
                                          logits_scratch,
                                          "scratch arena bytes"),
                         (size_t)GPT_SCRATCH_RESERVE_BYTES,
                         "scratch arena bytes"));
    mem.data_slot_bytes = size_max2(
        (size_t)GPT_MIN_DATA_SLOT_BYTES,
        checked_add_size(checked_mul_size(token_field_bytes,
                                          8U,
                                          "data arena packed token fields"),
                         64U * 1024U * 1024U,
                         "data arena bytes"));
    mem.scratch_slots = 1U;
    mem.data_slots = 2U;
    mem.default_alignment = GPT_ALIGNMENT;
    return mem;
}

static gd_tensor_spec tensor_spec_1d(gd_dtype dtype, int64_t dim)
{
    int64_t shape[1];
    shape[0] = dim;
    return gd_tensor_spec_make(dtype, gd_shape_make(1U, shape), GPT_ALIGNMENT);
}

static gd_tensor_spec tensor_spec_2d(gd_dtype dtype, int64_t dim0, int64_t dim1)
{
    int64_t shape[2];
    shape[0] = dim0;
    shape[1] = dim1;
    return gd_tensor_spec_make(dtype, gd_shape_make(2U, shape), GPT_ALIGNMENT);
}

static gd_linear_layer_config linear_config(int64_t in_features,
                                            int64_t out_features,
                                            uint64_t seed,
                                            float init_scale)
{
    gd_linear_layer_config cfg = gd_linear_layer_config_make(in_features,
                                                             out_features,
                                                             GD_DTYPE_F16,
                                                             seed);
    cfg.use_bias = false;
    cfg.alignment = GPT_ALIGNMENT;
    cfg.weight_low = -init_scale;
    cfg.weight_high = init_scale;
    return cfg;
}

static void gpt_block_init(gd_context *ctx,
                           gpt_block *block,
                           uint32_t index,
                           int n_layers,
                           uint64_t seed)
{
    const float base_scale = 0.02f;
    const float residual_scale = base_scale / sqrtf(2.0f * (float)n_layers);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec one = gd_init_one();
    const gd_linear_layer_config qkv_cfg = linear_config(GPT_D_MODEL,
                                                         3 * GPT_D_MODEL,
                                                         splitmix64(seed ^ ((uint64_t)index << 8) ^ 1U),
                                                         base_scale);
    const gd_linear_layer_config attn_cfg = linear_config(GPT_D_MODEL,
                                                          GPT_D_MODEL,
                                                          splitmix64(seed ^ ((uint64_t)index << 8) ^ 2U),
                                                          residual_scale);
    const gd_linear_layer_config up_gate_cfg = linear_config(GPT_D_MODEL,
                                                             2 * GPT_MLP_HIDDEN,
                                                             splitmix64(seed ^ ((uint64_t)index << 8) ^ 3U),
                                                             base_scale);
    const gd_linear_layer_config down_cfg = linear_config(GPT_MLP_HIDDEN,
                                                          GPT_D_MODEL,
                                                          splitmix64(seed ^ ((uint64_t)index << 8) ^ 4U),
                                                          residual_scale);
    char name[32];
    memset(block, 0, sizeof(*block));
    (void)snprintf(name, sizeof(name), "block_%u", (unsigned)index);
    TRY(ctx, gd_module_init(ctx, &block->mod, name));
    TRY(ctx, gd_module_param(ctx, &block->mod, "attn_norm_w", &norm_spec, &one, &block->attn_norm_w));
    TRY(ctx, gd_module_param(ctx, &block->mod, "mlp_norm_w", &norm_spec, &one, &block->mlp_norm_w));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "qkv_proj", &block->qkv_proj, &qkv_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "attn_proj", &block->attn_proj, &attn_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "up_gate", &block->up_gate, &up_gate_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "down_proj", &block->down_proj, &down_cfg));
}

static void gpt_lm_init(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    const gd_tensor_spec embed_spec = tensor_spec_2d(GD_DTYPE_F16, GPT_VOCAB_SIZE, GPT_D_MODEL);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec embed_init = gd_init_rand_uniform(splitmix64(config->seed ^ UINT64_C(0xabc001)),
                                                         -0.02f,
                                                         0.02f);
    const gd_init_spec one = gd_init_one();
    uint32_t i;
    memset(model, 0, sizeof(*model));
    model->n_layers = config->n_layers;
    model->vocab_size = GPT_VOCAB_SIZE;
    model->context_length = GPT_CONTEXT_LENGTH;
    model->d_model = GPT_D_MODEL;
    model->n_heads = GPT_N_HEADS;
    model->head_dim = GPT_HEAD_DIM;
    model->mlp_hidden = GPT_MLP_HIDDEN;
    model->sdpa_window = GPT_SDPA_WINDOW;
    model->dropout_p = config->dropout_p;
    model->rms_eps = GPT_DEFAULT_RMS_EPS;
    model->powlu_m = GPT_DEFAULT_POWLU_M;
    model->dropout_seed = splitmix64(config->seed ^ UINT64_C(0xd00d1234));

    TRY(ctx, gd_module_init(ctx, &model->mod, "gpt_lm"));
    TRY(ctx, gd_module_param(ctx,
                             &model->mod,
                             "token_embedding",
                             &embed_spec,
                             &embed_init,
                             &model->token_embedding));
    TRY(ctx, gd_module_param(ctx, &model->mod, "final_norm_w", &norm_spec, &one, &model->final_norm_w));
    model->block_items = (gpt_block *)calloc((size_t)model->n_layers, sizeof(model->block_items[0]));
    if (model->block_items == NULL) {
        fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "calloc blocks", __LINE__);
    }
    TRY(ctx, gd_module_list_init_child(ctx, &model->mod, "blocks", &model->blocks, (uint32_t)model->n_layers));
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        gpt_block_init(ctx, &model->block_items[i], i, model->n_layers, config->seed);
        TRY(ctx, gd_module_list_set(&model->blocks, i, &model->block_items[i].mod));
    }
}

static void gpt_block_deinit(gpt_block *block)
{
    if (block == NULL) {
        return;
    }
    gd_linear_layer_deinit(&block->down_proj);
    gd_linear_layer_deinit(&block->up_gate);
    gd_linear_layer_deinit(&block->attn_proj);
    gd_linear_layer_deinit(&block->qkv_proj);
    gd_module_deinit(&block->mod);
}

static void gpt_lm_deinit(gpt_lm *model)
{
    uint32_t i;
    if (model == NULL) {
        return;
    }
    if (model->block_items != NULL) {
        for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
            gpt_block_deinit(&model->block_items[i]);
        }
    }
    gd_module_list_deinit(&model->blocks);
    free(model->block_items);
    gd_module_deinit(&model->mod);
    memset(model, 0, sizeof(*model));
}

static gd_status gpt_kv_cache_init(gd_context *ctx,
                                   const gpt_lm *model,
                                   int batch_size,
                                   int max_seq,
                                   gpt_kv_cache *cache)
{
    int64_t shape[4];
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || batch_size <= 0 || max_seq <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(cache, 0, sizeof(*cache));
    cache->k = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->k[0]));
    cache->v = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->v[0]));
    cache->pos = (int32_t *)calloc((size_t)batch_size, sizeof(cache->pos[0]));
    if (cache->k == NULL || cache->v == NULL || cache->pos == NULL) {
        free(cache->k);
        free(cache->v);
        free(cache->pos);
        memset(cache, 0, sizeof(*cache));
        return GD_ERR_OUT_OF_MEMORY;
    }
    cache->batch_size = batch_size;
    cache->max_seq = max_seq;
    cache->n_layers = model->n_layers;
    cache->n_heads = model->n_heads;
    cache->head_dim = model->head_dim;
    shape[0] = batch_size;
    shape[1] = max_seq;
    shape[2] = model->n_heads;
    shape[3] = model->head_dim;
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        gd_status st = gd_tensor_empty(ctx,
                                       GD_ARENA_STATE,
                                       GD_DTYPE_F16,
                                       gd_shape_make(4U, shape),
                                       GPT_ALIGNMENT,
                                       &cache->k[i]);
        if (st != GD_OK) {
            free(cache->k);
            free(cache->v);
            free(cache->pos);
            memset(cache, 0, sizeof(*cache));
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F16,
                             gd_shape_make(4U, shape),
                             GPT_ALIGNMENT,
                             &cache->v[i]);
        if (st != GD_OK) {
            free(cache->k);
            free(cache->v);
            free(cache->pos);
            memset(cache, 0, sizeof(*cache));
            return st;
        }
        cache->k[i].is_leaf = false;
        cache->v[i].is_leaf = false;
    }
    return GD_OK;
}

static void gpt_kv_cache_deinit(gpt_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    free(cache->k);
    free(cache->v);
    free(cache->pos);
    memset(cache, 0, sizeof(*cache));
}

static gd_status gpt_block_forward(gd_context *ctx,
                                   gpt_lm *model,
                                   gpt_block *block,
                                   uint32_t block_index,
                                   const gd_tensor *x,
                                   const gd_tensor *positions,
                                   const gd_tensor *cu_seqlens,
                                   uint64_t step,
                                   gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_varlen_config sdpa_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_drop;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_tensor mlp_drop;
    gd_status st;
    int64_t n_tokens;
    int64_t q_shape[3];
    int64_t flat_shape[2];
    uint64_t site_seed;

    if (ctx == NULL || model == NULL || block == NULL || x == NULL || positions == NULL ||
        cu_seqlens == NULL || out == NULL || x->rank != 2U || x->shape[1] != GPT_D_MODEL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_tokens = x->shape[0];
    q_shape[0] = n_tokens;
    q_shape[1] = GPT_N_HEADS;
    q_shape[2] = GPT_HEAD_DIM;
    flat_shape[0] = n_tokens;
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(3U, q_shape), &q_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(3U, q_shape), &k_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(3U, q_shape), &v_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) {
        return st;
    }
    st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_ATTN ^ ((uint64_t)block_index << 32) ^ step);
    st = gd_dropout(ctx, &attn_proj, model->dropout_p, model->mod.training, site_seed, &attn_drop);
    if (st != GD_OK) {
        return st;
    }
    st = gd_add(ctx, &residual, &attn_drop, &attn_resid);
    if (st != GD_OK) {
        return st;
    }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) {
        return st;
    }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_MLP ^ ((uint64_t)block_index << 32) ^ step);
    st = gd_dropout(ctx, &mlp_proj, model->dropout_p, model->mod.training, site_seed, &mlp_drop);
    if (st != GD_OK) {
        return st;
    }
    return gd_add(ctx, &residual, &mlp_drop, out);
}

static gd_status gpt_lm_forward(gd_context *ctx,
                                gpt_lm *model,
                                const gd_tensor *input_ids,
                                const gd_tensor *target_ids,
                                const gd_tensor *positions,
                                const gd_tensor *cu_seqlens,
                                uint64_t step,
                                gd_tensor *loss)
{
    gd_tensor x;
    gd_tensor dropped;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_tensor logits;
    gd_status st;
    uint32_t i;
    uint64_t site_seed;

    if (ctx == NULL || model == NULL || input_ids == NULL || target_ids == NULL ||
        positions == NULL || cu_seqlens == NULL || loss == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (input_ids->rank != 1U || target_ids->rank != 1U || positions->rank != 1U ||
        input_ids->shape[0] != target_ids->shape[0] || input_ids->shape[0] != positions->shape[0]) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (input_ids->shape[0] <= 0 || (input_ids->shape[0] % GPT_CONTEXT_LENGTH) != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cu_seqlens->rank != 1U || cu_seqlens->shape[0] != input_ids->shape[0] / GPT_CONTEXT_LENGTH + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_EMBED ^ step);
    st = gd_dropout(ctx, &x, model->dropout_p, model->mod.training, site_seed, &dropped);
    if (st != GD_OK) {
        return st;
    }
    x = dropped;

    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_forward(ctx,
                               model,
                               &model->block_items[i],
                               i,
                               &x,
                               positions,
                               cu_seqlens,
                               step,
                               &block_out);
        if (st != GD_OK) {
            return st;
        }
        x = block_out;
    }

    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_transposed_weight(ctx, &final_norm, &model->token_embedding, NULL, &logits);
    if (st != GD_OK) {
        return st;
    }
    return gd_cross_entropy(ctx, &logits, target_ids, loss);
}

static gd_status gpt_block_prefill_cached(gd_context *ctx,
                                           gpt_lm *model,
                                           gpt_kv_cache *cache,
                                           gpt_block *block,
                                           uint32_t block_index,
                                           const gd_tensor *x,
                                           const gd_tensor *positions,
                                           const gd_tensor *cu_seqlens,
                                           const gd_tensor *cache_positions,
                                           gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_varlen_config sdpa_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t n_tokens;
    int64_t q_shape[3];
    int64_t flat_shape[2];

    if (ctx == NULL || model == NULL || cache == NULL || block == NULL || x == NULL ||
        positions == NULL || cu_seqlens == NULL || cache_positions == NULL || out == NULL ||
        x->rank != 2U || x->shape[1] != GPT_D_MODEL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_tokens = x->shape[0];
    if (n_tokens <= 0 || n_tokens > (int64_t)cache->batch_size * (int64_t)cache->max_seq ||
        block_index >= (uint32_t)cache->n_layers || cache_positions->rank != 1U ||
        cache_positions->shape[0] != cache->batch_size || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] != cache->batch_size + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    q_shape[0] = n_tokens;
    q_shape[1] = GPT_N_HEADS;
    q_shape[2] = GPT_HEAD_DIM;
    flat_shape[0] = n_tokens;
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(3U, q_shape), &q_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(3U, q_shape), &k_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(3U, q_shape), &v_view);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) { return st; }
    st = gd_kv_cache_append_packed(ctx,
                                    &cache->k[block_index],
                                    &cache->v[block_index],
                                    cache_positions,
                                    cu_seqlens,
                                    &k_rot,
                                    &v_view);
    if (st != GD_OK) { return st; }
    st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) { return st; }
    st = gd_add(ctx, &residual, &attn_proj, &attn_resid);
    if (st != GD_OK) { return st; }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) { return st; }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) { return st; }
    return gd_add(ctx, &residual, &mlp_proj, out);
}

static gd_status gpt_block_decode_cached(gd_context *ctx,
                                         gpt_lm *model,
                                         gpt_kv_cache *cache,
                                         gpt_block *block,
                                         uint32_t block_index,
                                         const gd_tensor *x,
                                         const gd_tensor *positions,
                                         const gd_tensor *cache_positions,
                                         gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_decode_config sdpa_cfg = {
        .scale = 0.0f,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t q_shape[4];
    int64_t flat_shape[2];

    if (ctx == NULL || model == NULL || cache == NULL || block == NULL || x == NULL ||
        positions == NULL || cache_positions == NULL || out == NULL || x->rank != 2U ||
        x->shape[1] != GPT_D_MODEL || x->shape[0] != cache->batch_size ||
        positions->rank != 1U || positions->shape[0] != cache->batch_size ||
        cache_positions->rank != 1U || cache_positions->shape[0] != cache->batch_size ||
        block_index >= (uint32_t)cache->n_layers) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    q_shape[0] = x->shape[0];
    q_shape[1] = 1;
    q_shape[2] = GPT_N_HEADS;
    q_shape[3] = GPT_HEAD_DIM;
    flat_shape[0] = x->shape[0];
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(4U, q_shape), &q_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(4U, q_shape), &k_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(4U, q_shape), &v_view);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) { return st; }
    st = gd_kv_cache_append_positions(ctx,
                                       &cache->k[block_index],
                                       &cache->v[block_index],
                                       cache_positions,
                                       &k_rot,
                                       &v_view);
    if (st != GD_OK) { return st; }
    st = gd_sdpa_decode_positions(ctx,
                                  &q_rot,
                                  &cache->k[block_index],
                                  &cache->v[block_index],
                                  cache_positions,
                                  &sdpa_cfg,
                                  &attn);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) { return st; }
    st = gd_add(ctx, &residual, &attn_proj, &attn_resid);
    if (st != GD_OK) { return st; }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) { return st; }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) { return st; }
    return gd_add(ctx, &residual, &mlp_proj, out);
}

static gd_status gpt_lm_prefill_logits(gd_context *ctx,
                                       gpt_lm *model,
                                       gpt_kv_cache *cache,
                                       const gd_tensor *input_ids,
                                       const gd_tensor *positions,
                                       const gd_tensor *cu_seqlens,
                                       const gd_tensor *cache_positions,
                                       gd_tensor *logits)
{
    gd_tensor x;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_status st;
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || input_ids == NULL || positions == NULL ||
        cu_seqlens == NULL || cache_positions == NULL || logits == NULL || input_ids->rank != 1U ||
        positions->rank != 1U || input_ids->shape[0] != positions->shape[0] ||
        input_ids->shape[0] <= 0 || cache_positions->rank != 1U ||
        cache_positions->shape[0] != cache->batch_size || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] != cache->batch_size + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_prefill_cached(ctx,
                                      model,
                                      cache,
                                      &model->block_items[i],
                                      i,
                                      &x,
                                      positions,
                                      cu_seqlens,
                                      cache_positions,
                                      &block_out);
        if (st != GD_OK) { return st; }
        x = block_out;
    }
    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) { return st; }
    st = gd_linear_transposed_weight(ctx, &final_norm, &model->token_embedding, NULL, logits);
    if (st != GD_OK) { return st; }
    return GD_OK;
}

static gd_status gpt_lm_decode_logits(gd_context *ctx,
                                      gpt_lm *model,
                                      gpt_kv_cache *cache,
                                      const gd_tensor *input_ids,
                                      const gd_tensor *positions,
                                      const gd_tensor *cache_positions,
                                      gd_tensor *logits)
{
    gd_tensor x;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_status st;
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || input_ids == NULL || positions == NULL ||
        cache_positions == NULL || logits == NULL || input_ids->rank != 1U || positions->rank != 1U ||
        input_ids->shape[0] != positions->shape[0] || input_ids->shape[0] != cache->batch_size ||
        cache_positions->rank != 1U || cache_positions->shape[0] != cache->batch_size) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_decode_cached(ctx,
                                     model,
                                     cache,
                                     &model->block_items[i],
                                     i,
                                     &x,
                                     positions,
                                     cache_positions,
                                     &block_out);
        if (st != GD_OK) { return st; }
        x = block_out;
    }
    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) { return st; }
    st = gd_linear_transposed_weight(ctx, &final_norm, &model->token_embedding, NULL, logits);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)cache->batch_size; ++i) {
        cache->pos[i] += 1;
    }
    return GD_OK;
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
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, name, line);
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

static void gpt_generate_vowels(gd_context *ctx,
                                gpt_lm *model,
                                const gpt_config *config,
                                size_t step);

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
    TRY(ctx, gd_optimizer_step_amp_lr(ctx, optimizer, scaler, lr));
    TRY(ctx, gd_end_step(ctx));
    TRY(ctx, gd_dataloader_release(loader, batch));
    TRY(ctx, gd_dataloader_prefetch(loader));

    if (report) {
        const double now = wall_seconds();
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

static void train_gpt(gd_context *ctx,
                      gpt_lm *model,
                      gd_dataset *dataset,
                      gd_optimizer *optimizer,
                      gd_amp_scaler *scaler,
                      const gd_lr_scheduler_config *lr_config,
                      const gpt_config *config,
                      size_t steps_per_epoch,
                      size_t total_steps)
{
    gd_sampler *sampler = NULL;
    gd_dataloader *loader = NULL;
    double last_report_time = wall_seconds();
    size_t last_report_step = 0U;
    size_t global_step = 0U;
    size_t epoch;

    gd_module_set_training(&model->mod, true);
    if (config->overfit_num_samples == 0U) {
        TRY(ctx, gd_sampler_create_random(dataset, config->seed ^ UINT64_C(0x51504c), &sampler));
        TRY(ctx, create_gdds_loader(ctx, dataset, sampler, config->batch_size, &loader));
        for (global_step = 0U; global_step < total_steps; ++global_step) {
            const size_t epoch_index = global_step / steps_per_epoch + 1U;
            const size_t epoch_step = global_step % steps_per_epoch + 1U;
            train_one_batch(ctx,
                            model,
                            loader,
                            optimizer,
                            scaler,
                            lr_config,
                            config,
                            global_step,
                            total_steps,
                            epoch_index,
                            epoch_step,
                            steps_per_epoch,
                            &last_report_time,
                            &last_report_step);
        }
        gd_dataloader_destroy(loader);
        gd_sampler_destroy(sampler);
        return;
    }

    for (epoch = 1U; epoch <= (size_t)config->epochs; ++epoch) {
        size_t epoch_step;
        TRY(ctx, create_gdds_loader(ctx, dataset, NULL, config->batch_size, &loader));
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
        loader = NULL;
    }
}

static char *gpt_default_tokenizer_path(const char *data_dir)
{
    const char *file = "tokenizer-v2048.json";
    const size_t data_len = strlen(data_dir);
    const bool need_sep = data_len == 0U || data_dir[data_len - 1U] != '/';
    const size_t file_len = strlen(file);
    char *path;
    if (data_len > SIZE_MAX - file_len - (need_sep ? 2U : 1U)) {
        return NULL;
    }
    path = (char *)malloc(data_len + file_len + (need_sep ? 2U : 1U));
    if (path == NULL) {
        return NULL;
    }
    (void)sprintf(path, "%s%s%s", data_dir, need_sep ? "/" : "", file);
    return path;
}

static gd_status gpt_data_i32_tensor(gd_context *ctx,
                                     const int32_t *values,
                                     int64_t count,
                                     gd_tensor *out)
{
    gd_status st;
    if (ctx == NULL || values == NULL || out == NULL || count <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_I32, GD_SHAPE(count), GPT_ALIGNMENT, out);
    if (st != GD_OK) {
        return st;
    }
    return gd_tensor_write(ctx, out, values, (size_t)count * sizeof(values[0]));
}

static int gpt_sample_next_token(const float *logits,
                                 int vocab_size,
                                 float temperature,
                                 uint64_t *rng)
{
    int best = 0;
    float best_value = logits[0];
    int i;
    for (i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    if (temperature <= 0.0f) {
        return best;
    }
    {
        double sum = 0.0;
        double target;
        uint64_t r;
        for (i = 0; i < vocab_size; ++i) {
            const double z = ((double)logits[i] - (double)best_value) / (double)temperature;
            sum += exp(z);
        }
        if (!(sum > 0.0) || !isfinite(sum)) {
            return best;
        }
        *rng = splitmix64(*rng);
        r = *rng >> 11;
        target = ((double)r * (1.0 / 9007199254740992.0)) * sum;
        for (i = 0; i < vocab_size; ++i) {
            const double z = ((double)logits[i] - (double)best_value) / (double)temperature;
            target -= exp(z);
            if (target <= 0.0) {
                return i;
            }
        }
    }
    return best;
}

static void gpt_generate_prompts(gd_context *ctx,
                                 gpt_lm *model,
                                 const gpt_config *config,
                                 const char *const *prompts,
                                 int n_prompts,
                                 const char *tag,
                                 size_t step,
                                 bool restore_training)
{
    gd_tokenizer *tok = NULL;
    gd_tokenizer_config tok_cfg;
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    int32_t **encoded = NULL;
    int32_t **seq_ids = NULL;
    int *n_encoded = NULL;
    int *prompt_offset = NULL;
    int *prompt_len = NULL;
    int *seq_len = NULL;
    int32_t *packed_ids = NULL;
    int32_t *packed_positions = NULL;
    int32_t *cu = NULL;
    int32_t *cache_pos_values = NULL;
    int32_t *decode_ids = NULL;
    int32_t *decode_positions = NULL;
    int *next_ids = NULL;
    gd_tensor *last_logits = NULL;
    float *logits_host = NULL;
    gpt_kv_cache cache;
    int total_prompt_tokens = 0;
    int max_new;
    int room_for_prompt;
    int generated = 0;
    int b;
    int i;
    uint64_t rng;
    double start;
    double elapsed;

    memset(&cache, 0, sizeof(cache));
    if (ctx == NULL || model == NULL || config == NULL || prompts == NULL || n_prompts <= 0) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "generation arguments", __LINE__);
    }
    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "tokenizer path allocation", __LINE__);
    }
    TRY(ctx, gd_bpe_tokenizer_load(tokenizer_path, &tok_cfg, &tok));

    encoded = (int32_t **)calloc((size_t)n_prompts, sizeof(encoded[0]));
    seq_ids = (int32_t **)calloc((size_t)n_prompts, sizeof(seq_ids[0]));
    n_encoded = (int *)calloc((size_t)n_prompts, sizeof(n_encoded[0]));
    prompt_offset = (int *)calloc((size_t)n_prompts, sizeof(prompt_offset[0]));
    prompt_len = (int *)calloc((size_t)n_prompts, sizeof(prompt_len[0]));
    seq_len = (int *)calloc((size_t)n_prompts, sizeof(seq_len[0]));
    cu = (int32_t *)calloc((size_t)n_prompts + 1U, sizeof(cu[0]));
    cache_pos_values = (int32_t *)calloc((size_t)n_prompts, sizeof(cache_pos_values[0]));
    decode_ids = (int32_t *)calloc((size_t)n_prompts, sizeof(decode_ids[0]));
    decode_positions = (int32_t *)calloc((size_t)n_prompts, sizeof(decode_positions[0]));
    next_ids = (int *)calloc((size_t)n_prompts, sizeof(next_ids[0]));
    last_logits = (gd_tensor *)calloc((size_t)n_prompts, sizeof(last_logits[0]));
    logits_host = (float *)malloc((size_t)n_prompts * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (encoded == NULL || seq_ids == NULL || n_encoded == NULL || prompt_offset == NULL ||
        prompt_len == NULL || seq_len == NULL || cu == NULL || cache_pos_values == NULL ||
        decode_ids == NULL || decode_positions == NULL || next_ids == NULL || last_logits == NULL ||
        logits_host == NULL) {
        fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation allocation", __LINE__);
    }

    max_new = config->max_new_tokens;
    room_for_prompt = GPT_CONTEXT_LENGTH - max_new;
    if (room_for_prompt <= 0) {
        fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "no context room for generation", __LINE__);
    }
    for (b = 0; b < n_prompts; ++b) {
        TRY(ctx, gd_tokenizer_encode(tok, prompts[b], &encoded[b], &n_encoded[b]));
        if (n_encoded[b] <= 0) {
            fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty generation prompt", __LINE__);
        }
        if (n_encoded[b] > room_for_prompt) {
            prompt_offset[b] = n_encoded[b] - room_for_prompt;
            printf("generate%s%s: prompt[%d] tokens=%d exceeds context budget; using last %d tokens\n",
                   tag != NULL ? ":" : "",
                   tag != NULL ? tag : "",
                   b,
                   n_encoded[b],
                   room_for_prompt);
        }
        prompt_len[b] = n_encoded[b] - prompt_offset[b];
        cu[b] = (int32_t)total_prompt_tokens;
        total_prompt_tokens += prompt_len[b];
        seq_ids[b] = (int32_t *)calloc((size_t)GPT_CONTEXT_LENGTH, sizeof(seq_ids[b][0]));
        if (seq_ids[b] == NULL) {
            fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation sequence allocation", __LINE__);
        }
        memcpy(seq_ids[b], encoded[b] + prompt_offset[b], (size_t)prompt_len[b] * sizeof(seq_ids[b][0]));
        seq_len[b] = prompt_len[b];
    }
    cu[n_prompts] = (int32_t)total_prompt_tokens;
    packed_ids = (int32_t *)calloc((size_t)total_prompt_tokens, sizeof(packed_ids[0]));
    packed_positions = (int32_t *)calloc((size_t)total_prompt_tokens, sizeof(packed_positions[0]));
    if (packed_ids == NULL || packed_positions == NULL) {
        fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation packed prompt allocation", __LINE__);
    }
    for (b = 0; b < n_prompts; ++b) {
        for (i = 0; i < prompt_len[b]; ++i) {
            const int dst = (int)cu[b] + i;
            packed_ids[dst] = seq_ids[b][i];
            packed_positions[dst] = cache_pos_values[b] + i;
        }
    }

    rng = splitmix64(config->seed ^ UINT64_C(0xdec0de1234567890) ^ (uint64_t)n_prompts ^ (uint64_t)step);
    TRY(ctx, gpt_kv_cache_init(ctx, model, n_prompts, GPT_CONTEXT_LENGTH, &cache));
    gd_module_set_training(&model->mod, false);
    start = wall_seconds();

    {
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
        TRY(ctx, gpt_data_i32_tensor(ctx, packed_ids, total_prompt_tokens, &ids_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, packed_positions, total_prompt_tokens, &pos_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, cu, n_prompts + 1, &cu_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, cache_pos_values, n_prompts, &cache_pos_t));
        TRY(ctx, gpt_lm_prefill_logits(ctx,
                                       model,
                                       &cache,
                                       &ids_t,
                                       &pos_t,
                                       &cu_t,
                                       &cache_pos_t,
                                       &logits));
        for (b = 0; b < n_prompts; ++b) {
            TRY(ctx, gd_tensor_slice(ctx, &logits, 0U, (int64_t)cu[b + 1] - 1, 1, &last_logits[b]));
        }
        TRY(ctx, gd_end_step(ctx));
        for (b = 0; b < n_prompts; ++b) {
            TRY(ctx, gd_tensor_read_f32(ctx,
                                        &last_logits[b],
                                        logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                        GPT_VOCAB_SIZE));
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                config->temperature,
                                                &rng);
            cache.pos[b] = prompt_len[b];
        }
    }

    while (generated < max_new) {
        for (b = 0; b < n_prompts; ++b) {
            seq_ids[b][seq_len[b]] = (int32_t)next_ids[b];
            seq_len[b] += 1;
        }
        generated += 1;
        if (generated >= max_new) {
            break;
        }
        for (b = 0; b < n_prompts; ++b) {
            decode_ids[b] = (int32_t)next_ids[b];
            decode_positions[b] = cache.pos[b];
            cache_pos_values[b] = cache.pos[b];
        }
        {
            gd_tensor ids_t;
            gd_tensor pos_t;
            gd_tensor cache_pos_t;
            gd_tensor logits;
            TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
            TRY(ctx, gpt_data_i32_tensor(ctx, decode_ids, n_prompts, &ids_t));
            TRY(ctx, gpt_data_i32_tensor(ctx, decode_positions, n_prompts, &pos_t));
            TRY(ctx, gpt_data_i32_tensor(ctx, cache_pos_values, n_prompts, &cache_pos_t));
            TRY(ctx, gpt_lm_decode_logits(ctx, model, &cache, &ids_t, &pos_t, &cache_pos_t, &logits));
            TRY(ctx, gd_end_step(ctx));
            TRY(ctx, gd_tensor_read_f32(ctx,
                                        &logits,
                                        logits_host,
                                        (size_t)n_prompts * (size_t)GPT_VOCAB_SIZE));
        }
        for (b = 0; b < n_prompts; ++b) {
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                config->temperature,
                                                &rng);
        }
    }

    elapsed = wall_seconds() - start;
    printf("generate%s%s: tokenizer=%s batch=%d generated=%d temperature=%.3f elapsed=%.3fs tok/s=%.1f",
           tag != NULL ? ":" : "",
           tag != NULL ? tag : "",
           tokenizer_path,
           n_prompts,
           generated,
           (double)config->temperature,
           elapsed,
           elapsed > 0.0 ? (double)((size_t)n_prompts * (size_t)generated) / elapsed : 0.0);
    if (step != 0U) {
        printf(" step=%zu", step);
    }
    printf("\n");
    for (b = 0; b < n_prompts; ++b) {
        char *decoded = NULL;
        char *decoded_new = NULL;
        TRY(ctx, gd_tokenizer_decode(tok, seq_ids[b], seq_len[b], &decoded));
        TRY(ctx, gd_tokenizer_decode(tok, seq_ids[b] + prompt_len[b], generated, &decoded_new));
        printf("  prefix=\"%s\" prompt_tokens=%d generated_text=%s\n",
               prompts[b],
               prompt_len[b],
               decoded_new != NULL ? decoded_new : "");
        if (n_prompts == 1) {
            printf("full_text:\n%s\n", decoded != NULL ? decoded : "");
        }
        gd_tokenizer_free(decoded_new);
        gd_tokenizer_free(decoded);
    }

    if (restore_training) {
        gd_module_set_training(&model->mod, true);
    }
    gpt_kv_cache_deinit(&cache);
    for (b = 0; b < n_prompts; ++b) {
        gd_tokenizer_free(encoded[b]);
        free(seq_ids[b]);
    }
    free(logits_host);
    free(last_logits);
    free(next_ids);
    free(decode_positions);
    free(decode_ids);
    free(cache_pos_values);
    free(cu);
    free(packed_positions);
    free(packed_ids);
    free(seq_len);
    free(prompt_len);
    free(prompt_offset);
    free(n_encoded);
    free(seq_ids);
    free(encoded);
    gd_tokenizer_destroy(tok);
    free(default_tok_path);
}

static void gpt_generate(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    const char *prompts[1];
    prompts[0] = config->generate_prompt;
    gpt_generate_prompts(ctx, model, config, prompts, 1, "user", 0U, false);
}

static void gpt_generate_vowels(gd_context *ctx,
                                gpt_lm *model,
                                const gpt_config *config,
                                size_t step)
{
    static const char *const prompts[5] = {"a", "e", "i", "o", "u"};
    gpt_generate_prompts(ctx, model, config, prompts, 5, "vowels", step, true);
}

int main(int argc, char **argv)
{
    const gpt_config config = parse_args(argc, argv);
    const gd_memory_config mem = gpt_memory_config(&config);
    gd_context *ctx = NULL;
    gd_status st = gd_context_create(&mem, &ctx);
    gd_dataset *dataset = NULL;
    gpt_lm model;
    gd_param_set params;
    gd_optimizer *optimizer = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_lr_scheduler_config lr_config;
    size_t dataset_samples = 0U;
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
        fail_status(ctx, st, "gd_context_create", __LINE__);
    }
    if (GPT_D_MODEL != GPT_N_HEADS * GPT_HEAD_DIM) {
        fail_status(ctx, GD_ERR_BAD_STATE, "invalid GPT head config", __LINE__);
    }

    if (config.epochs > 0) {
        TRY(ctx, gd_dataset_open_gdds_split(config.data_dir, "train", &dataset));
        dataset_samples = (size_t)gd_dataset_num_samples(dataset);
        steps_per_epoch = effective_steps_per_epoch((uint64_t)dataset_samples, &config, &samples_per_epoch);
        if (steps_per_epoch == 0U || (size_t)config.epochs > SIZE_MAX / steps_per_epoch) {
            fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "dataset too small for requested batch size", __LINE__);
        }
        total_steps = (size_t)config.epochs * steps_per_epoch;
    }

    gpt_lm_init(ctx, &model, &config);
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

    printf("dataset: dir=%s samples=%zu batch=%d context=%d steps_per_epoch=%zu samples_per_epoch=%zu epochs=%d total_steps=%zu%s\n",
           config.data_dir,
           dataset_samples,
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
        printf("optim: adamw lr_max=%.6g lr_min=%.6g warmup=%llu total=%llu weight_decay=%.4g amp_scale=%.1f\n",
               (double)lr_config.max_lr,
               (double)lr_config.min_lr,
               (unsigned long long)lr_config.warmup_steps,
               (unsigned long long)lr_config.total_steps,
               (double)config.weight_decay,
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
        train_gpt(ctx, &model, dataset, optimizer, scaler, &lr_config, &config, steps_per_epoch, total_steps);
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
    gd_dataset_destroy(dataset);
    gd_context_destroy(ctx);
    return exit_code;
}
