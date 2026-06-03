/*
 * Real GPT LM training example.
 *
 * End-to-end path for docs/plan_gpt_preparation.md without generation/KV cache:
 *   optional bootstrap BPE + packed gdtok dataset from plain text
 *   async background-worker dataloaders
 *   reusable graph runners with runtime-bound batch inputs
 *   fused tied-LM CE, AMP, grad clip, mutable LR scheduler, parameter groups
 *   periodic validation perplexity
 *
 * Intended run (same knobs as tests/gpt_bench.c):
 *   GD_BENCH_DTYPE=f16 GD_BENCH_AMP=1 GD_BENCH_ATTN_WINDOW=512 \
 *   GD_GPT_MLP_POWLU=1 GD_BENCH_FUSED_LMCE=1 GD_METAL_MPS=1 GD_DEVICE=metal \
 *   build/examples/gpt-lm/gpt_lm --steps 1000
 */

#include "gradients/gradients.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#define GD_LM_MAX_PATH 4096U

#define CHECK_OK(expr)                                                           \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());        \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_ZERO(expr)                                                         \
    do {                                                                         \
        int rc_ = (expr);                                                        \
        if (rc_ != 0) {                                                          \
            return 1;                                                            \
        }                                                                        \
    } while (0)

typedef struct train_graph {
    gd_graph *graph;
    gd_graph_runner *runner;
    gd_graph_input *tokens_in;
    gd_graph_input *positions_in;
    gd_graph_input *targets_in;
    gd_tensor *logits;
    gd_tensor *loss;
    gd_tensor *scaled_loss;
    gd_tensor *grad_norm;
} train_graph;

typedef struct app_config {
    char corpus_path[GD_LM_MAX_PATH];
    char data_dir[GD_LM_MAX_PATH];
    char tokenizer_path[GD_LM_MAX_PATH];
    char train_path[GD_LM_MAX_PATH];
    char val_path[GD_LM_MAX_PATH];
    int have_train;
    int have_val;
    int prepare;
    int steps;
    int batch_size;
    int context_len;
    int eval_interval;
    int eval_batches;
    int train_eval;
    int log_interval;
    int num_workers;
    int prefetch_factor;
    uint64_t seed;
    int vocab_from_env;
} app_config;

static int env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return 0;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

static int env_flag_default(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    long parsed = 0;

    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtol(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0' || parsed < -2147483647L ||
        parsed > 2147483647L) {
        return fallback;
    }
    return (int)parsed;
}

static float env_float(const char *name, float fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    float parsed = 0.0F;

    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtof(v, &end);
    if (errno != 0 || end == v || *end != '\0') {
        return fallback;
    }
    return parsed;
}

static uint64_t env_u64(const char *name, uint64_t fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    unsigned long long parsed = 0ULL;

    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    errno = 0;
    parsed = strtoull(v, &end, 10);
    if (errno != 0 || end == v || *end != '\0') {
        return fallback;
    }
    return (uint64_t)parsed;
}

static double now_ms(void)
{
    struct timeval tv;
    (void)gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int copy_path(char *dst, size_t cap, const char *src)
{
    size_t n = 0U;

    if (dst == NULL || src == NULL || cap == 0U) {
        return 1;
    }
    n = strlen(src);
    if (n + 1U > cap) {
        return 1;
    }
    memcpy(dst, src, n + 1U);
    return 0;
}

static int join_path(char *dst, size_t cap, const char *dir, const char *leaf)
{
    int n = 0;
    const char *slash = "/";
    size_t dir_len = 0U;

    if (dst == NULL || dir == NULL || leaf == NULL || cap == 0U) {
        return 1;
    }
    dir_len = strlen(dir);
    if (dir_len > 0U && dir[dir_len - 1U] == '/') {
        slash = "";
    }
    n = snprintf(dst, cap, "%s%s%s", dir, slash, leaf);
    return n < 0 || (size_t)n >= cap;
}

static int file_exists(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int mkdir_p(const char *path)
{
    char tmp[GD_LM_MAX_PATH];
    char *p = NULL;
    size_t len = 0U;

    if (path == NULL || path[0] == '\0') {
        return 1;
    }
    if (copy_path(tmp, sizeof(tmp), path) != 0) {
        return 1;
    }
    len = strlen(tmp);
    if (len > 1U && tmp[len - 1U] == '/') {
        tmp[len - 1U] = '\0';
    }
    for (p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
                return 1;
            }
            *p = '/';
        }
    }
    if (mkdir(tmp, 0777) != 0 && errno != EEXIST) {
        return 1;
    }
    return 0;
}

static int parse_int_arg(const char *s, int *out)
{
    char *end = NULL;
    long v = 0;

    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 0L || v > 2147483647L) {
        return 1;
    }
    *out = (int)v;
    return 0;
}

static int parse_u64_arg(const char *s, uint64_t *out)
{
    char *end = NULL;
    unsigned long long v = 0ULL;

    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 1;
    }
    *out = (uint64_t)v;
    return 0;
}

static int has_run_intent(int argc)
{
    if (argc > 1) {
        return 1;
    }
    return getenv("GD_LM_TRAIN") != NULL || getenv("GD_LM_VAL") != NULL ||
           getenv("GD_LM_PREPARE") != NULL || getenv("GD_LM_STEPS") != NULL ||
           getenv("GD_BENCH_DTYPE") != NULL || getenv("GD_BENCH_AMP") != NULL ||
           getenv("GD_BENCH_FUSED_LMCE") != NULL || getenv("GD_BENCH_ATTN_WINDOW") != NULL;
}

static void usage(FILE *f)
{
    fprintf(f,
            "usage: gpt_lm [--prepare] [--corpus PATH] [--data-dir DIR] "
            "[--train train.gdtok --val val.gdtok] [options]\n"
            "\n"
            "options:\n"
            "  --prepare          train BPE and build train/val gdtok shards first\n"
            "  --corpus PATH      bootstrap plain-text corpus (default: $HOME/projects/dnn.c/docs/promessi_sposi.txt)\n"
            "  --data-dir DIR     output/input dataset dir (default: data/gpt-lm-promessi)\n"
            "  --tokenizer PATH   tokenizer path (default: DIR/tokenizer.json)\n"
            "  --train PATH       train .gdtok shard\n"
            "  --val PATH         validation .gdtok shard\n"
            "  --steps N          train steps (default: GD_LM_STEPS or 100)\n"
            "  --batch-size N     batch size (default: GD_LM_BATCH_SIZE/GD_BENCH_B or 1)\n"
            "  --context-len N    block/context length (default: GD_BENCH_T/GD_LM_CONTEXT_LEN or 512)\n"
            "  --eval-interval N  validation interval, 0 disables (default: 50)\n"
            "  --eval-batches N   validation batches per eval (default: 8)\n"
            "  --train-eval       report eval-mode train_loss with val_loss (default)\n"
            "  --no-train-eval    disable eval-mode train_loss\n"
            "  --log-interval N   train log interval (default: 10)\n"
            "  --num-workers N    dataloader workers (default: GD_LM_NUM_WORKERS or 1)\n"
            "  --prefetch-factor N batches queued per worker (default: GD_LM_PREFETCH_FACTOR or 2)\n"
            "  --seed N           master seed (default: 1234)\n"
            "\n"
            "Model/training knobs intentionally mirror gpt_bench:\n"
            "  GD_DEVICE=metal GD_BENCH_DTYPE=f16 GD_BENCH_AMP=1\n"
            "  GD_BENCH_FUSED_LMCE=1 GD_BENCH_ATTN_WINDOW=512 GD_METAL_MPS=1\n"
            "  GD_BENCH_DMODEL/LAYERS/HEADS/KV_HEADS/HEAD_DIM/DFF/VOCAB/DROPOUT\n");
}

static void default_corpus_path(char *out, size_t cap)
{
    const char *env = getenv("GD_LM_CORPUS");
    const char *home = getenv("HOME");
    int n = 0;

    if (env != NULL && env[0] != '\0') {
        (void)copy_path(out, cap, env);
        return;
    }
    if (home != NULL && home[0] != '\0') {
        n = snprintf(out, cap, "%s/projects/dnn.c/docs/promessi_sposi.txt", home);
        if (n >= 0 && (size_t)n < cap) {
            return;
        }
    }
    (void)copy_path(out, cap, "/Users/mascit/projects/dnn.c/docs/promessi_sposi.txt");
}

static void init_app_config(app_config *cfg)
{
    const char *s = NULL;

    memset(cfg, 0, sizeof(*cfg));
    default_corpus_path(cfg->corpus_path, sizeof(cfg->corpus_path));
    s = getenv("GD_LM_DATA_DIR");
    if (s == NULL || s[0] == '\0') {
        s = "data/gpt-lm-promessi";
    }
    (void)copy_path(cfg->data_dir, sizeof(cfg->data_dir), s);
    (void)join_path(cfg->tokenizer_path, sizeof(cfg->tokenizer_path), cfg->data_dir,
                    "tokenizer.json");
    s = getenv("GD_LM_TRAIN");
    if (s != NULL && s[0] != '\0') {
        cfg->have_train = copy_path(cfg->train_path, sizeof(cfg->train_path), s) == 0;
    }
    s = getenv("GD_LM_VAL");
    if (s != NULL && s[0] != '\0') {
        cfg->have_val = copy_path(cfg->val_path, sizeof(cfg->val_path), s) == 0;
    }
    cfg->prepare = env_flag_enabled("GD_LM_PREPARE");
    cfg->steps = env_int("GD_LM_STEPS", 100);
    cfg->batch_size = env_int("GD_LM_BATCH_SIZE", env_int("GD_BENCH_B", 1));
    cfg->context_len = env_int("GD_BENCH_T", env_int("GD_LM_CONTEXT_LEN", 512));
    cfg->eval_interval = env_int("GD_LM_EVAL_INTERVAL", 50);
    cfg->eval_batches = env_int("GD_LM_EVAL_BATCHES", 8);
    cfg->train_eval = env_flag_default("GD_LM_TRAIN_EVAL", 1);
    cfg->log_interval = env_int("GD_LM_LOG_INTERVAL", 10);
    cfg->num_workers = env_int("GD_LM_NUM_WORKERS", 1);
    cfg->prefetch_factor = env_int("GD_LM_PREFETCH_FACTOR", 2);
    cfg->seed = env_u64("GD_LM_SEED", 1234U);
    cfg->vocab_from_env = getenv("GD_BENCH_VOCAB") != NULL;
}

static int parse_args(int argc, char **argv, app_config *cfg)
{
    int i = 0;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            return 2;
        } else if (strcmp(argv[i], "--prepare") == 0) {
            cfg->prepare = 1;
        } else if (strcmp(argv[i], "--corpus") == 0) {
            if (i + 1 >= argc || copy_path(cfg->corpus_path, sizeof(cfg->corpus_path), argv[++i]) != 0) {
                fprintf(stderr, "invalid --corpus\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--data-dir") == 0) {
            if (i + 1 >= argc || copy_path(cfg->data_dir, sizeof(cfg->data_dir), argv[++i]) != 0) {
                fprintf(stderr, "invalid --data-dir\n");
                return 1;
            }
            if (join_path(cfg->tokenizer_path, sizeof(cfg->tokenizer_path), cfg->data_dir,
                          "tokenizer.json") != 0) {
                fprintf(stderr, "data dir path too long\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 >= argc || copy_path(cfg->tokenizer_path, sizeof(cfg->tokenizer_path), argv[++i]) != 0) {
                fprintf(stderr, "invalid --tokenizer\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--train") == 0) {
            if (i + 1 >= argc || copy_path(cfg->train_path, sizeof(cfg->train_path), argv[++i]) != 0) {
                fprintf(stderr, "invalid --train\n");
                return 1;
            }
            cfg->have_train = 1;
        } else if (strcmp(argv[i], "--val") == 0) {
            if (i + 1 >= argc || copy_path(cfg->val_path, sizeof(cfg->val_path), argv[++i]) != 0) {
                fprintf(stderr, "invalid --val\n");
                return 1;
            }
            cfg->have_val = 1;
        } else if (strcmp(argv[i], "--steps") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->steps) != 0) {
                fprintf(stderr, "invalid --steps\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--batch-size") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->batch_size) != 0) {
                fprintf(stderr, "invalid --batch-size\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--context-len") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->context_len) != 0) {
                fprintf(stderr, "invalid --context-len\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--eval-interval") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->eval_interval) != 0) {
                fprintf(stderr, "invalid --eval-interval\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--eval-batches") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->eval_batches) != 0) {
                fprintf(stderr, "invalid --eval-batches\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--train-eval") == 0) {
            cfg->train_eval = 1;
        } else if (strcmp(argv[i], "--no-train-eval") == 0) {
            cfg->train_eval = 0;
        } else if (strcmp(argv[i], "--log-interval") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->log_interval) != 0) {
                fprintf(stderr, "invalid --log-interval\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--num-workers") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->num_workers) != 0) {
                fprintf(stderr, "invalid --num-workers\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--prefetch-factor") == 0) {
            if (i + 1 >= argc || parse_int_arg(argv[++i], &cfg->prefetch_factor) != 0) {
                fprintf(stderr, "invalid --prefetch-factor\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || parse_u64_arg(argv[++i], &cfg->seed) != 0) {
                fprintf(stderr, "invalid --seed\n");
                return 1;
            }
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(stderr);
            return 1;
        }
    }
    return 0;
}

static long count_params(gd_tensor **params, int n)
{
    long total = 0;
    int i = 0;

    for (i = 0; i < n; ++i) {
        long k = 1;
        int d = 0;
        for (d = 0; d < gd_tensor_ndim(params[i]); ++d) {
            k *= (long)gd_tensor_size(params[i], d);
        }
        total += k;
    }
    return total;
}

static int resolve_dtype(gd_dtype *dtype_out)
{
    const char *dtype_env = getenv("GD_BENCH_DTYPE");

    *dtype_out = GD_DTYPE_F32;
    if (dtype_env == NULL || dtype_env[0] == '\0') {
        return 0;
    }
    if (strcmp(dtype_env, "f16") == 0 || strcmp(dtype_env, "fp16") == 0) {
        *dtype_out = GD_DTYPE_F16;
        return 0;
    }
    if (strcmp(dtype_env, "f32") == 0 || strcmp(dtype_env, "fp32") == 0) {
        *dtype_out = GD_DTYPE_F32;
        return 0;
    }
    fprintf(stderr, "config error: GD_BENCH_DTYPE must be f32 or f16\n");
    return 1;
}

static gd_gpt_mlp_kind resolve_mlp_kind(void)
{
    const char *mlp_env = getenv("GD_BENCH_MLP");

    if (env_flag_enabled("GD_GPT_MLP_POWLU")) {
        return GD_GPT_MLP_POWLU;
    }
    if (mlp_env != NULL && strcmp(mlp_env, "swiglu") == 0) {
        return GD_GPT_MLP_SWIGLU;
    }
    if (mlp_env != NULL && strcmp(mlp_env, "gelu") == 0) {
        return GD_GPT_MLP_GELU;
    }
    return GD_GPT_MLP_POWLU;
}

static int prepare_dataset(const app_config *app,
                           int requested_vocab,
                           char *train_out,
                           size_t train_cap,
                           char *val_out,
                           size_t val_cap)
{
    const char *inputs[1];
    const char *specials[3] = {"<|pad|>", "<|im_start|>", "<|im_end|>"};
    gd_bpe_train_config bpe;
    gd_dataset_build_config ds_cfg;
    gd_dataset_build_result result;
    gd_tokenizer *tok = NULL;
    gd_status status = GD_OK;

    if (!file_exists(app->corpus_path)) {
        fprintf(stderr, "corpus missing: %s\n", app->corpus_path);
        return 1;
    }
    if (mkdir_p(app->data_dir) != 0) {
        fprintf(stderr, "failed to create data dir: %s\n", app->data_dir);
        return 1;
    }

    memset(&bpe, 0, sizeof(bpe));
    inputs[0] = app->corpus_path;
    bpe.vocab_size = requested_vocab;
    bpe.min_frequency = env_int("GD_LM_BPE_MIN_FREQ", 2);
    bpe.split_digits = 1;
    bpe.n_special_tokens = 3;
    bpe.special_tokens = specials;
    bpe.seed = app->seed;

    printf("prepare: train BPE vocab=%d corpus=%s\n", requested_vocab, app->corpus_path);
    status = gd_bpe_tokenizer_train(inputs, 1, &bpe, &tok);
    if (status != GD_OK) {
        fprintf(stderr, "BPE train failed: %s\n", gd_last_error());
        return 1;
    }
    status = gd_bpe_tokenizer_save(tok, app->tokenizer_path);
    if (status != GD_OK) {
        fprintf(stderr, "tokenizer save failed: %s\n", gd_last_error());
        gd_tokenizer_destroy(tok);
        return 1;
    }
    printf("prepare: tokenizer=%s actual_vocab=%d hash=%016" PRIx64 "\n",
           app->tokenizer_path,
           gd_tokenizer_vocab_size(tok),
           gd_tokenizer_hash(tok));
    gd_tokenizer_destroy(tok);

    memset(&ds_cfg, 0, sizeof(ds_cfg));
    memset(&result, 0, sizeof(result));
    ds_cfg.tokenizer_path = app->tokenizer_path;
    ds_cfg.input_paths = inputs;
    ds_cfg.n_input_paths = 1;
    ds_cfg.output_dir = app->data_dir;
    ds_cfg.block_len = app->context_len;
    ds_cfg.train_ratio = 0.9;
    ds_cfg.val_ratio = 0.1;
    ds_cfg.seed = app->seed ^ UINT64_C(0x9e3779b97f4a7c15);
    ds_cfg.wrap_plain_text = 1;
    ds_cfg.im_start = "<|im_start|>";
    ds_cfg.im_end = "<|im_end|>";

    printf("prepare: build dataset block_len=%d out=%s\n", app->context_len, app->data_dir);
    status = gd_dataset_build(&ds_cfg, &result);
    if (status != GD_OK) {
        fprintf(stderr, "dataset build failed: %s\n", gd_last_error());
        gd_dataset_build_result_clear(&result);
        return 1;
    }
    if (result.train.n_samples == 0U || result.val.n_samples == 0U) {
        fprintf(stderr, "dataset build produced empty split: train=%llu val=%llu\n",
                (unsigned long long)result.train.n_samples,
                (unsigned long long)result.val.n_samples);
        gd_dataset_build_result_clear(&result);
        return 1;
    }
    if (copy_path(train_out, train_cap, result.train.shard_path) != 0 ||
        copy_path(val_out, val_cap, result.val.shard_path) != 0) {
        fprintf(stderr, "dataset shard path too long\n");
        gd_dataset_build_result_clear(&result);
        return 1;
    }
    printf("prepare: train=%s samples=%llu dropped=%llu\n",
           result.train.shard_path,
           (unsigned long long)result.train.n_samples,
           (unsigned long long)result.train.dropped_tail_tokens);
    printf("prepare: val=%s samples=%llu dropped=%llu\n",
           result.val.shard_path,
           (unsigned long long)result.val.n_samples,
           (unsigned long long)result.val.dropped_tail_tokens);
    gd_dataset_build_result_clear(&result);
    return 0;
}

static int make_lr_tensor(gd_context *ctx, gd_device device, gd_tensor **out)
{
    gd_tensor_desc desc;
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 0, NULL, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, out));
    return 0;
}

static int add_batch_inputs(gd_context *ctx,
                            gd_graph *graph,
                            gd_device target,
                            int batch_size,
                            int block_len,
                            gd_tensor **tokens,
                            gd_tensor **positions,
                            gd_tensor **targets,
                            train_graph *out)
{
    gd_tensor_desc desc;
    int64_t sizes[2];

    sizes[0] = batch_size;
    sizes[1] = block_len;
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, target, 2, sizes, &desc));
    CHECK_OK(gd_graph_add_input(ctx, graph, "tokens", &desc, tokens, &out->tokens_in));
    CHECK_OK(gd_graph_add_input(ctx, graph, "positions", &desc, positions, &out->positions_in));
    CHECK_OK(gd_graph_add_input(ctx, graph, "targets", &desc, targets, &out->targets_in));
    return 0;
}

static int build_train_graph(gd_context *ctx,
                             gd_device target,
                             gd_gpt *model,
                             gd_optimizer *opt,
                             gd_amp_scaler *scaler,
                             int batch_size,
                             int block_len,
                             gd_tensor *lr_tensor,
                             gd_tensor **params,
                             int n_params,
                             int fused_lmce,
                             int amp_enabled,
                             float grad_clip,
                             train_graph *out)
{
    gd_tensor *tokens = NULL;
    gd_tensor *positions = NULL;
    gd_tensor *targets = NULL;

    memset(out, 0, sizeof(*out));
    gd_gpt_set_training(model, true);
    CHECK_OK(gd_graph_create(ctx, &out->graph));
    CHECK_OK(gd_graph_begin(ctx, out->graph));
    CHECK_ZERO(add_batch_inputs(ctx, out->graph, target, batch_size, block_len,
                                &tokens, &positions, &targets, out));
    if (fused_lmce) {
        CHECK_OK(gd_gpt_forward_loss(ctx, model, tokens, positions, targets, &out->loss));
    } else {
        CHECK_OK(gd_gpt_forward(ctx, model, tokens, positions, &out->logits));
        CHECK_OK(gd_cross_entropy(ctx, out->logits, targets, 2, &out->loss));
    }
    if (amp_enabled) {
        CHECK_OK(gd_amp_scaler_scale_loss(ctx, scaler, out->loss, &out->scaled_loss));
        CHECK_OK(gd_backward(ctx, out->scaled_loss));
        if (grad_clip > 0.0F) {
            CHECK_OK(gd_optimizer_step_amp_clip_lr(ctx, opt, scaler, grad_clip, &out->grad_norm,
                                                   lr_tensor));
        } else {
            CHECK_OK(gd_optimizer_step_amp_lr(ctx, opt, scaler, lr_tensor));
        }
    } else {
        CHECK_OK(gd_backward(ctx, out->loss));
        if (grad_clip > 0.0F) {
            CHECK_OK(gd_clip_grad_norm(ctx, params, n_params, grad_clip, &out->grad_norm));
        }
        CHECK_OK(gd_optimizer_step_lr(ctx, opt, lr_tensor));
    }
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(tokens);
    gd_tensor_release(positions);
    gd_tensor_release(targets);
    CHECK_OK(gd_graph_compile(out->graph, target));
    CHECK_OK(gd_graph_runner_create(out->graph, &out->runner));
    return 0;
}

static int build_eval_graph(gd_context *ctx,
                            gd_device target,
                            gd_gpt *model,
                            int batch_size,
                            int block_len,
                            train_graph *out)
{
    gd_tensor *tokens = NULL;
    gd_tensor *positions = NULL;
    gd_tensor *targets = NULL;

    memset(out, 0, sizeof(*out));
    gd_gpt_set_training(model, false);
    CHECK_OK(gd_graph_create(ctx, &out->graph));
    CHECK_OK(gd_graph_begin(ctx, out->graph));
    CHECK_ZERO(add_batch_inputs(ctx, out->graph, target, batch_size, block_len,
                                &tokens, &positions, &targets, out));
    CHECK_OK(gd_gpt_forward_loss(ctx, model, tokens, positions, targets, &out->loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(tokens);
    gd_tensor_release(positions);
    gd_tensor_release(targets);
    CHECK_OK(gd_graph_compile(out->graph, target));
    CHECK_OK(gd_graph_runner_create(out->graph, &out->runner));
    return 0;
}

static int bind_batch(train_graph *graph, gd_batch *batch)
{
    gd_tensor *tokens = gd_batch_tensor(batch, "tokens");
    gd_tensor *positions = gd_batch_tensor(batch, "positions");
    gd_tensor *targets = gd_batch_tensor(batch, "targets");
    if (tokens == NULL || positions == NULL || targets == NULL) {
        fprintf(stderr, "batch missing LM fields\n");
        return 1;
    }
    CHECK_OK(gd_graph_runner_bind(graph->runner, graph->tokens_in, tokens));
    CHECK_OK(gd_graph_runner_bind(graph->runner, graph->positions_in, positions));
    CHECK_OK(gd_graph_runner_bind(graph->runner, graph->targets_in, targets));
    return 0;
}

static int destroy_graph_slot(train_graph *slot)
{
    gd_graph_runner_destroy(slot->runner);
    slot->runner = NULL;
    gd_tensor_release(slot->logits);
    gd_tensor_release(slot->loss);
    gd_tensor_release(slot->scaled_loss);
    gd_tensor_release(slot->grad_norm);
    slot->logits = NULL;
    slot->loss = NULL;
    slot->scaled_loss = NULL;
    slot->grad_norm = NULL;
    if (slot->graph != NULL) {
        CHECK_OK(gd_graph_reset(slot->graph));
        CHECK_OK(gd_graph_destroy(slot->graph));
        slot->graph = NULL;
    }
    return 0;
}

static int prefetch_n(gd_dataloader *dl, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
        CHECK_OK(gd_dataloader_prefetch(dl));
    }
    return 0;
}

static int run_eval(gd_context *ctx,
                    gd_device target,
                    gd_dataloader *val_dl,
                    train_graph *graph,
                    int eval_batches,
                    float *loss_out)
{
    double sum = 0.0;
    int i = 0;

    if (eval_batches <= 0) {
        *loss_out = 0.0F;
        return 0;
    }
    for (i = 0; i < eval_batches; ++i) {
        gd_batch *batch = NULL;
        float loss = 0.0F;
        CHECK_OK(gd_dataloader_next(val_dl, &batch));
        CHECK_ZERO(bind_batch(graph, batch));
        CHECK_OK(gd_graph_runner_run(graph->runner));
        CHECK_OK(gd_synchronize(ctx, target));
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, graph->loss, &loss, sizeof(loss)));
        CHECK_OK(gd_dataloader_release(val_dl, batch));
        CHECK_OK(gd_dataloader_prefetch(val_dl));
        sum += (double)loss;
    }
    *loss_out = (float)(sum / (double)eval_batches);
    return 0;
}

static void init_lm_batch_fields(gd_batch_field_desc *fields,
                                 int batch_size,
                                 int block_len)
{
    memset(fields, 0, 3U * sizeof(fields[0]));
    fields[0].name = "tokens";
    fields[0].dtype = GD_DTYPE_I32;
    fields[0].rank = 2;
    fields[0].sizes[0] = batch_size;
    fields[0].sizes[1] = block_len;
    fields[1].name = "targets";
    fields[1].dtype = GD_DTYPE_I32;
    fields[1].rank = 2;
    fields[1].sizes[0] = batch_size;
    fields[1].sizes[1] = block_len;
    fields[2].name = "positions";
    fields[2].dtype = GD_DTYPE_I32;
    fields[2].rank = 2;
    fields[2].sizes[0] = batch_size;
    fields[2].sizes[1] = block_len;
}

int main(int argc, char **argv)
{
    app_config app;
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");
    gd_dtype param_dtype = GD_DTYPE_F32;
    int amp_enabled = env_flag_enabled("GD_BENCH_AMP");
    int fused_lmce = env_int("GD_BENCH_FUSED_LMCE", 0) != 0;
    float grad_clip = env_float("GD_BENCH_GRAD_CLIP", 1.0F);
    float weight_decay = env_float("GD_LM_WEIGHT_DECAY", 0.1F);
    gd_gpt_config model_cfg;
    gd_adamw_config opt_cfg;
    gd_lr_scheduler_config lr_cfg;
    gd_amp_scaler_config scaler_cfg;
    gd_dataset *train_ds = NULL;
    gd_dataset *val_ds = NULL;
    gd_dataloader *train_dl = NULL;
    gd_dataloader *train_eval_dl = NULL;
    gd_dataloader *val_dl = NULL;
    gd_dataloader_config dl_cfg;
    gd_batch_field_desc lm_fields[3];
    uint64_t train_block_len = 0U;
    uint64_t val_block_len = 0U;
    uint64_t train_vocab_size = 0U;
    uint64_t train_tokenizer_hash = 0U;
    uint64_t val_tokenizer_hash = 0U;
    gd_gpt *model = NULL;
    gd_param_group *groups = NULL;
    int n_groups = 0;
    gd_tensor **params = NULL;
    int n_params = 0;
    gd_optimizer *opt = NULL;
    gd_amp_scaler *scaler = NULL;
    gd_tensor *lr_tensor = NULL;
    train_graph train_step;
    train_graph eval_step;
    int train_slots = 0;
    int train_eval_slots = 0;
    int val_slots = 0;
    const char *train_paths[1];
    const char *val_paths[1];
    long total_params = 0;
    int step = 0;
    int do_eval = 0;
    int do_train_eval = 0;
    int rc = 1;
    double t_start = 0.0;
    double t_last = 0.0;
    double train_loss_sum = 0.0;
    int train_loss_count = 0;
    float loss_last = 0.0F;
    uint64_t last_batches = 0U;
    uint64_t last_wait_ns = 0U;

    memset(&train_step, 0, sizeof(train_step));
    memset(&eval_step, 0, sizeof(eval_step));
    setvbuf(stdout, NULL, _IONBF, 0);

    init_app_config(&app);
    if (!has_run_intent(argc)) {
        usage(stdout);
        return 0;
    }
    {
        int parse = parse_args(argc, argv, &app);
        if (parse == 2) {
            return 0;
        }
        if (parse != 0) {
            return 2;
        }
    }
    if (app.steps <= 0 || app.batch_size <= 0 || app.context_len <= 0 ||
        app.log_interval <= 0 || app.eval_batches < 0 || app.num_workers <= 0 ||
        app.prefetch_factor <= 0) {
        fprintf(stderr, "invalid train shape/control config\n");
        return 2;
    }
    do_eval = app.eval_interval > 0 && app.eval_batches > 0;
    do_train_eval = do_eval && app.train_eval != 0;
    if (resolve_dtype(&param_dtype) != 0) {
        return 2;
    }

    if (!app.have_train || !app.have_val) {
        char default_train[GD_LM_MAX_PATH];
        char default_val[GD_LM_MAX_PATH];
        if (join_path(default_train, sizeof(default_train), app.data_dir, "train-00000.gdtok") != 0 ||
            join_path(default_val, sizeof(default_val), app.data_dir, "val-00000.gdtok") != 0) {
            fprintf(stderr, "default dataset path too long\n");
            return 2;
        }
        if (!app.prepare && file_exists(default_train) && file_exists(default_val)) {
            (void)copy_path(app.train_path, sizeof(app.train_path), default_train);
            (void)copy_path(app.val_path, sizeof(app.val_path), default_val);
            app.have_train = 1;
            app.have_val = 1;
        } else {
            app.prepare = 1;
        }
    }

    memset(&model_cfg, 0, sizeof(model_cfg));
    model_cfg.vocab_size = env_int("GD_BENCH_VOCAB", 8000);
    model_cfg.d_model = env_int("GD_BENCH_DMODEL", 256);
    model_cfg.n_layers = env_int("GD_BENCH_LAYERS", 6);
    model_cfg.n_heads = env_int("GD_BENCH_HEADS", 4);
    model_cfg.n_kv_heads = env_int("GD_BENCH_KV_HEADS", model_cfg.n_heads);
    model_cfg.head_dim = env_int("GD_BENCH_HEAD_DIM", 64);
    model_cfg.d_ff = env_int("GD_BENCH_DFF", 4 * model_cfg.d_model);
    model_cfg.max_seq_len = app.context_len;
    model_cfg.rope_theta = 10000.0F;
    model_cfg.norm_eps = 1e-5F;
    model_cfg.mlp_kind = resolve_mlp_kind();
    model_cfg.powlu_m = env_float("GD_POWLU_M", 3.0F);
    model_cfg.attention_window = env_int("GD_BENCH_ATTN_WINDOW", 0);
    model_cfg.param_dtype = param_dtype;
    model_cfg.tie_embeddings = true;
    model_cfg.dropout_p = env_float("GD_BENCH_DROPOUT", 0.0F);

    if (model_cfg.d_model != model_cfg.n_heads * model_cfg.head_dim) {
        fprintf(stderr, "config error: d_model (%d) != n_heads*head_dim (%d*%d)\n",
                model_cfg.d_model, model_cfg.n_heads, model_cfg.head_dim);
        return 2;
    }
    if (model_cfg.attention_window < 0) {
        fprintf(stderr, "config error: attention window must be non-negative\n");
        return 2;
    }
    if (!isfinite(model_cfg.dropout_p) || model_cfg.dropout_p < 0.0F ||
        model_cfg.dropout_p >= 1.0F) {
        fprintf(stderr, "config error: GD_BENCH_DROPOUT must satisfy 0 <= p < 1\n");
        return 2;
    }
    if (param_dtype == GD_DTYPE_F16 && !amp_enabled) {
        fprintf(stderr, "config error: GD_BENCH_DTYPE=f16 requires GD_BENCH_AMP=1\n");
        return 2;
    }
    if (param_dtype == GD_DTYPE_F16 && !fused_lmce) {
        fprintf(stderr, "config error: GD_BENCH_DTYPE=f16 requires GD_BENCH_FUSED_LMCE=1\n");
        return 2;
    }

    if (app.prepare) {
        if (prepare_dataset(&app, model_cfg.vocab_size, app.train_path, sizeof(app.train_path),
                            app.val_path, sizeof(app.val_path)) != 0) {
            return 1;
        }
        app.have_train = 1;
        app.have_val = 1;
    }
    if (!app.have_train || !app.have_val) {
        fprintf(stderr, "missing dataset; pass --train/--val or --prepare\n");
        usage(stderr);
        return 2;
    }

    CHECK_OK(gd_context_create(&ctx));
    if (dev_env != NULL && strcmp(dev_env, "metal") == 0) {
        gd_device metal = {GD_DEVICE_METAL, 0};
        if (gd_synchronize(ctx, metal) == GD_OK) {
            target = metal;
        } else {
            printf("device: cpu (metal unavailable)\n");
        }
    }
    if (target.type == GD_DEVICE_METAL && param_dtype == GD_DTYPE_F16 &&
        !env_flag_enabled("GD_METAL_MPS")) {
        fprintf(stderr, "config error: Metal F16 GPT needs GD_METAL_MPS=1\n");
        gd_context_destroy(ctx);
        return 2;
    }
    CHECK_OK(gd_context_set_default_device(ctx, target));

    train_paths[0] = app.train_path;
    val_paths[0] = app.val_path;
    CHECK_OK(gd_dataset_open_gdtok(train_paths, 1, &train_ds));
    CHECK_OK(gd_dataset_open_gdtok(val_paths, 1, &val_ds));
    CHECK_OK(gd_dataset_get_u64(train_ds, "block_len", &train_block_len));
    CHECK_OK(gd_dataset_get_u64(val_ds, "block_len", &val_block_len));
    CHECK_OK(gd_dataset_get_u64(train_ds, "vocab_size", &train_vocab_size));
    CHECK_OK(gd_dataset_get_u64(train_ds, "tokenizer_hash", &train_tokenizer_hash));
    CHECK_OK(gd_dataset_get_u64(val_ds, "tokenizer_hash", &val_tokenizer_hash));
    if (train_block_len != (uint64_t)app.context_len ||
        val_block_len != (uint64_t)app.context_len) {
        fprintf(stderr, "dataset block_len mismatch: train=%llu val=%llu expected=%d\n",
                (unsigned long long)train_block_len, (unsigned long long)val_block_len,
                app.context_len);
        goto cleanup;
    }
    if (train_tokenizer_hash != val_tokenizer_hash) {
        fprintf(stderr, "train/val tokenizer hash mismatch\n");
        goto cleanup;
    }
    if (!app.vocab_from_env) {
        model_cfg.vocab_size = (int)train_vocab_size;
    } else if (train_vocab_size > (uint64_t)model_cfg.vocab_size) {
        fprintf(stderr, "dataset vocab %llu exceeds model vocab %d\n",
                (unsigned long long)train_vocab_size, model_cfg.vocab_size);
        goto cleanup;
    }

    init_lm_batch_fields(lm_fields, app.batch_size, app.context_len);
    memset(&dl_cfg, 0, sizeof(dl_cfg));
    dl_cfg.batch_size = app.batch_size;
    dl_cfg.device = target;
    dl_cfg.num_workers = app.num_workers;
    dl_cfg.prefetch_factor = app.prefetch_factor;

    dl_cfg.seed = app.seed ^ UINT64_C(0x243f6a8885a308d3);
    dl_cfg.sampler = GD_SAMPLER_RANDOM_REPLACEMENT;
    dl_cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(train_ds);
    CHECK_OK(gd_dataloader_create(ctx, train_ds, &dl_cfg, lm_fields, 3,
                                  gd_collate_gdtok_lm, NULL, &train_dl));
    train_slots = gd_dataloader_slot_count(train_dl);

    if (do_train_eval) {
        dl_cfg.seed = app.seed ^ UINT64_C(0xa4093822299f31d0);
        dl_cfg.sampler = GD_SAMPLER_SEQUENTIAL;
        dl_cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(train_ds);
        CHECK_OK(gd_dataloader_create(ctx, train_ds, &dl_cfg, lm_fields, 3,
                                      gd_collate_gdtok_lm, NULL, &train_eval_dl));
        train_eval_slots = gd_dataloader_slot_count(train_eval_dl);
    }

    if (do_eval) {
        dl_cfg.seed = app.seed ^ UINT64_C(0x13198a2e03707344);
        dl_cfg.sampler = GD_SAMPLER_SEQUENTIAL;
        dl_cfg.expected_dataset_fingerprint = gd_dataset_fingerprint(val_ds);
        CHECK_OK(gd_dataloader_create(ctx, val_ds, &dl_cfg, lm_fields, 3,
                                      gd_collate_gdtok_lm, NULL, &val_dl));
        val_slots = gd_dataloader_slot_count(val_dl);
    }

    CHECK_ZERO(prefetch_n(train_dl, train_slots));
    if (do_train_eval) {
        CHECK_ZERO(prefetch_n(train_eval_dl, train_eval_slots));
    }
    if (do_eval) {
        CHECK_ZERO(prefetch_n(val_dl, val_slots));
    }

    CHECK_OK(gd_gpt_create(ctx, &model_cfg, app.seed ^ UINT64_C(0xdeadbeef), &model));
    CHECK_OK(gd_gpt_parameters(model, &params, &n_params));
    total_params = count_params(params, n_params);
    CHECK_OK(gd_gpt_parameter_groups(model, weight_decay, &groups, &n_groups));

    memset(&opt_cfg, 0, sizeof(opt_cfg));
    opt_cfg.lr = 0.0F;
    opt_cfg.beta1 = env_float("GD_LM_BETA1", 0.9F);
    opt_cfg.beta2 = env_float("GD_LM_BETA2", 0.999F);
    opt_cfg.eps = env_float("GD_LM_ADAM_EPS", 1e-8F);
    opt_cfg.weight_decay = 0.0F;
    opt_cfg.master_param_policy = GD_MASTER_PARAM_AUTO;
    CHECK_OK(gd_adamw_create_groups(ctx, groups, n_groups, &opt_cfg, &opt));
    gd_gpt_parameter_groups_free(groups, n_groups);
    groups = NULL;

    memset(&lr_cfg, 0, sizeof(lr_cfg));
    lr_cfg.max_lr = env_float("GD_BENCH_LR_MAX", 0.001F);
    lr_cfg.min_lr = env_float("GD_BENCH_LR_MIN", lr_cfg.max_lr * 0.1F);
    lr_cfg.warmup_steps = env_int("GD_BENCH_LR_WARMUP", app.steps < 100 ? app.steps / 10 : 100);
    lr_cfg.total_steps = env_int("GD_BENCH_LR_TOTAL", app.steps);
    if (lr_cfg.total_steps <= 0) {
        lr_cfg.total_steps = app.steps;
    }
    CHECK_ZERO(make_lr_tensor(ctx, target, &lr_tensor));
    if (amp_enabled) {
        memset(&scaler_cfg, 0, sizeof(scaler_cfg));
        scaler_cfg.init_scale = env_float("GD_BENCH_AMP_SCALE", 32768.0F);
        scaler_cfg.growth_factor = 2.0F;
        scaler_cfg.backoff_factor = 0.5F;
        scaler_cfg.growth_interval = env_int("GD_BENCH_AMP_GROWTH", 2000);
        scaler_cfg.min_scale = 1.0F;
        scaler_cfg.max_scale = 1048576.0F;
        CHECK_OK(gd_amp_scaler_create(ctx, &scaler_cfg, &scaler));
    }

    printf("gpt_lm\n");
    printf("  device      : %s\n", target.type == GD_DEVICE_METAL ? "metal" : "cpu");
    printf("  dtype       : %s%s%s\n", gd_dtype_name(model_cfg.param_dtype),
           amp_enabled ? " + amp" : "", fused_lmce ? " + fused_lmce" : "");
    printf("  data        : train=%s val=%s\n", app.train_path, app.val_path);
    printf("  samples     : train=%llu val=%llu block=%d batch=%d\n",
           (unsigned long long)gd_dataset_num_samples(train_ds),
           (unsigned long long)gd_dataset_num_samples(val_ds), app.context_len,
           app.batch_size);
    printf("  loader      : workers=%d prefetch_factor=%d slots=%d\n",
           app.num_workers, app.prefetch_factor, train_slots);
    printf("  tokenizer   : hash=%016" PRIx64 " dataset_vocab=%llu model_vocab=%d\n",
           train_tokenizer_hash, (unsigned long long)train_vocab_size,
           model_cfg.vocab_size);
    printf("  model       : d=%d L=%d H=%d Hkv=%d Dh=%d dff=%d params=%ld (%.2fM)\n",
           model_cfg.d_model, model_cfg.n_layers, model_cfg.n_heads, model_cfg.n_kv_heads,
           model_cfg.head_dim, model_cfg.d_ff, total_params, (double)total_params / 1e6);
    printf("  mlp/window  : %s window=%d dropout=%.3g\n",
           model_cfg.mlp_kind == GD_GPT_MLP_POWLU ? "powlu" :
           (model_cfg.mlp_kind == GD_GPT_MLP_SWIGLU ? "swiglu" : "gelu"),
           model_cfg.attention_window, (double)model_cfg.dropout_p);
    printf("  optim       : lr %.4g -> %.4g warmup=%d total=%d wd=%.4g clip=%.4g\n",
           (double)lr_cfg.max_lr, (double)lr_cfg.min_lr, lr_cfg.warmup_steps,
           lr_cfg.total_steps, (double)weight_decay, (double)grad_clip);
    if (amp_enabled) {
        printf("  amp         : scale=%.4g growth_interval=%d\n",
               (double)gd_amp_scaler_scale(scaler), scaler_cfg.growth_interval);
    }

    printf("  compiling train/eval graphs...\n");
    CHECK_ZERO(build_train_graph(ctx, target, model, opt, scaler, app.batch_size,
                                 app.context_len, lr_tensor, params, n_params, fused_lmce,
                                 amp_enabled, grad_clip, &train_step));
    if (do_eval) {
        CHECK_ZERO(build_eval_graph(ctx, target, model, app.batch_size, app.context_len,
                                    &eval_step));
    }

    printf("  training    : steps=%d log_interval=%d eval_interval=%d eval_batches=%d train_eval=%s\n",
           app.steps, app.log_interval, app.eval_interval, app.eval_batches,
           do_train_eval ? "on" : "off");
    {
        gd_dataloader_metrics m;
        gd_dataloader_metrics_get(train_dl, &m);
        last_batches = m.batches_returned;
        last_wait_ns = m.wait_for_batch_ns;
    }
    t_start = now_ms();
    t_last = t_start;
    for (step = 0; step < app.steps; ++step) {
        gd_batch *batch = NULL;
        train_graph *tg = &train_step;
        float lr = 0.0F;
        bool stepped = true;
        CHECK_OK(gd_dataloader_next(train_dl, &batch));
        CHECK_ZERO(bind_batch(tg, batch));
        CHECK_OK(gd_lr_scheduler_write(ctx, &lr_cfg, step, lr_tensor, &lr));
        CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
        CHECK_OK(gd_graph_runner_run(tg->runner));
        CHECK_OK(gd_synchronize(ctx, target));
        if (amp_enabled) {
            CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
        }
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, tg->loss, &loss_last, sizeof(loss_last)));
        if (!isfinite(loss_last)) {
            fprintf(stderr, "non-finite train loss\n");
            goto cleanup;
        }
        train_loss_sum += (double)loss_last;
        train_loss_count += 1;

        {
            int reset_log_window = (step + 1) % app.log_interval == 0 || step + 1 == app.steps;
            int emit_log = step == 0 || reset_log_window;
            if (emit_log) {
                gd_dataloader_metrics m;
                double t_now = now_ms();
                double dt = (t_now - t_last) / 1000.0;
                double loss_avg = train_loss_count > 0 ?
                    train_loss_sum / (double)train_loss_count : (double)loss_last;
                uint64_t db = 0U;
                uint64_t dwait = 0U;
                float norm = 0.0F;
                if (tg->grad_norm != NULL) {
                    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tg->grad_norm, &norm, sizeof(norm)));
                }
                gd_dataloader_metrics_get(train_dl, &m);
                db = m.batches_returned - last_batches;
                dwait = m.wait_for_batch_ns - last_wait_ns;
                printf("step %6d loss %.5f loss_last %.5f lr %.4g",
                       step + 1, loss_avg, (double)loss_last, (double)lr);
                if (tg->grad_norm != NULL) {
                    printf(" grad_norm %.4g", (double)norm);
                }
                if (amp_enabled) {
                    printf(" amp_scale %.4g %s", (double)gd_amp_scaler_scale(scaler),
                           stepped ? "step" : "skip");
                }
                if (dt > 0.0) {
                    double toks = (double)db * (double)app.batch_size * (double)app.context_len;
                    printf(" tok/s %.1f", toks / dt);
                }
                printf(" loader_wait_ms %.3f\n", (double)dwait / 1.0e6);
                if (reset_log_window) {
                    t_last = t_now;
                    last_batches = m.batches_returned;
                    last_wait_ns = m.wait_for_batch_ns;
                    train_loss_sum = 0.0;
                    train_loss_count = 0;
                }
            }
        }

        CHECK_OK(gd_dataloader_release(train_dl, batch));
        CHECK_OK(gd_dataloader_prefetch(train_dl));

        if (do_eval && ((step + 1) % app.eval_interval == 0 || step + 1 == app.steps)) {
            float train_eval_loss = 0.0F;
            float val_loss = 0.0F;
            double ppl = 0.0;
            if (do_train_eval) {
                CHECK_ZERO(run_eval(ctx, target, train_eval_dl, &eval_step,
                                    app.eval_batches, &train_eval_loss));
                if (!isfinite(train_eval_loss)) {
                    fprintf(stderr, "non-finite train eval loss\n");
                    goto cleanup;
                }
            }
            CHECK_ZERO(run_eval(ctx, target, val_dl, &eval_step, app.eval_batches, &val_loss));
            ppl = val_loss < 80.0F ? exp((double)val_loss) : HUGE_VAL;
            if (do_train_eval) {
                printf("eval step %6d train_loss %.5f val_loss %.5f gap %.5f ppl %.5g\n",
                       step + 1, (double)train_eval_loss, (double)val_loss,
                       (double)(val_loss - train_eval_loss), ppl);
            } else {
                printf("eval step %6d val_loss %.5f ppl %.5g\n",
                       step + 1, (double)val_loss, ppl);
            }
            if (!isfinite(val_loss)) {
                fprintf(stderr, "non-finite val loss\n");
                goto cleanup;
            }
        }
    }
    {
        gd_dataloader_metrics m;
        double elapsed = (now_ms() - t_start) / 1000.0;
        double toks = (double)app.steps * (double)app.batch_size * (double)app.context_len;
        gd_dataloader_metrics_get(train_dl, &m);
        printf("done steps=%d elapsed_s=%.3f avg_tok/s=%.1f\n", app.steps, elapsed,
               elapsed > 0.0 ? toks / elapsed : 0.0);
        if (m.batches_prepared > 0U) {
            printf("loader final prepared=%llu returned=%llu fill_ms/batch=%.3f copy_ms/batch=%.3f wait_ms_total=%.3f requests=%llu max_ready=%llu\n",
                   (unsigned long long)m.batches_prepared,
                   (unsigned long long)m.batches_returned,
                   (double)m.host_fill_ns / (double)m.batches_prepared / 1.0e6,
                   (double)m.host_to_device_copy_ns / (double)m.batches_prepared / 1.0e6,
                   (double)m.wait_for_batch_ns / 1.0e6,
                   (unsigned long long)m.prefetch_requests,
                   (unsigned long long)m.max_ready_depth);
        }
    }
    rc = 0;

cleanup:
    (void)destroy_graph_slot(&train_step);
    (void)destroy_graph_slot(&eval_step);
    gd_tensor_release(lr_tensor);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    gd_gpt_parameter_groups_free(groups, n_groups);
    gd_gpt_destroy(model);
    gd_dataloader_destroy(train_dl);
    gd_dataloader_destroy(train_eval_dl);
    gd_dataloader_destroy(val_dl);
    gd_dataset_destroy(train_ds);
    gd_dataset_destroy(val_ds);
    gd_context_destroy(ctx);
    return rc;
}
