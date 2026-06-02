/*
 * GPT-VLM ImageNet trainer skeleton.
 *
 * Uses packed variable-length sequences and gd_sdpa_varlen. Config is forced to
 * F16 + Dh=64 + causal prefix-window attention so Metal picks optimized
 * gd_sdpa_varlen_prefix_window_lane8_dh64_f16 kernels.
 */

#include "gradients/gradients.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#define GD_VLM_MAX_PATH 4096U
#define GD_VLM_IGNORE_INDEX (-100)
#define GD_VLM_GRAPH_MAX_BATCH 256

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

typedef struct app_config {
    char data_dir[GD_VLM_MAX_PATH];
    char split[64];
    int epochs;
    int steps;
    int batch_size;
    int max_text_len;
    int log_interval;
    int skip_batches;
    uint64_t seed;
    float lr;
    float weight_decay;
    float grad_clip;
    int d_model;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int d_ff;
    int attention_window;
    int text_bucket_multiple;
    int graph_cache_max;
    int graph_cache_footprint_mb;
} app_config;

typedef struct packed_batch {
    int batch_size;
    int prefix_len;
    int patch_dim;
    int max_seq;
    int n_tokens;
    int n_text;
    int text_bucket;
    int *text_lens;       /* padded text lengths used by graph [B] */
    int *actual_text_lens;/* true truncated text lengths [B], debug/loss mask */
    uint64_t *sample_ids;/* dataset ids [B], debug only */
    uint16_t *patches;   /* f16 [B*P, patch_dim] */
    int32_t *text_tokens;/* i32 [Ntext], shift-right suffix inputs */
    int32_t *positions;  /* i32 [N] */
    int32_t *targets;    /* i32 [N], ignore prefix */
    int32_t *cu;         /* i32 [B+1] */
} packed_batch;

typedef struct train_graph {
    gd_graph *graph;
    gd_graph_runner *runner;
    gd_graph_input *patches_in;
    gd_graph_input *text_tokens_in;
    gd_graph_input *positions_in;
    gd_graph_input *targets_in;
    gd_graph_input *cu_in;
    gd_tensor *loss;
    gd_tensor *scaled_loss;
    gd_tensor *grad_norm;
} train_graph;

typedef struct materialized_batch {
    gd_tensor *patches;
    gd_tensor *text_tokens;
    gd_tensor *positions;
    gd_tensor *targets;
    gd_tensor *cu;
} materialized_batch;

typedef struct train_graph_cache_entry {
    int batch_size;
    int text_bucket;
    uint64_t last_used;
    uint64_t runs;
    train_graph graph;
    materialized_batch inputs;
    double footprint_mb;
} train_graph_cache_entry;

typedef struct train_graph_cache {
    train_graph_cache_entry *entries;
    int n_entries;
    int cap;
    int max_entries;
    int footprint_limit_mb;
    double avg_entry_footprint_mb;
    uint64_t builds;
    uint64_t hits;
    uint64_t evictions;
} train_graph_cache;

typedef struct epoch_sampler {
    uint64_t *order;
    uint64_t n_samples;
    uint64_t cursor;
    int epoch;
    uint64_t rng_state;
} epoch_sampler;

static int env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return 0;
    }
    return strcmp(v, "0") != 0 && strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
           strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
}

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    long parsed;
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
    float parsed;
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
    unsigned long long parsed;
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

static void format_eta(double seconds, char *out, size_t out_cap)
{
    uint64_t total;
    uint64_t days;
    uint64_t hours;
    uint64_t minutes;
    uint64_t secs;

    if (out == NULL || out_cap == 0U) {
        return;
    }
    if (!isfinite(seconds) || seconds < 0.0) {
        seconds = 0.0;
    }
    total = (uint64_t)(seconds + 0.5);
    days = total / 86400U;
    hours = (total / 3600U) % 24U;
    minutes = (total / 60U) % 60U;
    secs = total % 60U;
    if (days > 0U) {
        (void)snprintf(out, out_cap, "%llud %lluh",
                       (unsigned long long)days, (unsigned long long)hours);
    } else if (hours > 0U) {
        (void)snprintf(out, out_cap, "%lluh %02llum",
                       (unsigned long long)hours, (unsigned long long)minutes);
    } else if (minutes > 0U) {
        (void)snprintf(out, out_cap, "%llum %02llus",
                       (unsigned long long)minutes, (unsigned long long)secs);
    } else {
        (void)snprintf(out, out_cap, "%llus", (unsigned long long)secs);
    }
}

static double current_footprint_mb(void)
{
#ifdef __APPLE__
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr;
    memset(&info, 0, sizeof(info));
    kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        return (double)NAN;
    }
    return (double)info.phys_footprint / (1024.0 * 1024.0);
#else
    return (double)NAN;
#endif
}

static int copy_path(char *dst, size_t cap, const char *src)
{
    size_t n;
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

static int parse_int_arg(const char *s, int *out)
{
    char *end = NULL;
    long v;
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

static int parse_float_arg(const char *s, float *out)
{
    char *end = NULL;
    float v;
    if (s == NULL || out == NULL) {
        return 1;
    }
    errno = 0;
    v = strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) {
        return 1;
    }
    *out = v;
    return 0;
}

static void usage(FILE *f)
{
    fprintf(f, "usage: gpt_vlm --data-dir DIR [options]\n\n");
    fprintf(f, "options:\n");
    fprintf(f, "  --data-dir DIR       directory with train.idx/train-*.gdvlm\n");
    fprintf(f, "  --split NAME         split tag (default: train)\n");
    fprintf(f, "  --epochs N           full shuffled epochs (default: GD_VLM_EPOCHS or 1)\n");
    fprintf(f, "  --steps N            debug cap in train steps; 0 disables (default: GD_VLM_STEPS or 0)\n");
    fprintf(f, "  --batch-size N       batch size (default: GD_VLM_BATCH_SIZE or 2)\n");
    fprintf(f, "  --max-text-len N     truncate suffix tokens to N (default: dataset max)\n");
    fprintf(f, "  --log-interval N     log interval (default: 1)\n");
    fprintf(f, "  --skip-batches N     advance sampler by N batches before step 1 (debug)\n");
    fprintf(f, "  --lr F               learning rate (default: 1e-4)\n");
    fprintf(f, "  --seed N             seed (default: 1234)\n\n");
    fprintf(f, "Bucket/cache knobs: GD_VLM_TEXT_BUCKET_MULTIPLE (default 8), GD_VLM_GRAPH_CACHE_MAX (default 16), GD_VLM_GRAPH_CACHE_FOOTPRINT_MB (default 0 disables).\n");
    fprintf(f, "Model knobs: GD_BENCH_DMODEL/LAYERS/HEADS/KV_HEADS/DFF and GD_BENCH_ATTN_WINDOW.\n");
    fprintf(f, "Hard requirements: dtype=f16, head_dim=64, prefix_len>0, sliding_window>0.\n");
}

static void init_app(app_config *app)
{
    memset(app, 0, sizeof(*app));
    (void)copy_path(app->split, sizeof(app->split), "train");
    app->epochs = env_int("GD_VLM_EPOCHS", 1);
    app->steps = env_int("GD_VLM_STEPS", 0);
    app->batch_size = env_int("GD_VLM_BATCH_SIZE", 2);
    app->max_text_len = env_int("GD_VLM_MAX_TEXT_LEN", 0);
    app->log_interval = env_int("GD_VLM_LOG_INTERVAL", 1);
    app->skip_batches = env_int("GD_VLM_SKIP_BATCHES", 0);
    app->seed = env_u64("GD_VLM_SEED", 1234U);
    app->lr = env_float("GD_VLM_LR", 1.0e-4F);
    app->weight_decay = env_float("GD_VLM_WEIGHT_DECAY", 4.0e-3F);
    app->grad_clip = env_float("GD_VLM_GRAD_CLIP", 1.0F);
    app->d_model = env_int("GD_BENCH_DMODEL", 256);
    app->n_layers = env_int("GD_BENCH_LAYERS", 6);
    app->n_heads = env_int("GD_BENCH_HEADS", 4);
    app->n_kv_heads = env_int("GD_BENCH_KV_HEADS", app->n_heads);
    app->d_ff = env_int("GD_BENCH_DFF", app->d_model * 4);
    app->attention_window = env_int("GD_BENCH_ATTN_WINDOW", 128);
    app->text_bucket_multiple = env_int("GD_VLM_TEXT_BUCKET_MULTIPLE", 8);
    app->graph_cache_max = env_int("GD_VLM_GRAPH_CACHE_MAX", 16);
    app->graph_cache_footprint_mb = env_int("GD_VLM_GRAPH_CACHE_FOOTPRINT_MB", 0);
}

static int parse_args(int argc, char **argv, app_config *app)
{
    int i;
    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0) {
            usage(stdout);
            return 2;
        } else if (strcmp(a, "--data-dir") == 0 && i + 1 < argc) {
            if (copy_path(app->data_dir, sizeof(app->data_dir), argv[++i]) != 0) {
                fprintf(stderr, "data dir too long\n");
                return 1;
            }
        } else if (strcmp(a, "--split") == 0 && i + 1 < argc) {
            if (copy_path(app->split, sizeof(app->split), argv[++i]) != 0) {
                fprintf(stderr, "split too long\n");
                return 1;
            }
        } else if (strcmp(a, "--epochs") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->epochs) != 0) { return 1; }
        } else if (strcmp(a, "--steps") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->steps) != 0) { return 1; }
        } else if (strcmp(a, "--batch-size") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->batch_size) != 0) { return 1; }
        } else if (strcmp(a, "--max-text-len") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->max_text_len) != 0) { return 1; }
        } else if (strcmp(a, "--log-interval") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->log_interval) != 0) { return 1; }
        } else if (strcmp(a, "--skip-batches") == 0 && i + 1 < argc) {
            if (parse_int_arg(argv[++i], &app->skip_batches) != 0) { return 1; }
        } else if (strcmp(a, "--lr") == 0 && i + 1 < argc) {
            if (parse_float_arg(argv[++i], &app->lr) != 0) { return 1; }
        } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            uint64_t seed;
            char *end = NULL;
            errno = 0;
            seed = (uint64_t)strtoull(argv[++i], &end, 10);
            if (errno != 0 || end == argv[i] || *end != '\0') { return 1; }
            app->seed = seed;
        } else {
            fprintf(stderr, "unknown/invalid arg: %s\n", a);
            return 1;
        }
    }
    return 0;
}

static uint64_t splitmix64(uint64_t *state)
{
    uint64_t z;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static int epoch_sampler_shuffle(epoch_sampler *s)
{
    uint64_t i;
    if (s == NULL || s->order == NULL || s->n_samples == 0U) {
        fprintf(stderr, "invalid epoch sampler\n");
        return 1;
    }
    for (i = s->n_samples; i > 1U; --i) {
        uint64_t j = splitmix64(&s->rng_state) % i;
        uint64_t tmp = s->order[(size_t)(i - 1U)];
        s->order[(size_t)(i - 1U)] = s->order[(size_t)j];
        s->order[(size_t)j] = tmp;
    }
    return 0;
}

static int epoch_sampler_begin(epoch_sampler *s, int epoch)
{
    if (s == NULL || epoch < 0) {
        fprintf(stderr, "invalid epoch sampler begin\n");
        return 1;
    }
    s->epoch = epoch;
    s->cursor = 0U;
    return epoch_sampler_shuffle(s);
}

static int epoch_sampler_init(epoch_sampler *s, uint64_t n_samples, uint64_t seed)
{
    uint64_t i;
    if (s == NULL || n_samples == 0U) {
        fprintf(stderr, "invalid epoch sampler init\n");
        return 1;
    }
    memset(s, 0, sizeof(*s));
    if (n_samples > (uint64_t)(SIZE_MAX / sizeof(*s->order))) {
        fprintf(stderr, "epoch sampler too large\n");
        return 1;
    }
    s->order = (uint64_t *)malloc((size_t)n_samples * sizeof(*s->order));
    if (s->order == NULL) {
        fprintf(stderr, "epoch sampler allocation failed\n");
        return 1;
    }
    s->n_samples = n_samples;
    s->rng_state = seed;
    for (i = 0U; i < n_samples; ++i) {
        s->order[(size_t)i] = i;
    }
    return epoch_sampler_begin(s, 0);
}

static void epoch_sampler_destroy(epoch_sampler *s)
{
    if (s == NULL) {
        return;
    }
    free(s->order);
    memset(s, 0, sizeof(*s));
}

static int epoch_sampler_take_batch(epoch_sampler *s,
                                    int max_batch,
                                    uint64_t *ids,
                                    int *batch_out,
                                    int *epoch_out,
                                    uint64_t *offset_out)
{
    uint64_t remaining;
    uint64_t take;
    uint64_t i;
    if (s == NULL || max_batch <= 0 || s->n_samples == 0U) {
        fprintf(stderr, "invalid epoch sampler batch\n");
        return 1;
    }
    if (s->cursor >= s->n_samples) {
        if (s->epoch >= INT32_MAX) {
            fprintf(stderr, "epoch index overflow\n");
            return 1;
        }
        CHECK_ZERO(epoch_sampler_begin(s, s->epoch + 1));
    }
    remaining = s->n_samples - s->cursor;
    take = remaining < (uint64_t)max_batch ? remaining : (uint64_t)max_batch;
    if (ids != NULL) {
        for (i = 0U; i < take; ++i) {
            ids[(size_t)i] = s->order[(size_t)(s->cursor + i)];
        }
    }
    if (batch_out != NULL) {
        *batch_out = (int)take;
    }
    if (epoch_out != NULL) {
        *epoch_out = s->epoch;
    }
    if (offset_out != NULL) {
        *offset_out = s->cursor;
    }
    s->cursor += take;
    return 0;
}

static int epoch_sampler_skip_batches(epoch_sampler *s, int n_batches, int batch_size)
{
    int i;
    if (n_batches < 0) {
        fprintf(stderr, "invalid skip batches\n");
        return 1;
    }
    for (i = 0; i < n_batches; ++i) {
        CHECK_ZERO(epoch_sampler_take_batch(s, batch_size, NULL, NULL, NULL, NULL));
    }
    return 0;
}

static uint16_t f32_to_f16_bits(float f)
{
    union { float f; uint32_t u; } v;
    uint32_t sign;
    uint32_t mant;
    int exp;
    v.f = f;
    sign = (v.u >> 16U) & 0x8000U;
    exp = (int)((v.u >> 23U) & 0xffU) - 127 + 15;
    mant = v.u & 0x7fffffU;
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000U) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000U) >> 13U));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10U) | ((mant + 0x1000U) >> 13U));
}

static float f16_bits_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    uint32_t sign = ((uint32_t)h & 0x8000U) << 16U;
    uint32_t exp = ((uint32_t)h >> 10U) & 0x1fU;
    uint32_t mant = (uint32_t)h & 0x03ffU;
    if (exp == 0U) {
        if (mant == 0U) {
            v.u = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1U;
                --e;
            }
            mant &= 0x03ffU;
            v.u = sign | ((uint32_t)(e + 127) << 23U) | (mant << 13U);
        }
    } else if (exp == 31U) {
        v.u = sign | 0x7f800000U | (mant << 13U);
    } else {
        v.u = sign | ((exp + 112U) << 23U) | (mant << 13U);
    }
    return v.f;
}

static int64_t tensor_numel_public(gd_tensor *t)
{
    int ndim;
    int i;
    int64_t n = 1;
    if (t == NULL) {
        return 0;
    }
    ndim = gd_tensor_ndim(t);
    for (i = 0; i < ndim; ++i) {
        n *= gd_tensor_size(t, i);
    }
    return n;
}

static void debug_print_shape(gd_tensor *t)
{
    int ndim;
    int i;
    if (t == NULL) {
        fprintf(stderr, "<null>");
        return;
    }
    ndim = gd_tensor_ndim(t);
    fprintf(stderr, "[");
    for (i = 0; i < ndim; ++i) {
        fprintf(stderr, "%s%lld", i == 0 ? "" : ",", (long long)gd_tensor_size(t, i));
    }
    fprintf(stderr, "]");
}

static int debug_tensor_stats(gd_context *ctx, const char *name, gd_tensor *t, int force_print)
{
    gd_dtype dtype;
    int64_t n;
    size_t elem;
    size_t nbytes;
    void *buf = NULL;
    int64_t finite = 0;
    int64_t nonfinite = 0;
    double sum = 0.0;
    float mn = INFINITY;
    float mx = -INFINITY;
    int ret = 0;
    int64_t first_nonfinite = -1;
    int64_t i;

    if (t == NULL) {
        if (force_print) {
            fprintf(stderr, "vlm_debug tensor %s=null\n", name);
        }
        return 0;
    }
    dtype = gd_tensor_dtype(t);
    n = tensor_numel_public(t);
    elem = gd_dtype_sizeof(dtype);
    if (n < 0 || elem == 0U || (uint64_t)n > (uint64_t)SIZE_MAX / elem) {
        fprintf(stderr, "vlm_debug tensor %s unsupported dtype=%s n=%lld\n",
                name, gd_dtype_name(dtype), (long long)n);
        return 0;
    }
    nbytes = (size_t)n * elem;
    if (nbytes == 0U) {
        if (force_print) {
            fprintf(stderr, "vlm_debug tensor %s dtype=%s shape=", name, gd_dtype_name(dtype));
            debug_print_shape(t);
            fprintf(stderr, " n=0\n");
        }
        return 0;
    }
    buf = malloc(nbytes);
    if (buf == NULL) {
        fprintf(stderr, "vlm_debug tensor %s stats allocation failed bytes=%zu\n", name, nbytes);
        return 0;
    }
    if (gd_tensor_copy_to_cpu(ctx, t, buf, nbytes) != GD_OK) {
        fprintf(stderr, "vlm_debug tensor %s copy failed: %s\n", name, gd_last_error());
        free(buf);
        return 0;
    }
    if (dtype == GD_DTYPE_F32) {
        const float *x = (const float *)buf;
        for (i = 0; i < n; ++i) {
            float v = x[i];
            if (isfinite(v)) {
                ++finite;
                sum += (double)v;
                if (v < mn) { mn = v; }
                if (v > mx) { mx = v; }
            } else {
                if (first_nonfinite < 0) { first_nonfinite = i; }
                ++nonfinite;
            }
        }
    } else if (dtype == GD_DTYPE_F16) {
        const uint16_t *x = (const uint16_t *)buf;
        for (i = 0; i < n; ++i) {
            float v = f16_bits_to_f32(x[i]);
            if (isfinite(v)) {
                ++finite;
                sum += (double)v;
                if (v < mn) { mn = v; }
                if (v > mx) { mx = v; }
            } else {
                if (first_nonfinite < 0) { first_nonfinite = i; }
                ++nonfinite;
            }
        }
    } else if (dtype == GD_DTYPE_I32) {
        const int32_t *x = (const int32_t *)buf;
        finite = n;
        for (i = 0; i < n; ++i) {
            float v = (float)x[i];
            sum += (double)v;
            if (v < mn) { mn = v; }
            if (v > mx) { mx = v; }
        }
    }
    if (nonfinite != 0) {
        ret = 1;
    }
    if (force_print || nonfinite != 0 || env_flag_enabled("GD_VLM_DEBUG_TENSORS")) {
        fprintf(stderr, "vlm_debug tensor %s dtype=%s shape=", name, gd_dtype_name(dtype));
        debug_print_shape(t);
        fprintf(stderr,
                " n=%lld finite=%lld nonfinite=%lld first_nonfinite=%lld min=%.9g max=%.9g mean=%.9g\n",
                (long long)n, (long long)finite, (long long)nonfinite,
                (long long)first_nonfinite,
                finite > 0 ? (double)mn : (double)NAN,
                finite > 0 ? (double)mx : (double)NAN,
                finite > 0 ? sum / (double)finite : (double)NAN);
    }
    free(buf);
    return ret;
}

static void param_debug_name(int index, char *buf, size_t cap)
{
    static const char *per_layer[] = {
        "ln1", "wq", "wk", "wv", "wo", "b_wo", "ln2", "w_gate", "b_gate", "w_up", "w_down"
    };
    int rel;
    int layer;
    int slot;
    if (buf == NULL || cap == 0U) {
        return;
    }
    if (index == 0) {
        (void)snprintf(buf, cap, "patch_proj");
        return;
    }
    if (index == 1) {
        (void)snprintf(buf, cap, "patch_proj.bias");
        return;
    }
    if (index == 2) {
        (void)snprintf(buf, cap, "image_pos");
        return;
    }
    if (index == 3) {
        (void)snprintf(buf, cap, "img_norm");
        return;
    }
    if (index == 4) {
        (void)snprintf(buf, cap, "txt_norm");
        return;
    }
    if (index == 5) {
        (void)snprintf(buf, cap, "wte");
        return;
    }
    if (index == 6) {
        (void)snprintf(buf, cap, "ln_f");
        return;
    }
    rel = index - 7;
    layer = rel / 11;
    slot = rel % 11;
    (void)snprintf(buf, cap, "blk.%d.%s", layer, per_layer[slot]);
}

static void debug_params_and_grads(gd_context *ctx, gd_tensor **params, int n_params, int force)
{
    int i;
    int print_params = env_flag_enabled("GD_VLM_DEBUG_PARAMS");
    int print_grads = env_flag_enabled("GD_VLM_DEBUG_GRADS");
    char label[64];
    char name[96];

    if (!force && !print_params && !print_grads) {
        return;
    }
    for (i = 0; i < n_params; ++i) {
        gd_tensor *grad = NULL;
        param_debug_name(i, label, sizeof(label));
        if (print_params || force) {
            (void)snprintf(name, sizeof(name), "param[%d:%s]", i, label);
            (void)debug_tensor_stats(ctx, name, params[i], print_params || force);
        }
        if (print_grads || force) {
            if (gd_tensor_grad(params[i], &grad) == GD_OK && grad != NULL) {
                (void)snprintf(name, sizeof(name), "grad[%d:%s]", i, label);
                (void)debug_tensor_stats(ctx, name, grad, print_grads || force);
            }
        }
    }
}

static void debug_nonfinite_params_and_grads(gd_context *ctx, gd_tensor **params, int n_params)
{
    int i;
    char label[64];
    char name[96];
    for (i = 0; i < n_params; ++i) {
        gd_tensor *grad = NULL;
        param_debug_name(i, label, sizeof(label));
        (void)snprintf(name, sizeof(name), "param[%d:%s]", i, label);
        (void)debug_tensor_stats(ctx, name, params[i], 0);
        if (gd_tensor_grad(params[i], &grad) == GD_OK && grad != NULL) {
            (void)snprintf(name, sizeof(name), "grad[%d:%s]", i, label);
            (void)debug_tensor_stats(ctx, name, grad, 0);
        }
    }
}

static void debug_packed_batch(int step, const packed_batch *b, int vocab_size, int force)
{
    int i;
    int64_t patch_n;
    int64_t patch_finite = 0;
    int64_t patch_nonfinite = 0;
    double patch_sum = 0.0;
    float patch_min = INFINITY;
    float patch_max = -INFINITY;
    int32_t text_min = INT32_MAX;
    int32_t text_max = INT32_MIN;
    int32_t target_min = INT32_MAX;
    int32_t target_max = INT32_MIN;
    int64_t target_valid = 0;
    int64_t target_ignore = 0;
    int64_t target_bad = 0;
    int cu_bad = 0;

    if (!force && !env_flag_enabled("GD_VLM_DEBUG")) {
        return;
    }
    if (b == NULL) {
        fprintf(stderr, "vlm_debug step=%d batch=null\n", step);
        return;
    }
    for (i = 0; i < b->n_text; ++i) {
        int32_t v = b->text_tokens[i];
        if (v < text_min) { text_min = v; }
        if (v > text_max) { text_max = v; }
    }
    for (i = 0; i < b->n_tokens; ++i) {
        int32_t v = b->targets[i];
        if (v == GD_VLM_IGNORE_INDEX) {
            ++target_ignore;
        } else {
            ++target_valid;
            if (v < target_min) { target_min = v; }
            if (v > target_max) { target_max = v; }
            if (v < 0 || v >= vocab_size) { ++target_bad; }
        }
    }
    if (b->cu == NULL || b->cu[0] != 0 || b->cu[b->batch_size] != b->n_tokens) {
        cu_bad = 1;
    } else {
        for (i = 1; i <= b->batch_size; ++i) {
            if (b->cu[i] < b->cu[i - 1]) { cu_bad = 1; }
        }
    }
    patch_n = (int64_t)b->batch_size * (int64_t)b->prefix_len * (int64_t)b->patch_dim;
    for (i = 0; i < patch_n; ++i) {
        float v = f16_bits_to_f32(b->patches[i]);
        if (isfinite(v)) {
            ++patch_finite;
            patch_sum += (double)v;
            if (v < patch_min) { patch_min = v; }
            if (v > patch_max) { patch_max = v; }
        } else {
            ++patch_nonfinite;
        }
    }
    fprintf(stderr,
            "vlm_debug step=%d batch B=%d n_tokens=%d n_text=%d text_bucket=%d max_seq=%d prefix=%d patch_dim=%d cu_bad=%d\n",
            step, b->batch_size, b->n_tokens, b->n_text, b->text_bucket,
            b->max_seq, b->prefix_len, b->patch_dim, cu_bad);
    fprintf(stderr, "vlm_debug step=%d ids=[", step);
    for (i = 0; i < b->batch_size; ++i) {
        fprintf(stderr, "%s%" PRIu64, i == 0 ? "" : ",", b->sample_ids ? b->sample_ids[i] : 0U);
    }
    fprintf(stderr, "] text_lens=[");
    for (i = 0; i < b->batch_size; ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : ",", b->text_lens ? b->text_lens[i] : 0);
    }
    fprintf(stderr, "] actual_text_lens=[");
    for (i = 0; i < b->batch_size; ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : ",",
                b->actual_text_lens ? b->actual_text_lens[i] : 0);
    }
    fprintf(stderr, "] cu=[");
    for (i = 0; i <= b->batch_size; ++i) {
        fprintf(stderr, "%s%d", i == 0 ? "" : ",", b->cu ? b->cu[i] : 0);
    }
    fprintf(stderr, "]\n");
    fprintf(stderr,
            "vlm_debug step=%d text_input_range=[%d,%d] target_valid=%lld ignore=%lld bad=%lld target_range=[%d,%d] vocab=%d\n",
            step,
            b->n_text > 0 ? text_min : 0, b->n_text > 0 ? text_max : 0,
            (long long)target_valid, (long long)target_ignore, (long long)target_bad,
            target_valid > 0 ? target_min : 0, target_valid > 0 ? target_max : 0,
            vocab_size);
    fprintf(stderr,
            "vlm_debug step=%d patch_f16 n=%lld finite=%lld nonfinite=%lld min=%.9g max=%.9g mean=%.9g\n",
            step, (long long)patch_n, (long long)patch_finite, (long long)patch_nonfinite,
            patch_finite > 0 ? (double)patch_min : (double)NAN,
            patch_finite > 0 ? (double)patch_max : (double)NAN,
            patch_finite > 0 ? patch_sum / (double)patch_finite : (double)NAN);
}

static float rand_uniform01(uint64_t *state)
{
    uint64_t z = splitmix64(state);
    return ((float)(z >> 40U) + 0.5F) * (1.0F / 16777216.0F);
}

static float rand_normal(uint64_t *state)
{
    float u;
    float v;
    float s;
    do {
        u = 2.0F * rand_uniform01(state) - 1.0F;
        v = 2.0F * rand_uniform01(state) - 1.0F;
        s = u * u + v * v;
    } while (s >= 1.0F || s <= 0.0F);
    return u * sqrtf(-2.0F * logf(s) / s);
}

typedef enum vlm_param_init {
    VLM_PARAM_NORMAL,
    VLM_PARAM_UNIFORM,
    VLM_PARAM_CONSTANT
} vlm_param_init;

static int make_f16_param(gd_context *ctx,
                          gd_device device,
                          const char *name,
                          int ndim,
                          const int64_t *sizes,
                          vlm_param_init init,
                          float scale,
                          uint64_t *rng,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_tensor *t = NULL;
    uint16_t *buf = NULL;
    int64_t n = 1;
    int64_t i;
    int d;
    (void)name;
    if (ctx == NULL || sizes == NULL || out == NULL || ndim <= 0 || ndim > 4) {
        return 1;
    }
    *out = NULL;
    for (d = 0; d < ndim; ++d) {
        if (sizes[d] <= 0 || n > INT64_MAX / sizes[d]) {
            return 1;
        }
        n *= sizes[d];
    }
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F16, device, ndim, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, &t));
    if ((uint64_t)n > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
        gd_tensor_release(t);
        return 1;
    }
    buf = (uint16_t *)malloc((size_t)n * sizeof(uint16_t));
    if (buf == NULL) {
        gd_tensor_release(t);
        return 1;
    }
    for (i = 0; i < n; ++i) {
        float v = scale;
        if (init == VLM_PARAM_NORMAL) {
            v = scale * rand_normal(rng);
        } else if (init == VLM_PARAM_UNIFORM) {
            v = (2.0F * rand_uniform01(rng) - 1.0F) * scale;
        }
        buf[i] = f32_to_f16_bits(v);
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, t, buf, (size_t)n * sizeof(uint16_t)));
    free(buf);
    CHECK_OK(gd_tensor_set_requires_grad(t, true));
    *out = t;
    return 0;
}

static int make_lr_tensor(gd_context *ctx, gd_device device, gd_tensor **out)
{
    gd_tensor_desc desc;
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 0, NULL, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, out));
    return 0;
}

static void packed_batch_clear(packed_batch *b)
{
    if (b == NULL) {
        return;
    }
    free(b->text_lens);
    free(b->actual_text_lens);
    free(b->sample_ids);
    free(b->patches);
    free(b->text_tokens);
    free(b->positions);
    free(b->targets);
    free(b->cu);
    memset(b, 0, sizeof(*b));
}

static int choose_text_bucket(int max_actual, int max_text_cap, int bucket_multiple)
{
    int bucket;
    if (max_actual <= 0) {
        max_actual = 1;
    }
    if (bucket_multiple <= 0) {
        bucket = max_actual;
    } else {
        bucket = ((max_actual + bucket_multiple - 1) / bucket_multiple) * bucket_multiple;
    }
    if (max_text_cap > 0 && bucket > max_text_cap) {
        bucket = max_text_cap;
    }
    if (bucket < max_actual) {
        bucket = max_actual;
    }
    return bucket;
}

static int build_packed_batch(gd_dataset *ds,
                              const uint64_t *sample_ids,
                              int batch_size,
                              int prefix_len,
                              int patch_dim,
                              int max_text_cap,
                              int bucket_multiple,
                              int pad_token_id,
                              packed_batch *out)
{
    uint64_t n_samples = gd_dataset_num_samples(ds);
    gd_gdvlm_sample_info *infos = NULL;
    uint64_t *ids = NULL;
    int32_t *tmp_tokens = NULL;
    int b;
    int max_actual_text = 0;
    int text_total = 0;
    int token_total = 0;
    int max_seq = 0;
    size_t patch_elems_per_sample = (size_t)prefix_len * (size_t)patch_dim;

    memset(out, 0, sizeof(*out));
    if (sample_ids == NULL || batch_size <= 0 || prefix_len <= 0 || patch_dim <= 0 ||
        n_samples == 0U) {
        fprintf(stderr, "invalid packed batch config\n");
        return 1;
    }
    infos = (gd_gdvlm_sample_info *)calloc((size_t)batch_size, sizeof(*infos));
    ids = (uint64_t *)calloc((size_t)batch_size, sizeof(*ids));
    out->text_lens = (int *)calloc((size_t)batch_size, sizeof(int));
    out->actual_text_lens = (int *)calloc((size_t)batch_size, sizeof(int));
    if (infos == NULL || ids == NULL || out->text_lens == NULL ||
        out->actual_text_lens == NULL) {
        goto oom;
    }
    for (b = 0; b < batch_size; ++b) {
        ids[b] = sample_ids[b];
        if (ids[b] >= n_samples) {
            fprintf(stderr, "sample id out of range: %" PRIu64 " >= %" PRIu64 "\n",
                    ids[b], n_samples);
            packed_batch_clear(out);
            free(tmp_tokens);
            free(infos);
            free(ids);
            return 1;
        }
        CHECK_OK(gd_gdvlm_dataset_sample_info(ds, ids[b], &infos[b]));
        out->actual_text_lens[b] = (int)infos[b].token_len;
        if (max_text_cap > 0 && out->actual_text_lens[b] > max_text_cap) {
            out->actual_text_lens[b] = max_text_cap;
        }
        if (out->actual_text_lens[b] <= 0) {
            out->actual_text_lens[b] = 1;
        }
        if (out->actual_text_lens[b] > max_actual_text) {
            max_actual_text = out->actual_text_lens[b];
        }
    }
    out->text_bucket = choose_text_bucket(max_actual_text, max_text_cap, bucket_multiple);
    for (b = 0; b < batch_size; ++b) {
        out->text_lens[b] = out->text_bucket;
    }
    text_total = batch_size * out->text_bucket;
    token_total = batch_size * (prefix_len + out->text_bucket);
    max_seq = prefix_len + out->text_bucket;
    out->batch_size = batch_size;
    out->prefix_len = prefix_len;
    out->patch_dim = patch_dim;
    out->max_seq = max_seq;
    out->n_tokens = token_total;
    out->n_text = text_total;
    out->patches = (uint16_t *)calloc((size_t)batch_size * patch_elems_per_sample,
                                      sizeof(uint16_t));
    out->text_tokens = (int32_t *)calloc((size_t)text_total, sizeof(int32_t));
    out->positions = (int32_t *)calloc((size_t)token_total, sizeof(int32_t));
    out->targets = (int32_t *)calloc((size_t)token_total, sizeof(int32_t));
    out->cu = (int32_t *)calloc((size_t)batch_size + 1U, sizeof(int32_t));
    tmp_tokens = (int32_t *)calloc((size_t)out->text_bucket, sizeof(int32_t));
    if (out->patches == NULL || out->text_tokens == NULL || out->positions == NULL ||
        out->targets == NULL || out->cu == NULL || tmp_tokens == NULL) {
        goto oom;
    }
    {
        int text_cursor = 0;
        int token_cursor = 0;
        out->cu[0] = 0;
        for (b = 0; b < batch_size; ++b) {
            int tlen = out->text_lens[b];
            int actual = out->actual_text_lens[b];
            int j;
            memset(tmp_tokens, 0, (size_t)tlen * sizeof(int32_t));
            CHECK_OK(gd_gdvlm_dataset_read_sample(ds, ids[b], NULL, tmp_tokens, actual,
                                                  &out->patches[(size_t)b * patch_elems_per_sample],
                                                  patch_elems_per_sample * sizeof(uint16_t)));
            for (j = 0; j < prefix_len; ++j) {
                out->positions[token_cursor + j] = j;
                out->targets[token_cursor + j] = GD_VLM_IGNORE_INDEX;
            }
            for (j = 0; j < tlen; ++j) {
                if (j == 0) {
                    out->text_tokens[text_cursor + j] = pad_token_id;
                } else if (j - 1 < actual) {
                    out->text_tokens[text_cursor + j] = tmp_tokens[j - 1];
                } else {
                    out->text_tokens[text_cursor + j] = pad_token_id;
                }
                out->positions[token_cursor + prefix_len + j] = prefix_len + j;
                out->targets[token_cursor + prefix_len + j] =
                    j < actual ? tmp_tokens[j] : GD_VLM_IGNORE_INDEX;
            }
            text_cursor += tlen;
            token_cursor += prefix_len + tlen;
            out->cu[b + 1] = token_cursor;
        }
    }
    out->sample_ids = ids;
    ids = NULL;
    free(tmp_tokens);
    free(infos);
    free(ids);
    return 0;

oom:
    fprintf(stderr, "packed batch allocation failed\n");
    free(tmp_tokens);
    free(infos);
    free(ids);
    packed_batch_clear(out);
    return 1;
}

static void materialized_batch_destroy(materialized_batch *b)
{
    if (b == NULL) {
        return;
    }
    gd_tensor_release(b->patches);
    gd_tensor_release(b->text_tokens);
    gd_tensor_release(b->positions);
    gd_tensor_release(b->targets);
    gd_tensor_release(b->cu);
    memset(b, 0, sizeof(*b));
}

static int empty_tensor(gd_context *ctx,
                        gd_device device,
                        gd_dtype dtype,
                        int ndim,
                        const int64_t *sizes,
                        gd_tensor **out)
{
    gd_tensor_desc desc;
    CHECK_OK(gd_tensor_desc_contiguous(dtype, device, ndim, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, out));
    return 0;
}

static int materialized_batch_create(gd_context *ctx,
                                     gd_device device,
                                     const packed_batch *b,
                                     materialized_batch *out)
{
    int64_t patch_sizes[2];
    int64_t text_sizes[1];
    int64_t tok_sizes[1];
    int64_t cu_sizes[1];
    memset(out, 0, sizeof(*out));
    patch_sizes[0] = (int64_t)b->batch_size * (int64_t)b->prefix_len;
    patch_sizes[1] = b->patch_dim;
    text_sizes[0] = b->n_text;
    tok_sizes[0] = b->n_tokens;
    cu_sizes[0] = b->batch_size + 1;
    CHECK_ZERO(empty_tensor(ctx, device, GD_DTYPE_F16, 2, patch_sizes, &out->patches));
    CHECK_ZERO(empty_tensor(ctx, device, GD_DTYPE_I32, 1, text_sizes, &out->text_tokens));
    CHECK_ZERO(empty_tensor(ctx, device, GD_DTYPE_I32, 1, tok_sizes, &out->positions));
    CHECK_ZERO(empty_tensor(ctx, device, GD_DTYPE_I32, 1, tok_sizes, &out->targets));
    CHECK_ZERO(empty_tensor(ctx, device, GD_DTYPE_I32, 1, cu_sizes, &out->cu));
    return 0;
}

static int materialized_batch_upload(gd_context *ctx,
                                     const packed_batch *b,
                                     materialized_batch *out)
{
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, out->patches, b->patches,
                                     (size_t)b->batch_size * (size_t)b->prefix_len *
                                         (size_t)b->patch_dim * sizeof(uint16_t)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, out->text_tokens, b->text_tokens,
                                     (size_t)b->n_text * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, out->positions, b->positions,
                                     (size_t)b->n_tokens * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, out->targets, b->targets,
                                     (size_t)b->n_tokens * sizeof(int32_t)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, out->cu, b->cu,
                                     ((size_t)b->batch_size + 1U) * sizeof(int32_t)));
    return 0;
}

static int add_input(gd_context *ctx,
                     gd_graph *graph,
                     const char *name,
                     gd_dtype dtype,
                     gd_device device,
                     int ndim,
                     const int64_t *sizes,
                     gd_tensor **tensor_out,
                     gd_graph_input **input_out)
{
    gd_tensor_desc desc;
    CHECK_OK(gd_tensor_desc_contiguous(dtype, device, ndim, sizes, &desc));
    CHECK_OK(gd_graph_add_input(ctx, graph, name, &desc, tensor_out, input_out));
    return 0;
}

static int build_train_graph(gd_context *ctx,
                             gd_device device,
                             gd_gpt *gpt,
                             gd_tensor *w_patch,
                             gd_tensor *b_patch,
                             gd_tensor *image_pos,
                             gd_tensor *img_norm,
                             gd_tensor *txt_norm,
                             gd_optimizer *opt,
                             gd_amp_scaler *scaler,
                             gd_tensor *lr_tensor,
                             const packed_batch *batch,
                             int amp_enabled,
                             float grad_clip,
                             train_graph *out)
{
    gd_tensor *patches = NULL;
    gd_tensor *text_tokens = NULL;
    gd_tensor *positions = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *cu = NULL;
    gd_tensor *patch_embeds = NULL;
    gd_tensor *text_embeds_raw = NULL;
    gd_tensor *text_embeds = NULL;
    gd_tensor *inputs = NULL;
    gd_tensor **seq_chunks = NULL;
    gd_gpt_forward_config fwd;
    int64_t patch_sizes[2];
    int64_t text_sizes[1];
    int64_t tok_sizes[1];
    int64_t cu_sizes[1];
    int b;
    int text_cursor = 0;

    memset(out, 0, sizeof(*out));
    patch_sizes[0] = (int64_t)batch->batch_size * (int64_t)batch->prefix_len;
    patch_sizes[1] = batch->patch_dim;
    text_sizes[0] = batch->n_text;
    tok_sizes[0] = batch->n_tokens;
    cu_sizes[0] = batch->batch_size + 1;
    seq_chunks = (gd_tensor **)calloc((size_t)batch->batch_size, sizeof(gd_tensor *));
    if (seq_chunks == NULL) {
        fprintf(stderr, "graph chunk allocation failed\n");
        return 1;
    }
    CHECK_OK(gd_graph_create(ctx, &out->graph));
    CHECK_OK(gd_graph_begin(ctx, out->graph));
    CHECK_ZERO(add_input(ctx, out->graph, "patches", GD_DTYPE_F16, device, 2,
                         patch_sizes, &patches, &out->patches_in));
    CHECK_ZERO(add_input(ctx, out->graph, "text_tokens", GD_DTYPE_I32, device, 1,
                         text_sizes, &text_tokens, &out->text_tokens_in));
    CHECK_ZERO(add_input(ctx, out->graph, "positions", GD_DTYPE_I32, device, 1,
                         tok_sizes, &positions, &out->positions_in));
    CHECK_ZERO(add_input(ctx, out->graph, "targets", GD_DTYPE_I32, device, 1,
                         tok_sizes, &targets, &out->targets_in));
    CHECK_ZERO(add_input(ctx, out->graph, "cu_seqlens", GD_DTYPE_I32, device, 1,
                         cu_sizes, &cu, &out->cu_in));
    CHECK_OK(gd_linear(ctx, patches, w_patch, b_patch, &patch_embeds));
    CHECK_OK(gd_gpt_embed_tokens(ctx, gpt, text_tokens, &text_embeds_raw));
    CHECK_OK(gd_rms_norm(ctx, text_embeds_raw, txt_norm, 1.0e-5F, &text_embeds));
    for (b = 0; b < batch->batch_size; ++b) {
        gd_tensor *p_slice = NULL;
        gd_tensor *p_pos = NULL;
        gd_tensor *p_norm = NULL;
        gd_tensor *t_slice = NULL;
        gd_tensor *parts[2];
        CHECK_OK(gd_slice(ctx, patch_embeds, 0, (int64_t)b * batch->prefix_len,
                          batch->prefix_len, &p_slice));
        CHECK_OK(gd_add(ctx, p_slice, image_pos, &p_pos));
        CHECK_OK(gd_rms_norm(ctx, p_pos, img_norm, 1.0e-5F, &p_norm));
        CHECK_OK(gd_slice(ctx, text_embeds, 0, text_cursor, batch->text_lens[b], &t_slice));
        parts[0] = p_norm;
        parts[1] = t_slice;
        CHECK_OK(gd_concat(ctx, parts, 2, 0, &seq_chunks[b]));
        gd_tensor_release(p_slice);
        gd_tensor_release(p_pos);
        gd_tensor_release(p_norm);
        gd_tensor_release(t_slice);
        text_cursor += batch->text_lens[b];
    }
    CHECK_OK(gd_concat(ctx, seq_chunks, batch->batch_size, 0, &inputs));
    memset(&fwd, 0, sizeof(fwd));
    fwd.prefix_len = batch->prefix_len;
    fwd.has_ignore_index = true;
    fwd.ignore_index = GD_VLM_IGNORE_INDEX;
    fwd.max_seqlen = batch->max_seq;
    /* Loss path is fused LMCE: hidden @ W^T + CE in one op, no logits tensor. */
    CHECK_OK(gd_gpt_forward_embeds_varlen_loss(ctx, gpt, inputs, positions, cu, targets,
                                               &fwd, &out->loss));
    if (!env_flag_enabled("GD_VLM_FORWARD_ONLY")) {
        if (amp_enabled) {
            CHECK_OK(gd_amp_scaler_scale_loss(ctx, scaler, out->loss, &out->scaled_loss));
            CHECK_OK(gd_backward(ctx, out->scaled_loss));
            if (!env_flag_enabled("GD_VLM_BACKWARD_ONLY")) {
                CHECK_OK(gd_optimizer_step_amp_clip_lr(ctx, opt, scaler, grad_clip,
                                                       &out->grad_norm, lr_tensor));
            }
        } else {
            CHECK_OK(gd_backward(ctx, out->loss));
            if (!env_flag_enabled("GD_VLM_BACKWARD_ONLY")) {
                CHECK_OK(gd_optimizer_step_lr(ctx, opt, lr_tensor));
            }
        }
    }
    CHECK_OK(gd_graph_end(ctx));
    {
        const char *dump_path = getenv("GD_VLM_DUMP_GRAPH");
        if (dump_path != NULL && dump_path[0] != '\0') {
            CHECK_OK(gd_graph_dump(out->graph, GD_DUMP_TEXT, dump_path));
        }
    }
    gd_tensor_release(patches);
    gd_tensor_release(text_tokens);
    gd_tensor_release(positions);
    gd_tensor_release(targets);
    gd_tensor_release(cu);
    gd_tensor_release(patch_embeds);
    gd_tensor_release(text_embeds_raw);
    gd_tensor_release(text_embeds);
    gd_tensor_release(inputs);
    for (b = 0; b < batch->batch_size; ++b) {
        gd_tensor_release(seq_chunks[b]);
    }
    free(seq_chunks);
    CHECK_OK(gd_graph_compile(out->graph, device));
    CHECK_OK(gd_graph_runner_create(out->graph, &out->runner));
    return 0;
}

static int destroy_train_graph(train_graph *g)
{
    if (g == NULL) {
        return 0;
    }
    gd_graph_runner_destroy(g->runner);
    gd_tensor_release(g->loss);
    gd_tensor_release(g->scaled_loss);
    gd_tensor_release(g->grad_norm);
    if (g->graph != NULL) {
        CHECK_OK(gd_graph_reset(g->graph));
        CHECK_OK(gd_graph_destroy(g->graph));
    }
    memset(g, 0, sizeof(*g));
    return 0;
}

static int bind_graph(train_graph *g, const materialized_batch *b)
{
    CHECK_OK(gd_graph_runner_bind(g->runner, g->patches_in, b->patches));
    CHECK_OK(gd_graph_runner_bind(g->runner, g->text_tokens_in, b->text_tokens));
    CHECK_OK(gd_graph_runner_bind(g->runner, g->positions_in, b->positions));
    CHECK_OK(gd_graph_runner_bind(g->runner, g->targets_in, b->targets));
    CHECK_OK(gd_graph_runner_bind(g->runner, g->cu_in, b->cu));
    return 0;
}

static void train_graph_cache_destroy(train_graph_cache *cache)
{
    int i;
    if (cache == NULL) {
        return;
    }
    for (i = 0; i < cache->n_entries; ++i) {
        materialized_batch_destroy(&cache->entries[i].inputs);
        (void)destroy_train_graph(&cache->entries[i].graph);
    }
    free(cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static int train_graph_cache_init(train_graph_cache *cache,
                                  int max_entries,
                                  int footprint_limit_mb)
{
    if (cache == NULL || max_entries <= 0 || footprint_limit_mb < 0) {
        return 1;
    }
    memset(cache, 0, sizeof(*cache));
    cache->max_entries = max_entries;
    cache->footprint_limit_mb = footprint_limit_mb;
    return 0;
}

static train_graph_cache_entry *train_graph_cache_find(train_graph_cache *cache,
                                                       int batch_size,
                                                       int text_bucket)
{
    int i;
    if (cache == NULL) {
        return NULL;
    }
    for (i = 0; i < cache->n_entries; ++i) {
        if (cache->entries[i].batch_size == batch_size &&
            cache->entries[i].text_bucket == text_bucket) {
            return &cache->entries[i];
        }
    }
    return NULL;
}

static int train_graph_cache_evict_one(gd_context *ctx,
                                       gd_device device,
                                       train_graph_cache *cache,
                                       double *freed_mb)
{
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    int i;
    if (freed_mb != NULL) {
        *freed_mb = 0.0;
    }
    if (cache == NULL || cache->n_entries <= 0) {
        return 1;
    }
    for (i = 0; i < cache->n_entries; ++i) {
        if (cache->entries[i].last_used < oldest) {
            oldest = cache->entries[i].last_used;
            victim = i;
        }
    }
    if (victim < 0) {
        return 1;
    }
    if (freed_mb != NULL) {
        *freed_mb = cache->entries[victim].footprint_mb;
    }
    CHECK_OK(gd_synchronize(ctx, device));
    materialized_batch_destroy(&cache->entries[victim].inputs);
    CHECK_ZERO(destroy_train_graph(&cache->entries[victim].graph));
    if (victim + 1 < cache->n_entries) {
        memmove(&cache->entries[victim], &cache->entries[victim + 1],
                (size_t)(cache->n_entries - victim - 1) * sizeof(cache->entries[0]));
    }
    cache->n_entries -= 1;
    cache->evictions += 1U;
    return 0;
}

static int train_graph_cache_trim_for_new_entry(gd_context *ctx,
                                                gd_device device,
                                                train_graph_cache *cache)
{
    double estimate = 0.0;

    if (cache == NULL || cache->footprint_limit_mb <= 0 || cache->n_entries <= 0) {
        return 0;
    }
    estimate = cache->avg_entry_footprint_mb > 1.0 ? cache->avg_entry_footprint_mb : 600.0;
    {
        double footprint = current_footprint_mb();
        double projected = isfinite(footprint) ? footprint + estimate : 0.0;
        while (cache->n_entries > 0 && projected > (double)cache->footprint_limit_mb) {
            double freed_mb = 0.0;
            CHECK_ZERO(train_graph_cache_evict_one(ctx, device, cache, &freed_mb));
            if (freed_mb <= 1.0) {
                break;
            }
            projected -= freed_mb;
        }
    }
    return 0;
}

static int train_graph_cache_reserve(gd_context *ctx,
                                     gd_device device,
                                     train_graph_cache *cache)
{
    train_graph_cache_entry *grown = NULL;
    if (cache->n_entries >= cache->max_entries) {
        CHECK_ZERO(train_graph_cache_evict_one(ctx, device, cache, NULL));
    }
    if (cache->n_entries < cache->cap) {
        return 0;
    }
    {
        int new_cap = cache->cap == 0 ? 4 : cache->cap * 2;
        if (new_cap > cache->max_entries) {
            new_cap = cache->max_entries;
        }
        if (new_cap <= cache->cap) {
            return 1;
        }
        grown = (train_graph_cache_entry *)realloc(cache->entries,
                                                   (size_t)new_cap * sizeof(*grown));
        if (grown == NULL) {
            return 1;
        }
        cache->entries = grown;
        cache->cap = new_cap;
    }
    return 0;
}

static int train_graph_cache_get(gd_context *ctx,
                                 gd_device device,
                                 gd_gpt *gpt,
                                 gd_tensor *w_patch,
                                 gd_tensor *b_patch,
                                 gd_tensor *image_pos,
                                 gd_tensor *img_norm,
                                 gd_tensor *txt_norm,
                                 gd_optimizer *opt,
                                 gd_amp_scaler *scaler,
                                 gd_tensor *lr_tensor,
                                 const packed_batch *batch,
                                 int amp_enabled,
                                 float grad_clip,
                                 uint64_t step_id,
                                 train_graph_cache *cache,
                                 train_graph_cache_entry **entry_out)
{
    train_graph_cache_entry *entry = NULL;
    if (cache == NULL || batch == NULL || entry_out == NULL) {
        return 1;
    }
    *entry_out = NULL;
    entry = train_graph_cache_find(cache, batch->batch_size, batch->text_bucket);
    if (entry != NULL) {
        cache->hits += 1U;
        entry->last_used = step_id;
        entry->runs += 1U;
        CHECK_ZERO(materialized_batch_upload(ctx, batch, &entry->inputs));
        *entry_out = entry;
        return 0;
    }

    CHECK_ZERO(train_graph_cache_trim_for_new_entry(ctx, device, cache));
    CHECK_ZERO(train_graph_cache_reserve(ctx, device, cache));
    entry = &cache->entries[cache->n_entries];
    memset(entry, 0, sizeof(*entry));
    entry->batch_size = batch->batch_size;
    entry->text_bucket = batch->text_bucket;
    entry->last_used = step_id;
    entry->runs = 1U;
    {
        double footprint_before = current_footprint_mb();
        double footprint_after = 0.0;
        if (materialized_batch_create(ctx, device, batch, &entry->inputs) != 0 ||
            materialized_batch_upload(ctx, batch, &entry->inputs) != 0 ||
            build_train_graph(ctx, device, gpt, w_patch, b_patch, image_pos, img_norm,
                              txt_norm, opt, scaler, lr_tensor, batch, amp_enabled,
                              grad_clip, &entry->graph) != 0 ||
            bind_graph(&entry->graph, &entry->inputs) != 0) {
            materialized_batch_destroy(&entry->inputs);
            (void)destroy_train_graph(&entry->graph);
            memset(entry, 0, sizeof(*entry));
            return 1;
        }
        footprint_after = current_footprint_mb();
        if (isfinite(footprint_before) && isfinite(footprint_after) &&
            footprint_after > footprint_before) {
            entry->footprint_mb = footprint_after - footprint_before;
            if (cache->avg_entry_footprint_mb <= 1.0) {
                cache->avg_entry_footprint_mb = entry->footprint_mb;
            } else {
                cache->avg_entry_footprint_mb = 0.75 * cache->avg_entry_footprint_mb +
                                                0.25 * entry->footprint_mb;
            }
        }
    }
    cache->n_entries += 1;
    cache->builds += 1U;
    if (env_flag_enabled("GD_VLM_DEBUG")) {
        fprintf(stderr, "vlm_debug graph_cache_build B=%d text_bucket=%d entries=%d\n",
                entry->batch_size, entry->text_bucket, cache->n_entries);
    }
    *entry_out = entry;
    return 0;
}

int main(int argc, char **argv)
{
    app_config app;
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");
    gd_dataset *ds = NULL;
    gd_gpt_config gpt_cfg;
    gd_gpt *gpt = NULL;
    gd_tensor *w_patch = NULL;
    gd_tensor *b_patch = NULL;
    gd_tensor *image_pos = NULL;
    gd_tensor *img_norm = NULL;
    gd_tensor *txt_norm = NULL;
    gd_tensor **gpt_params = NULL;
    gd_tensor **all_params = NULL;
    int n_gpt_params = 0;
    int n_params = 0;
    gd_optimizer *opt = NULL;
    gd_adamw_config opt_cfg;
    gd_amp_scaler *scaler = NULL;
    gd_amp_scaler_config scaler_cfg;
    gd_lr_scheduler_config lr_cfg;
    gd_tensor *lr_tensor = NULL;
    uint64_t vocab_size = 0U;
    uint64_t prefix_len_u64 = 0U;
    uint64_t patch_dim_u64 = 0U;
    uint64_t max_text_u64 = 0U;
    int prefix_len = 0;
    int patch_dim = 0;
    int max_text = 0;
    uint64_t n_samples = 0U;
    uint64_t steps_per_epoch = 0U;
    uint64_t planned_steps = 0U;
    int run_steps = 0;
    int schedule_steps = 0;
    int step;
    uint64_t init_rng;
    uint64_t *batch_ids = NULL;
    epoch_sampler sampler = {0};
    train_graph_cache graph_cache = {0};
    double t0;
    int rc = 1;

    init_app(&app);
    if (argc == 1) {
        usage(stdout);
        return 0;
    }
    {
        int parse = parse_args(argc, argv, &app);
        if (parse == 2) {
            return 0;
        }
        if (parse != 0 || app.data_dir[0] == '\0') {
            usage(stderr);
            return 2;
        }
    }
    if (app.epochs <= 0 || app.steps < 0 || app.batch_size <= 0 ||
        app.log_interval <= 0 || app.skip_batches < 0 ||
        app.text_bucket_multiple < 0 || app.graph_cache_max <= 0 ||
        app.graph_cache_footprint_mb < 0 || app.attention_window <= 0 ||
        app.n_heads <= 0 || app.n_kv_heads <= 0 ||
        app.n_heads % app.n_kv_heads != 0 || app.d_model != app.n_heads * 64) {
        fprintf(stderr,
                "invalid config: require epochs/batch/log/cache >0, steps>=0, bucket/cache_footprint>=0, window>0, d_model=n_heads*64\n");
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
    if (target.type == GD_DEVICE_METAL && !env_flag_enabled("GD_METAL_MPS")) {
        fprintf(stderr, "config error: Metal F16 GPT-VLM needs GD_METAL_MPS=1\n");
        gd_context_destroy(ctx);
        return 2;
    }
    CHECK_OK(gd_context_set_default_device(ctx, target));
    CHECK_OK(gd_dataset_open_gdvlm_split(app.data_dir, app.split, &ds));
    CHECK_OK(gd_dataset_get_u64(ds, "vocab_size", &vocab_size));
    CHECK_OK(gd_dataset_get_u64(ds, "image_prefix_tokens", &prefix_len_u64));
    CHECK_OK(gd_dataset_get_u64(ds, "patch_dim", &patch_dim_u64));
    CHECK_OK(gd_dataset_get_u64(ds, "max_text_len", &max_text_u64));
    if (prefix_len_u64 == 0U || prefix_len_u64 > (uint64_t)INT32_MAX ||
        patch_dim_u64 == 0U || patch_dim_u64 > (uint64_t)INT32_MAX ||
        max_text_u64 == 0U || max_text_u64 > (uint64_t)INT32_MAX ||
        vocab_size == 0U || vocab_size > (uint64_t)INT32_MAX) {
        fprintf(stderr, "dataset metadata out of supported range\n");
        goto cleanup;
    }
    prefix_len = (int)prefix_len_u64;
    patch_dim = (int)patch_dim_u64;
    max_text = app.max_text_len > 0 ? app.max_text_len : (int)max_text_u64;
    n_samples = gd_dataset_num_samples(ds);
    if (n_samples == 0U) {
        fprintf(stderr, "dataset has no samples\n");
        goto cleanup;
    }
    steps_per_epoch = ((n_samples - 1U) / (uint64_t)app.batch_size) + 1U;
    if (steps_per_epoch > UINT64_MAX / (uint64_t)app.epochs) {
        fprintf(stderr, "epoch plan overflows step counter\n");
        goto cleanup;
    }
    planned_steps = steps_per_epoch * (uint64_t)app.epochs;
    if (planned_steps > (uint64_t)INT32_MAX) {
        fprintf(stderr, "too many train steps for lr scheduler: %" PRIu64 "\n",
                planned_steps);
        goto cleanup;
    }
    schedule_steps = (int)planned_steps;
    if (app.steps > 0 && (uint64_t)app.steps < planned_steps) {
        run_steps = app.steps;
    } else {
        run_steps = schedule_steps;
    }
    if (run_steps <= 0) {
        fprintf(stderr, "no training steps scheduled\n");
        goto cleanup;
    }
    if ((uint64_t)app.batch_size > (uint64_t)(SIZE_MAX / sizeof(*batch_ids))) {
        fprintf(stderr, "batch id allocation too large\n");
        goto cleanup;
    }
    batch_ids = (uint64_t *)malloc((size_t)app.batch_size * sizeof(*batch_ids));
    if (batch_ids == NULL) {
        fprintf(stderr, "batch id allocation failed\n");
        goto cleanup;
    }
    if (epoch_sampler_init(&sampler, n_samples, app.seed ^ UINT64_C(0xa5a5a5a5)) != 0) {
        goto cleanup;
    }
    if (train_graph_cache_init(&graph_cache, app.graph_cache_max,
                               app.graph_cache_footprint_mb) != 0) {
        fprintf(stderr, "graph cache init failed\n");
        goto cleanup;
    }

    memset(&gpt_cfg, 0, sizeof(gpt_cfg));
    gpt_cfg.vocab_size = (int)vocab_size;
    gpt_cfg.d_model = app.d_model;
    gpt_cfg.n_layers = app.n_layers;
    gpt_cfg.n_heads = app.n_heads;
    gpt_cfg.n_kv_heads = app.n_kv_heads;
    gpt_cfg.head_dim = 64;
    gpt_cfg.d_ff = app.d_ff;
    gpt_cfg.max_seq_len = prefix_len + max_text;
    gpt_cfg.mlp_kind = GD_GPT_MLP_POWLU;
    gpt_cfg.attention_window = app.attention_window;
    gpt_cfg.param_dtype = GD_DTYPE_F16;
    gpt_cfg.tie_embeddings = true;
    init_rng = app.seed ^ UINT64_C(0xf00d1234);
    CHECK_OK(gd_gpt_create(ctx, &gpt_cfg, app.seed ^ UINT64_C(0xdeadbeef), &gpt));
    gd_gpt_set_training(gpt, true);
    {
        int64_t patch_proj_sizes[2] = {patch_dim, app.d_model};
        int64_t patch_bias_sizes[1] = {app.d_model};
        int64_t image_pos_sizes[2] = {prefix_len, app.d_model};
        int64_t norm_sizes[1] = {app.d_model};
        float patch_bound = 1.0F / sqrtf((float)patch_dim);
        float mod_gamma = 1.0F / sqrtf((float)app.d_model);
        CHECK_ZERO(make_f16_param(ctx, target, "patch_proj", 2, patch_proj_sizes,
                                  VLM_PARAM_UNIFORM, patch_bound, &init_rng, &w_patch));
        CHECK_ZERO(make_f16_param(ctx, target, "patch_proj.bias", 1, patch_bias_sizes,
                                  VLM_PARAM_UNIFORM, patch_bound, &init_rng, &b_patch));
        CHECK_ZERO(make_f16_param(ctx, target, "image_pos", 2, image_pos_sizes,
                                  VLM_PARAM_NORMAL, 0.02F, &init_rng, &image_pos));
        CHECK_ZERO(make_f16_param(ctx, target, "img_norm", 1, norm_sizes,
                                  VLM_PARAM_CONSTANT, mod_gamma, &init_rng, &img_norm));
        CHECK_ZERO(make_f16_param(ctx, target, "txt_norm", 1, norm_sizes,
                                  VLM_PARAM_CONSTANT, mod_gamma, &init_rng, &txt_norm));
    }
    CHECK_OK(gd_gpt_parameters(gpt, &gpt_params, &n_gpt_params));
    n_params = n_gpt_params + 5;
    all_params = (gd_tensor **)calloc((size_t)n_params, sizeof(gd_tensor *));
    if (all_params == NULL) {
        fprintf(stderr, "param list allocation failed\n");
        goto cleanup;
    }
    all_params[0] = w_patch;
    all_params[1] = b_patch;
    all_params[2] = image_pos;
    all_params[3] = img_norm;
    all_params[4] = txt_norm;
    memcpy(&all_params[5], gpt_params, (size_t)n_gpt_params * sizeof(gd_tensor *));
    memset(&opt_cfg, 0, sizeof(opt_cfg));
    opt_cfg.lr = 0.0F;
    opt_cfg.beta1 = 0.9F;
    opt_cfg.beta2 = 0.999F;
    opt_cfg.eps = 1e-8F;
    opt_cfg.weight_decay = app.weight_decay;
    opt_cfg.master_param_policy = GD_MASTER_PARAM_AUTO;
    CHECK_OK(gd_adamw_create(ctx, all_params, n_params, &opt_cfg, &opt));
    memset(&scaler_cfg, 0, sizeof(scaler_cfg));
    scaler_cfg.init_scale = env_float("GD_VLM_AMP_SCALE", 8.0F);
    scaler_cfg.growth_factor = 2.0F;
    scaler_cfg.backoff_factor = 0.5F;
    scaler_cfg.growth_interval = env_int("GD_VLM_AMP_GROWTH", 2000);
    scaler_cfg.min_scale = 1.0F;
    scaler_cfg.max_scale = 1048576.0F;
    CHECK_OK(gd_amp_scaler_create(ctx, &scaler_cfg, &scaler));
    CHECK_ZERO(make_lr_tensor(ctx, target, &lr_tensor));
    memset(&lr_cfg, 0, sizeof(lr_cfg));
    lr_cfg.max_lr = app.lr;
    lr_cfg.min_lr = app.lr * 0.1F;
    lr_cfg.warmup_steps = (schedule_steps + 19) / 20;
    lr_cfg.total_steps = schedule_steps;

    printf("gpt_vlm\n");
    printf("  device      : %s\n", target.type == GD_DEVICE_METAL ? "metal" : "cpu");
    printf("  data        : dir=%s split=%s samples=%llu\n", app.data_dir, app.split,
           (unsigned long long)n_samples);
    printf("  model       : dtype=f16 d=%d L=%d H=%d Hkv=%d Dh=64 dff=%d\n",
           app.d_model, app.n_layers, app.n_heads, app.n_kv_heads, app.d_ff);
    printf("  attention   : varlen causal prefix_len=%d sliding_window=%d max_text=%d\n",
           prefix_len, app.attention_window, max_text);
    printf("  fusion      : patch_proj+bias + image_pos + modality RMSNorm(gamma=1/sqrt(d))\n");
    printf("  loss        : fused lm_cross_entropy (LMCE), logits not materialized\n");
    printf("  sampler     : random without replacement steps_per_epoch=%llu epochs=%d%s\n",
           (unsigned long long)steps_per_epoch, app.epochs,
           app.steps > 0 ? " steps_debug_cap=on" : "");
    printf("  training    : steps=%d lr_schedule_steps=%d batch=%d log_interval=%d\n",
           run_steps, schedule_steps, app.batch_size, app.log_interval);
    printf("  graph_cache : max_entries=%d text_bucket_multiple=%d footprint_limit_mb=%d\n",
           app.graph_cache_max, app.text_bucket_multiple, app.graph_cache_footprint_mb);
    printf("  optimized   : requires F16 && Dh==64 && causal && prefix_len>0 && sliding_window>0 (satisfied)\n");
    if (env_flag_enabled("GD_VLM_DEBUG")) {
        fprintf(stderr,
                "vlm_debug enabled params=%d grads=%d tensors=%d metal=%d forward_only=%d\n",
                env_flag_enabled("GD_VLM_DEBUG_PARAMS"),
                env_flag_enabled("GD_VLM_DEBUG_GRADS"),
                env_flag_enabled("GD_VLM_DEBUG_TENSORS"),
                env_flag_enabled("GD_VLM_DEBUG_METAL"),
                env_flag_enabled("GD_VLM_FORWARD_ONLY"));
    }
    fflush(stdout);

    if (app.skip_batches > 0) {
        if (epoch_sampler_skip_batches(&sampler, app.skip_batches, app.batch_size) != 0) {
            goto cleanup;
        }
        fprintf(stderr, "vlm_debug skipped_batches=%d skipped_samples_stream=%llu\n",
                app.skip_batches,
                (unsigned long long)((uint64_t)app.skip_batches * (uint64_t)app.batch_size));
    }
    t0 = now_ms();
    for (step = 0; step < run_steps; ++step) {
        packed_batch pb;
        train_graph_cache_entry *entry = NULL;
        train_graph *graph = NULL;
        float loss = 0.0F;
        float lr = 0.0F;
        float grad_norm = NAN;
        bool found_inf = false;
        bool stepped = true;
        bool forward_only = env_flag_enabled("GD_VLM_FORWARD_ONLY");
        int current_batch_size = 0;
        int epoch_index = 0;
        uint64_t epoch_offset = 0U;
        double ts = now_ms();

        CHECK_ZERO(epoch_sampler_take_batch(&sampler, app.batch_size, batch_ids,
                                            &current_batch_size, &epoch_index,
                                            &epoch_offset));
        CHECK_ZERO(build_packed_batch(ds, batch_ids, current_batch_size, prefix_len,
                                      patch_dim, max_text, app.text_bucket_multiple, 0, &pb));
        debug_packed_batch(step + 1, &pb, gpt_cfg.vocab_size, 0);
        CHECK_OK(gd_lr_scheduler_write(ctx, &lr_cfg, step, lr_tensor, &lr));
        if (!forward_only) {
            CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
        }
        CHECK_ZERO(train_graph_cache_get(ctx, target, gpt, w_patch, b_patch, image_pos,
                                         img_norm, txt_norm, opt, scaler, lr_tensor,
                                         &pb, 1, app.grad_clip, (uint64_t)step + 1U,
                                         &graph_cache, &entry));
        graph = &entry->graph;
        CHECK_OK(gd_graph_runner_run(graph->runner));
        CHECK_OK(gd_synchronize(ctx, target));
        CHECK_OK(gd_amp_scaler_found_inf(ctx, scaler, &found_inf));
        if (graph->grad_norm != NULL) {
            CHECK_OK(gd_tensor_copy_to_cpu(ctx, graph->grad_norm, &grad_norm,
                                           sizeof(grad_norm)));
        }
        if (env_flag_enabled("GD_VLM_DEBUG")) {
            CHECK_OK(gd_tensor_copy_to_cpu(ctx, graph->loss, &loss, sizeof(loss)));
            fprintf(stderr,
                    "vlm_debug step=%d after_run loss=%.9g found_inf=%d grad_norm=%.9g amp_scale=%.9g footprint_mb=%.1f\n",
                    step + 1, (double)loss, found_inf ? 1 : 0,
                    (double)grad_norm, (double)gd_amp_scaler_scale(scaler),
                    current_footprint_mb());
            (void)debug_tensor_stats(ctx, "loss", graph->loss, env_flag_enabled("GD_VLM_DEBUG_TENSORS"));
            if (graph->grad_norm != NULL) {
                (void)debug_tensor_stats(ctx, "grad_norm", graph->grad_norm,
                                         env_flag_enabled("GD_VLM_DEBUG_TENSORS"));
            }
        }
        if (env_flag_enabled("GD_VLM_DEBUG_PATCH_GRAD")) {
            gd_tensor *patch_grad = NULL;
            if (gd_tensor_grad(all_params[0], &patch_grad) == GD_OK && patch_grad != NULL) {
                (void)debug_tensor_stats(ctx, "grad[0:patch_proj]", patch_grad, 1);
            }
        }
        if (env_flag_enabled("GD_VLM_DEBUG_GRADS_ALL")) {
            debug_params_and_grads(ctx, all_params, n_params, 1);
        }
        if (found_inf && env_flag_enabled("GD_VLM_DEBUG_GRADS")) {
            debug_nonfinite_params_and_grads(ctx, all_params, n_params);
        }
        CHECK_OK(gd_amp_scaler_update(ctx, scaler, &stepped));
        if (!env_flag_enabled("GD_VLM_DEBUG")) {
            CHECK_OK(gd_tensor_copy_to_cpu(ctx, graph->loss, &loss, sizeof(loss)));
        }
        if (!isfinite(loss)) {
            fprintf(stderr, "non-finite loss at step %d\n", step + 1);
            debug_packed_batch(step + 1, &pb, gpt_cfg.vocab_size, 1);
            (void)debug_tensor_stats(ctx, "loss", graph->loss, 1);
            if (graph->grad_norm != NULL) {
                (void)debug_tensor_stats(ctx, "grad_norm", graph->grad_norm, 1);
            }
            debug_params_and_grads(ctx, all_params, n_params, 1);
            packed_batch_clear(&pb);
            goto cleanup;
        }
        if (step == 0 || (step + 1) % app.log_interval == 0 || step + 1 == run_steps) {
            double now = now_ms();
            double dt = (now - ts) / 1000.0;
            double elapsed = (now - t0) / 1000.0;
            double eta = (step + 1) > 0 ? (elapsed / (double)(step + 1)) *
                         (double)(run_steps - step - 1) : 0.0;
            uint64_t batch_idx = ((uint64_t)app.batch_size > 0U ?
                                  epoch_offset / (uint64_t)app.batch_size : 0U) + 1U;
            char eta_buf[32];
            format_eta(eta, eta_buf, sizeof(eta_buf));
            printf("epoch %d/%d idx=%llu/%llu loss %.5f lr %.4g tok/s %.1f grad_norm %.4g eta %s\n",
                   epoch_index + 1, app.epochs,
                   (unsigned long long)batch_idx,
                   (unsigned long long)steps_per_epoch,
                   (double)loss, (double)lr,
                   dt > 0.0 ? (double)pb.n_tokens / dt : 0.0,
                   (double)grad_norm, eta_buf);
        }
        packed_batch_clear(&pb);
        if (env_flag_enabled("GD_VLM_DEBUG")) {
            fprintf(stderr, "vlm_debug step=%d after_cleanup footprint_mb=%.1f\n",
                    step + 1, current_footprint_mb());
        }
    }
    printf("done steps=%d epochs=%d elapsed_s=%.3f graph_builds=%llu graph_hits=%llu graph_evictions=%llu\n",
           run_steps, app.epochs, (now_ms() - t0) / 1000.0,
           (unsigned long long)graph_cache.builds,
           (unsigned long long)graph_cache.hits,
           (unsigned long long)graph_cache.evictions);
    rc = 0;

cleanup:
    train_graph_cache_destroy(&graph_cache);
    free(batch_ids);
    epoch_sampler_destroy(&sampler);
    gd_tensor_release(lr_tensor);
    gd_amp_scaler_destroy(scaler);
    gd_optimizer_destroy(opt);
    free(all_params);
    gd_tensor_release(txt_norm);
    gd_tensor_release(img_norm);
    gd_tensor_release(image_pos);
    gd_tensor_release(b_patch);
    gd_tensor_release(w_patch);
    gd_gpt_destroy(gpt);
    gd_dataset_destroy(ds);
    gd_context_destroy(ctx);
    return rc;
}
