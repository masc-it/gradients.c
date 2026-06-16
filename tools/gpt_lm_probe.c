/* Temporary GPT-LM checkpoint diagnostic probe.
 *
 * Builds a one-file executable that reuses examples/gpt_lm/gpt_lm_shared.c
 * internals to expose prompt token cross-entropy and generation-step logits.
 *
 * Example build from repo root:
 *   cc -std=c11 -O2 -Iinclude -Iexamples/common tools/gpt_lm_probe.c \
 *      build-gpt_lm/libgradients.a -pthread -lm -framework Foundation -framework Metal \
 *      -o tools/gpt_lm_probe
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "../examples/gpt_lm/gpt_lm_shared.h"

/* Diagnostic-only trick: gpt_lm_shared.c keeps the prefill/decode/logit helper
 * functions file-local.  Include it in this translation unit with `static`
 * erased after all relevant headers are guarded/included, so the probe can call
 * those helpers without modifying example sources. */
#define static
#include "../examples/gpt_lm/gpt_lm_shared.c"
#undef static

#define PROBE_DEFAULT_CHECKPOINT "examples/gpt_lm/checkpoints/gpt_lm_latest.gdckpt"
#define PROBE_DEFAULT_DATA_DIR "examples/gpt_lm/data"
#define PROBE_DEFAULT_PROMPT "<|im_start|>Termine: casa\nDefinizioni:"

typedef struct probe_config {
    gpt_config model;
    const char *prompt;
    const char *prompt_file;
    int top_k;
    int trace_limit;
    bool no_prompt_ce;
    bool no_generate;
} probe_config;

static int parse_i64_probe(const char *text, int64_t min_value, int64_t max_value, int64_t *out)
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

static int parse_u64_probe(const char *text, uint64_t max_value, uint64_t *out)
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

static int parse_float_probe(const char *text, float min_value, float max_value, float *out)
{
    char *end = NULL;
    float parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) {
        return 0;
    }
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) || parsed < min_value || parsed > max_value) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static const char *arg_value_probe(int argc, char **argv, int *index, const char *name)
{
    const size_t name_len = strlen(name);
    const char *arg = argv[*index];
    if (strncmp(arg, name, name_len) == 0 && arg[name_len] == '=') {
        return arg + name_len + 1U;
    }
    if (strcmp(arg, name) == 0) {
        if (*index + 1 >= argc) {
            fprintf(stderr, "gpt_lm_probe: missing value for %s\n", name);
            exit(2);
        }
        *index += 1;
        return argv[*index];
    }
    return NULL;
}

static void usage(const char *argv0)
{
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --checkpoint PATH       model checkpoint (default: %s)\n", PROBE_DEFAULT_CHECKPOINT);
    printf("  --data-dir PATH         data dir for tokenizer (default: %s)\n", PROBE_DEFAULT_DATA_DIR);
    printf("  --tokenizer-path PATH   tokenizer JSON; checkpoint metadata used when available\n");
    printf("  --prompt TEXT           prompt/prefix to inspect (default: dictionary casa prefix)\n");
    printf("  --prompt-file PATH      read prompt text from file\n");
    printf("  --max-new-tokens N      generation budget (default: 32)\n");
    printf("  --temperature T         0 means greedy (default: 0)\n");
    printf("  --min-p P               min-p sampling cutoff (default: 0)\n");
    printf("  --repetition-penalty P  repetition penalty (default: 1)\n");
    printf("  --top-k N               top logits printed per step/token (default: 8)\n");
    printf("  --trace-limit N         max CE token rows to print, from tail if long (default: 160)\n");
    printf("  --layers N              override checkpoint metadata layers\n");
    printf("  --logits-softcap C      override checkpoint metadata softcap\n");
    printf("  --seed N                generation seed\n");
    printf("  --no-prompt-ce          skip prompt token CE trace\n");
    printf("  --no-generate           skip generation trace\n");
    printf("  --help                  show this help\n");
}

static gpt_config probe_model_defaults(void)
{
    gpt_config c;
    memset(&c, 0, sizeof(c));
    c.data_dir = PROBE_DEFAULT_DATA_DIR;
    c.tokenizer_path = NULL;
    c.generate_prompt = PROBE_DEFAULT_PROMPT;
    c.checkpoint_path = PROBE_DEFAULT_CHECKPOINT;
    c.load_checkpoint_path = NULL;
    c.val_split = "val";
    c.epochs = 0;
    c.batch_size = GPT_DEFAULT_BATCH_SIZE;
    c.n_layers = GPT_DEFAULT_LAYERS;
    c.report_every = GPT_DEFAULT_REPORT_EVERY;
    c.lr_warmup_steps = -1;
    c.max_new_tokens = 32;
    c.generate_every_n_steps = 0;
    c.epochs_set = true;
    c.save_best = false;
    c.save_latest = false;
    c.metrics_enabled = false;
    c.overfit_num_samples = 0U;
    c.seed = GPT_DEFAULT_SEED;
    c.dropout_p = GPT_DEFAULT_DROPOUT_P;
    c.lr_max = GPT_DEFAULT_LR_MAX;
    c.lr_min = GPT_DEFAULT_LR_MIN;
    c.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
    c.grad_clip_norm = GPT_DEFAULT_GRAD_CLIP_NORM;
    c.temperature = 0.0f;
    c.min_p = GPT_DEFAULT_MIN_P;
    c.repetition_penalty = GPT_DEFAULT_REPETITION_PENALTY;
    c.logits_softcap = 0.0f;
    return c;
}

static probe_config parse_probe_args(int argc,
                                     char **argv,
                                     bool *layers_set,
                                     bool *tokenizer_set,
                                     bool *softcap_set)
{
    probe_config p;
    int i;
    memset(&p, 0, sizeof(p));
    p.model = probe_model_defaults();
    p.prompt = PROBE_DEFAULT_PROMPT;
    p.prompt_file = NULL;
    p.top_k = 8;
    p.trace_limit = 160;
    *layers_set = false;
    *tokenizer_set = false;
    *softcap_set = false;
    for (i = 1; i < argc; ++i) {
        const char *value;
        int64_t parsed_i64 = 0;
        uint64_t parsed_u64 = 0U;
        float parsed_f32 = 0.0f;
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            exit(0);
        }
        value = arg_value_probe(argc, argv, &i, "--checkpoint");
        if (value != NULL) { p.model.checkpoint_path = value; continue; }
        value = arg_value_probe(argc, argv, &i, "--checkpoint-path");
        if (value != NULL) { p.model.checkpoint_path = value; continue; }
        value = arg_value_probe(argc, argv, &i, "--data-dir");
        if (value != NULL) { p.model.data_dir = value; continue; }
        value = arg_value_probe(argc, argv, &i, "--tokenizer-path");
        if (value != NULL) { p.model.tokenizer_path = value; *tokenizer_set = true; continue; }
        value = arg_value_probe(argc, argv, &i, "--prompt");
        if (value != NULL) { p.prompt = value; continue; }
        value = arg_value_probe(argc, argv, &i, "--prompt-file");
        if (value != NULL) { p.prompt_file = value; continue; }
        value = arg_value_probe(argc, argv, &i, "--max-new-tokens");
        if (value != NULL) {
            if (!parse_i64_probe(value, 1, GPT_CONTEXT_LENGTH, &parsed_i64)) { fprintf(stderr, "invalid --max-new-tokens\n"); exit(2); }
            p.model.max_new_tokens = (int)parsed_i64; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--temperature");
        if (value != NULL) {
            if (!parse_float_probe(value, 0.0f, 10.0f, &parsed_f32)) { fprintf(stderr, "invalid --temperature\n"); exit(2); }
            p.model.temperature = parsed_f32; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--min-p");
        if (value != NULL) {
            if (!parse_float_probe(value, 0.0f, 1.0f, &parsed_f32)) { fprintf(stderr, "invalid --min-p\n"); exit(2); }
            p.model.min_p = parsed_f32; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--repetition-penalty");
        if (value != NULL) {
            if (!parse_float_probe(value, 1.0f, 10.0f, &parsed_f32)) { fprintf(stderr, "invalid --repetition-penalty\n"); exit(2); }
            p.model.repetition_penalty = parsed_f32; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--top-k");
        if (value != NULL) {
            if (!parse_i64_probe(value, 1, 64, &parsed_i64)) { fprintf(stderr, "invalid --top-k\n"); exit(2); }
            p.top_k = (int)parsed_i64; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--trace-limit");
        if (value != NULL) {
            if (!parse_i64_probe(value, 0, GPT_CONTEXT_LENGTH, &parsed_i64)) { fprintf(stderr, "invalid --trace-limit\n"); exit(2); }
            p.trace_limit = (int)parsed_i64; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--layers");
        if (value != NULL) {
            if (!parse_i64_probe(value, 1, 96, &parsed_i64)) { fprintf(stderr, "invalid --layers\n"); exit(2); }
            p.model.n_layers = (int)parsed_i64; *layers_set = true; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_probe(value, 0.0f, 1000000.0f, &parsed_f32)) { fprintf(stderr, "invalid --logits-softcap\n"); exit(2); }
            p.model.logits_softcap = parsed_f32; *softcap_set = true; continue;
        }
        value = arg_value_probe(argc, argv, &i, "--seed");
        if (value != NULL) {
            if (!parse_u64_probe(value, UINT64_MAX, &parsed_u64)) { fprintf(stderr, "invalid --seed\n"); exit(2); }
            p.model.seed = parsed_u64; continue;
        }
        if (strcmp(argv[i], "--no-prompt-ce") == 0) { p.no_prompt_ce = true; continue; }
        if (strcmp(argv[i], "--no-generate") == 0) { p.no_generate = true; continue; }
        fprintf(stderr, "gpt_lm_probe: unknown argument %s\n", argv[i]);
        usage(argv[0]);
        exit(2);
    }
    return p;
}

static bool metadata_value_probe(const char *metadata,
                                 size_t metadata_len,
                                 const char *key,
                                 char *out,
                                 size_t out_size)
{
    const size_t key_len = strlen(key);
    size_t offset = 0U;
    if (metadata == NULL || key == NULL || out == NULL || out_size == 0U) { return false; }
    while (offset < metadata_len) {
        const size_t line_start = offset;
        size_t line_end = line_start;
        while (line_end < metadata_len && metadata[line_end] != '\n') { line_end += 1U; }
        if (line_end > line_start + key_len && strncmp(metadata + line_start, key, key_len) == 0 &&
            metadata[line_start + key_len] == '=') {
            const size_t value_start = line_start + key_len + 1U;
            const size_t value_len = line_end - value_start;
            if (value_len >= out_size) { return false; }
            memcpy(out, metadata + value_start, value_len);
            out[value_len] = '\0';
            return true;
        }
        offset = line_end < metadata_len ? line_end + 1U : line_end;
    }
    return false;
}

static void apply_metadata_probe(gpt_config *config,
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
    if (config == NULL || metadata == NULL) { return; }
    if (!layers_set && metadata_value_probe(metadata, metadata_len, "n_layers", value, sizeof(value)) &&
        parse_i64_probe(value, 1, 96, &parsed_i64)) {
        config->n_layers = (int)parsed_i64;
    }
    if (!softcap_set && metadata_value_probe(metadata, metadata_len, "logits_softcap", value, sizeof(value)) &&
        parse_float_probe(value, 0.0f, 1000000.0f, &parsed_f32)) {
        config->logits_softcap = parsed_f32;
    }
    if (!tokenizer_set && metadata_value_probe(metadata, metadata_len, "tokenizer_path", tokenizer_storage, tokenizer_storage_size)) {
        config->tokenizer_path = tokenizer_storage;
    }
}

static char *read_file_text(const char *path)
{
    FILE *f;
    long size_long;
    size_t size;
    char *buf;
    if (path == NULL) { return NULL; }
    f = fopen(path, "rb");
    if (f == NULL) { perror(path); exit(2); }
    if (fseek(f, 0L, SEEK_END) != 0) { perror("fseek"); exit(2); }
    size_long = ftell(f);
    if (size_long < 0) { perror("ftell"); exit(2); }
    if (fseek(f, 0L, SEEK_SET) != 0) { perror("fseek"); exit(2); }
    size = (size_t)size_long;
    buf = (char *)malloc(size + 1U);
    if (buf == NULL) { fprintf(stderr, "out of memory reading prompt file\n"); exit(2); }
    if (size != 0U && fread(buf, 1U, size, f) != size) { perror("fread"); exit(2); }
    if (fclose(f) != 0) { perror("fclose"); exit(2); }
    buf[size] = '\0';
    return buf;
}

static char *escape_piece(const char *text)
{
    size_t len = text != NULL ? strlen(text) : 0U;
    size_t cap = len * 4U + 1U;
    char *out = (char *)malloc(cap);
    size_t j = 0U;
    size_t i;
    if (out == NULL) { return NULL; }
    for (i = 0U; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n') { out[j++] = '\\'; out[j++] = 'n'; }
        else if (c == '\r') { out[j++] = '\\'; out[j++] = 'r'; }
        else if (c == '\t') { out[j++] = '\\'; out[j++] = 't'; }
        else if (c == '\\') { out[j++] = '\\'; out[j++] = '\\'; }
        else if (c == '"') { out[j++] = '\\'; out[j++] = '"'; }
        else if (c < 32U || c == 127U) {
            static const char hex[] = "0123456789abcdef";
            out[j++] = '\\'; out[j++] = 'x'; out[j++] = hex[c >> 4]; out[j++] = hex[c & 15U];
        } else {
            out[j++] = (char)c;
        }
    }
    out[j] = '\0';
    return out;
}

static char *token_piece(gd_tokenizer *tok, int32_t id)
{
    char *raw = NULL;
    char *escaped;
    if (gd_tokenizer_decode(tok, &id, 1, &raw) != GD_OK || raw == NULL) {
        raw = NULL;
        escaped = (char *)malloc(32U);
        if (escaped != NULL) { (void)snprintf(escaped, 32U, "<id:%d>", (int)id); }
        return escaped;
    }
    escaped = escape_piece(raw);
    gd_tokenizer_free(raw);
    return escaped;
}

static double row_logsumexp(const float *x, int n)
{
    double m = -HUGE_VAL;
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) {
        if (isfinite((double)x[i]) && (double)x[i] > m) { m = (double)x[i]; }
    }
    if (!isfinite(m)) { return m; }
    for (i = 0; i < n; ++i) {
        if (isfinite((double)x[i])) { s += exp((double)x[i] - m); }
    }
    return m + log(s);
}

static double row_entropy_from_logits(const float *x, int n)
{
    const double lse = row_logsumexp(x, n);
    double h = 0.0;
    int i;
    if (!isfinite(lse)) { return NAN; }
    for (i = 0; i < n; ++i) {
        if (isfinite((double)x[i])) {
            const double logp = (double)x[i] - lse;
            const double p = exp(logp);
            h -= p * logp;
        }
    }
    return h;
}

static double target_nll_rank_prob(const float *x, int n, int target, int *rank_out, double *prob_out)
{
    const double lse = row_logsumexp(x, n);
    int rank = 1;
    int i;
    if (target < 0 || target >= n || !isfinite(lse)) {
        if (rank_out != NULL) { *rank_out = -1; }
        if (prob_out != NULL) { *prob_out = NAN; }
        return NAN;
    }
    for (i = 0; i < n; ++i) {
        if (i != target && x[i] > x[target]) { rank += 1; }
    }
    if (rank_out != NULL) { *rank_out = rank; }
    if (prob_out != NULL) { *prob_out = exp((double)x[target] - lse); }
    return lse - (double)x[target];
}

static void topk_indices(const float *x, int n, int k, int *idx)
{
    int i;
    int j;
    for (j = 0; j < k; ++j) { idx[j] = -1; }
    for (i = 0; i < n; ++i) {
        for (j = 0; j < k; ++j) {
            if (idx[j] < 0 || x[i] > x[idx[j]]) {
                int m;
                for (m = k - 1; m > j; --m) { idx[m] = idx[m - 1]; }
                idx[j] = i;
                break;
            }
        }
    }
}

static void print_topk(gd_tokenizer *tok, const float *scores, int vocab, int k)
{
    int *idx = (int *)calloc((size_t)k, sizeof(idx[0]));
    double lse;
    int j;
    if (idx == NULL) { return; }
    topk_indices(scores, vocab, k, idx);
    lse = row_logsumexp(scores, vocab);
    printf(" top%d=[", k);
    for (j = 0; j < k; ++j) {
        char *piece;
        double p;
        if (idx[j] < 0) { continue; }
        piece = token_piece(tok, (int32_t)idx[j]);
        p = isfinite(lse) ? exp((double)scores[idx[j]] - lse) : NAN;
        printf("%s%d:%s:%.4f", j == 0 ? "" : ", ", idx[j], piece != NULL ? piece : "?", p);
        free(piece);
    }
    printf("]");
    free(idx);
}

static void write_i32_tensor_or_die(gd_context *ctx, const int32_t *values, int64_t count, gd_tensor *out)
{
    TRY(ctx, gpt_data_i32_tensor(ctx, values, count, out));
}

static void trace_prompt_ce(gd_context *ctx,
                            gpt_lm *model,
                            const gpt_config *config,
                            gd_tokenizer *tok,
                            const char *prompt,
                            int top_k,
                            int trace_limit)
{
    int32_t *encoded = NULL;
    int n_encoded = 0;
    int offset = 0;
    int n;
    int32_t *ids = NULL;
    int32_t *positions = NULL;
    int32_t cu[2];
    int32_t cache_pos[1];
    gpt_kv_cache *cache = NULL;
    float *logits_host = NULL;
    int i;
    double sum_nll = 0.0;
    double sum_prob = 0.0;
    int rows = 0;
    int print_start;

    TRY(ctx, gd_tokenizer_encode(tok, prompt, &encoded, &n_encoded));
    if (n_encoded < 2) {
        printf("prompt_ce: encoded tokens=%d; need at least 2\n", n_encoded);
        gd_tokenizer_free(encoded);
        return;
    }
    if (n_encoded > GPT_CONTEXT_LENGTH) {
        offset = n_encoded - GPT_CONTEXT_LENGTH;
        printf("prompt_ce: prompt tokens=%d exceeds context=%d; tracing last %d tokens\n", n_encoded, GPT_CONTEXT_LENGTH, GPT_CONTEXT_LENGTH);
    }
    n = n_encoded - offset;
    ids = (int32_t *)calloc((size_t)n, sizeof(ids[0]));
    positions = (int32_t *)calloc((size_t)n, sizeof(positions[0]));
    logits_host = (float *)malloc((size_t)n * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (ids == NULL || positions == NULL || logits_host == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "prompt CE allocations", __LINE__);
    }
    for (i = 0; i < n; ++i) {
        ids[i] = encoded[offset + i];
        positions[i] = (int32_t)i;
    }
    cu[0] = 0;
    cu[1] = (int32_t)n;
    cache_pos[0] = 0;

    TRY(ctx, gpt_lm_prepare_generation_cache(ctx, model, config, 1, GPT_CONTEXT_LENGTH, &cache));
    gd_module_set_training(&model->mod, false);
    {
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
        write_i32_tensor_or_die(ctx, ids, n, &ids_t);
        write_i32_tensor_or_die(ctx, positions, n, &pos_t);
        write_i32_tensor_or_die(ctx, cu, 2, &cu_t);
        write_i32_tensor_or_die(ctx, cache_pos, 1, &cache_pos_t);
        TRY(ctx, gpt_lm_prefill_logits(ctx, model, cache, &ids_t, &pos_t, &cu_t, &cache_pos_t, &logits));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_tensor_read_f32(ctx, &logits, logits_host, (size_t)n * (size_t)GPT_VOCAB_SIZE));
    }

    for (i = 0; i < n - 1; ++i) {
        float *row = logits_host + (size_t)i * (size_t)GPT_VOCAB_SIZE;
        int rank;
        double prob;
        double nll;
        gpt_apply_logits_softcap(row, GPT_VOCAB_SIZE, model->logits_softcap);
        nll = target_nll_rank_prob(row, GPT_VOCAB_SIZE, ids[i + 1], &rank, &prob);
        if (isfinite(nll)) {
            sum_nll += nll;
            sum_prob += prob;
            rows += 1;
        }
    }
    printf("\nprompt_ce: checkpoint=%s tokens=%d traced_tokens=%d avg_nll=%.6f nats avg_bits=%.4f avg_p_target=%.5f logits_softcap=%.3f\n",
           config->checkpoint_path,
           n_encoded,
           n,
           rows > 0 ? sum_nll / (double)rows : NAN,
           rows > 0 ? (sum_nll / (double)rows) / log(2.0) : NAN,
           rows > 0 ? sum_prob / (double)rows : NAN,
           (double)model->logits_softcap);
    print_start = 0;
    if (trace_limit > 0 && n - 1 > trace_limit) {
        print_start = (n - 1) - trace_limit;
        printf("prompt_ce: printing tail rows %d..%d of %d predicted-token rows\n", print_start, n - 2, n - 1);
    }
    printf("idx pos input_id input_piece -> target_id target_piece nll bits p_target rank top1\n");
    for (i = print_start; i < n - 1; ++i) {
        float *row = logits_host + (size_t)i * (size_t)GPT_VOCAB_SIZE;
        int rank;
        double prob;
        double nll = target_nll_rank_prob(row, GPT_VOCAB_SIZE, ids[i + 1], &rank, &prob);
        int top1 = 0;
        char *in_piece;
        char *target_piece;
        char *top_piece;
        int j;
        for (j = 1; j < GPT_VOCAB_SIZE; ++j) { if (row[j] > row[top1]) { top1 = j; } }
        in_piece = token_piece(tok, ids[i]);
        target_piece = token_piece(tok, ids[i + 1]);
        top_piece = token_piece(tok, (int32_t)top1);
        printf("%4d %3d %4d %-16s -> %4d %-16s nll=%7.4f bits=%6.3f p=%8.5f rank=%4d top1=%4d:%s",
               i,
               i,
               (int)ids[i],
               in_piece != NULL ? in_piece : "?",
               (int)ids[i + 1],
               target_piece != NULL ? target_piece : "?",
               nll,
               nll / log(2.0),
               prob,
               rank,
               top1,
               top_piece != NULL ? top_piece : "?");
        if (top_k > 1) { print_topk(tok, row, GPT_VOCAB_SIZE, top_k); }
        printf("\n");
        free(in_piece);
        free(target_piece);
        free(top_piece);
    }

    free(logits_host);
    free(positions);
    free(ids);
    gd_tokenizer_free(encoded);
}

static void trace_generation(gd_context *ctx,
                             gpt_lm *model,
                             const gpt_config *config,
                             gd_tokenizer *tok,
                             const char *prompt,
                             int top_k)
{
    int32_t *encoded = NULL;
    int n_encoded = 0;
    int offset = 0;
    int prompt_len;
    int room_for_prompt;
    int32_t *seq_ids = NULL;
    int32_t *packed_positions = NULL;
    int32_t cu[2];
    int32_t cache_pos[1];
    int32_t decode_id[1];
    int32_t decode_pos[1];
    float *logits_host = NULL;
    gpt_kv_cache *cache = NULL;
    uint64_t rng;
    int next_id;
    int seq_len;
    int generated = 0;
    int stop_token = -1;

    TRY(ctx, gd_tokenizer_encode(tok, prompt, &encoded, &n_encoded));
    if (n_encoded <= 0) {
        printf("generation: empty prompt after tokenization\n");
        gd_tokenizer_free(encoded);
        return;
    }
    room_for_prompt = config->max_new_tokens >= GPT_CONTEXT_LENGTH ? GPT_CONTEXT_LENGTH - 1 : GPT_CONTEXT_LENGTH - config->max_new_tokens;
    if (room_for_prompt <= 0) { room_for_prompt = GPT_CONTEXT_LENGTH - 1; }
    if (n_encoded > room_for_prompt) {
        offset = n_encoded - room_for_prompt;
        printf("generation: prompt tokens=%d exceeds generation room=%d; using last %d tokens\n", n_encoded, room_for_prompt, room_for_prompt);
    }
    prompt_len = n_encoded - offset;
    seq_len = prompt_len;
    seq_ids = (int32_t *)calloc((size_t)GPT_CONTEXT_LENGTH, sizeof(seq_ids[0]));
    packed_positions = (int32_t *)calloc((size_t)prompt_len, sizeof(packed_positions[0]));
    logits_host = (float *)malloc((size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (seq_ids == NULL || packed_positions == NULL || logits_host == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation trace allocations", __LINE__);
    }
    memcpy(seq_ids, encoded + offset, (size_t)prompt_len * sizeof(seq_ids[0]));
    for (int i = 0; i < prompt_len; ++i) { packed_positions[i] = (int32_t)i; }
    cu[0] = 0;
    cu[1] = (int32_t)prompt_len;
    cache_pos[0] = 0;
    if (gd_tokenizer_id(tok, "<|im_end|>", &stop_token) != GD_OK) { stop_token = -1; }
    rng = splitmix64(config->seed ^ UINT64_C(0xdec0de1234567890) ^ UINT64_C(1));

    TRY(ctx, gpt_lm_prepare_generation_cache(ctx, model, config, 1, GPT_CONTEXT_LENGTH, &cache));
    gd_module_set_training(&model->mod, false);
    {
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        gd_tensor last_logits;
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
        write_i32_tensor_or_die(ctx, seq_ids, prompt_len, &ids_t);
        write_i32_tensor_or_die(ctx, packed_positions, prompt_len, &pos_t);
        write_i32_tensor_or_die(ctx, cu, 2, &cu_t);
        write_i32_tensor_or_die(ctx, cache_pos, 1, &cache_pos_t);
        TRY(ctx, gpt_lm_prefill_logits(ctx, model, cache, &ids_t, &pos_t, &cu_t, &cache_pos_t, &logits));
        TRY(ctx, gd_tensor_slice(ctx, &logits, 0U, (int64_t)prompt_len - 1, 1, &last_logits));
        TRY(ctx, gd_end_step(ctx));
        TRY(ctx, gd_tensor_read_f32(ctx, &last_logits, logits_host, GPT_VOCAB_SIZE));
        cache->pos[0] = prompt_len;
    }

    printf("\ngeneration_trace: prompt_tokens=%d max_new=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f stop_token=%d\n",
           prompt_len,
           config->max_new_tokens,
           (double)config->temperature,
           (double)config->min_p,
           (double)config->repetition_penalty,
           stop_token);
    while (generated < config->max_new_tokens) {
        float *display_scores = (float *)malloc((size_t)GPT_VOCAB_SIZE * sizeof(display_scores[0]));
        float *sample_scores = (float *)malloc((size_t)GPT_VOCAB_SIZE * sizeof(sample_scores[0]));
        double entropy;
        double lse;
        double selected_prob;
        char *piece;
        if (display_scores == NULL || sample_scores == NULL) {
            gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation score copies", __LINE__);
        }
        memcpy(display_scores, logits_host, (size_t)GPT_VOCAB_SIZE * sizeof(display_scores[0]));
        memcpy(sample_scores, logits_host, (size_t)GPT_VOCAB_SIZE * sizeof(sample_scores[0]));
        gpt_apply_logits_softcap(display_scores, GPT_VOCAB_SIZE, model->logits_softcap);
        gpt_apply_logits_softcap(sample_scores, GPT_VOCAB_SIZE, model->logits_softcap);
        gpt_apply_repetition_penalty(display_scores,
                                     GPT_VOCAB_SIZE,
                                     seq_ids,
                                     seq_len,
                                     config->repetition_penalty);
        next_id = gpt_sample_next_token(sample_scores,
                                        GPT_VOCAB_SIZE,
                                        seq_ids,
                                        seq_len,
                                        config->temperature,
                                        config->min_p,
                                        config->repetition_penalty,
                                        &rng);
        entropy = row_entropy_from_logits(display_scores, GPT_VOCAB_SIZE);
        lse = row_logsumexp(display_scores, GPT_VOCAB_SIZE);
        selected_prob = isfinite(lse) ? exp((double)display_scores[next_id] - lse) : NAN;
        piece = token_piece(tok, (int32_t)next_id);
        printf("gen_step=%3d input_pos=%3d next_id=%4d piece=%-16s p=%.5f nll=%.4f entropy=%.4f bits=%.3f",
               generated,
               seq_len - 1,
               next_id,
               piece != NULL ? piece : "?",
               selected_prob,
               isfinite(selected_prob) && selected_prob > 0.0 ? -log(selected_prob) : NAN,
               entropy,
               entropy / log(2.0));
        print_topk(tok, display_scores, GPT_VOCAB_SIZE, top_k);
        printf("\n");
        free(piece);
        if (stop_token >= 0 && next_id == stop_token) {
            free(display_scores);
            free(sample_scores);
            printf("generation_trace: stop token selected before append\n");
            break;
        }
        if (seq_len >= GPT_CONTEXT_LENGTH) {
            free(display_scores);
            free(sample_scores);
            printf("generation_trace: context full\n");
            break;
        }
        seq_ids[seq_len] = (int32_t)next_id;
        seq_len += 1;
        generated += 1;
        free(display_scores);
        free(sample_scores);
        if (generated >= config->max_new_tokens) { break; }
        decode_id[0] = (int32_t)next_id;
        decode_pos[0] = cache->pos[0];
        cache_pos[0] = cache->pos[0];
        {
            gd_tensor ids_t;
            gd_tensor pos_t;
            gd_tensor cache_pos_t;
            gd_tensor logits;
            TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
            write_i32_tensor_or_die(ctx, decode_id, 1, &ids_t);
            write_i32_tensor_or_die(ctx, decode_pos, 1, &pos_t);
            write_i32_tensor_or_die(ctx, cache_pos, 1, &cache_pos_t);
            TRY(ctx, gpt_lm_decode_logits(ctx, model, cache, &ids_t, &pos_t, &cache_pos_t, &logits));
            TRY(ctx, gd_end_step(ctx));
            TRY(ctx, gd_tensor_read_f32(ctx, &logits, logits_host, GPT_VOCAB_SIZE));
        }
    }
    {
        char *decoded = NULL;
        TRY(ctx, gd_tokenizer_decode(tok, seq_ids + prompt_len, seq_len - prompt_len, &decoded));
        printf("generation_trace: generated_tokens=%d text=%s\n", seq_len - prompt_len, decoded != NULL ? decoded : "");
        gd_tokenizer_free(decoded);
    }
    free(logits_host);
    free(packed_positions);
    free(seq_ids);
    gd_tokenizer_free(encoded);
}

int main(int argc, char **argv)
{
    bool layers_set;
    bool tokenizer_set;
    bool softcap_set;
    probe_config probe = parse_probe_args(argc, argv, &layers_set, &tokenizer_set, &softcap_set);
    char *metadata = NULL;
    size_t metadata_len = 0U;
    char tokenizer_from_metadata[1024];
    char *prompt_owned = NULL;
    gd_memory_config mem;
    gd_context *ctx = NULL;
    gd_status st;
    gpt_lm model;
    gpt_generation_tokenizer generation_tokenizer;
    gd_module_load_options load_options;
    int exit_code = 1;

    memset(&model, 0, sizeof(model));
    memset(&generation_tokenizer, 0, sizeof(generation_tokenizer));
    tokenizer_from_metadata[0] = '\0';
    st = gd_checkpoint_read_metadata(probe.model.checkpoint_path, &metadata, &metadata_len);
    if (st == GD_OK) {
        apply_metadata_probe(&probe.model,
                             metadata,
                             metadata_len,
                             layers_set,
                             tokenizer_set,
                             softcap_set,
                             tokenizer_from_metadata,
                             sizeof(tokenizer_from_metadata));
    } else {
        fprintf(stderr, "gpt_lm_probe: warning: could not read checkpoint metadata (%s)\n", gd_status_string(st));
    }
    if (probe.prompt_file != NULL) {
        prompt_owned = read_file_text(probe.prompt_file);
        probe.prompt = prompt_owned;
    }
    probe.model.generate_prompt = probe.prompt;

    mem = gpt_memory_config(&probe.model);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm_probe: skipped (no supported gradients.c backend)\n");
        free(prompt_owned);
        free(metadata);
        return 0;
    }
    if (st != GD_OK) { gpt_fail_status(ctx, st, "gd_context_create", __LINE__); }
    if (GPT_D_MODEL != GPT_N_HEADS * GPT_HEAD_DIM) {
        gpt_fail_status(ctx, GD_ERR_BAD_STATE, "invalid GPT head config", __LINE__);
    }
    printf("probe_config: checkpoint=%s data_dir=%s tokenizer=%s layers=%d max_new=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f top_k=%d\n",
           probe.model.checkpoint_path,
           probe.model.data_dir,
           probe.model.tokenizer_path != NULL ? probe.model.tokenizer_path : "<default>",
           probe.model.n_layers,
           probe.model.max_new_tokens,
           (double)probe.model.temperature,
           (double)probe.model.min_p,
           (double)probe.model.repetition_penalty,
           (double)probe.model.logits_softcap,
           probe.top_k);
    if (metadata != NULL) {
        char value[128];
        if (metadata_value_probe(metadata, metadata_len, "epoch", value, sizeof(value))) { printf("checkpoint_metadata: epoch=%s", value); }
        if (metadata_value_probe(metadata, metadata_len, "global_step", value, sizeof(value))) { printf(" global_step=%s", value); }
        if (metadata_value_probe(metadata, metadata_len, "val_loss", value, sizeof(value))) { printf(" val_loss=%s", value); }
        if (metadata_value_probe(metadata, metadata_len, "best_val_loss", value, sizeof(value))) { printf(" best_val_loss=%s", value); }
        printf("\n");
    }

    gpt_lm_init(ctx, &model, &probe.model);
    load_options.strict = true;
    load_options.load_buffers = true;
    TRY(ctx, gd_module_load_state(ctx, &model.mod, probe.model.checkpoint_path, &load_options));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, false);
    gpt_generation_tokenizer_init(ctx, &probe.model, &generation_tokenizer);

    if (!probe.no_prompt_ce) {
        trace_prompt_ce(ctx, &model, &probe.model, generation_tokenizer.tok, probe.prompt, probe.top_k, probe.trace_limit);
    }
    if (!probe.no_generate) {
        trace_generation(ctx, &model, &probe.model, generation_tokenizer.tok, probe.prompt, probe.top_k);
    }
    exit_code = 0;

    gpt_generation_tokenizer_deinit(&generation_tokenizer);
    gpt_lm_deinit(&model);
    gd_context_destroy(ctx);
    free(prompt_owned);
    free(metadata);
    return exit_code;
}
