/* Temporary GPT-LM semantic-content NLL probe.
 *
 * For each raw_ita_dict entry, condition on:
 *   <|im_start|>Termine: <term>\nDefinizioni:\n1.
 * and measure the NLL of the first definition content tokens, excluding the
 * static/template header and excluding the term token itself.
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../examples/gpt_lm/gpt_lm_shared.h"

#define static
#include "../examples/gpt_lm/gpt_lm_shared.c"
#undef static

#define DEFAULT_CHECKPOINT "examples/gpt_lm/checkpoints/gpt_lm_latest.gdckpt"
#define DEFAULT_DATA_DIR "examples/gpt_lm/data"
#define DEFAULT_SPLIT "val"

typedef struct entry_result {
    char *term;
    char *first_piece;
    char *definition;
    int tokens;
    double first_nll;
    double first4_nll;
    double first8_nll;
    double all_nll;
    int first_rank;
} entry_result;

typedef struct result_list {
    entry_result *items;
    size_t count;
    size_t cap;
} result_list;

static int parse_i64_arg(const char *text, int64_t min_value, int64_t max_value, int64_t *out)
{
    char *end = NULL;
    long long parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) { return 0; }
    errno = 0;
    parsed = strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < min_value || parsed > max_value) { return 0; }
    *out = (int64_t)parsed;
    return 1;
}

static int parse_float_arg(const char *text, float min_value, float max_value, float *out)
{
    char *end = NULL;
    float parsed;
    if (text == NULL || text[0] == '\0' || out == NULL) { return 0; }
    errno = 0;
    parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed) || parsed < min_value || parsed > max_value) { return 0; }
    *out = parsed;
    return 1;
}

static const char *arg_value(int argc, char **argv, int *index, const char *name)
{
    const size_t name_len = strlen(name);
    const char *arg = argv[*index];
    if (strncmp(arg, name, name_len) == 0 && arg[name_len] == '=') { return arg + name_len + 1U; }
    if (strcmp(arg, name) == 0) {
        if (*index + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); exit(2); }
        *index += 1;
        return argv[*index];
    }
    return NULL;
}

static bool metadata_value(const char *metadata, size_t metadata_len, const char *key, char *out, size_t out_size)
{
    const size_t key_len = strlen(key);
    size_t offset = 0U;
    if (metadata == NULL || key == NULL || out == NULL || out_size == 0U) { return false; }
    while (offset < metadata_len) {
        const size_t line_start = offset;
        size_t line_end = line_start;
        while (line_end < metadata_len && metadata[line_end] != '\n') { line_end += 1U; }
        if (line_end > line_start + key_len && strncmp(metadata + line_start, key, key_len) == 0 && metadata[line_start + key_len] == '=') {
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

static gpt_config default_config(void)
{
    gpt_config c;
    memset(&c, 0, sizeof(c));
    c.data_dir = DEFAULT_DATA_DIR;
    c.tokenizer_path = NULL;
    c.generate_prompt = "x";
    c.checkpoint_path = DEFAULT_CHECKPOINT;
    c.val_split = "val";
    c.epochs = 0;
    c.batch_size = GPT_DEFAULT_BATCH_SIZE;
    c.n_layers = GPT_DEFAULT_LAYERS;
    c.lr_warmup_steps = -1;
    c.max_new_tokens = 1;
    c.epochs_set = true;
    c.save_best = false;
    c.save_latest = false;
    c.seed = GPT_DEFAULT_SEED;
    c.dropout_p = GPT_DEFAULT_DROPOUT_P;
    c.lr_max = GPT_DEFAULT_LR_MAX;
    c.lr_min = GPT_DEFAULT_LR_MIN;
    c.weight_decay = GPT_DEFAULT_WEIGHT_DECAY;
    c.grad_clip_norm = GPT_DEFAULT_GRAD_CLIP_NORM;
    c.temperature = 0.0f;
    c.min_p = 0.0f;
    c.repetition_penalty = 1.0f;
    c.logits_softcap = 0.0f;
    return c;
}

static char *xstrndup(const char *s, size_t n)
{
    char *out = (char *)malloc(n + 1U);
    if (out == NULL) { return NULL; }
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *read_file_text(const char *path)
{
    FILE *f = fopen(path, "rb");
    long sz;
    char *buf;
    if (f == NULL) { perror(path); exit(2); }
    if (fseek(f, 0L, SEEK_END) != 0) { perror("fseek"); exit(2); }
    sz = ftell(f);
    if (sz < 0) { perror("ftell"); exit(2); }
    if (fseek(f, 0L, SEEK_SET) != 0) { perror("fseek"); exit(2); }
    buf = (char *)malloc((size_t)sz + 1U);
    if (buf == NULL) { fprintf(stderr, "out of memory reading %s\n", path); exit(2); }
    if (sz != 0 && fread(buf, 1U, (size_t)sz, f) != (size_t)sz) { perror("fread"); exit(2); }
    if (fclose(f) != 0) { perror("fclose"); exit(2); }
    buf[(size_t)sz] = '\0';
    return buf;
}

static char *path_join3(const char *a, const char *b, const char *c)
{
    size_t na = strlen(a), nb = strlen(b), nc = strlen(c);
    bool sep1 = na == 0U || a[na - 1U] != '/';
    bool sep2 = nb == 0U || b[nb - 1U] != '/';
    char *out = (char *)malloc(na + nb + nc + (sep1 ? 1U : 0U) + (sep2 ? 1U : 0U) + 1U);
    if (out == NULL) { return NULL; }
    sprintf(out, "%s%s%s%s%s", a, sep1 ? "/" : "", b, sep2 ? "/" : "", c);
    return out;
}

static char *make_header_prompt(const char *term)
{
    const char *prefix = "<|im_start|>Termine: ";
    const char *suffix = "\nDefinizioni:\n1.";
    size_t n = strlen(prefix) + strlen(term) + strlen(suffix);
    char *out = (char *)malloc(n + 1U);
    if (out == NULL) { return NULL; }
    sprintf(out, "%s%s%s", prefix, term, suffix);
    return out;
}

static char *make_full_prompt(const char *term, const char *definition)
{
    char *header = make_header_prompt(term);
    size_t n;
    char *out;
    if (header == NULL) { return NULL; }
    n = strlen(header) + 1U + strlen(definition);
    out = (char *)malloc(n + 1U);
    if (out == NULL) { free(header); return NULL; }
    sprintf(out, "%s %s", header, definition);
    free(header);
    return out;
}

static char *escape_piece(const char *text)
{
    size_t len = text != NULL ? strlen(text) : 0U;
    char *out = (char *)malloc(len * 4U + 1U);
    size_t j = 0U;
    size_t i;
    if (out == NULL) { return NULL; }
    for (i = 0U; i < len; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c == '\n') { out[j++] = '\\'; out[j++] = 'n'; }
        else if (c == '\r') { out[j++] = '\\'; out[j++] = 'r'; }
        else if (c == '\t') { out[j++] = '\\'; out[j++] = 't'; }
        else if (c == '\\') { out[j++] = '\\'; out[j++] = '\\'; }
        else if (c < 32U || c == 127U) {
            static const char hex[] = "0123456789abcdef";
            out[j++] = '\\'; out[j++] = 'x'; out[j++] = hex[c >> 4]; out[j++] = hex[c & 15U];
        } else { out[j++] = (char)c; }
    }
    out[j] = '\0';
    return out;
}

static char *token_piece(gd_tokenizer *tok, int32_t id)
{
    char *raw = NULL;
    char *escaped;
    if (gd_tokenizer_decode(tok, &id, 1, &raw) != GD_OK || raw == NULL) {
        escaped = (char *)malloc(32U);
        if (escaped != NULL) { sprintf(escaped, "<id:%d>", (int)id); }
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
    for (i = 0; i < n; ++i) { if (isfinite((double)x[i]) && (double)x[i] > m) { m = (double)x[i]; } }
    if (!isfinite(m)) { return m; }
    for (i = 0; i < n; ++i) { if (isfinite((double)x[i])) { s += exp((double)x[i] - m); } }
    return m + log(s);
}

static double target_nll_rank(const float *x, int n, int target, int *rank_out)
{
    const double lse = row_logsumexp(x, n);
    int rank = 1;
    int i;
    if (target < 0 || target >= n || !isfinite(lse)) {
        if (rank_out != NULL) { *rank_out = -1; }
        return NAN;
    }
    for (i = 0; i < n; ++i) { if (i != target && x[i] > x[target]) { rank += 1; } }
    if (rank_out != NULL) { *rank_out = rank; }
    return lse - (double)x[target];
}

static void result_list_append(result_list *list, entry_result item)
{
    entry_result *grown;
    if (list->count == list->cap) {
        size_t new_cap = list->cap == 0U ? 128U : list->cap * 2U;
        grown = (entry_result *)realloc(list->items, new_cap * sizeof(list->items[0]));
        if (grown == NULL) { fprintf(stderr, "out of memory growing results\n"); exit(2); }
        list->items = grown;
        list->cap = new_cap;
    }
    list->items[list->count++] = item;
}

static int cmp_first_desc(const void *a, const void *b)
{
    const entry_result *x = (const entry_result *)a;
    const entry_result *y = (const entry_result *)b;
    return x->first_nll < y->first_nll ? 1 : (x->first_nll > y->first_nll ? -1 : 0);
}

static int cmp_all_asc_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static gd_status eval_entry(gd_context *ctx,
                            gpt_lm *model,
                            const gpt_config *config,
                            gd_tokenizer *tok,
                            const char *term,
                            const char *definition,
                            entry_result *out)
{
    char *header = NULL;
    char *full = NULL;
    int32_t *header_ids = NULL;
    int32_t *ids = NULL;
    int n_header = 0;
    int n_ids = 0;
    int common = 0;
    int content_start;
    int32_t *positions = NULL;
    int32_t cu[2];
    int32_t cache_pos[1];
    gpt_kv_cache *cache = NULL;
    float *logits_host = NULL;
    double sum_all = 0.0;
    double sum4 = 0.0;
    double sum8 = 0.0;
    int cnt_all = 0;
    int cnt4 = 0;
    int cnt8 = 0;
    int first_rank = -1;
    double first_nll = NAN;
    int i;
    gd_status st = GD_OK;

    memset(out, 0, sizeof(*out));
    header = make_header_prompt(term);
    full = make_full_prompt(term, definition);
    if (header == NULL || full == NULL) { st = GD_ERR_OUT_OF_MEMORY; goto done; }
    st = gd_tokenizer_encode(tok, header, &header_ids, &n_header);
    if (st != GD_OK) { goto done; }
    st = gd_tokenizer_encode(tok, full, &ids, &n_ids);
    if (st != GD_OK) { goto done; }
    while (common < n_header && common < n_ids && header_ids[common] == ids[common]) { common += 1; }
    content_start = common;
    if (content_start <= 0 || content_start >= n_ids || n_ids > GPT_CONTEXT_LENGTH) {
        st = GD_ERR_INVALID_ARGUMENT;
        goto done;
    }
    positions = (int32_t *)calloc((size_t)n_ids, sizeof(positions[0]));
    logits_host = (float *)malloc((size_t)n_ids * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (positions == NULL || logits_host == NULL) { st = GD_ERR_OUT_OF_MEMORY; goto done; }
    for (i = 0; i < n_ids; ++i) { positions[i] = (int32_t)i; }
    cu[0] = 0; cu[1] = (int32_t)n_ids; cache_pos[0] = 0;
    st = gpt_lm_prepare_generation_cache(ctx, model, config, 1, GPT_CONTEXT_LENGTH, &cache);
    if (st != GD_OK) { goto done; }
    {
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        st = gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty());
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, ids, n_ids, &ids_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, positions, n_ids, &pos_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, cu, 2, &cu_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, cache_pos, 1, &cache_pos_t);
        if (st != GD_OK) { goto done; }
        st = gpt_lm_prefill_logits(ctx, model, cache, &ids_t, &pos_t, &cu_t, &cache_pos_t, &logits);
        if (st != GD_OK) { goto done; }
        st = gd_end_step(ctx);
        if (st != GD_OK) { goto done; }
        st = gd_tensor_read_f32(ctx, &logits, logits_host, (size_t)n_ids * (size_t)GPT_VOCAB_SIZE);
        if (st != GD_OK) { goto done; }
    }
    for (i = content_start; i < n_ids; ++i) {
        float *row = logits_host + (size_t)(i - 1) * (size_t)GPT_VOCAB_SIZE;
        int rank;
        double nll;
        gpt_apply_logits_softcap(row, GPT_VOCAB_SIZE, model->logits_softcap);
        nll = target_nll_rank(row, GPT_VOCAB_SIZE, ids[i], &rank);
        if (!isfinite(nll)) { continue; }
        if (cnt_all == 0) { first_nll = nll; first_rank = rank; }
        sum_all += nll; cnt_all += 1;
        if (cnt4 < 4) { sum4 += nll; cnt4 += 1; }
        if (cnt8 < 8) { sum8 += nll; cnt8 += 1; }
    }
    if (cnt_all == 0) { st = GD_ERR_INVALID_ARGUMENT; goto done; }
    out->term = xstrndup(term, strlen(term));
    out->definition = xstrndup(definition, strlen(definition) > 220U ? 220U : strlen(definition));
    out->first_piece = token_piece(tok, ids[content_start]);
    out->tokens = cnt_all;
    out->first_nll = first_nll;
    out->first_rank = first_rank;
    out->first4_nll = cnt4 > 0 ? sum4 / (double)cnt4 : NAN;
    out->first8_nll = cnt8 > 0 ? sum8 / (double)cnt8 : NAN;
    out->all_nll = sum_all / (double)cnt_all;
    if (out->term == NULL || out->definition == NULL || out->first_piece == NULL) { st = GD_ERR_OUT_OF_MEMORY; }

done:
    free(logits_host);
    free(positions);
    gd_tokenizer_free(ids);
    gd_tokenizer_free(header_ids);
    free(full);
    free(header);
    return st;
}

static void free_result(entry_result *r)
{
    if (r == NULL) { return; }
    free(r->term);
    free(r->first_piece);
    free(r->definition);
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [--checkpoint PATH] [--data-dir PATH] [--split train|val] [--limit N] [--top-worst N]\n", argv0);
}

int main(int argc, char **argv)
{
    gpt_config config = default_config();
    const char *split = DEFAULT_SPLIT;
    const char *raw_path_arg = NULL;
    const char *tokenizer_arg = NULL;
    int limit = 0;
    int top_worst = 12;
    bool layers_set = false;
    bool softcap_set = false;
    int argi;
    char *metadata = NULL;
    size_t metadata_len = 0U;
    gd_memory_config mem;
    gd_context *ctx = NULL;
    gd_status st;
    gpt_lm model;
    gd_module_load_options load_options;
    gpt_generation_tokenizer generation_tokenizer;
    char tokenizer_storage[1024];
    char *raw_path = NULL;
    char *text = NULL;
    char *line;
    char *current_term = NULL;
    int entries_seen = 0;
    int entries_eval = 0;
    int entries_skipped = 0;
    result_list results;
    double sum_first = 0.0;
    double sum_first4 = 0.0;
    double sum_first8 = 0.0;
    double sum_all_entry = 0.0;
    double sum_all_token_weighted = 0.0;
    uint64_t total_content_tokens = 0U;

    memset(&model, 0, sizeof(model));
    memset(&generation_tokenizer, 0, sizeof(generation_tokenizer));
    memset(&results, 0, sizeof(results));
    tokenizer_storage[0] = '\0';

    for (argi = 1; argi < argc; ++argi) {
        const char *value;
        int64_t parsed_i64;
        float parsed_f32;
        if (strcmp(argv[argi], "--help") == 0 || strcmp(argv[argi], "-h") == 0) { print_usage(argv[0]); return 0; }
        value = arg_value(argc, argv, &argi, "--checkpoint");
        if (value != NULL) { config.checkpoint_path = value; continue; }
        value = arg_value(argc, argv, &argi, "--data-dir");
        if (value != NULL) { config.data_dir = value; continue; }
        value = arg_value(argc, argv, &argi, "--tokenizer-path");
        if (value != NULL) { tokenizer_arg = value; config.tokenizer_path = value; continue; }
        value = arg_value(argc, argv, &argi, "--split");
        if (value != NULL) { split = value; continue; }
        value = arg_value(argc, argv, &argi, "--raw-path");
        if (value != NULL) { raw_path_arg = value; continue; }
        value = arg_value(argc, argv, &argi, "--limit");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000000000, &parsed_i64)) { fprintf(stderr, "invalid --limit\n"); return 2; }
            limit = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &argi, "--top-worst");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 1000, &parsed_i64)) { fprintf(stderr, "invalid --top-worst\n"); return 2; }
            top_worst = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &argi, "--layers");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 96, &parsed_i64)) { fprintf(stderr, "invalid --layers\n"); return 2; }
            config.n_layers = (int)parsed_i64; layers_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) { fprintf(stderr, "invalid --logits-softcap\n"); return 2; }
            config.logits_softcap = parsed_f32; softcap_set = true; continue;
        }
        fprintf(stderr, "unknown argument: %s\n", argv[argi]);
        return 2;
    }

    st = gd_checkpoint_read_metadata(config.checkpoint_path, &metadata, &metadata_len);
    if (st == GD_OK) {
        char value[128];
        int64_t parsed_i64;
        float parsed_f32;
        if (!layers_set && metadata_value(metadata, metadata_len, "n_layers", value, sizeof(value)) && parse_i64_arg(value, 1, 96, &parsed_i64)) {
            config.n_layers = (int)parsed_i64;
        }
        if (!softcap_set && metadata_value(metadata, metadata_len, "logits_softcap", value, sizeof(value)) && parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) {
            config.logits_softcap = parsed_f32;
        }
        if (tokenizer_arg == NULL && metadata_value(metadata, metadata_len, "tokenizer_path", tokenizer_storage, sizeof(tokenizer_storage))) {
            config.tokenizer_path = tokenizer_storage;
        }
    }
    if (raw_path_arg != NULL) {
        raw_path = xstrndup(raw_path_arg, strlen(raw_path_arg));
    } else {
        char fname[64];
        snprintf(fname, sizeof(fname), "%s.txt", split);
        raw_path = path_join3(config.data_dir, "raw_ita_dict", fname);
    }
    if (raw_path == NULL) { fprintf(stderr, "out of memory for raw path\n"); return 2; }

    mem = gpt_memory_config(&config);
    st = gd_context_create(&mem, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("gpt_lm_content_nll: skipped (no supported gradients.c backend)\n");
        free(raw_path); free(metadata); return 0;
    }
    if (st != GD_OK) { gpt_fail_status(ctx, st, "gd_context_create", __LINE__); }
    gpt_lm_init(ctx, &model, &config);
    load_options.strict = true;
    load_options.load_buffers = true;
    TRY(ctx, gd_module_load_state(ctx, &model.mod, config.checkpoint_path, &load_options));
    TRY(ctx, gd_context_seal_params(ctx));
    gd_module_set_training(&model.mod, false);
    gpt_generation_tokenizer_init(ctx, &config, &generation_tokenizer);

    printf("content_nll: checkpoint=%s data_dir=%s raw=%s split=%s layers=%d logits_softcap=%.3f limit=%d tokenizer=%s\n",
           config.checkpoint_path,
           config.data_dir,
           raw_path,
           split,
           config.n_layers,
           (double)config.logits_softcap,
           limit,
           generation_tokenizer.path);
    text = read_file_text(raw_path);
    line = text;
    while (line != NULL && *line != '\0') {
        char *next = strchr(line, '\n');
        size_t len = next != NULL ? (size_t)(next - line) : strlen(line);
        if (len > 0U && line[len - 1U] == '\r') { len -= 1U; }
        if (len >= 9U && memcmp(line, "Termine: ", 9U) == 0) {
            free(current_term);
            current_term = xstrndup(line + 9U, len - 9U);
        } else if (current_term != NULL && len >= 3U && line[0] == '1' && line[1] == '.' && line[2] == ' ') {
            char *definition = xstrndup(line + 3U, len - 3U);
            entry_result r;
            entries_seen += 1;
            if (definition == NULL) { fprintf(stderr, "out of memory definition\n"); exit(2); }
            st = eval_entry(ctx, &model, &config, generation_tokenizer.tok, current_term, definition, &r);
            if (st == GD_OK) {
                result_list_append(&results, r);
                entries_eval += 1;
                sum_first += r.first_nll;
                sum_first4 += r.first4_nll;
                sum_first8 += r.first8_nll;
                sum_all_entry += r.all_nll;
                sum_all_token_weighted += r.all_nll * (double)r.tokens;
                total_content_tokens += (uint64_t)r.tokens;
            } else {
                entries_skipped += 1;
            }
            free(definition);
            free(current_term);
            current_term = NULL;
            if (limit > 0 && entries_eval >= limit) { break; }
        }
        if (next == NULL) { break; }
        line = next + 1;
    }
    free(current_term);

    if (entries_eval > 0) {
        double *first_vals = (double *)malloc(results.count * sizeof(first_vals[0]));
        double *all_vals = (double *)malloc(results.count * sizeof(all_vals[0]));
        size_t i;
        for (i = 0U; i < results.count; ++i) { first_vals[i] = results.items[i].first_nll; all_vals[i] = results.items[i].all_nll; }
        qsort(first_vals, results.count, sizeof(first_vals[0]), cmp_all_asc_double);
        qsort(all_vals, results.count, sizeof(all_vals[0]), cmp_all_asc_double);
        printf("summary: entries_seen=%d entries_eval=%d skipped=%d content_tokens=%llu\n",
               entries_seen,
               entries_eval,
               entries_skipped,
               (unsigned long long)total_content_tokens);
        printf("  first_content_token: mean=%.6f bits=%.4f median=%.6f p90=%.6f p99=%.6f\n",
               sum_first / (double)entries_eval,
               (sum_first / (double)entries_eval) / log(2.0),
               first_vals[results.count / 2U],
               first_vals[(size_t)((double)(results.count - 1U) * 0.90)],
               first_vals[(size_t)((double)(results.count - 1U) * 0.99)]);
        printf("  first4_content_tokens: mean=%.6f bits=%.4f\n", sum_first4 / (double)entries_eval, (sum_first4 / (double)entries_eval) / log(2.0));
        printf("  first8_content_tokens: mean=%.6f bits=%.4f\n", sum_first8 / (double)entries_eval, (sum_first8 / (double)entries_eval) / log(2.0));
        printf("  all_def1_content_tokens entry_mean=%.6f token_weighted=%.6f bits=%.4f median=%.6f p90=%.6f p99=%.6f\n",
               sum_all_entry / (double)entries_eval,
               sum_all_token_weighted / (double)total_content_tokens,
               (sum_all_token_weighted / (double)total_content_tokens) / log(2.0),
               all_vals[results.count / 2U],
               all_vals[(size_t)((double)(results.count - 1U) * 0.90)],
               all_vals[(size_t)((double)(results.count - 1U) * 0.99)]);
        free(first_vals);
        free(all_vals);
        qsort(results.items, results.count, sizeof(results.items[0]), cmp_first_desc);
        if (top_worst > 0) {
            printf("worst_first_content_tokens:\n");
            for (i = 0U; i < results.count && i < (size_t)top_worst; ++i) {
                const entry_result *r = &results.items[i];
                printf("  nll=%7.4f bits=%6.3f rank=%4d first=%-14s first8=%.4f all=%.4f tokens=%d term=%s def=%s\n",
                       r->first_nll,
                       r->first_nll / log(2.0),
                       r->first_rank,
                       r->first_piece != NULL ? r->first_piece : "?",
                       r->first8_nll,
                       r->all_nll,
                       r->tokens,
                       r->term != NULL ? r->term : "?",
                       r->definition != NULL ? r->definition : "?");
            }
        }
    } else {
        printf("summary: no entries evaluated\n");
    }

    for (size_t i = 0U; i < results.count; ++i) { free_result(&results.items[i]); }
    free(results.items);
    free(text);
    free(raw_path);
    gpt_generation_tokenizer_deinit(&generation_tokenizer);
    gpt_lm_deinit(&model);
    gd_context_destroy(ctx);
    free(metadata);
    return 0;
}
