/* GPT-LM document NLL probe for examples/gpt_lm ita_dict_v2.
 *
 * Reads raw_ita_dict/{train,val}.jsonl, extracts the JSON `text` document,
 * wraps it exactly as the dataset tokenizer does:
 *
 *   <|im_start|>{text}<|im_end|>
 *
 * and measures teacher-forced next-token NLL.  By default it scores every token
 * after <|im_start|>, including the final <|im_end|>.  With --prefix TEXT it
 * scores only the continuation after that prefix.  Prefix TEXT may include or
 * omit <|im_start|>; if omitted the probe adds it for matching/scoring.
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
#define PROBE_IM_START "<|im_start|>"
#define PROBE_IM_END "<|im_end|>"

typedef struct doc_result {
    int doc_index;
    char *kind;
    char *label;
    char *first_piece;
    int tokens;
    double first_nll;
    double first4_nll;
    double first8_nll;
    double all_nll;
    int first_rank;
} doc_result;

typedef struct result_list {
    doc_result *items;
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

static int parse_architecture_arg(const char *text, gpt_architecture *out)
{
    if (text == NULL || out == NULL) { return 0; }
    if (strcmp(text, "gpt") == 0 || strcmp(text, "dense") == 0) {
        *out = GPT_ARCH_GPT;
        return 1;
    }
    if (strcmp(text, "minimax_m3") == 0 || strcmp(text, "minimax") == 0 || strcmp(text, "m3") == 0) {
        *out = GPT_ARCH_MINIMAX_M3;
        return 1;
    }
    return 0;
}

static const char *arch_name(gpt_architecture architecture)
{
    return architecture == GPT_ARCH_MINIMAX_M3 ? "minimax_m3" : "gpt";
}

static bool file_exists(const char *path)
{
    FILE *f;
    if (path == NULL || path[0] == '\0') { return false; }
    f = fopen(path, "rb");
    if (f == NULL) { return false; }
    (void)fclose(f);
    return true;
}

static bool starts_with(const char *text, const char *prefix)
{
    size_t n;
    if (text == NULL || prefix == NULL) { return false; }
    n = strlen(prefix);
    return strncmp(text, prefix, n) == 0;
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
    c.architecture = GPT_ARCH_GPT;
    c.minimax_m3_topk_blocks = GPT_MINIMAX_M3_TOPK_BLOCKS;
    c.minimax_m3_init_blocks = GPT_MINIMAX_M3_INIT_BLOCKS;
    c.minimax_m3_local_blocks = GPT_MINIMAX_M3_LOCAL_BLOCKS;
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
    char *out;
    if (s == NULL) { return NULL; }
    out = (char *)malloc(n + 1U);
    if (out == NULL) { return NULL; }
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *xstrdup2(const char *s)
{
    return xstrndup(s, s != NULL ? strlen(s) : 0U);
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

static char *wrap_document(const char *text)
{
    const size_t a = strlen(PROBE_IM_START);
    const size_t b = text != NULL ? strlen(text) : 0U;
    const size_t c = strlen(PROBE_IM_END);
    char *out = (char *)malloc(a + b + c + 1U);
    if (out == NULL) { return NULL; }
    memcpy(out, PROBE_IM_START, a);
    if (b > 0U) { memcpy(out + a, text, b); }
    memcpy(out + a + b, PROBE_IM_END, c);
    out[a + b + c] = '\0';
    return out;
}

static char *make_scoring_prefix(const char *prefix)
{
    const size_t start_len = strlen(PROBE_IM_START);
    size_t prefix_len;
    char *out;
    if (prefix == NULL) { return xstrdup2(PROBE_IM_START); }
    if (starts_with(prefix, PROBE_IM_START)) { return xstrdup2(prefix); }
    prefix_len = strlen(prefix);
    out = (char *)malloc(start_len + prefix_len + 1U);
    if (out == NULL) { return NULL; }
    memcpy(out, PROBE_IM_START, start_len);
    memcpy(out + start_len, prefix, prefix_len);
    out[start_len + prefix_len] = '\0';
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

static void result_list_append(result_list *list, doc_result item)
{
    doc_result *grown;
    if (list->count == list->cap) {
        size_t new_cap = list->cap == 0U ? 128U : list->cap * 2U;
        grown = (doc_result *)realloc(list->items, new_cap * sizeof(list->items[0]));
        if (grown == NULL) { fprintf(stderr, "out of memory growing results\n"); exit(2); }
        list->items = grown;
        list->cap = new_cap;
    }
    list->items[list->count++] = item;
}

static int cmp_first_desc(const void *a, const void *b)
{
    const doc_result *x = (const doc_result *)a;
    const doc_result *y = (const doc_result *)b;
    return x->first_nll < y->first_nll ? 1 : (x->first_nll > y->first_nll ? -1 : 0);
}

static int cmp_all_desc(const void *a, const void *b)
{
    const doc_result *x = (const doc_result *)a;
    const doc_result *y = (const doc_result *)b;
    return x->all_nll < y->all_nll ? 1 : (x->all_nll > y->all_nll ? -1 : 0);
}

static int cmp_asc_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return -1;
}

static bool parse_hex4(const char *p, uint32_t *out)
{
    int a, b, c, d;
    if (p == NULL || out == NULL) { return false; }
    a = hex_value(p[0]); b = hex_value(p[1]); c = hex_value(p[2]); d = hex_value(p[3]);
    if (a < 0 || b < 0 || c < 0 || d < 0) { return false; }
    *out = ((uint32_t)a << 12) | ((uint32_t)b << 8) | ((uint32_t)c << 4) | (uint32_t)d;
    return true;
}

static size_t append_utf8(char *out, uint32_t cp)
{
    if (cp == 0U) { out[0] = '?'; return 1U; }
    if (cp <= 0x7FU) { out[0] = (char)cp; return 1U; }
    if (cp <= 0x7FFU) {
        out[0] = (char)(0xC0U | (cp >> 6));
        out[1] = (char)(0x80U | (cp & 0x3FU));
        return 2U;
    }
    if (cp <= 0xFFFFU) {
        out[0] = (char)(0xE0U | (cp >> 12));
        out[1] = (char)(0x80U | ((cp >> 6) & 0x3FU));
        out[2] = (char)(0x80U | (cp & 0x3FU));
        return 3U;
    }
    out[0] = (char)(0xF0U | (cp >> 18));
    out[1] = (char)(0x80U | ((cp >> 12) & 0x3FU));
    out[2] = (char)(0x80U | ((cp >> 6) & 0x3FU));
    out[3] = (char)(0x80U | (cp & 0x3FU));
    return 4U;
}

static char *json_extract_text_field(const char *line)
{
    const char *p = line;
    char *out;
    char *w;
    if (line == NULL) { return NULL; }
    p = strstr(p, "\"text\"");
    if (p == NULL) { return NULL; }
    p += 6;
    while (*p == ' ' || *p == '\t') { p += 1; }
    if (*p != ':') { return NULL; }
    p += 1;
    while (*p == ' ' || *p == '\t') { p += 1; }
    if (*p != '"') { return NULL; }
    p += 1;
    out = (char *)malloc(strlen(p) + 1U);
    if (out == NULL) { return NULL; }
    w = out;
    while (*p != '\0') {
        unsigned char ch = (unsigned char)*p;
        if (ch == '"') {
            *w = '\0';
            return out;
        }
        if (ch != '\\') {
            *w++ = (char)ch;
            p += 1;
            continue;
        }
        p += 1;
        if (*p == '\0') { break; }
        switch (*p) {
            case '"': *w++ = '"'; p += 1; break;
            case '\\': *w++ = '\\'; p += 1; break;
            case '/': *w++ = '/'; p += 1; break;
            case 'b': *w++ = '\b'; p += 1; break;
            case 'f': *w++ = '\f'; p += 1; break;
            case 'n': *w++ = '\n'; p += 1; break;
            case 'r': *w++ = '\r'; p += 1; break;
            case 't': *w++ = '\t'; p += 1; break;
            case 'u': {
                uint32_t cp;
                if (!parse_hex4(p + 1, &cp)) { free(out); return NULL; }
                p += 5;
                if (cp >= 0xD800U && cp <= 0xDBFFU && p[0] == '\\' && p[1] == 'u') {
                    uint32_t lo;
                    if (parse_hex4(p + 2, &lo) && lo >= 0xDC00U && lo <= 0xDFFFU) {
                        cp = 0x10000U + ((cp - 0xD800U) << 10) + (lo - 0xDC00U);
                        p += 6;
                    }
                }
                w += append_utf8(w, cp);
                break;
            }
            default:
                *w++ = *p++;
                break;
        }
    }
    free(out);
    return NULL;
}

static char *first_line_label(const char *text)
{
    const char *end;
    if (text == NULL) { return xstrdup2("?"); }
    end = strchr(text, '\n');
    if (end == NULL) { end = text + strlen(text); }
    return xstrndup(text, (size_t)(end - text));
}

static const char *document_kind(const char *text)
{
    if (text == NULL) { return "unknown"; }
    if (strstr(text, "\n\n## Quiz\n") != NULL || strstr(text, "\n## Quiz\n") != NULL) { return "quiz"; }
    if (strstr(text, "\n\n## Story\n") != NULL || strstr(text, "\n## Story\n") != NULL) { return "story"; }
    if (strstr(text, "\n\n## Definizioni") != NULL || strstr(text, "\n## Definizioni") != NULL) { return "term"; }
    return "unknown";
}

static gd_status eval_document(gd_context *ctx,
                               gpt_lm *model,
                               const gpt_config *config,
                               gd_tokenizer *tok,
                               int doc_index,
                               const char *kind,
                               const char *text,
                               const char *prefix_arg,
                               int trace_tokens,
                               doc_result *out)
{
    char *full = NULL;
    char *prefix = NULL;
    int32_t *prefix_ids = NULL;
    int32_t *ids = NULL;
    int n_prefix = 0;
    int n_ids = 0;
    int common = 0;
    int content_start;
    int first_target = -1;
    int32_t positions[GPT_CONTEXT_LENGTH];
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
    int start;
    int i;
    gd_status st = GD_OK;

    memset(out, 0, sizeof(*out));
    full = wrap_document(text);
    prefix = make_scoring_prefix(prefix_arg);
    if (full == NULL || prefix == NULL) { st = GD_ERR_OUT_OF_MEMORY; goto done; }
    if (!starts_with(full, prefix)) { st = GD_ERR_INVALID_ARGUMENT; goto done; }
    st = gd_tokenizer_encode(tok, prefix, &prefix_ids, &n_prefix);
    if (st != GD_OK) { goto done; }
    st = gd_tokenizer_encode(tok, full, &ids, &n_ids);
    if (st != GD_OK) { goto done; }
    while (common < n_prefix && common < n_ids && prefix_ids[common] == ids[common]) { common += 1; }
    content_start = common;
    if (content_start <= 0 || content_start >= n_ids) { st = GD_ERR_INVALID_ARGUMENT; goto done; }
    logits_host = (float *)malloc((size_t)GPT_CONTEXT_LENGTH * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (logits_host == NULL) { st = GD_ERR_OUT_OF_MEMORY; goto done; }
    for (i = 0; i < GPT_CONTEXT_LENGTH; ++i) { positions[i] = (int32_t)i; }

    for (start = 0; start < n_ids - 1; start += GPT_CONTEXT_LENGTH) {
        const int remaining = n_ids - 1 - start;
        const int input_len = remaining > GPT_CONTEXT_LENGTH ? GPT_CONTEXT_LENGTH : remaining;
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        if (input_len <= 0) { break; }
        cu[0] = 0;
        cu[1] = (int32_t)input_len;
        cache_pos[0] = 0;
        st = gpt_lm_prepare_generation_cache(ctx, model, config, 1, GPT_CONTEXT_LENGTH, &cache);
        if (st != GD_OK) { goto done; }
        st = gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty());
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, ids + start, input_len, &ids_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, positions, input_len, &pos_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, cu, 2, &cu_t);
        if (st != GD_OK) { goto done; }
        st = gpt_data_i32_tensor(ctx, cache_pos, 1, &cache_pos_t);
        if (st != GD_OK) { goto done; }
        st = gpt_lm_prefill_logits(ctx, model, cache, &ids_t, &pos_t, &cu_t, &cache_pos_t, &logits);
        if (st != GD_OK) { goto done; }
        st = gd_end_step(ctx);
        if (st != GD_OK) { goto done; }
        st = gd_tensor_read_f32(ctx, &logits, logits_host, (size_t)input_len * (size_t)GPT_VOCAB_SIZE);
        if (st != GD_OK) { goto done; }
        for (i = 0; i < input_len; ++i) {
            const int target_index = start + i + 1;
            float *row;
            int rank;
            double nll;
            if (target_index < content_start) { continue; }
            row = logits_host + (size_t)i * (size_t)GPT_VOCAB_SIZE;
            gpt_apply_logits_softcap(row, GPT_VOCAB_SIZE, model->logits_softcap);
            nll = target_nll_rank(row, GPT_VOCAB_SIZE, ids[target_index], &rank);
            if (!isfinite(nll)) { continue; }
            if (cnt_all < trace_tokens) {
                char *piece = token_piece(tok, ids[target_index]);
                printf("  trace[%d] token_index=%d nll=%.6f bits=%.4f rank=%d piece=%s\n",
                       cnt_all,
                       target_index,
                       nll,
                       nll / log(2.0),
                       rank,
                       piece != NULL ? piece : "?");
                free(piece);
            }
            if (cnt_all == 0) { first_nll = nll; first_rank = rank; first_target = target_index; }
            sum_all += nll;
            cnt_all += 1;
            if (cnt4 < 4) { sum4 += nll; cnt4 += 1; }
            if (cnt8 < 8) { sum8 += nll; cnt8 += 1; }
        }
    }
    if (cnt_all == 0 || first_target < 0) { st = GD_ERR_INVALID_ARGUMENT; goto done; }
    out->doc_index = doc_index;
    out->kind = xstrdup2(kind != NULL ? kind : "unknown");
    out->label = first_line_label(text);
    out->first_piece = token_piece(tok, ids[first_target]);
    out->tokens = cnt_all;
    out->first_nll = first_nll;
    out->first_rank = first_rank;
    out->first4_nll = cnt4 > 0 ? sum4 / (double)cnt4 : NAN;
    out->first8_nll = cnt8 > 0 ? sum8 / (double)cnt8 : NAN;
    out->all_nll = sum_all / (double)cnt_all;
    if (out->kind == NULL || out->label == NULL || out->first_piece == NULL) { st = GD_ERR_OUT_OF_MEMORY; }

done:
    free(logits_host);
    gd_tokenizer_free(ids);
    gd_tokenizer_free(prefix_ids);
    free(prefix);
    free(full);
    return st;
}

static void free_result(doc_result *r)
{
    if (r == NULL) { return; }
    free(r->kind);
    free(r->label);
    free(r->first_piece);
}

static void print_usage(const char *argv0)
{
    printf("usage: %s [--checkpoint PATH] [--data-dir PATH] [--split train|val] [options]\n", argv0);
    printf("\n");
    printf("Options:\n");
    printf("  --raw-path PATH       JSONL path; default data_dir/raw_ita_dict/<split>.jsonl\n");
    printf("  --tokenizer-path PATH tokenizer JSON; by default data_dir/tokenizer-v2048.json\n");
    printf("  --kind all|term|story|quiz|unknown  filter documents (default: all)\n");
    printf("  --prefix TEXT         score only continuation after TEXT; <|im_start|> is optional\n");
    printf("  --limit N             stop after N evaluated docs (default: 0 = all)\n");
    printf("  --top-worst N         print N worst docs by first/all NLL (default: 12)\n");
    printf("  --trace-tokens N      print first N scored gold tokens with rank/NLL (use --limit 1)\n");
    printf("  --layers N            override checkpoint metadata layer count\n");
    printf("  --architecture NAME   override checkpoint metadata architecture\n");
    printf("  --logits-softcap C    override checkpoint metadata logits softcap\n");
}

int main(int argc, char **argv)
{
    gpt_config config = default_config();
    const char *split = DEFAULT_SPLIT;
    const char *raw_path_arg = NULL;
    const char *tokenizer_arg = NULL;
    const char *kind_filter = "all";
    const char *prefix_arg = NULL;
    int limit = 0;
    int top_worst = 12;
    int trace_tokens = 0;
    bool layers_set = false;
    bool softcap_set = false;
    bool architecture_set = false;
    bool minimax_topk_set = false;
    bool minimax_init_set = false;
    bool minimax_local_set = false;
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
    int docs_seen = 0;
    int docs_kind_matched = 0;
    int docs_eval = 0;
    int docs_skipped = 0;
    int json_errors = 0;
    int kind_counts_term = 0;
    int kind_counts_story = 0;
    int kind_counts_quiz = 0;
    int kind_counts_unknown = 0;
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
        value = arg_value(argc, argv, &argi, "--kind");
        if (value != NULL) { kind_filter = value; continue; }
        value = arg_value(argc, argv, &argi, "--prefix");
        if (value != NULL) { prefix_arg = value; continue; }
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
        value = arg_value(argc, argv, &argi, "--trace-tokens");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 10000, &parsed_i64)) { fprintf(stderr, "invalid --trace-tokens\n"); return 2; }
            trace_tokens = (int)parsed_i64;
            continue;
        }
        value = arg_value(argc, argv, &argi, "--layers");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 96, &parsed_i64)) { fprintf(stderr, "invalid --layers\n"); return 2; }
            config.n_layers = (int)parsed_i64; layers_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--architecture");
        if (value != NULL) {
            if (!parse_architecture_arg(value, &config.architecture)) { fprintf(stderr, "invalid --architecture\n"); return 2; }
            architecture_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--minimax-topk-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 1, 16, &parsed_i64)) { fprintf(stderr, "invalid --minimax-topk-blocks\n"); return 2; }
            config.minimax_m3_topk_blocks = (int)parsed_i64; minimax_topk_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--minimax-init-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 16, &parsed_i64)) { fprintf(stderr, "invalid --minimax-init-blocks\n"); return 2; }
            config.minimax_m3_init_blocks = (int)parsed_i64; minimax_init_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--minimax-local-blocks");
        if (value != NULL) {
            if (!parse_i64_arg(value, 0, 16, &parsed_i64)) { fprintf(stderr, "invalid --minimax-local-blocks\n"); return 2; }
            config.minimax_m3_local_blocks = (int)parsed_i64; minimax_local_set = true; continue;
        }
        value = arg_value(argc, argv, &argi, "--logits-softcap");
        if (value != NULL) {
            if (!parse_float_arg(value, 0.0f, 1000000.0f, &parsed_f32)) { fprintf(stderr, "invalid --logits-softcap\n"); return 2; }
            config.logits_softcap = parsed_f32; softcap_set = true; continue;
        }
        fprintf(stderr, "unknown argument: %s\n", argv[argi]);
        return 2;
    }
    if (!(strcmp(kind_filter, "all") == 0 || strcmp(kind_filter, "term") == 0 || strcmp(kind_filter, "story") == 0 || strcmp(kind_filter, "quiz") == 0 || strcmp(kind_filter, "unknown") == 0)) {
        fprintf(stderr, "invalid --kind %s\n", kind_filter);
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
        if (!architecture_set && metadata_value(metadata, metadata_len, "architecture", value, sizeof(value))) {
            (void)parse_architecture_arg(value, &config.architecture);
        }
        if (!minimax_topk_set && metadata_value(metadata, metadata_len, "minimax_m3_topk_blocks", value, sizeof(value)) && parse_i64_arg(value, 1, 16, &parsed_i64)) {
            config.minimax_m3_topk_blocks = (int)parsed_i64;
        }
        if (!minimax_init_set && metadata_value(metadata, metadata_len, "minimax_m3_init_blocks", value, sizeof(value)) && parse_i64_arg(value, 0, 16, &parsed_i64)) {
            config.minimax_m3_init_blocks = (int)parsed_i64;
        }
        if (!minimax_local_set && metadata_value(metadata, metadata_len, "minimax_m3_local_blocks", value, sizeof(value)) && parse_i64_arg(value, 0, 16, &parsed_i64)) {
            config.minimax_m3_local_blocks = (int)parsed_i64;
        }
        if (tokenizer_arg == NULL && metadata_value(metadata, metadata_len, "tokenizer_path", tokenizer_storage, sizeof(tokenizer_storage)) && file_exists(tokenizer_storage)) {
            config.tokenizer_path = tokenizer_storage;
        }
    }
    if (raw_path_arg != NULL) {
        raw_path = xstrndup(raw_path_arg, strlen(raw_path_arg));
    } else {
        char fname[64];
        snprintf(fname, sizeof(fname), "%s.jsonl", split);
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

    printf("content_nll: checkpoint=%s data_dir=%s raw=%s split=%s kind=%s prefix=%s layers=%d arch=%s logits_softcap=%.3f limit=%d tokenizer=%s\n",
           config.checkpoint_path,
           config.data_dir,
           raw_path,
           split,
           kind_filter,
           prefix_arg != NULL ? prefix_arg : "<none>",
           config.n_layers,
           arch_name(config.architecture),
           (double)config.logits_softcap,
           limit,
           generation_tokenizer.path);

    text = read_file_text(raw_path);
    line = text;
    while (line != NULL && *line != '\0') {
        char *next = strchr(line, '\n');
        size_t len = next != NULL ? (size_t)(next - line) : strlen(line);
        char *json_line;
        char *doc_text;
        const char *kind;
        if (len > 0U && line[len - 1U] == '\r') { len -= 1U; }
        if (len == 0U) {
            if (next == NULL) { break; }
            line = next + 1;
            continue;
        }
        docs_seen += 1;
        json_line = xstrndup(line, len);
        if (json_line == NULL) { fprintf(stderr, "out of memory json line\n"); exit(2); }
        doc_text = json_extract_text_field(json_line);
        free(json_line);
        if (doc_text == NULL) {
            json_errors += 1;
            docs_skipped += 1;
            if (next == NULL) { break; }
            line = next + 1;
            continue;
        }
        kind = document_kind(doc_text);
        if (strcmp(kind, "term") == 0) { kind_counts_term += 1; }
        else if (strcmp(kind, "story") == 0) { kind_counts_story += 1; }
        else if (strcmp(kind, "quiz") == 0) { kind_counts_quiz += 1; }
        else { kind_counts_unknown += 1; }
        if (strcmp(kind_filter, "all") == 0 || strcmp(kind_filter, kind) == 0) {
            doc_result r;
            docs_kind_matched += 1;
            st = eval_document(ctx, &model, &config, generation_tokenizer.tok, docs_seen - 1, kind, doc_text, prefix_arg, docs_eval == 0 ? trace_tokens : 0, &r);
            if (st == GD_OK) {
                result_list_append(&results, r);
                docs_eval += 1;
                sum_first += r.first_nll;
                sum_first4 += r.first4_nll;
                sum_first8 += r.first8_nll;
                sum_all_entry += r.all_nll;
                sum_all_token_weighted += r.all_nll * (double)r.tokens;
                total_content_tokens += (uint64_t)r.tokens;
            } else {
                docs_skipped += 1;
            }
            if (limit > 0 && docs_eval >= limit) {
                free(doc_text);
                break;
            }
        }
        free(doc_text);
        if (next == NULL) { break; }
        line = next + 1;
    }

    printf("docs: seen=%d matched_kind=%d eval=%d skipped=%d json_errors=%d kinds={term:%d story:%d quiz:%d unknown:%d}\n",
           docs_seen,
           docs_kind_matched,
           docs_eval,
           docs_skipped,
           json_errors,
           kind_counts_term,
           kind_counts_story,
           kind_counts_quiz,
           kind_counts_unknown);

    if (docs_eval > 0) {
        double *first_vals = (double *)malloc(results.count * sizeof(first_vals[0]));
        double *all_vals = (double *)malloc(results.count * sizeof(all_vals[0]));
        size_t i;
        if (first_vals == NULL || all_vals == NULL) { fprintf(stderr, "out of memory summary\n"); exit(2); }
        for (i = 0U; i < results.count; ++i) { first_vals[i] = results.items[i].first_nll; all_vals[i] = results.items[i].all_nll; }
        qsort(first_vals, results.count, sizeof(first_vals[0]), cmp_asc_double);
        qsort(all_vals, results.count, sizeof(all_vals[0]), cmp_asc_double);
        printf("summary: scored_tokens=%llu\n", (unsigned long long)total_content_tokens);
        printf("  first_scored_token: mean=%.6f bits=%.4f median=%.6f p90=%.6f p99=%.6f\n",
               sum_first / (double)docs_eval,
               (sum_first / (double)docs_eval) / log(2.0),
               first_vals[results.count / 2U],
               first_vals[(size_t)((double)(results.count - 1U) * 0.90)],
               first_vals[(size_t)((double)(results.count - 1U) * 0.99)]);
        printf("  first4_scored_tokens: mean=%.6f bits=%.4f\n", sum_first4 / (double)docs_eval, (sum_first4 / (double)docs_eval) / log(2.0));
        printf("  first8_scored_tokens: mean=%.6f bits=%.4f\n", sum_first8 / (double)docs_eval, (sum_first8 / (double)docs_eval) / log(2.0));
        printf("  all_scored_tokens: entry_mean=%.6f token_weighted=%.6f bits=%.4f median=%.6f p90=%.6f p99=%.6f\n",
               sum_all_entry / (double)docs_eval,
               sum_all_token_weighted / (double)total_content_tokens,
               (sum_all_token_weighted / (double)total_content_tokens) / log(2.0),
               all_vals[results.count / 2U],
               all_vals[(size_t)((double)(results.count - 1U) * 0.90)],
               all_vals[(size_t)((double)(results.count - 1U) * 0.99)]);
        free(first_vals);
        free(all_vals);

        if (top_worst > 0) {
            printf("worst_first_scored_tokens:\n");
            qsort(results.items, results.count, sizeof(results.items[0]), cmp_first_desc);
            for (i = 0U; i < results.count && i < (size_t)top_worst; ++i) {
                const doc_result *r = &results.items[i];
                printf("  nll=%7.4f bits=%6.3f rank=%4d first=%-14s first8=%.4f all=%.4f tokens=%d doc=%d kind=%s label=%s\n",
                       r->first_nll,
                       r->first_nll / log(2.0),
                       r->first_rank,
                       r->first_piece != NULL ? r->first_piece : "?",
                       r->first8_nll,
                       r->all_nll,
                       r->tokens,
                       r->doc_index,
                       r->kind != NULL ? r->kind : "?",
                       r->label != NULL ? r->label : "?");
            }
            printf("worst_all_scored_tokens:\n");
            qsort(results.items, results.count, sizeof(results.items[0]), cmp_all_desc);
            for (i = 0U; i < results.count && i < (size_t)top_worst; ++i) {
                const doc_result *r = &results.items[i];
                printf("  all=%7.4f bits=%6.3f first=%.4f rank=%4d first_piece=%-14s tokens=%d doc=%d kind=%s label=%s\n",
                       r->all_nll,
                       r->all_nll / log(2.0),
                       r->first_nll,
                       r->first_rank,
                       r->first_piece != NULL ? r->first_piece : "?",
                       r->tokens,
                       r->doc_index,
                       r->kind != NULL ? r->kind : "?",
                       r->label != NULL ? r->label : "?");
            }
        }
    } else {
        printf("summary: no documents evaluated\n");
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
