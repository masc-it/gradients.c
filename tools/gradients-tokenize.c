#include <gradients/tokenizer.h>

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_TOK_SHARD_MAX_BYTES UINT64_C(2147483648)
#define GD_TOK_PATH_MAX 4096U
#define GD_TOK_JSON_MAX_DEPTH 128
#define GD_TOK_MAX_SPECIALS 64
#define GD_TOK_DEFAULT_VOCAB_SIZE 32768
#define GD_TOK_DEFAULT_MIN_FREQUENCY 2
#define GD_TOK_DEFAULT_SEED UINT64_C(17)
#define GD_TOK_STATUS_EOF ((gd_status)-1000)
#define GD_TOK_STATUS_NOT_FOUND ((gd_status)-1001)

#if defined(_WIN32)
#error "gradients-tokenize sharded output currently requires POSIX mkdir semantics"
#endif

typedef enum gd_tok_mode {
    GD_TOK_MODE_LEGACY_FLAT = 0,
    GD_TOK_MODE_PACKED_TEXT = 1,
    GD_TOK_MODE_JSONL = 2
} gd_tok_mode;

typedef struct gd_tok_args {
    const char *tokenizer_path;
    const char *use_tokenizer_path;
    const char *input_path;
    const char *input_jsonl_path;
    const char *output_path;
    const char *output_dir;
    const char *jsonl_text_field;
    const char *im_start;
    const char *im_end;
    uint64_t num_tokens_per_sequence;
    uint64_t max_length;
    const char *specials[GD_TOK_MAX_SPECIALS];
    int n_specials;
    int vocab_size;
    int min_frequency;
    uint64_t seed;
    int no_special;
} gd_tok_args;

typedef struct gd_tok_shard_info {
    char tokens_file[64];
    char offsets_file[64];
    uint64_t sequences;
    uint64_t tokens;
    uint64_t bytes;
} gd_tok_shard_info;

typedef struct gd_tok_writer {
    const char *output_dir;
    int variable_length;
    uint64_t max_shard_bytes;

    FILE *tokens_f;
    FILE *offsets_f;
    char cur_tokens_file[64];
    char cur_offsets_file[64];
    uint64_t next_shard_index;
    uint64_t cur_sequences;
    uint64_t cur_tokens;
    uint64_t cur_token_bytes;
    uint64_t cur_offset_bytes;

    gd_tok_shard_info *infos;
    size_t n_infos;
    size_t cap_infos;
    uint64_t total_sequences;
    uint64_t total_tokens;
} gd_tok_writer;

typedef struct gd_tok_manifest_cfg {
    const char *mode;
    const char *tokenizer_path;
    const char *jsonl_text_field;
    uint64_t tokenizer_hash;
    uint64_t sequence_length;
    uint64_t stride;
    uint64_t max_length;
    uint64_t shard_size_limit_bytes;
    int32_t im_start_id;
    int32_t im_end_id;
} gd_tok_manifest_cfg;

static void usage(FILE *f)
{
    fprintf(f,
            "usage:\n"
            "  legacy flat token stream:\n"
            "    gradients-tokenize --tokenizer tokenizer.json --input text.txt "
            "--output tokens.i32 [--vocab-size N] [--no-special]\n"
            "    gradients-tokenize --use-tokenizer tokenizer.json --input text.txt "
            "--output tokens.i32 [--no-special]\n"
            "\n"
            "  packed text file corpus:\n"
            "    gradients-tokenize --tokenizer tokenizer.json --input text.txt "
            "--output-dir out --im-start '<|im_start|>' --im-end '<|im_end|>' "
            "--num-tokens-per-sequence N [--max-length N] [--vocab-size N]\n"
            "    gradients-tokenize --use-tokenizer tokenizer.json --input text.txt "
            "--output-dir out --im-start '<|im_start|>' --im-end '<|im_end|>' "
            "--num-tokens-per-sequence N [--max-length N]\n"
            "\n"
            "  jsonl corpus, one sequence per line:\n"
            "    gradients-tokenize --tokenizer tokenizer.json --input-jsonl data.jsonl "
            "--jsonl-text-field text --output-dir out --im-start '<|im_start|>' "
            "--im-end '<|im_end|>' --max-length N [--vocab-size N]\n"
            "    gradients-tokenize --use-tokenizer tokenizer.json --input-jsonl data.jsonl "
            "--jsonl-text-field text --output-dir out --im-start '<|im_start|>' "
            "--im-end '<|im_end|>' --max-length N\n"
            "\n"
            "training options when --use-tokenizer is absent:\n"
            "  --vocab-size N       target BPE vocabulary (default: 32768)\n"
            "  --min-frequency N    minimum pair frequency (default: 2)\n"
            "  --special TOKEN      reserve extra special token; repeatable\n"
            "  --seed N             deterministic BPE tie-break seed (default: 17)\n"
            "\n"
            "digits are always split: 123 tokenizes as 1, 2, 3 before BPE merges.\n");
}

static gd_status gd_tok_parse_u64(const char *s, uint64_t min_value, uint64_t *out)
{
    unsigned long long v;
    char *end = NULL;
    if (s == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || (uint64_t)v < min_value) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = (uint64_t)v;
    return GD_OK;
}

static gd_status gd_tok_parse_int(const char *s, int min_value, int *out)
{
    long v;
    char *end = NULL;
    if (s == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < min_value || v > INT_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = (int)v;
    return GD_OK;
}

static gd_status gd_tok_join_path(char *out, size_t cap, const char *dir, const char *name)
{
    size_t dir_len;
    size_t name_len;
    int needs_slash;
    int n;
    if (out == NULL || cap == 0U || dir == NULL || name == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dir_len = strlen(dir);
    name_len = strlen(name);
    needs_slash = dir_len > 0U && dir[dir_len - 1U] != '/';
    if (dir_len + (needs_slash != 0 ? 1U : 0U) + name_len + 1U > cap) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = snprintf(out, cap, "%s%s%s", dir, needs_slash != 0 ? "/" : "", name);
    if (n < 0 || (size_t)n >= cap) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_tok_ensure_dir(const char *path)
{
    struct stat st;
    if (path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (stat(path, &st) == 0) {
        return S_ISDIR(st.st_mode) ? GD_OK : GD_ERR_INVALID_ARGUMENT;
    }
    if (errno != ENOENT) {
        return GD_ERR_IO;
    }
    if (mkdir(path, 0775) != 0 && errno != EEXIST) {
        return GD_ERR_IO;
    }
    if (stat(path, &st) != 0 || !S_ISDIR(st.st_mode)) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_tok_mkdir_p(const char *path)
{
    char tmp[GD_TOK_PATH_MAX];
    size_t len;
    char *p;

    if (path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    len = strlen(path);
    if (len >= sizeof(tmp)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memcpy(tmp, path, len + 1U);
    while (len > 1U && tmp[len - 1U] == '/') {
        tmp[len - 1U] = '\0';
        len -= 1U;
    }
    for (p = tmp + 1; *p != '\0'; ++p) {
        if (*p == '/') {
            *p = '\0';
            if (tmp[0] != '\0') {
                gd_status status = gd_tok_ensure_dir(tmp);
                if (status != GD_OK) {
                    return status;
                }
            }
            *p = '/';
        }
    }
    return gd_tok_ensure_dir(tmp);
}

static gd_status gd_tok_ensure_parent_dir(const char *path)
{
    char tmp[GD_TOK_PATH_MAX];
    char *slash;
    size_t len;
    if (path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    len = strlen(path);
    if (len >= sizeof(tmp)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memcpy(tmp, path, len + 1U);
    slash = strrchr(tmp, '/');
    if (slash == NULL) {
        return GD_OK;
    }
    if (slash == tmp) {
        return GD_OK;
    }
    *slash = '\0';
    return gd_tok_mkdir_p(tmp);
}

static gd_status read_file(const char *path, char **text_out)
{
    FILE *f;
    long end;
    char *text;
    size_t nread;

    if (path == NULL || text_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *text_out = NULL;
    f = fopen(path, "rb");
    if (f == NULL) {
        return GD_ERR_IO;
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    end = ftell(f);
    if (end < 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    text = (char *)malloc((size_t)end + 1U);
    if (text == NULL) {
        (void)fclose(f);
        return GD_ERR_OUT_OF_MEMORY;
    }
    nread = fread(text, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(text);
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fclose(f) != 0) {
        free(text);
        return GD_ERR_IO;
    }
    text[(size_t)end] = '\0';
    *text_out = text;
    return GD_OK;
}

static gd_status write_tokens(const char *path, const int32_t *ids, int n_ids)
{
    FILE *f;
    size_t nwrite;
    if (path == NULL || ids == NULL || n_ids < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        return GD_ERR_IO;
    }
    nwrite = fwrite(ids, sizeof(int32_t), (size_t)n_ids, f);
    if (nwrite != (size_t)n_ids) {
        (void)fclose(f);
        return GD_ERR_IO;
    }
    if (fclose(f) != 0) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_tok_read_line(FILE *f, char **line, size_t *cap, size_t *len_out)
{
    int c;
    size_t len = 0U;
    if (f == NULL || line == NULL || cap == NULL || len_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (*line == NULL || *cap == 0U) {
        *cap = 4096U;
        *line = (char *)malloc(*cap);
        if (*line == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    while ((c = fgetc(f)) != EOF) {
        if (len + 1U >= *cap) {
            char *new_line;
            if (*cap > SIZE_MAX / 2U) {
                return GD_ERR_OUT_OF_MEMORY;
            }
            *cap *= 2U;
            new_line = (char *)realloc(*line, *cap);
            if (new_line == NULL) {
                return GD_ERR_OUT_OF_MEMORY;
            }
            *line = new_line;
        }
        (*line)[len] = (char)c;
        len += 1U;
        if (c == '\n') {
            break;
        }
    }
    if (len == 0U && c == EOF) {
        *len_out = 0U;
        return feof(f) ? GD_TOK_STATUS_EOF : GD_ERR_IO;
    }
    (*line)[len] = '\0';
    *len_out = len;
    return GD_OK;
}

static void gd_tok_json_write_string(FILE *f, const char *s)
{
    const unsigned char *p = (const unsigned char *)s;
    fputc('"', f);
    while (*p != (unsigned char)'\0') {
        unsigned char c = *p;
        if (c == (unsigned char)'"' || c == (unsigned char)'\\') {
            fputc('\\', f);
            fputc((int)c, f);
        } else if (c == (unsigned char)'\n') {
            fputs("\\n", f);
        } else if (c == (unsigned char)'\r') {
            fputs("\\r", f);
        } else if (c == (unsigned char)'\t') {
            fputs("\\t", f);
        } else if (c < 0x20U) {
            fprintf(f, "\\u%04x", (unsigned int)c);
        } else {
            fputc((int)c, f);
        }
        ++p;
    }
    fputc('"', f);
}

static int gd_tok_hex_value(char c)
{
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

static gd_status gd_tok_append_byte(char **buf, size_t *len, size_t *cap, unsigned char c)
{
    char *new_buf;
    if (buf == NULL || len == NULL || cap == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (*len + 1U >= *cap) {
        if (*cap > SIZE_MAX / 2U) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        *cap *= 2U;
        new_buf = (char *)realloc(*buf, *cap);
        if (new_buf == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        *buf = new_buf;
    }
    (*buf)[*len] = (char)c;
    *len += 1U;
    return GD_OK;
}

static gd_status gd_tok_append_utf8(char **buf, size_t *len, size_t *cap, uint32_t cp)
{
    gd_status status;
    if (cp == 0U || cp > 0x10FFFFU || (cp >= 0xD800U && cp <= 0xDFFFU)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cp <= 0x7FU) {
        return gd_tok_append_byte(buf, len, cap, (unsigned char)cp);
    }
    if (cp <= 0x7FFU) {
        status = gd_tok_append_byte(buf, len, cap, (unsigned char)(0xC0U | (cp >> 6U)));
        if (status != GD_OK) {
            return status;
        }
        return gd_tok_append_byte(buf, len, cap, (unsigned char)(0x80U | (cp & 0x3FU)));
    }
    if (cp <= 0xFFFFU) {
        status = gd_tok_append_byte(buf, len, cap, (unsigned char)(0xE0U | (cp >> 12U)));
        if (status != GD_OK) {
            return status;
        }
        status = gd_tok_append_byte(buf, len, cap,
                                    (unsigned char)(0x80U | ((cp >> 6U) & 0x3FU)));
        if (status != GD_OK) {
            return status;
        }
        return gd_tok_append_byte(buf, len, cap, (unsigned char)(0x80U | (cp & 0x3FU)));
    }
    status = gd_tok_append_byte(buf, len, cap, (unsigned char)(0xF0U | (cp >> 18U)));
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_append_byte(buf, len, cap,
                                (unsigned char)(0x80U | ((cp >> 12U) & 0x3FU)));
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_append_byte(buf, len, cap,
                                (unsigned char)(0x80U | ((cp >> 6U) & 0x3FU)));
    if (status != GD_OK) {
        return status;
    }
    return gd_tok_append_byte(buf, len, cap, (unsigned char)(0x80U | (cp & 0x3FU)));
}

static gd_status gd_tok_parse_u_escape(const char **p, uint32_t *cp_out)
{
    int i;
    uint32_t cp = 0U;
    if (p == NULL || *p == NULL || cp_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < 4; ++i) {
        int hv = gd_tok_hex_value((*p)[i]);
        if (hv < 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        cp = (cp << 4U) | (uint32_t)hv;
    }
    *p += 4;
    *cp_out = cp;
    return GD_OK;
}

static gd_status gd_tok_json_parse_string(const char **p_in, char **out)
{
    const char *p;
    char *buf;
    size_t cap = 64U;
    size_t len = 0U;
    gd_status status;

    if (p_in == NULL || *p_in == NULL || out == NULL || **p_in != '"') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    p = *p_in + 1;
    buf = (char *)malloc(cap);
    if (buf == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    while (*p != '\0' && *p != '"') {
        unsigned char c = (unsigned char)*p;
        if (c < 0x20U) {
            free(buf);
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (*p == '\\') {
            ++p;
            if (*p == '\0') {
                free(buf);
                return GD_ERR_INVALID_ARGUMENT;
            }
            switch (*p) {
                case '"':
                case '\\':
                case '/':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)*p);
                    ++p;
                    break;
                case 'b':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)'\b');
                    ++p;
                    break;
                case 'f':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)'\f');
                    ++p;
                    break;
                case 'n':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)'\n');
                    ++p;
                    break;
                case 'r':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)'\r');
                    ++p;
                    break;
                case 't':
                    status = gd_tok_append_byte(&buf, &len, &cap, (unsigned char)'\t');
                    ++p;
                    break;
                case 'u': {
                    uint32_t cp;
                    ++p;
                    status = gd_tok_parse_u_escape(&p, &cp);
                    if (status != GD_OK) {
                        break;
                    }
                    if (cp >= 0xD800U && cp <= 0xDBFFU) {
                        uint32_t low;
                        if (p[0] != '\\' || p[1] != 'u') {
                            status = GD_ERR_INVALID_ARGUMENT;
                            break;
                        }
                        p += 2;
                        status = gd_tok_parse_u_escape(&p, &low);
                        if (status != GD_OK) {
                            break;
                        }
                        if (low < 0xDC00U || low > 0xDFFFU) {
                            status = GD_ERR_INVALID_ARGUMENT;
                            break;
                        }
                        cp = 0x10000U + (((cp - 0xD800U) << 10U) | (low - 0xDC00U));
                    } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
                        status = GD_ERR_INVALID_ARGUMENT;
                        break;
                    }
                    status = gd_tok_append_utf8(&buf, &len, &cap, cp);
                    break;
                }
                default:
                    status = GD_ERR_INVALID_ARGUMENT;
                    break;
            }
            if (status != GD_OK) {
                free(buf);
                return status;
            }
        } else {
            status = gd_tok_append_byte(&buf, &len, &cap, c);
            if (status != GD_OK) {
                free(buf);
                return status;
            }
            ++p;
        }
    }
    if (*p != '"') {
        free(buf);
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (len + 1U >= cap) {
        char *new_buf = (char *)realloc(buf, len + 1U);
        if (new_buf == NULL) {
            free(buf);
            return GD_ERR_OUT_OF_MEMORY;
        }
        buf = new_buf;
    }
    buf[len] = '\0';
    *p_in = p + 1;
    *out = buf;
    return GD_OK;
}

static void gd_tok_json_skip_ws(const char **p)
{
    while (**p == ' ' || **p == '\t' || **p == '\n' || **p == '\r') {
        *p += 1;
    }
}

static gd_status gd_tok_json_skip_string(const char **p)
{
    char *tmp = NULL;
    gd_status status = gd_tok_json_parse_string(p, &tmp);
    free(tmp);
    return status;
}

static gd_status gd_tok_json_skip_value(const char **p)
{
    int stack[GD_TOK_JSON_MAX_DEPTH];
    int depth = 0;
    if (p == NULL || *p == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_tok_json_skip_ws(p);
    if (**p == '"') {
        return gd_tok_json_skip_string(p);
    }
    if (**p == '{' || **p == '[') {
        stack[depth++] = **p == '{' ? '}' : ']';
        *p += 1;
        while (**p != '\0' && depth > 0) {
            if (**p == '"') {
                gd_status status = gd_tok_json_skip_string(p);
                if (status != GD_OK) {
                    return status;
                }
                continue;
            }
            if (**p == '{' || **p == '[') {
                if (depth >= GD_TOK_JSON_MAX_DEPTH) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
                stack[depth++] = **p == '{' ? '}' : ']';
                *p += 1;
                continue;
            }
            if (**p == stack[depth - 1]) {
                depth -= 1;
                *p += 1;
                continue;
            }
            *p += 1;
        }
        return depth == 0 ? GD_OK : GD_ERR_INVALID_ARGUMENT;
    }
    if (**p == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    while (**p != '\0' && **p != ',' && **p != '}' && **p != ']') {
        *p += 1;
    }
    return GD_OK;
}

static gd_status gd_tok_jsonl_extract_string(const char *line,
                                             const char *field,
                                             char **text_out)
{
    const char *p = line;
    if (line == NULL || field == NULL || text_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *text_out = NULL;
    gd_tok_json_skip_ws(&p);
    if (*p != '{') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    ++p;
    for (;;) {
        char *key = NULL;
        gd_tok_json_skip_ws(&p);
        if (*p == '}') {
            return GD_TOK_STATUS_NOT_FOUND;
        }
        if (*p != '"') {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (gd_tok_json_parse_string(&p, &key) != GD_OK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        gd_tok_json_skip_ws(&p);
        if (*p != ':') {
            free(key);
            return GD_ERR_INVALID_ARGUMENT;
        }
        ++p;
        gd_tok_json_skip_ws(&p);
        if (strcmp(key, field) == 0) {
            gd_status status;
            free(key);
            if (*p != '"') {
                return GD_ERR_INVALID_ARGUMENT;
            }
            status = gd_tok_json_parse_string(&p, text_out);
            return status;
        }
        free(key);
        if (gd_tok_json_skip_value(&p) != GD_OK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        gd_tok_json_skip_ws(&p);
        if (*p == ',') {
            ++p;
            continue;
        }
        if (*p == '}') {
            return GD_TOK_STATUS_NOT_FOUND;
        }
        return GD_ERR_INVALID_ARGUMENT;
    }
}

static const char *gd_tok_effective_tokenizer_path(const gd_tok_args *args)
{
    return args->use_tokenizer_path != NULL ? args->use_tokenizer_path : args->tokenizer_path;
}

static gd_status gd_tok_add_unique_special(const char **specials, int *n_specials, const char *s)
{
    int i;
    if (specials == NULL || n_specials == NULL || s == NULL || s[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < *n_specials; ++i) {
        if (strcmp(specials[i], s) == 0) {
            return GD_OK;
        }
    }
    if (*n_specials >= GD_TOK_MAX_SPECIALS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    specials[*n_specials] = s;
    *n_specials += 1;
    return GD_OK;
}

static gd_status gd_tok_extract_jsonl_training_file(const gd_tok_args *args, char **path_out)
{
    FILE *in;
    FILE *out;
    int fd;
    char tmpl[GD_TOK_PATH_MAX];
    char *line = NULL;
    size_t line_cap = 0U;
    size_t line_len = 0U;
    uint64_t line_no = 0U;
    gd_status status;

    if (args == NULL || args->input_jsonl_path == NULL || args->output_dir == NULL ||
        args->jsonl_text_field == NULL || path_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *path_out = NULL;
    status = gd_tok_mkdir_p(args->output_dir);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_join_path(tmpl,
                              sizeof(tmpl),
                              args->output_dir,
                              ".gradients-tokenize-train-XXXXXX.tmp");
    if (status != GD_OK) {
        return status;
    }
    fd = mkstemp(tmpl);
    if (fd < 0) {
        return GD_ERR_IO;
    }
    out = fdopen(fd, "wb");
    if (out == NULL) {
        (void)close(fd);
        (void)unlink(tmpl);
        return GD_ERR_IO;
    }
    in = fopen(args->input_jsonl_path, "rb");
    if (in == NULL) {
        (void)fclose(out);
        (void)unlink(tmpl);
        return GD_ERR_IO;
    }
    for (;;) {
        char *text = NULL;
        status = gd_tok_read_line(in, &line, &line_cap, &line_len);
        if (status == GD_TOK_STATUS_EOF) {
            status = GD_OK;
            break;
        }
        if (status != GD_OK) {
            break;
        }
        line_no += 1U;
        (void)line_len;
        status = gd_tok_jsonl_extract_string(line, args->jsonl_text_field, &text);
        if (status != GD_OK) {
            fprintf(stderr,
                    "jsonl parse failed while preparing tokenizer training data at line %"
                    PRIu64 ": missing/string field '%s'\n",
                    line_no,
                    args->jsonl_text_field);
            break;
        }
        if (fputs(text, out) == EOF || fputc('\n', out) == EOF) {
            free(text);
            status = GD_ERR_IO;
            break;
        }
        free(text);
    }
    free(line);
    if (fclose(in) != 0 && status == GD_OK) {
        status = GD_ERR_IO;
    }
    if (fclose(out) != 0 && status == GD_OK) {
        status = GD_ERR_IO;
    }
    if (status != GD_OK) {
        (void)unlink(tmpl);
        return status;
    }
    *path_out = (char *)malloc(strlen(tmpl) + 1U);
    if (*path_out == NULL) {
        (void)unlink(tmpl);
        return GD_ERR_OUT_OF_MEMORY;
    }
    memcpy(*path_out, tmpl, strlen(tmpl) + 1U);
    return GD_OK;
}

static gd_status gd_tok_load_existing_tokenizer(const gd_tok_args *args,
                                                const char *path,
                                                int allow_special,
                                                gd_tokenizer **tok_out)
{
    gd_tokenizer_config cfg;
    if (args == NULL || path == NULL || tok_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    cfg.split_digits = 1;
    cfg.allow_special = allow_special;
    return gd_bpe_tokenizer_load(path, &cfg, tok_out);
}

static gd_status gd_tok_load_or_train_tokenizer(const gd_tok_args *args,
                                                const char *train_path,
                                                int include_im_specials,
                                                int allow_special,
                                                gd_tokenizer **tok_out,
                                                int *trained_out)
{
    gd_tokenizer *trained = NULL;
    gd_bpe_train_config cfg;
    const char *specials[GD_TOK_MAX_SPECIALS];
    const char *inputs[1];
    const char *path;
    int n_specials = 0;
    int i;
    gd_status status;

    if (args == NULL || tok_out == NULL || trained_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *tok_out = NULL;
    *trained_out = 0;
    path = gd_tok_effective_tokenizer_path(args);
    if (path == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (args->use_tokenizer_path != NULL) {
        return gd_tok_load_existing_tokenizer(args, path, allow_special, tok_out);
    }
    if (train_path == NULL || args->tokenizer_path == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < args->n_specials; ++i) {
        status = gd_tok_add_unique_special(specials, &n_specials, args->specials[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    if (include_im_specials != 0) {
        status = gd_tok_add_unique_special(specials, &n_specials, args->im_start);
        if (status != GD_OK) {
            return status;
        }
        status = gd_tok_add_unique_special(specials, &n_specials, args->im_end);
        if (status != GD_OK) {
            return status;
        }
    }
    memset(&cfg, 0, sizeof(cfg));
    cfg.vocab_size = args->vocab_size;
    cfg.min_frequency = args->min_frequency;
    cfg.split_digits = 1;
    cfg.n_special_tokens = n_specials;
    cfg.special_tokens = specials;
    cfg.seed = args->seed;
    inputs[0] = train_path;
    status = gd_bpe_tokenizer_train(inputs, 1, &cfg, &trained);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_ensure_parent_dir(args->tokenizer_path);
    if (status != GD_OK) {
        gd_tokenizer_destroy(trained);
        return status;
    }
    status = gd_bpe_tokenizer_save(trained, args->tokenizer_path);
    gd_tokenizer_destroy(trained);
    if (status != GD_OK) {
        return status;
    }
    *trained_out = 1;
    return gd_tok_load_existing_tokenizer(args, args->tokenizer_path, allow_special, tok_out);
}

static gd_status gd_tok_writer_add_info(gd_tok_writer *w, const gd_tok_shard_info *info)
{
    gd_tok_shard_info *new_infos;
    size_t new_cap;
    if (w == NULL || info == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (w->n_infos == w->cap_infos) {
        new_cap = w->cap_infos == 0U ? 8U : w->cap_infos * 2U;
        if (new_cap < w->cap_infos) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        new_infos = (gd_tok_shard_info *)realloc(w->infos, new_cap * sizeof(*w->infos));
        if (new_infos == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        w->infos = new_infos;
        w->cap_infos = new_cap;
    }
    w->infos[w->n_infos] = *info;
    w->n_infos += 1U;
    return GD_OK;
}

static gd_status gd_tok_writer_open(gd_tok_writer *w,
                                    const char *output_dir,
                                    int variable_length,
                                    uint64_t max_shard_bytes)
{
    if (w == NULL || output_dir == NULL || max_shard_bytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(w, 0, sizeof(*w));
    w->output_dir = output_dir;
    w->variable_length = variable_length;
    w->max_shard_bytes = max_shard_bytes;
    return gd_tok_mkdir_p(output_dir);
}

static gd_status gd_tok_writer_start_shard(gd_tok_writer *w)
{
    char token_path[GD_TOK_PATH_MAX];
    char offset_path[GD_TOK_PATH_MAX];
    int n;
    uint64_t zero = 0U;

    if (w == NULL || w->tokens_f != NULL || w->offsets_f != NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = snprintf(w->cur_tokens_file,
                 sizeof(w->cur_tokens_file),
                 "tokens-%06" PRIu64 ".i32",
                 w->next_shard_index);
    if (n < 0 || (size_t)n >= sizeof(w->cur_tokens_file)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = snprintf(w->cur_offsets_file,
                 sizeof(w->cur_offsets_file),
                 "offsets-%06" PRIu64 ".u64",
                 w->next_shard_index);
    if (n < 0 || (size_t)n >= sizeof(w->cur_offsets_file)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    w->next_shard_index += 1U;
    if (gd_tok_join_path(token_path, sizeof(token_path), w->output_dir, w->cur_tokens_file) != GD_OK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    w->tokens_f = fopen(token_path, "wb");
    if (w->tokens_f == NULL) {
        return GD_ERR_IO;
    }
    if (w->variable_length != 0) {
        if (gd_tok_join_path(offset_path,
                             sizeof(offset_path),
                             w->output_dir,
                             w->cur_offsets_file) != GD_OK) {
            (void)fclose(w->tokens_f);
            w->tokens_f = NULL;
            return GD_ERR_INVALID_ARGUMENT;
        }
        w->offsets_f = fopen(offset_path, "wb");
        if (w->offsets_f == NULL) {
            (void)fclose(w->tokens_f);
            w->tokens_f = NULL;
            return GD_ERR_IO;
        }
        if (fwrite(&zero, sizeof(zero), 1U, w->offsets_f) != 1U) {
            (void)fclose(w->offsets_f);
            (void)fclose(w->tokens_f);
            w->offsets_f = NULL;
            w->tokens_f = NULL;
            return GD_ERR_IO;
        }
        w->cur_offset_bytes = sizeof(uint64_t);
    } else {
        w->cur_offsets_file[0] = '\0';
        w->cur_offset_bytes = 0U;
    }
    w->cur_sequences = 0U;
    w->cur_tokens = 0U;
    w->cur_token_bytes = 0U;
    return GD_OK;
}

static gd_status gd_tok_writer_finish_shard(gd_tok_writer *w)
{
    gd_tok_shard_info info;
    int close_status = 0;
    if (w == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (w->tokens_f == NULL) {
        return GD_OK;
    }
    if (w->offsets_f != NULL && fclose(w->offsets_f) != 0) {
        close_status = 1;
    }
    if (fclose(w->tokens_f) != 0) {
        close_status = 1;
    }
    w->tokens_f = NULL;
    w->offsets_f = NULL;
    if (close_status != 0) {
        return GD_ERR_IO;
    }
    memset(&info, 0, sizeof(info));
    memcpy(info.tokens_file, w->cur_tokens_file, strlen(w->cur_tokens_file) + 1U);
    if (w->variable_length != 0) {
        memcpy(info.offsets_file, w->cur_offsets_file, strlen(w->cur_offsets_file) + 1U);
    }
    info.sequences = w->cur_sequences;
    info.tokens = w->cur_tokens;
    info.bytes = w->cur_token_bytes + w->cur_offset_bytes;
    return gd_tok_writer_add_info(w, &info);
}

static void gd_tok_writer_destroy(gd_tok_writer *w)
{
    if (w == NULL) {
        return;
    }
    if (w->offsets_f != NULL) {
        (void)fclose(w->offsets_f);
    }
    if (w->tokens_f != NULL) {
        (void)fclose(w->tokens_f);
    }
    free(w->infos);
    memset(w, 0, sizeof(*w));
}

static gd_status gd_tok_writer_write_i32(FILE *f, const int32_t *data, uint64_t n)
{
    if (n == 0U) {
        return GD_OK;
    }
    if (f == NULL || data == NULL || n > (uint64_t)SIZE_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (fwrite(data, sizeof(int32_t), (size_t)n, f) != (size_t)n) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_tok_writer_write_sequence(gd_tok_writer *w,
                                              const int32_t *prefix,
                                              int prefix_n,
                                              const int32_t *body,
                                              int body_n,
                                              const int32_t *suffix,
                                              int suffix_n)
{
    uint64_t total_n;
    uint64_t seq_token_bytes;
    uint64_t projected_bytes;
    gd_status status;

    if (w == NULL || prefix_n < 0 || body_n < 0 || suffix_n < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    total_n = (uint64_t)prefix_n + (uint64_t)body_n + (uint64_t)suffix_n;
    if (total_n == 0U || total_n > UINT64_MAX / sizeof(int32_t)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    seq_token_bytes = total_n * sizeof(int32_t);
    if (w->variable_length != 0) {
        if (seq_token_bytes + sizeof(uint64_t) + sizeof(uint64_t) > w->max_shard_bytes) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    } else if (seq_token_bytes > w->max_shard_bytes) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    if (w->tokens_f == NULL) {
        status = gd_tok_writer_start_shard(w);
        if (status != GD_OK) {
            return status;
        }
    }
    projected_bytes = w->cur_token_bytes + w->cur_offset_bytes + seq_token_bytes +
                      (w->variable_length != 0 ? sizeof(uint64_t) : 0U);
    if (w->cur_sequences > 0U && projected_bytes > w->max_shard_bytes) {
        status = gd_tok_writer_finish_shard(w);
        if (status != GD_OK) {
            return status;
        }
        status = gd_tok_writer_start_shard(w);
        if (status != GD_OK) {
            return status;
        }
    }

    status = gd_tok_writer_write_i32(w->tokens_f, prefix, (uint64_t)prefix_n);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_writer_write_i32(w->tokens_f, body, (uint64_t)body_n);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tok_writer_write_i32(w->tokens_f, suffix, (uint64_t)suffix_n);
    if (status != GD_OK) {
        return status;
    }
    w->cur_token_bytes += seq_token_bytes;
    w->cur_tokens += total_n;
    w->cur_sequences += 1U;
    w->total_tokens += total_n;
    w->total_sequences += 1U;

    if (w->variable_length != 0) {
        uint64_t next_offset = w->cur_tokens;
        if (fwrite(&next_offset, sizeof(next_offset), 1U, w->offsets_f) != 1U) {
            return GD_ERR_IO;
        }
        w->cur_offset_bytes += sizeof(uint64_t);
    }
    return GD_OK;
}

static gd_status gd_tok_write_manifest(const char *output_dir,
                                       const gd_tok_writer *w,
                                       const gd_tok_manifest_cfg *cfg)
{
    char path[GD_TOK_PATH_MAX];
    FILE *f;
    size_t i;
    if (output_dir == NULL || w == NULL || cfg == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (gd_tok_join_path(path, sizeof(path), output_dir, "manifest.json") != GD_OK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        return GD_ERR_IO;
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"gd-tokenized-corpus-v1\",\n");
    fprintf(f, "  \"mode\": ");
    gd_tok_json_write_string(f, cfg->mode);
    fprintf(f, ",\n");
    fprintf(f, "  \"dtype\": \"int32\",\n");
    fprintf(f, "  \"offset_dtype\": %s,\n", w->variable_length != 0 ? "\"uint64\"" : "null");
    fprintf(f, "  \"tokenizer\": ");
    gd_tok_json_write_string(f, cfg->tokenizer_path);
    fprintf(f, ",\n");
    fprintf(f, "  \"tokenizer_hash\": \"%016" PRIx64 "\",\n", cfg->tokenizer_hash);
    fprintf(f, "  \"im_start_id\": %d,\n", cfg->im_start_id);
    fprintf(f, "  \"im_end_id\": %d,\n", cfg->im_end_id);
    fprintf(f, "  \"sequence_length\": %" PRIu64 ",\n", cfg->sequence_length);
    if (cfg->stride != 0U) {
        fprintf(f, "  \"stride\": %" PRIu64 ",\n", cfg->stride);
    } else {
        fprintf(f, "  \"stride\": null,\n");
    }
    fprintf(f, "  \"max_length\": %" PRIu64 ",\n", cfg->max_length);
    if (cfg->jsonl_text_field != NULL) {
        fprintf(f, "  \"jsonl_text_field\": ");
        gd_tok_json_write_string(f, cfg->jsonl_text_field);
        fprintf(f, ",\n");
    }
    fprintf(f,
            "  \"shard_size_limit_bytes\": %" PRIu64 ",\n",
            cfg->shard_size_limit_bytes);
    fprintf(f, "  \"num_sequences\": %" PRIu64 ",\n", w->total_sequences);
    fprintf(f, "  \"num_tokens\": %" PRIu64 ",\n", w->total_tokens);
    fprintf(f, "  \"shards\": [\n");
    for (i = 0U; i < w->n_infos; ++i) {
        const gd_tok_shard_info *info = &w->infos[i];
        fprintf(f, "    {\"tokens\":");
        gd_tok_json_write_string(f, info->tokens_file);
        if (w->variable_length != 0) {
            fprintf(f, ",\"offsets\":");
            gd_tok_json_write_string(f, info->offsets_file);
        }
        fprintf(f,
                ",\"sequences\":%" PRIu64 ",\"tokens_count\":%" PRIu64
                ",\"bytes\":%" PRIu64 "}%s\n",
                info->sequences,
                info->tokens,
                info->bytes,
                i + 1U == w->n_infos ? "" : ",");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    if (fclose(f) != 0) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_tok_run_legacy_flat(const gd_tok_args *args)
{
    gd_tokenizer *tok = NULL;
    char *text = NULL;
    int32_t *ids = NULL;
    int n_ids = 0;
    int trained = 0;
    const char *tok_path = gd_tok_effective_tokenizer_path(args);
    gd_status status;

    status = gd_tok_load_or_train_tokenizer(args,
                                            args->input_path,
                                            0,
                                            args->no_special == 0 ? 1 : 0,
                                            &tok,
                                            &trained);
    if (status != GD_OK) {
        fprintf(stderr, "%s failed: %s\n", trained != 0 ? "train" : "load", gd_status_string(status));
        return status;
    }
    status = read_file(args->input_path, &text);
    if (status != GD_OK) {
        fprintf(stderr, "read failed: %s\n", gd_status_string(status));
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_tokenizer_encode(tok, text, &ids, &n_ids);
    if (status != GD_OK) {
        fprintf(stderr, "encode failed: %s\n", gd_status_string(status));
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = write_tokens(args->output_path, ids, n_ids);
    if (status != GD_OK) {
        fprintf(stderr, "write failed: %s\n", gd_status_string(status));
        gd_tokenizer_free(ids);
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    printf("{\"mode\":\"flat\",\"trained\":%s,\"tokenizer\":\"%s\","
           "\"tokens\":%d,\"output\":\"%s\"}\n",
           trained != 0 ? "true" : "false",
           tok_path,
           n_ids,
           args->output_path);

    gd_tokenizer_free(ids);
    free(text);
    gd_tokenizer_destroy(tok);
    return GD_OK;
}

static gd_status gd_tok_resolve_specials(gd_tokenizer *tok,
                                         const char *im_start,
                                         const char *im_end,
                                         int32_t *im_start_id,
                                         int32_t *im_end_id)
{
    gd_status status;
    if (tok == NULL || im_start == NULL || im_end == NULL || im_start_id == NULL ||
        im_end_id == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_tokenizer_id(tok, im_start, im_start_id);
    if (status != GD_OK) {
        fprintf(stderr, "special token not found: %s\n", im_start);
        return status;
    }
    status = gd_tokenizer_id(tok, im_end, im_end_id);
    if (status != GD_OK) {
        fprintf(stderr, "special token not found: %s\n", im_end);
        return status;
    }
    return GD_OK;
}

static gd_status gd_tok_run_packed_text(const gd_tok_args *args)
{
    gd_tokenizer *tok = NULL;
    gd_tok_writer writer;
    gd_tok_manifest_cfg manifest;
    char *text = NULL;
    int32_t *ids = NULL;
    int n_ids = 0;
    int32_t im_start_id;
    int32_t im_end_id;
    uint64_t chunk_tokens;
    uint64_t stride;
    uint64_t body_limit;
    uint64_t full_chunks;
    uint64_t emitted_body_tokens;
    uint64_t chunk;
    int trained = 0;
    const char *tok_path = gd_tok_effective_tokenizer_path(args);
    gd_status status;

    memset(&writer, 0, sizeof(writer));
    memset(&manifest, 0, sizeof(manifest));
    if (args->num_tokens_per_sequence < 3U || args->num_tokens_per_sequence > (uint64_t)INT_MAX) {
        fprintf(stderr, "--num-tokens-per-sequence must be in [3, INT_MAX]\n");
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (args->max_length != 0U &&
        (args->max_length < 2U || args->max_length > (uint64_t)INT_MAX)) {
        fprintf(stderr, "--max-length must be in [2, INT_MAX] when provided\n");
        return GD_ERR_INVALID_ARGUMENT;
    }
    chunk_tokens = args->num_tokens_per_sequence;
    stride = chunk_tokens - 1U;
    status = gd_tok_load_or_train_tokenizer(args, args->input_path, 1, 0, &tok, &trained);
    if (status != GD_OK) {
        fprintf(stderr, "tokenizer prepare failed: %s\n", gd_status_string(status));
        return status;
    }
    status = gd_tok_resolve_specials(tok, args->im_start, args->im_end, &im_start_id, &im_end_id);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = read_file(args->input_path, &text);
    if (status != GD_OK) {
        fprintf(stderr, "read failed: %s\n", gd_status_string(status));
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_tokenizer_encode(tok, text, &ids, &n_ids);
    if (status != GD_OK) {
        fprintf(stderr, "encode failed: %s\n", gd_status_string(status));
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_tok_writer_open(&writer, args->output_dir, 0, GD_TOK_SHARD_MAX_BYTES);
    if (status != GD_OK) {
        fprintf(stderr, "open output failed: %s\n", gd_status_string(status));
        gd_tokenizer_free(ids);
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    body_limit = (uint64_t)n_ids;
    if (args->max_length != 0U && body_limit > args->max_length - 2U) {
        body_limit = args->max_length - 2U;
    }
    full_chunks = (body_limit + 1U) / stride;
    if (full_chunks > 0U) {
        emitted_body_tokens = full_chunks * stride - 1U;
        for (chunk = 0U; chunk < full_chunks; ++chunk) {
            uint64_t lo = chunk * stride;
            uint64_t hi = lo + chunk_tokens;
            int prefix_n = lo == 0U ? 1 : 0;
            int suffix_n = hi == emitted_body_tokens + 2U ? 1 : 0;
            uint64_t body_start = lo < 1U ? 0U : lo - 1U;
            uint64_t body_end = hi <= emitted_body_tokens + 1U ? hi - 1U : emitted_body_tokens;
            int body_n = body_end > body_start ? (int)(body_end - body_start) : 0;
            status = gd_tok_writer_write_sequence(&writer,
                                                  prefix_n != 0 ? &im_start_id : NULL,
                                                  prefix_n,
                                                  body_n != 0 ? &ids[body_start] : NULL,
                                                  body_n,
                                                  suffix_n != 0 ? &im_end_id : NULL,
                                                  suffix_n);
            if (status != GD_OK) {
                fprintf(stderr, "write sequence failed: %s\n", gd_status_string(status));
                gd_tok_writer_destroy(&writer);
                gd_tokenizer_free(ids);
                free(text);
                gd_tokenizer_destroy(tok);
                return status;
            }
        }
    }
    status = gd_tok_writer_finish_shard(&writer);
    if (status != GD_OK) {
        fprintf(stderr, "finalize shard failed: %s\n", gd_status_string(status));
        gd_tok_writer_destroy(&writer);
        gd_tokenizer_free(ids);
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    manifest.mode = "packed";
    manifest.tokenizer_path = tok_path;
    manifest.tokenizer_hash = gd_tokenizer_hash(tok);
    manifest.sequence_length = args->num_tokens_per_sequence;
    manifest.stride = stride;
    manifest.max_length = args->max_length != 0U ? args->max_length : 0U;
    manifest.shard_size_limit_bytes = GD_TOK_SHARD_MAX_BYTES;
    manifest.im_start_id = im_start_id;
    manifest.im_end_id = im_end_id;
    status = gd_tok_write_manifest(args->output_dir, &writer, &manifest);
    if (status != GD_OK) {
        fprintf(stderr, "write manifest failed: %s\n", gd_status_string(status));
        gd_tok_writer_destroy(&writer);
        gd_tokenizer_free(ids);
        free(text);
        gd_tokenizer_destroy(tok);
        return status;
    }
    printf("{\"mode\":\"packed\",\"trained\":%s,\"tokenizer\":\"%s\","
           "\"sequences\":%" PRIu64 ",\"tokens\":%" PRIu64
           ",\"shards\":%zu,\"output_dir\":\"%s\"}\n",
           trained != 0 ? "true" : "false",
           tok_path,
           writer.total_sequences,
           writer.total_tokens,
           writer.n_infos,
           args->output_dir);
    gd_tok_writer_destroy(&writer);
    gd_tokenizer_free(ids);
    free(text);
    gd_tokenizer_destroy(tok);
    return GD_OK;
}

static gd_status gd_tok_run_jsonl(const gd_tok_args *args)
{
    FILE *f;
    gd_tokenizer *tok = NULL;
    gd_tok_writer writer;
    gd_tok_manifest_cfg manifest;
    char *line = NULL;
    size_t line_cap = 0U;
    size_t line_len = 0U;
    uint64_t line_no = 0U;
    int32_t im_start_id;
    int32_t im_end_id;
    int body_cap;
    int trained = 0;
    char *train_path = NULL;
    const char *tok_path = gd_tok_effective_tokenizer_path(args);
    gd_status status;

    memset(&writer, 0, sizeof(writer));
    memset(&manifest, 0, sizeof(manifest));
    if (args->max_length < 2U || args->max_length > (uint64_t)INT_MAX) {
        fprintf(stderr, "--max-length must be in [2, INT_MAX] for jsonl mode\n");
        return GD_ERR_INVALID_ARGUMENT;
    }
    body_cap = (int)args->max_length - 2;
    if (args->use_tokenizer_path == NULL) {
        status = gd_tok_extract_jsonl_training_file(args, &train_path);
        if (status != GD_OK) {
            fprintf(stderr, "prepare jsonl training data failed: %s\n", gd_status_string(status));
            return status;
        }
    }
    status = gd_tok_load_or_train_tokenizer(args,
                                            train_path != NULL ? train_path : args->input_jsonl_path,
                                            1,
                                            0,
                                            &tok,
                                            &trained);
    if (train_path != NULL) {
        (void)unlink(train_path);
        free(train_path);
        train_path = NULL;
    }
    if (status != GD_OK) {
        fprintf(stderr, "tokenizer prepare failed: %s\n", gd_status_string(status));
        return status;
    }
    status = gd_tok_resolve_specials(tok, args->im_start, args->im_end, &im_start_id, &im_end_id);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    f = fopen(args->input_jsonl_path, "rb");
    if (f == NULL) {
        fprintf(stderr, "read failed: %s\n", gd_status_string(GD_ERR_IO));
        gd_tokenizer_destroy(tok);
        return GD_ERR_IO;
    }
    status = gd_tok_writer_open(&writer, args->output_dir, 1, GD_TOK_SHARD_MAX_BYTES);
    if (status != GD_OK) {
        fprintf(stderr, "open output failed: %s\n", gd_status_string(status));
        (void)fclose(f);
        gd_tokenizer_destroy(tok);
        return status;
    }
    for (;;) {
        char *text = NULL;
        int32_t *ids = NULL;
        int n_ids = 0;
        int keep;
        status = gd_tok_read_line(f, &line, &line_cap, &line_len);
        if (status == GD_TOK_STATUS_EOF) {
            status = GD_OK;
            break;
        }
        if (status != GD_OK) {
            fprintf(stderr, "read jsonl failed at line %" PRIu64 ": %s\n",
                    line_no + 1U,
                    gd_status_string(status));
            break;
        }
        line_no += 1U;
        (void)line_len;
        status = gd_tok_jsonl_extract_string(line, args->jsonl_text_field, &text);
        if (status != GD_OK) {
            fprintf(stderr,
                    "jsonl parse failed at line %" PRIu64 ": missing/string field '%s'\n",
                    line_no,
                    args->jsonl_text_field);
            break;
        }
        status = gd_tokenizer_encode(tok, text, &ids, &n_ids);
        free(text);
        if (status != GD_OK) {
            fprintf(stderr,
                    "encode failed at line %" PRIu64 ": %s\n",
                    line_no,
                    gd_status_string(status));
            break;
        }
        keep = n_ids < body_cap ? n_ids : body_cap;
        status = gd_tok_writer_write_sequence(&writer,
                                              &im_start_id,
                                              1,
                                              ids,
                                              keep,
                                              &im_end_id,
                                              1);
        gd_tokenizer_free(ids);
        if (status != GD_OK) {
            fprintf(stderr,
                    "write sequence failed at line %" PRIu64 ": %s\n",
                    line_no,
                    gd_status_string(status));
            break;
        }
    }
    free(line);
    if (fclose(f) != 0 && status == GD_OK) {
        status = GD_ERR_IO;
    }
    if (status != GD_OK) {
        gd_tok_writer_destroy(&writer);
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_tok_writer_finish_shard(&writer);
    if (status != GD_OK) {
        fprintf(stderr, "finalize shard failed: %s\n", gd_status_string(status));
        gd_tok_writer_destroy(&writer);
        gd_tokenizer_destroy(tok);
        return status;
    }
    manifest.mode = "jsonl";
    manifest.tokenizer_path = tok_path;
    manifest.jsonl_text_field = args->jsonl_text_field;
    manifest.tokenizer_hash = gd_tokenizer_hash(tok);
    manifest.sequence_length = 0U;
    manifest.stride = 0U;
    manifest.max_length = args->max_length;
    manifest.shard_size_limit_bytes = GD_TOK_SHARD_MAX_BYTES;
    manifest.im_start_id = im_start_id;
    manifest.im_end_id = im_end_id;
    status = gd_tok_write_manifest(args->output_dir, &writer, &manifest);
    if (status != GD_OK) {
        fprintf(stderr, "write manifest failed: %s\n", gd_status_string(status));
        gd_tok_writer_destroy(&writer);
        gd_tokenizer_destroy(tok);
        return status;
    }
    printf("{\"mode\":\"jsonl\",\"trained\":%s,\"tokenizer\":\"%s\","
           "\"sequences\":%" PRIu64 ",\"tokens\":%" PRIu64
           ",\"shards\":%zu,\"output_dir\":\"%s\"}\n",
           trained != 0 ? "true" : "false",
           tok_path,
           writer.total_sequences,
           writer.total_tokens,
           writer.n_infos,
           args->output_dir);
    gd_tok_writer_destroy(&writer);
    gd_tokenizer_destroy(tok);
    return GD_OK;
}

static gd_status gd_tok_parse_args(int argc, char **argv, gd_tok_args *args)
{
    int i;
    if (args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(args, 0, sizeof(*args));
    args->jsonl_text_field = "text";
    args->vocab_size = GD_TOK_DEFAULT_VOCAB_SIZE;
    args->min_frequency = GD_TOK_DEFAULT_MIN_FREQUENCY;
    args->seed = GD_TOK_DEFAULT_SEED;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0) {
            usage(stdout);
            exit(0);
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--use-tokenizer") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->use_tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->input_path = argv[++i];
        } else if (strcmp(argv[i], "--input-jsonl") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->input_jsonl_path = argv[++i];
        } else if (strcmp(argv[i], "--jsonl-text-field") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->jsonl_text_field = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->output_path = argv[++i];
        } else if (strcmp(argv[i], "--output-dir") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->output_dir = argv[++i];
        } else if (strcmp(argv[i], "--im-start") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->im_start = argv[++i];
        } else if (strcmp(argv[i], "--im-end") == 0) {
            if (i + 1 >= argc) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->im_end = argv[++i];
        } else if (strcmp(argv[i], "--num-tokens-per-sequence") == 0) {
            if (i + 1 >= argc ||
                gd_tok_parse_u64(argv[++i], 1U, &args->num_tokens_per_sequence) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        } else if (strcmp(argv[i], "--max-length") == 0) {
            if (i + 1 >= argc || gd_tok_parse_u64(argv[++i], 1U, &args->max_length) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        } else if (strcmp(argv[i], "--vocab-size") == 0) {
            if (i + 1 >= argc || gd_tok_parse_int(argv[++i], 1, &args->vocab_size) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        } else if (strcmp(argv[i], "--min-frequency") == 0) {
            if (i + 1 >= argc || gd_tok_parse_int(argv[++i], 1, &args->min_frequency) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        } else if (strcmp(argv[i], "--special") == 0) {
            if (i + 1 >= argc || args->n_specials >= GD_TOK_MAX_SPECIALS) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            args->specials[args->n_specials] = argv[++i];
            args->n_specials += 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc || gd_tok_parse_u64(argv[++i], 0U, &args->seed) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        } else if (strcmp(argv[i], "--no-special") == 0) {
            args->no_special = 1;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_tok_mode gd_tok_select_mode(const gd_tok_args *args, gd_status *status_out)
{
    gd_status status = GD_OK;
    gd_tok_mode mode = GD_TOK_MODE_LEGACY_FLAT;
    if (args == NULL || status_out == NULL) {
        if (status_out != NULL) {
            *status_out = GD_ERR_INVALID_ARGUMENT;
        }
        return GD_TOK_MODE_LEGACY_FLAT;
    }
    if ((args->tokenizer_path == NULL && args->use_tokenizer_path == NULL) ||
        (args->tokenizer_path != NULL && args->use_tokenizer_path != NULL)) {
        status = GD_ERR_INVALID_ARGUMENT;
    } else if (args->input_path != NULL && args->input_jsonl_path != NULL) {
        status = GD_ERR_INVALID_ARGUMENT;
    } else if (args->output_dir != NULL) {
        if (args->output_path != NULL || args->im_start == NULL || args->im_end == NULL) {
            status = GD_ERR_INVALID_ARGUMENT;
        } else if (args->input_jsonl_path != NULL) {
            if (args->input_path != NULL || args->max_length == 0U ||
                args->num_tokens_per_sequence != 0U) {
                status = GD_ERR_INVALID_ARGUMENT;
            } else {
                mode = GD_TOK_MODE_JSONL;
            }
        } else if (args->input_path != NULL) {
            if (args->num_tokens_per_sequence == 0U) {
                status = GD_ERR_INVALID_ARGUMENT;
            } else {
                mode = GD_TOK_MODE_PACKED_TEXT;
            }
        } else {
            status = GD_ERR_INVALID_ARGUMENT;
        }
    } else if (args->output_path != NULL) {
        if (args->input_path == NULL || args->input_jsonl_path != NULL || args->im_start != NULL ||
            args->im_end != NULL || args->max_length != 0U ||
            args->num_tokens_per_sequence != 0U) {
            status = GD_ERR_INVALID_ARGUMENT;
        } else {
            mode = GD_TOK_MODE_LEGACY_FLAT;
        }
    } else {
        status = GD_ERR_INVALID_ARGUMENT;
    }
    *status_out = status;
    return mode;
}

int main(int argc, char **argv)
{
    gd_tok_args args;
    gd_status status;
    gd_tok_mode mode;

    status = gd_tok_parse_args(argc, argv, &args);
    if (status != GD_OK) {
        usage(stderr);
        return 2;
    }
    mode = gd_tok_select_mode(&args, &status);
    if (status != GD_OK) {
        usage(stderr);
        return 2;
    }
    if (mode == GD_TOK_MODE_LEGACY_FLAT) {
        status = gd_tok_run_legacy_flat(&args);
    } else if (mode == GD_TOK_MODE_PACKED_TEXT) {
        status = gd_tok_run_packed_text(&args);
    } else {
        status = gd_tok_run_jsonl(&args);
    }
    return status == GD_OK ? 0 : 1;
}
