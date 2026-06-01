#include "tokenizer_internal.h"

#include "../core/internal.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void gd_json_write_string(FILE *f, const char *s)
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

static void gd_write_hex(FILE *f, const uint8_t *bytes, size_t len)
{
    static const char hex[] = "0123456789abcdef";
    size_t i;
    for (i = 0U; i < len; ++i) {
        fputc(hex[bytes[i] >> 4U], f);
        fputc(hex[bytes[i] & 15U], f);
    }
}

gd_status gd_bpe_tokenizer_save(gd_tokenizer *tok, const char *tokenizer_path)
{
    FILE *f;
    int i;

    if (tok == NULL || tokenizer_path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer save arguments");
    }
    f = fopen(tokenizer_path, "wb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open tokenizer output");
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"gd-bpe-tokenizer-v1\",\n");
    fprintf(f, "  \"split_digits\": %s,\n", tok->split_digits != 0 ? "true" : "false");
    fprintf(f, "  \"vocab_size\": %d,\n", tok->n_tokens);
    fprintf(f, "  \"n_special_tokens\": %d,\n", tok->n_specials);
    fprintf(f, "  \"hash\": \"%016" PRIx64 "\",\n", tok->hash);
    fprintf(f, "  \"tokens\": [\n");
    for (i = 0; i < tok->n_tokens; ++i) {
        const gd_bpe_token *t = &tok->tokens[i];
        fprintf(f, "    {\"id\":%d,", i);
        if (t->is_special != 0) {
            fprintf(f, "\"kind\":\"special\",\"text\":");
            gd_json_write_string(f, (const char *)t->bytes);
        } else if (t->left >= 0 && t->right >= 0) {
            fprintf(f, "\"kind\":\"merge\",\"left\":%d,\"right\":%d,\"hex\":\"",
                    t->left,
                    t->right);
            gd_write_hex(f, t->bytes, t->len);
            fprintf(f, "\"");
        } else {
            fprintf(f, "\"kind\":\"byte\",\"hex\":\"%02x\"", (unsigned int)t->bytes[0]);
        }
        fprintf(f, "}%s\n", i + 1 == tok->n_tokens ? "" : ",");
    }
    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close tokenizer output");
    }
    return GD_OK;
}

static const char *gd_find_field(const char *line, const char *field)
{
    return strstr(line, field);
}

static gd_status gd_parse_int_field(const char *line, const char *field, int *out)
{
    const char *p = gd_find_field(line, field);
    char *end = NULL;
    long v;
    if (p == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing tokenizer int field");
    }
    p += strlen(field);
    errno = 0;
    v = strtol(p, &end, 10);
    if (errno != 0 || end == p || v < INT_MIN || v > INT_MAX) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer int field");
    }
    *out = (int)v;
    return GD_OK;
}

static int gd_hex_value(char c)
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

static gd_status gd_parse_hex_field(const char *line,
                                    const char *field,
                                    uint8_t **bytes_out,
                                    size_t *len_out)
{
    const char *p = gd_find_field(line, field);
    const char *q;
    size_t hex_len;
    size_t len;
    uint8_t *bytes;
    size_t i;

    if (p == NULL || bytes_out == NULL || len_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing tokenizer hex field");
    }
    *bytes_out = NULL;
    *len_out = 0U;
    p += strlen(field);
    q = strchr(p, '"');
    if (q == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unterminated tokenizer hex field");
    }
    hex_len = (size_t)(q - p);
    if ((hex_len % 2U) != 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer hex length");
    }
    len = hex_len / 2U;
    bytes = (uint8_t *)gd_xmalloc(len);
    if (bytes == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer hex allocation failed");
    }
    for (i = 0U; i < len; ++i) {
        int hi = gd_hex_value(p[2U * i]);
        int lo = gd_hex_value(p[2U * i + 1U]);
        if (hi < 0 || lo < 0) {
            free(bytes);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer hex byte");
        }
        bytes[i] = (uint8_t)((hi << 4) | lo);
    }
    *bytes_out = bytes;
    *len_out = len;
    return GD_OK;
}

static gd_status gd_parse_json_text_field(const char *line,
                                          const char *field,
                                          char **text_out)
{
    const char *p = gd_find_field(line, field);
    char *text;
    size_t cap = 64U;
    size_t len = 0U;

    if (p == NULL || text_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing tokenizer text field");
    }
    *text_out = NULL;
    p += strlen(field);
    text = (char *)gd_xmalloc(cap);
    if (text == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer text allocation failed");
    }
    while (*p != '\0' && *p != '"') {
        char c = *p;
        if (c == '\\') {
            ++p;
            if (*p == '\0') {
                free(text);
                return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer escape");
            }
            if (*p == 'n') {
                c = '\n';
            } else if (*p == 'r') {
                c = '\r';
            } else if (*p == 't') {
                c = '\t';
            } else if (*p == '"' || *p == '\\') {
                c = *p;
            } else {
                free(text);
                return _gd_error(GD_ERR_UNSUPPORTED, "unsupported tokenizer JSON escape");
            }
        }
        if (len + 1U >= cap) {
            char *new_text;
            if (cap > SIZE_MAX / 2U) {
                free(text);
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer text too large");
            }
            cap *= 2U;
            new_text = (char *)realloc(text, cap);
            if (new_text == NULL) {
                free(text);
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer text allocation failed");
            }
            text = new_text;
        }
        text[len] = c;
        len += 1U;
        ++p;
    }
    if (*p != '"') {
        free(text);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unterminated tokenizer text");
    }
    text[len] = '\0';
    *text_out = text;
    return GD_OK;
}

gd_status gd_bpe_tokenizer_load(const char *tokenizer_path,
                                const gd_tokenizer_config *cfg,
                                gd_tokenizer **out)
{
    FILE *f;
    gd_tokenizer *tok = NULL;
    char line[4096];
    int split_digits = 1;
    int allow_special = 1;
    gd_status status;

    if (tokenizer_path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer load arguments");
    }
    *out = NULL;
    if (cfg != NULL) {
        split_digits = cfg->split_digits;
        allow_special = cfg->allow_special;
    }
    f = fopen(tokenizer_path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open tokenizer file");
    }
    status = gd_tokenizer_create_empty(split_digits, allow_special, &tok);
    if (status != GD_OK) {
        fclose(f);
        return status;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strstr(line, "\"split_digits\": true") != NULL && cfg == NULL) {
            tok->split_digits = 1;
        } else if (strstr(line, "\"split_digits\": false") != NULL && cfg == NULL) {
            tok->split_digits = 0;
        }
        if (strstr(line, "{\"id\":") != NULL) {
            int id = -1;
            int left = -1;
            int right = -1;
            uint8_t *bytes = NULL;
            size_t len = 0U;
            char *text = NULL;
            status = gd_parse_int_field(line, "\"id\":", &id);
            if (status != GD_OK) {
                gd_tokenizer_destroy(tok);
                fclose(f);
                return status;
            }
            if (id != tok->n_tokens) {
                gd_tokenizer_destroy(tok);
                fclose(f);
                return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer ids must be contiguous");
            }
            if (strstr(line, "\"kind\":\"special\"") != NULL) {
                status = gd_parse_json_text_field(line, "\"text\":\"", &text);
                if (status != GD_OK) {
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
                status = gd_tokenizer_add_token(tok,
                                                (const uint8_t *)text,
                                                strlen(text),
                                                1,
                                                -1,
                                                -1,
                                                NULL);
                free(text);
                if (status != GD_OK) {
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
            } else if (strstr(line, "\"kind\":\"merge\"") != NULL) {
                status = gd_parse_int_field(line, "\"left\":", &left);
                if (status == GD_OK) {
                    status = gd_parse_int_field(line, "\"right\":", &right);
                }
                if (status == GD_OK) {
                    status = gd_parse_hex_field(line, "\"hex\":\"", &bytes, &len);
                }
                if (status != GD_OK) {
                    free(bytes);
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
                status = gd_tokenizer_add_token(tok, bytes, len, 0, (int32_t)left,
                                                (int32_t)right, NULL);
                free(bytes);
                if (status != GD_OK) {
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
            } else if (strstr(line, "\"kind\":\"byte\"") != NULL) {
                status = gd_parse_hex_field(line, "\"hex\":\"", &bytes, &len);
                if (status != GD_OK) {
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
                if (len != 1U) {
                    free(bytes);
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return _gd_error(GD_ERR_INVALID_ARGUMENT, "byte token must have len 1");
                }
                status = gd_tokenizer_add_token(tok, bytes, len, 0, -1, -1, NULL);
                free(bytes);
                if (status != GD_OK) {
                    gd_tokenizer_destroy(tok);
                    fclose(f);
                    return status;
                }
            }
        }
    }
    if (ferror(f) != 0) {
        gd_tokenizer_destroy(tok);
        fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read tokenizer file");
    }
    if (fclose(f) != 0) {
        gd_tokenizer_destroy(tok);
        return _gd_error(GD_ERR_IO, "failed to close tokenizer file");
    }
    if (tok->n_tokens < GD_BPE_N_BYTES) {
        gd_tokenizer_destroy(tok);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer missing byte vocabulary");
    }
    gd_tokenizer_recompute_hash(tok);
    *out = tok;
    return GD_OK;
}
