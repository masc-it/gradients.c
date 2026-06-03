#include "gradients/tokenizer.h"

/*
 * Native byte-level BPE tokenizer for gradients.c.
 * Encode path follows tiktoken's rank-ordered byte-pair merge design
 * (data/tiktoken/src/lib.rs) while keeping runtime dependency-free C.
 */

#include "../core/internal.h"
#include "tokenizer_internal.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t gd_hash_mix_u64(uint64_t h, uint64_t v)
{
    h ^= v;
    h *= GD_FNV_PRIME;
    return h;
}

uint64_t gd_hash_bytes(const uint8_t *bytes, size_t len)
{
    uint64_t h = GD_FNV_OFFSET;
    size_t i;
    for (i = 0; i < len; ++i) {
        h ^= (uint64_t)bytes[i];
        h *= GD_FNV_PRIME;
    }
    h = gd_hash_mix_u64(h, (uint64_t)len);
    return h;
}

uint64_t gd_pair_hash(int32_t left, int32_t right)
{
    uint64_t x = (uint64_t)(uint32_t)left;
    uint64_t y = (uint64_t)(uint32_t)right;
    uint64_t h = GD_FNV_OFFSET;
    h = gd_hash_mix_u64(h, x + UINT64_C(0x9e3779b97f4a7c15));
    h = gd_hash_mix_u64(h, y + UINT64_C(0xbf58476d1ce4e5b9));
    return h;
}

bool gd_mul_overflows_size(size_t a, size_t b)
{
    return b != 0U && a > SIZE_MAX / b;
}

size_t gd_next_pow2(size_t n)
{
    size_t p = 16U;
    while (p < n && p <= (SIZE_MAX / 2U)) {
        p *= 2U;
    }
    return p;
}

void *gd_xmalloc(size_t n)
{
    if (n == 0U) {
        n = 1U;
    }
    return malloc(n);
}

gd_status gd_memdup_bytes(const uint8_t *src, size_t len, uint8_t **out)
{
    uint8_t *dst;
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer memdup output is null");
    }
    *out = NULL;
    dst = (uint8_t *)gd_xmalloc(len);
    if (dst == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer allocation failed");
    }
    if (len > 0U) {
        memcpy(dst, src, len);
    }
    *out = dst;
    return GD_OK;
}

gd_status gd_memdup_cstr(const uint8_t *src, size_t len, char **out)
{
    char *dst;
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer string output is null");
    }
    *out = NULL;
    if (len >= SIZE_MAX) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer string too large");
    }
    dst = (char *)gd_xmalloc(len + 1U);
    if (dst == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer string allocation failed");
    }
    if (len > 0U) {
        memcpy(dst, src, len);
    }
    dst[len] = '\0';
    *out = dst;
    return GD_OK;
}

static int gd_ascii_is_digit(uint8_t c)
{
    return c >= (uint8_t)'0' && c <= (uint8_t)'9';
}

static int gd_ascii_is_space(uint8_t c)
{
    return c == (uint8_t)' ' || c == (uint8_t)'\n' || c == (uint8_t)'\r' ||
           c == (uint8_t)'\t' || c == (uint8_t)'\v' || c == (uint8_t)'\f';
}

static int gd_ascii_is_alpha(uint8_t c)
{
    return (c >= (uint8_t)'A' && c <= (uint8_t)'Z') ||
           (c >= (uint8_t)'a' && c <= (uint8_t)'z') || c >= 0x80U;
}

typedef enum gd_pretok_class {
    GD_PRETOK_SPACE = 0,
    GD_PRETOK_DIGIT = 1,
    GD_PRETOK_ALPHA = 2,
    GD_PRETOK_OTHER = 3
} gd_pretok_class;

static gd_pretok_class gd_pretok_classify(uint8_t c)
{
    if (gd_ascii_is_space(c) != 0) {
        return GD_PRETOK_SPACE;
    }
    if (gd_ascii_is_digit(c) != 0) {
        return GD_PRETOK_DIGIT;
    }
    if (gd_ascii_is_alpha(c) != 0) {
        return GD_PRETOK_ALPHA;
    }
    return GD_PRETOK_OTHER;
}

void gd_tokenizer_recompute_hash(gd_tokenizer *tok)
{
    uint64_t h = GD_FNV_OFFSET;
    int i;
    if (tok == NULL) {
        return;
    }
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)tok->split_digits);
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)tok->n_tokens);
    for (i = 0; i < tok->n_tokens; ++i) {
        const gd_bpe_token *t = &tok->tokens[i];
        size_t j;
        h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)i);
        h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)t->is_special);
        h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)t->left);
        h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)t->right);
        h = gd_hash_mix_u64(h, (uint64_t)t->len);
        for (j = 0U; j < t->len; ++j) {
            h ^= (uint64_t)t->bytes[j];
            h *= GD_FNV_PRIME;
        }
    }
    tok->hash = h;
}

gd_status gd_tokenizer_create_empty(int split_digits,
                                           int allow_special,
                                           gd_tokenizer **out)
{
    gd_tokenizer *tok;
    gd_status status;
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer output is null");
    }
    *out = NULL;
    tok = (gd_tokenizer *)calloc(1U, sizeof(gd_tokenizer));
    if (tok == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer allocation failed");
    }
    tok->split_digits = split_digits != 0 ? 1 : 0;
    tok->allow_special = allow_special != 0 ? 1 : 0;
    status = gd_pair_map_init(&tok->pair_map, 1024U);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_bytes_map_init(&tok->bytes_map, 1024U);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    *out = tok;
    return GD_OK;
}

static gd_status gd_tokenizer_reserve_tokens(gd_tokenizer *tok, int needed)
{
    int new_cap;
    gd_bpe_token *new_tokens;
    if (needed <= tok->token_cap) {
        return GD_OK;
    }
    new_cap = tok->token_cap == 0 ? 512 : tok->token_cap;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer vocab too large");
        }
        new_cap *= 2;
    }
    if (gd_mul_overflows_size((size_t)new_cap, sizeof(gd_bpe_token))) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer vocab allocation overflow");
    }
    new_tokens = (gd_bpe_token *)realloc(tok->tokens, (size_t)new_cap * sizeof(gd_bpe_token));
    if (new_tokens == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer vocab allocation failed");
    }
    memset(&new_tokens[tok->token_cap], 0,
           (size_t)(new_cap - tok->token_cap) * sizeof(gd_bpe_token));
    tok->tokens = new_tokens;
    tok->token_cap = new_cap;
    return GD_OK;
}

static gd_status gd_tokenizer_reserve_specials(gd_tokenizer *tok, int needed)
{
    int new_cap;
    char **new_texts;
    int32_t *new_ids;
    if (needed <= tok->special_cap) {
        return GD_OK;
    }
    new_cap = tok->special_cap == 0 ? 8 : tok->special_cap;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "too many special tokens");
        }
        new_cap *= 2;
    }
    if (gd_mul_overflows_size((size_t)new_cap, sizeof(char *)) ||
        gd_mul_overflows_size((size_t)new_cap, sizeof(int32_t))) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "special token allocation overflow");
    }
    new_texts = (char **)realloc(tok->special_texts, (size_t)new_cap * sizeof(char *));
    if (new_texts == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "special token allocation failed");
    }
    tok->special_texts = new_texts;
    new_ids = (int32_t *)realloc(tok->special_ids, (size_t)new_cap * sizeof(int32_t));
    if (new_ids == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "special id allocation failed");
    }
    tok->special_ids = new_ids;
    tok->special_cap = new_cap;
    return GD_OK;
}

gd_status gd_tokenizer_add_token(gd_tokenizer *tok,
                                        const uint8_t *bytes,
                                        size_t len,
                                        int is_special,
                                        int32_t left,
                                        int32_t right,
                                        int32_t *id_out)
{
    int32_t id;
    gd_status status;

    if (tok == NULL || bytes == NULL || len == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer token");
    }
    if (tok->n_tokens == INT32_MAX) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer vocab exceeds int32");
    }
    status = gd_tokenizer_reserve_tokens(tok, tok->n_tokens + 1);
    if (status != GD_OK) {
        return status;
    }
    id = (int32_t)tok->n_tokens;
    tok->tokens[tok->n_tokens].bytes = NULL;
    status = gd_memdup_bytes(bytes, len, &tok->tokens[tok->n_tokens].bytes);
    if (status != GD_OK) {
        return status;
    }
    tok->tokens[tok->n_tokens].len = len;
    tok->tokens[tok->n_tokens].is_special = is_special != 0 ? 1 : 0;
    tok->tokens[tok->n_tokens].left = left;
    tok->tokens[tok->n_tokens].right = right;
    tok->n_tokens += 1;

    if (is_special != 0) {
        char *text = NULL;
        if (tok->n_specials == GD_BPE_MAX_SPECIALS) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "too many special tokens");
        }
        status = gd_tokenizer_reserve_specials(tok, tok->n_specials + 1);
        if (status != GD_OK) {
            return status;
        }
        status = gd_memdup_cstr(bytes, len, &text);
        if (status != GD_OK) {
            return status;
        }
        tok->special_texts[tok->n_specials] = text;
        tok->special_ids[tok->n_specials] = id;
        tok->n_specials += 1;
    } else {
        status = gd_bytes_map_insert(&tok->bytes_map, tok->tokens[id].bytes, len, id);
        if (status != GD_OK) {
            return status;
        }
        if (left >= 0 && right >= 0) {
            status = gd_pair_map_insert(&tok->pair_map, left, right, id);
            if (status != GD_OK) {
                return status;
            }
        }
    }
    if (id_out != NULL) {
        *id_out = id;
    }
    return GD_OK;
}

gd_status gd_tokenizer_add_byte_tokens(gd_tokenizer *tok)
{
    int i;
    gd_status status;
    for (i = 0; i < GD_BPE_N_BYTES; ++i) {
        uint8_t b = (uint8_t)i;
        status = gd_tokenizer_add_token(tok, &b, 1U, 0, -1, -1, NULL);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static size_t gd_match_special(const char **specials,
                               const int32_t *ids,
                               int n_specials,
                               const uint8_t *data,
                               size_t len,
                               size_t pos,
                               int32_t *id_out)
{
    size_t best_len = 0U;
    int32_t best_id = -1;
    int i;
    for (i = 0; i < n_specials; ++i) {
        size_t slen;
        if (specials[i] == NULL) {
            continue;
        }
        slen = strlen(specials[i]);
        if (slen > best_len && pos + slen <= len &&
            memcmp(&data[pos], specials[i], slen) == 0) {
            best_len = slen;
            best_id = ids != NULL ? ids[i] : (int32_t)i;
        }
    }
    if (best_len > 0U && id_out != NULL) {
        *id_out = best_id;
    }
    return best_len;
}

static gd_status gd_emit_normal_piece(const uint8_t *bytes,
                                      size_t len,
                                      gd_normal_piece_cb normal_cb,
                                      void *user)
{
    size_t offset = 0U;
    gd_status status;
    while (offset < len) {
        size_t chunk = len - offset;
        if (chunk > GD_BPE_MAX_PIECE_BYTES) {
            chunk = GD_BPE_MAX_PIECE_BYTES;
        }
        status = normal_cb(&bytes[offset], chunk, user);
        if (status != GD_OK) {
            return status;
        }
        offset += chunk;
    }
    return GD_OK;
}

gd_status gd_pretokenize(const uint8_t *data,
                                size_t len,
                                const char **specials,
                                const int32_t *special_ids,
                                int n_specials,
                                int split_digits,
                                int allow_special,
                                gd_normal_piece_cb normal_cb,
                                gd_special_piece_cb special_cb,
                                void *user)
{
    size_t i = 0U;
    if (data == NULL || normal_cb == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid pretokenizer input");
    }
    while (i < len) {
        size_t match_len = 0U;
        int32_t special_id = -1;
        if (allow_special != 0) {
            match_len = gd_match_special(specials, special_ids, n_specials, data, len, i,
                                         &special_id);
        }
        if (match_len > 0U) {
            if (special_cb != NULL) {
                gd_status status = special_cb(special_id, user);
                if (status != GD_OK) {
                    return status;
                }
            }
            i += match_len;
            continue;
        }
        if (split_digits != 0 && gd_ascii_is_digit(data[i]) != 0) {
            gd_status status = normal_cb(&data[i], 1U, user);
            if (status != GD_OK) {
                return status;
            }
            i += 1U;
            continue;
        }
        if (gd_ascii_is_space(data[i]) != 0) {
            size_t start = i;
            if (data[i] == (uint8_t)' ' && i + 1U < len &&
                gd_ascii_is_space(data[i + 1U]) == 0 &&
                !(split_digits != 0 && gd_ascii_is_digit(data[i + 1U]) != 0) &&
                (allow_special == 0 || gd_match_special(specials,
                                                         special_ids,
                                                         n_specials,
                                                         data,
                                                         len,
                                                         i + 1U,
                                                         NULL) == 0U)) {
                gd_pretok_class cls;
                i += 1U;
                cls = gd_pretok_classify(data[i]);
                while (i < len && gd_pretok_classify(data[i]) == cls &&
                       !(split_digits != 0 && gd_ascii_is_digit(data[i]) != 0) &&
                       (allow_special == 0 || gd_match_special(specials,
                                                               special_ids,
                                                               n_specials,
                                                               data,
                                                               len,
                                                               i,
                                                               NULL) == 0U)) {
                    i += 1U;
                }
            } else {
                i += 1U;
                while (i < len && gd_ascii_is_space(data[i]) != 0 &&
                       (allow_special == 0 || gd_match_special(specials,
                                                               special_ids,
                                                               n_specials,
                                                               data,
                                                               len,
                                                               i,
                                                               NULL) == 0U)) {
                    i += 1U;
                }
            }
            {
                gd_status status = gd_emit_normal_piece(&data[start], i - start, normal_cb, user);
                if (status != GD_OK) {
                    return status;
                }
            }
            continue;
        }
        {
            size_t start = i;
            gd_pretok_class cls = gd_pretok_classify(data[i]);
            i += 1U;
            while (i < len && gd_pretok_classify(data[i]) == cls &&
                   !(split_digits != 0 && gd_ascii_is_digit(data[i]) != 0) &&
                   (allow_special == 0 || gd_match_special(specials,
                                                           special_ids,
                                                           n_specials,
                                                           data,
                                                           len,
                                                           i,
                                                           NULL) == 0U)) {
                i += 1U;
            }
            {
                gd_status status = gd_emit_normal_piece(&data[start], i - start, normal_cb, user);
                if (status != GD_OK) {
                    return status;
                }
            }
        }
    }
    return GD_OK;
}

void gd_tokenizer_destroy(gd_tokenizer *tok)
{
    int i;
    if (tok == NULL) {
        return;
    }
    if (tok->tokens != NULL) {
        for (i = 0; i < tok->n_tokens; ++i) {
            free(tok->tokens[i].bytes);
        }
    }
    for (i = 0; i < tok->n_specials; ++i) {
        free(tok->special_texts[i]);
    }
    free(tok->tokens);
    free(tok->special_texts);
    free(tok->special_ids);
    gd_pair_map_free(&tok->pair_map);
    gd_bytes_map_free(&tok->bytes_map);
    free(tok);
}

gd_status gd_tokenizer_id(gd_tokenizer *tok, const char *special, int32_t *id_out)
{
    int i;
    if (tok == NULL || special == NULL || id_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer id arguments");
    }
    for (i = 0; i < tok->n_specials; ++i) {
        if (strcmp(tok->special_texts[i], special) == 0) {
            *id_out = tok->special_ids[i];
            return GD_OK;
        }
    }
    return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown special token");
}

int gd_tokenizer_vocab_size(const gd_tokenizer *tok)
{
    if (tok == NULL) {
        return 0;
    }
    return tok->n_tokens;
}

uint64_t gd_tokenizer_hash(const gd_tokenizer *tok)
{
    if (tok == NULL) {
        return 0U;
    }
    return tok->hash;
}

void gd_tokenizer_free(void *ptr)
{
    free(ptr);
}
