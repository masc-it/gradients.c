#include "tokenizer_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct gd_encode_user {
    gd_tokenizer *tok;
    gd_i32_vec *ids;
} gd_encode_user;

static gd_status gd_i32_vec_reserve(gd_i32_vec *vec, int needed)
{
    int new_cap;
    int32_t *new_data;
    if (vec == NULL || needed < 0) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid token vector reserve");
    }
    if (needed <= vec->cap) {
        return GD_OK;
    }
    new_cap = vec->cap == 0 ? 64 : vec->cap;
    while (new_cap < needed) {
        if (new_cap > INT_MAX / 2) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "token vector too large");
        }
        new_cap *= 2;
    }
    if (gd_mul_overflows_size((size_t)new_cap, sizeof(int32_t))) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "token vector too large");
    }
    new_data = (int32_t *)realloc(vec->data, (size_t)new_cap * sizeof(int32_t));
    if (new_data == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "token vector allocation failed");
    }
    vec->data = new_data;
    vec->cap = new_cap;
    return GD_OK;
}

static gd_status gd_i32_vec_push(gd_i32_vec *vec, int32_t value)
{
    gd_status status;
    if (vec->len == vec->cap) {
        status = gd_i32_vec_reserve(vec, vec->len + 1);
        if (status != GD_OK) {
            return status;
        }
    }
    vec->data[vec->len] = value;
    vec->len += 1;
    return GD_OK;
}

static void gd_i32_vec_free(gd_i32_vec *vec)
{
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0;
        vec->cap = 0;
    }
}

static gd_status gd_encode_piece(gd_tokenizer *tok,
                                 const uint8_t *bytes,
                                 size_t len,
                                 gd_i32_vec *out)
{
    int32_t parts[GD_BPE_MAX_PIECE_BYTES];
    int32_t ranks[GD_BPE_MAX_PIECE_BYTES];
    int n_parts;
    int i;
    gd_status status;

    if (len == 0U) {
        return GD_OK;
    }
    {
        int32_t direct = gd_bytes_map_get(&tok->bytes_map, bytes, len);
        if (direct >= 0) {
            return gd_i32_vec_push(out, direct);
        }
    }
    if (len > GD_BPE_MAX_PIECE_BYTES) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "tokenizer piece too large");
    }
    n_parts = (int)len;
    for (i = 0; i < n_parts; ++i) {
        parts[i] = (int32_t)bytes[i];
    }
    for (i = 0; i + 1 < n_parts; ++i) {
        int32_t rank = gd_pair_map_get(&tok->pair_map, parts[i], parts[i + 1]);
        ranks[i] = rank >= 0 ? rank : INT32_MAX;
    }
    ranks[n_parts - 1] = INT32_MAX;
    while (n_parts > 1) {
        int best_index = -1;
        int32_t best_rank = INT32_MAX;
        for (i = 0; i + 1 < n_parts; ++i) {
            if (ranks[i] < best_rank) {
                best_rank = ranks[i];
                best_index = i;
            }
        }
        if (best_index < 0) {
            break;
        }
        parts[best_index] = best_rank;
        if (best_index + 2 < n_parts) {
            memmove(&parts[best_index + 1],
                    &parts[best_index + 2],
                    (size_t)(n_parts - best_index - 2) * sizeof(int32_t));
            memmove(&ranks[best_index + 1],
                    &ranks[best_index + 2],
                    (size_t)(n_parts - best_index - 2) * sizeof(int32_t));
        }
        n_parts -= 1;
        if (best_index > 0) {
            int32_t rank = gd_pair_map_get(&tok->pair_map,
                                           parts[best_index - 1],
                                           parts[best_index]);
            ranks[best_index - 1] = rank >= 0 ? rank : INT32_MAX;
        }
        if (best_index + 1 < n_parts) {
            int32_t rank = gd_pair_map_get(&tok->pair_map,
                                           parts[best_index],
                                           parts[best_index + 1]);
            ranks[best_index] = rank >= 0 ? rank : INT32_MAX;
        } else {
            ranks[best_index] = INT32_MAX;
        }
    }
    for (i = 0; i < n_parts; ++i) {
        status = gd_i32_vec_push(out, parts[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status gd_encode_normal_cb(const uint8_t *bytes, size_t len, void *user)
{
    gd_encode_user *u = (gd_encode_user *)user;
    return gd_encode_piece(u->tok, bytes, len, u->ids);
}

static gd_status gd_encode_special_cb(int32_t id, void *user)
{
    gd_encode_user *u = (gd_encode_user *)user;
    return gd_i32_vec_push(u->ids, id);
}

gd_status gd_tokenizer_encode(gd_tokenizer *tok,
                              const char *text,
                              int32_t **ids_out,
                              int *n_ids_out)
{
    gd_i32_vec ids = {0};
    gd_encode_user user;
    gd_status status;

    if (tok == NULL || text == NULL || ids_out == NULL || n_ids_out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer encode arguments");
    }
    *ids_out = NULL;
    *n_ids_out = 0;
    user.tok = tok;
    user.ids = &ids;
    {
        size_t text_len = strlen(text);
        size_t estimate = text_len / 2U + 64U;
        if (estimate > (size_t)INT_MAX) {
            estimate = (size_t)INT_MAX;
        }
        status = gd_i32_vec_reserve(&ids, (int)estimate);
        if (status != GD_OK) {
            return status;
        }
    }
    status = gd_pretokenize((const uint8_t *)text,
                            strlen(text),
                            (const char **)tok->special_texts,
                            tok->special_ids,
                            tok->n_specials,
                            tok->split_digits,
                            tok->allow_special,
                            gd_encode_normal_cb,
                            gd_encode_special_cb,
                            &user);
    if (status != GD_OK) {
        gd_i32_vec_free(&ids);
        return status;
    }
    *ids_out = ids.data;
    *n_ids_out = ids.len;
    return GD_OK;
}

gd_status gd_tokenizer_decode(gd_tokenizer *tok,
                              const int32_t *ids,
                              int n_ids,
                              char **text_out)
{
    size_t total = 0U;
    char *text;
    size_t offset = 0U;
    int i;

    if (tok == NULL || text_out == NULL || n_ids < 0 || (n_ids > 0 && ids == NULL)) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer decode arguments");
    }
    *text_out = NULL;
    for (i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->n_tokens) {
            return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "token id out of range");
        }
        if (tok->tokens[id].len > SIZE_MAX - total - 1U) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "decoded text too large");
        }
        total += tok->tokens[id].len;
    }
    text = (char *)gd_xmalloc(total + 1U);
    if (text == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "decoded text allocation failed");
    }
    for (i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        memcpy(&text[offset], tok->tokens[id].bytes, tok->tokens[id].len);
        offset += tok->tokens[id].len;
    }
    text[offset] = '\0';
    *text_out = text;
    return GD_OK;
}
