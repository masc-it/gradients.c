#include "gradients/tokenizer.h"

/*
 * Native byte-level BPE tokenizer for gradients.c.
 * Encode path follows tiktoken's rank-ordered byte-pair merge design
 * (data/tiktoken/src/lib.rs) while keeping runtime dependency-free C.
 */

#include "../core/internal.h"

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

#define GD_BPE_N_BYTES 256
#define GD_BPE_MAX_SPECIALS 64
#define GD_BPE_MAX_PIECE_BYTES 256
#define GD_BPE_HASH_LOAD_NUM 7U
#define GD_BPE_HASH_LOAD_DEN 10U
#define GD_FNV_OFFSET UINT64_C(14695981039346656037)
#define GD_FNV_PRIME UINT64_C(1099511628211)

typedef struct gd_bpe_token {
    uint8_t *bytes;
    size_t len;
    int is_special;
    int32_t left;
    int32_t right;
} gd_bpe_token;

typedef struct gd_pair_entry {
    int used;
    int32_t left;
    int32_t right;
    int32_t id;
} gd_pair_entry;

typedef struct gd_pair_map {
    gd_pair_entry *entries;
    size_t cap;
    size_t size;
} gd_pair_map;

typedef struct gd_bytes_entry {
    int used;
    uint64_t hash;
    const uint8_t *bytes;
    size_t len;
    int32_t id;
} gd_bytes_entry;

typedef struct gd_bytes_map {
    gd_bytes_entry *entries;
    size_t cap;
    size_t size;
} gd_bytes_map;

typedef struct gd_word_map_entry {
    int used;
    uint64_t hash;
    uint8_t *bytes;
    size_t len;
    uint64_t count;
} gd_word_map_entry;

typedef struct gd_word_map {
    gd_word_map_entry *entries;
    size_t cap;
    size_t size;
} gd_word_map;

typedef struct gd_bpe_word {
    int32_t *ids;
    int len;
    uint64_t count;
} gd_bpe_word;

typedef struct gd_pair_stat_entry {
    int used;
    int32_t left;
    int32_t right;
    uint64_t count;
} gd_pair_stat_entry;

typedef struct gd_pair_stats {
    gd_pair_stat_entry *entries;
    size_t cap;
    size_t size;
} gd_pair_stats;

typedef struct gd_i32_vec {
    int32_t *data;
    int len;
    int cap;
} gd_i32_vec;

struct gd_tokenizer {
    int split_digits;
    int allow_special;
    int min_frequency;
    gd_bpe_token *tokens;
    int n_tokens;
    int token_cap;
    char **special_texts;
    int32_t *special_ids;
    int n_specials;
    int special_cap;
    gd_pair_map pair_map;
    gd_bytes_map bytes_map;
    uint64_t hash;
};

static uint64_t gd_hash_mix_u64(uint64_t h, uint64_t v)
{
    h ^= v;
    h *= GD_FNV_PRIME;
    return h;
}

static uint64_t gd_hash_bytes(const uint8_t *bytes, size_t len)
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

static uint64_t gd_pair_hash(int32_t left, int32_t right)
{
    uint64_t x = (uint64_t)(uint32_t)left;
    uint64_t y = (uint64_t)(uint32_t)right;
    uint64_t h = GD_FNV_OFFSET;
    h = gd_hash_mix_u64(h, x + UINT64_C(0x9e3779b97f4a7c15));
    h = gd_hash_mix_u64(h, y + UINT64_C(0xbf58476d1ce4e5b9));
    return h;
}

static bool gd_mul_overflows_size(size_t a, size_t b)
{
    return b != 0U && a > SIZE_MAX / b;
}

static size_t gd_next_pow2(size_t n)
{
    size_t p = 16U;
    while (p < n && p <= (SIZE_MAX / 2U)) {
        p *= 2U;
    }
    return p;
}

static void *gd_xmalloc(size_t n)
{
    if (n == 0U) {
        n = 1U;
    }
    return malloc(n);
}

static gd_status gd_memdup_bytes(const uint8_t *src, size_t len, uint8_t **out)
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

static gd_status gd_memdup_cstr(const uint8_t *src, size_t len, char **out)
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

static gd_status gd_i32_vec_push(gd_i32_vec *vec, int32_t value)
{
    int new_cap;
    int32_t *new_data;
    if (vec->len == vec->cap) {
        new_cap = vec->cap == 0 ? 64 : vec->cap * 2;
        if (new_cap <= vec->cap || gd_mul_overflows_size((size_t)new_cap, sizeof(int32_t))) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "token vector too large");
        }
        new_data = (int32_t *)realloc(vec->data, (size_t)new_cap * sizeof(int32_t));
        if (new_data == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "token vector allocation failed");
        }
        vec->data = new_data;
        vec->cap = new_cap;
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

static void gd_pair_map_free(gd_pair_map *map)
{
    if (map != NULL) {
        free(map->entries);
        map->entries = NULL;
        map->cap = 0U;
        map->size = 0U;
    }
}

static gd_status gd_pair_map_init(gd_pair_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "pair map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_pair_entry *)calloc(cap, sizeof(gd_pair_entry));
    if (map->entries == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "pair map allocation failed");
    }
    map->cap = cap;
    map->size = 0U;
    return GD_OK;
}

static gd_status gd_pair_map_rehash(gd_pair_map *map, size_t requested_cap)
{
    gd_pair_entry *old_entries = map->entries;
    size_t old_cap = map->cap;
    gd_pair_map new_map = {0};
    size_t i;
    gd_status status;

    status = gd_pair_map_init(&new_map, requested_cap);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0U; i < old_cap; ++i) {
        if (old_entries[i].used != 0) {
            uint64_t h = gd_pair_hash(old_entries[i].left, old_entries[i].right);
            size_t mask = new_map.cap - 1U;
            size_t j = (size_t)h & mask;
            while (new_map.entries[j].used != 0) {
                j = (j + 1U) & mask;
            }
            new_map.entries[j] = old_entries[i];
            new_map.size += 1U;
        }
    }
    free(old_entries);
    *map = new_map;
    return GD_OK;
}

static gd_status gd_pair_map_insert(gd_pair_map *map,
                                    int32_t left,
                                    int32_t right,
                                    int32_t id)
{
    uint64_t h;
    size_t mask;
    size_t i;
    gd_status status;

    if ((map->size + 1U) * GD_BPE_HASH_LOAD_DEN > map->cap * GD_BPE_HASH_LOAD_NUM) {
        status = gd_pair_map_rehash(map, map->cap * 2U);
        if (status != GD_OK) {
            return status;
        }
    }

    h = gd_pair_hash(left, right);
    mask = map->cap - 1U;
    i = (size_t)h & mask;
    while (map->entries[i].used != 0) {
        if (map->entries[i].left == left && map->entries[i].right == right) {
            map->entries[i].id = id;
            return GD_OK;
        }
        i = (i + 1U) & mask;
    }
    map->entries[i].used = 1;
    map->entries[i].left = left;
    map->entries[i].right = right;
    map->entries[i].id = id;
    map->size += 1U;
    return GD_OK;
}

static int32_t gd_pair_map_get(const gd_pair_map *map, int32_t left, int32_t right)
{
    uint64_t h;
    size_t mask;
    size_t i;
    if (map == NULL || map->cap == 0U) {
        return -1;
    }
    h = gd_pair_hash(left, right);
    mask = map->cap - 1U;
    i = (size_t)h & mask;
    while (map->entries[i].used != 0) {
        if (map->entries[i].left == left && map->entries[i].right == right) {
            return map->entries[i].id;
        }
        i = (i + 1U) & mask;
    }
    return -1;
}

static void gd_bytes_map_free(gd_bytes_map *map)
{
    if (map != NULL) {
        free(map->entries);
        map->entries = NULL;
        map->cap = 0U;
        map->size = 0U;
    }
}

static gd_status gd_bytes_map_init(gd_bytes_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "bytes map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_bytes_entry *)calloc(cap, sizeof(gd_bytes_entry));
    if (map->entries == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "bytes map allocation failed");
    }
    map->cap = cap;
    map->size = 0U;
    return GD_OK;
}

static gd_status gd_bytes_map_rehash(gd_bytes_map *map, size_t requested_cap)
{
    gd_bytes_entry *old_entries = map->entries;
    size_t old_cap = map->cap;
    gd_bytes_map new_map = {0};
    size_t i;
    gd_status status;

    status = gd_bytes_map_init(&new_map, requested_cap);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0U; i < old_cap; ++i) {
        if (old_entries[i].used != 0) {
            size_t mask = new_map.cap - 1U;
            size_t j = (size_t)old_entries[i].hash & mask;
            while (new_map.entries[j].used != 0) {
                j = (j + 1U) & mask;
            }
            new_map.entries[j] = old_entries[i];
            new_map.size += 1U;
        }
    }
    free(old_entries);
    *map = new_map;
    return GD_OK;
}

static gd_status gd_bytes_map_insert(gd_bytes_map *map,
                                     const uint8_t *bytes,
                                     size_t len,
                                     int32_t id)
{
    uint64_t h;
    size_t mask;
    size_t i;
    gd_status status;

    if ((map->size + 1U) * GD_BPE_HASH_LOAD_DEN > map->cap * GD_BPE_HASH_LOAD_NUM) {
        status = gd_bytes_map_rehash(map, map->cap * 2U);
        if (status != GD_OK) {
            return status;
        }
    }

    h = gd_hash_bytes(bytes, len);
    mask = map->cap - 1U;
    i = (size_t)h & mask;
    while (map->entries[i].used != 0) {
        if (map->entries[i].hash == h && map->entries[i].len == len &&
            memcmp(map->entries[i].bytes, bytes, len) == 0) {
            map->entries[i].id = id;
            return GD_OK;
        }
        i = (i + 1U) & mask;
    }
    map->entries[i].used = 1;
    map->entries[i].hash = h;
    map->entries[i].bytes = bytes;
    map->entries[i].len = len;
    map->entries[i].id = id;
    map->size += 1U;
    return GD_OK;
}

static int32_t gd_bytes_map_get(const gd_bytes_map *map, const uint8_t *bytes, size_t len)
{
    uint64_t h;
    size_t mask;
    size_t i;
    if (map == NULL || map->cap == 0U) {
        return -1;
    }
    h = gd_hash_bytes(bytes, len);
    mask = map->cap - 1U;
    i = (size_t)h & mask;
    while (map->entries[i].used != 0) {
        if (map->entries[i].hash == h && map->entries[i].len == len &&
            memcmp(map->entries[i].bytes, bytes, len) == 0) {
            return map->entries[i].id;
        }
        i = (i + 1U) & mask;
    }
    return -1;
}

static void gd_word_map_free(gd_word_map *map)
{
    size_t i;
    if (map == NULL) {
        return;
    }
    if (map->entries != NULL) {
        for (i = 0U; i < map->cap; ++i) {
            if (map->entries[i].used != 0) {
                free(map->entries[i].bytes);
            }
        }
    }
    free(map->entries);
    map->entries = NULL;
    map->cap = 0U;
    map->size = 0U;
}

static gd_status gd_word_map_init(gd_word_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "word map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_word_map_entry *)calloc(cap, sizeof(gd_word_map_entry));
    if (map->entries == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "word map allocation failed");
    }
    map->cap = cap;
    map->size = 0U;
    return GD_OK;
}

static gd_status gd_word_map_rehash(gd_word_map *map, size_t requested_cap)
{
    gd_word_map_entry *old_entries = map->entries;
    size_t old_cap = map->cap;
    gd_word_map new_map = {0};
    size_t i;
    gd_status status;

    status = gd_word_map_init(&new_map, requested_cap);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0U; i < old_cap; ++i) {
        if (old_entries[i].used != 0) {
            size_t mask = new_map.cap - 1U;
            size_t j = (size_t)old_entries[i].hash & mask;
            while (new_map.entries[j].used != 0) {
                j = (j + 1U) & mask;
            }
            new_map.entries[j] = old_entries[i];
            new_map.size += 1U;
        }
    }
    free(old_entries);
    *map = new_map;
    return GD_OK;
}

static gd_status gd_word_map_add(gd_word_map *map, const uint8_t *bytes, size_t len)
{
    uint64_t h;
    size_t mask;
    size_t i;
    gd_status status;

    if (len == 0U) {
        return GD_OK;
    }
    if ((map->size + 1U) * GD_BPE_HASH_LOAD_DEN > map->cap * GD_BPE_HASH_LOAD_NUM) {
        status = gd_word_map_rehash(map, map->cap * 2U);
        if (status != GD_OK) {
            return status;
        }
    }
    h = gd_hash_bytes(bytes, len);
    mask = map->cap - 1U;
    i = (size_t)h & mask;
    while (map->entries[i].used != 0) {
        if (map->entries[i].hash == h && map->entries[i].len == len &&
            memcmp(map->entries[i].bytes, bytes, len) == 0) {
            map->entries[i].count += 1U;
            return GD_OK;
        }
        i = (i + 1U) & mask;
    }
    map->entries[i].bytes = NULL;
    status = gd_memdup_bytes(bytes, len, &map->entries[i].bytes);
    if (status != GD_OK) {
        return status;
    }
    map->entries[i].used = 1;
    map->entries[i].hash = h;
    map->entries[i].len = len;
    map->entries[i].count = 1U;
    map->size += 1U;
    return GD_OK;
}

static void gd_pair_stats_free(gd_pair_stats *stats)
{
    if (stats != NULL) {
        free(stats->entries);
        stats->entries = NULL;
        stats->cap = 0U;
        stats->size = 0U;
    }
}

static gd_status gd_pair_stats_init(gd_pair_stats *stats, size_t requested_cap)
{
    size_t cap;
    if (stats == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "pair stats is null");
    }
    cap = gd_next_pow2(requested_cap);
    stats->entries = (gd_pair_stat_entry *)calloc(cap, sizeof(gd_pair_stat_entry));
    if (stats->entries == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "pair stats allocation failed");
    }
    stats->cap = cap;
    stats->size = 0U;
    return GD_OK;
}

static gd_status gd_pair_stats_rehash(gd_pair_stats *stats, size_t requested_cap)
{
    gd_pair_stat_entry *old_entries = stats->entries;
    size_t old_cap = stats->cap;
    gd_pair_stats new_stats = {0};
    size_t i;
    gd_status status;

    status = gd_pair_stats_init(&new_stats, requested_cap);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0U; i < old_cap; ++i) {
        if (old_entries[i].used != 0) {
            uint64_t h = gd_pair_hash(old_entries[i].left, old_entries[i].right);
            size_t mask = new_stats.cap - 1U;
            size_t j = (size_t)h & mask;
            while (new_stats.entries[j].used != 0) {
                j = (j + 1U) & mask;
            }
            new_stats.entries[j] = old_entries[i];
            new_stats.size += 1U;
        }
    }
    free(old_entries);
    *stats = new_stats;
    return GD_OK;
}

static gd_status gd_pair_stats_add(gd_pair_stats *stats,
                                   int32_t left,
                                   int32_t right,
                                   uint64_t count)
{
    uint64_t h;
    size_t mask;
    size_t i;
    gd_status status;

    if ((stats->size + 1U) * GD_BPE_HASH_LOAD_DEN > stats->cap * GD_BPE_HASH_LOAD_NUM) {
        status = gd_pair_stats_rehash(stats, stats->cap * 2U);
        if (status != GD_OK) {
            return status;
        }
    }
    h = gd_pair_hash(left, right);
    mask = stats->cap - 1U;
    i = (size_t)h & mask;
    while (stats->entries[i].used != 0) {
        if (stats->entries[i].left == left && stats->entries[i].right == right) {
            stats->entries[i].count += count;
            return GD_OK;
        }
        i = (i + 1U) & mask;
    }
    stats->entries[i].used = 1;
    stats->entries[i].left = left;
    stats->entries[i].right = right;
    stats->entries[i].count = count;
    stats->size += 1U;
    return GD_OK;
}

static void gd_tokenizer_recompute_hash(gd_tokenizer *tok)
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

static gd_status gd_tokenizer_create_empty(int split_digits,
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

static gd_status gd_tokenizer_add_token(gd_tokenizer *tok,
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

static gd_status gd_tokenizer_add_byte_tokens(gd_tokenizer *tok)
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

typedef gd_status (*gd_normal_piece_cb)(const uint8_t *bytes, size_t len, void *user);
typedef gd_status (*gd_special_piece_cb)(int32_t id, void *user);

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

static gd_status gd_pretokenize(const uint8_t *data,
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

static gd_status gd_read_file(const char *path, uint8_t **data_out, size_t *len_out)
{
    FILE *f;
    long end;
    size_t nread;
    uint8_t *data;

    if (path == NULL || data_out == NULL || len_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid file read arguments");
    }
    *data_out = NULL;
    *len_out = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open tokenizer input");
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        fclose(f);
        return _gd_error(GD_ERR_IO, "failed to seek tokenizer input");
    }
    end = ftell(f);
    if (end < 0) {
        fclose(f);
        return _gd_error(GD_ERR_IO, "failed to size tokenizer input");
    }
    if ((unsigned long)end > (unsigned long)SIZE_MAX) {
        fclose(f);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer input too large");
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        fclose(f);
        return _gd_error(GD_ERR_IO, "failed to rewind tokenizer input");
    }
    data = (uint8_t *)gd_xmalloc((size_t)end);
    if (data == NULL) {
        fclose(f);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer input allocation failed");
    }
    nread = fread(data, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(data);
        fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read tokenizer input");
    }
    if (fclose(f) != 0) {
        free(data);
        return _gd_error(GD_ERR_IO, "failed to close tokenizer input");
    }
    *data_out = data;
    *len_out = (size_t)end;
    return GD_OK;
}

typedef struct gd_collect_user {
    gd_word_map *words;
} gd_collect_user;

static gd_status gd_collect_piece_cb(const uint8_t *bytes, size_t len, void *user)
{
    gd_collect_user *u = (gd_collect_user *)user;
    return gd_word_map_add(u->words, bytes, len);
}

static gd_status gd_build_words_from_map(const gd_word_map *map,
                                         gd_bpe_word **words_out,
                                         int *n_words_out)
{
    gd_bpe_word *words;
    int n_words = 0;
    size_t i;

    if (map == NULL || words_out == NULL || n_words_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid word map conversion");
    }
    *words_out = NULL;
    *n_words_out = 0;
    if (map->size > (size_t)INT_MAX) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "too many unique tokenizer pieces");
    }
    if (gd_mul_overflows_size(map->size, sizeof(gd_bpe_word))) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "word array allocation overflow");
    }
    words = (gd_bpe_word *)calloc(map->size == 0U ? 1U : map->size, sizeof(gd_bpe_word));
    if (words == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "word array allocation failed");
    }
    for (i = 0U; i < map->cap; ++i) {
        const gd_word_map_entry *entry = &map->entries[i];
        size_t j;
        if (entry->used == 0) {
            continue;
        }
        if (entry->len > (size_t)INT_MAX) {
            free(words);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "tokenizer piece too large");
        }
        words[n_words].ids = (int32_t *)gd_xmalloc(entry->len * sizeof(int32_t));
        if (words[n_words].ids == NULL) {
            int k;
            for (k = 0; k < n_words; ++k) {
                free(words[k].ids);
            }
            free(words);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "word token allocation failed");
        }
        for (j = 0U; j < entry->len; ++j) {
            words[n_words].ids[j] = (int32_t)entry->bytes[j];
        }
        words[n_words].len = (int)entry->len;
        words[n_words].count = entry->count;
        n_words += 1;
    }
    *words_out = words;
    *n_words_out = n_words;
    return GD_OK;
}

static void gd_bpe_words_free(gd_bpe_word *words, int n_words)
{
    int i;
    if (words == NULL) {
        return;
    }
    for (i = 0; i < n_words; ++i) {
        free(words[i].ids);
    }
    free(words);
}

static gd_status gd_count_word_pairs(const gd_bpe_word *words,
                                     int n_words,
                                     gd_pair_stats *stats)
{
    int i;
    gd_status status;
    status = gd_pair_stats_init(stats, 4096U);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < n_words; ++i) {
        int j;
        const gd_bpe_word *word = &words[i];
        if (word->len < 2) {
            continue;
        }
        for (j = 0; j + 1 < word->len; ++j) {
            status = gd_pair_stats_add(stats, word->ids[j], word->ids[j + 1], word->count);
            if (status != GD_OK) {
                gd_pair_stats_free(stats);
                return status;
            }
        }
    }
    return GD_OK;
}

static uint64_t gd_pair_tie_key(uint64_t seed, int32_t left, int32_t right)
{
    uint64_t h = GD_FNV_OFFSET;
    h = gd_hash_mix_u64(h, seed);
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)left);
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)right);
    return h;
}

static int gd_choose_best_pair(const gd_pair_stats *stats,
                               uint64_t min_frequency,
                               uint64_t seed,
                               int32_t *left_out,
                               int32_t *right_out,
                               uint64_t *count_out)
{
    uint64_t best_count = 0U;
    uint64_t best_tie = UINT64_MAX;
    int32_t best_left = INT32_MAX;
    int32_t best_right = INT32_MAX;
    size_t i;

    for (i = 0U; i < stats->cap; ++i) {
        const gd_pair_stat_entry *entry = &stats->entries[i];
        uint64_t tie;
        if (entry->used == 0) {
            continue;
        }
        tie = gd_pair_tie_key(seed, entry->left, entry->right);
        if (entry->count > best_count ||
            (entry->count == best_count &&
             (tie < best_tie ||
              (tie == best_tie &&
               (entry->left < best_left ||
                (entry->left == best_left && entry->right < best_right)))))) {
            best_count = entry->count;
            best_tie = tie;
            best_left = entry->left;
            best_right = entry->right;
        }
    }
    if (best_count < min_frequency || best_count == 0U) {
        return 0;
    }
    *left_out = best_left;
    *right_out = best_right;
    *count_out = best_count;
    return 1;
}

static void gd_merge_words(gd_bpe_word *words,
                           int n_words,
                           int32_t left,
                           int32_t right,
                           int32_t merged)
{
    int i;
    for (i = 0; i < n_words; ++i) {
        gd_bpe_word *word = &words[i];
        int read = 0;
        int write = 0;
        while (read < word->len) {
            if (read + 1 < word->len && word->ids[read] == left &&
                word->ids[read + 1] == right) {
                word->ids[write] = merged;
                write += 1;
                read += 2;
            } else {
                word->ids[write] = word->ids[read];
                write += 1;
                read += 1;
            }
        }
        word->len = write;
    }
}

static gd_status gd_add_special_tokens(gd_tokenizer *tok,
                                       const char **special_tokens,
                                       int n_special_tokens)
{
    int i;
    gd_status status;
    for (i = 0; i < n_special_tokens; ++i) {
        int j;
        const char *s = special_tokens[i];
        if (s == NULL || s[0] == '\0') {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid special token");
        }
        for (j = 0; j < i; ++j) {
            if (strcmp(s, special_tokens[j]) == 0) {
                return _gd_error(GD_ERR_INVALID_ARGUMENT, "duplicate special token");
            }
        }
        status = gd_tokenizer_add_token(tok, (const uint8_t *)s, strlen(s), 1, -1, -1, NULL);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status gd_bpe_tokenizer_train(const char **input_paths,
                                 int n_input_paths,
                                 const gd_bpe_train_config *cfg,
                                 gd_tokenizer **out)
{
    gd_tokenizer *tok = NULL;
    gd_word_map word_map = {0};
    gd_bpe_word *words = NULL;
    int n_words = 0;
    int target_vocab;
    int min_frequency;
    int i;
    gd_status status;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer output is null");
    }
    *out = NULL;
    if (input_paths == NULL || n_input_paths <= 0 || cfg == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE train arguments");
    }
    if (cfg->n_special_tokens < 0 || cfg->n_special_tokens > GD_BPE_MAX_SPECIALS ||
        (cfg->n_special_tokens > 0 && cfg->special_tokens == NULL)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid special token config");
    }
    if (cfg->vocab_size < GD_BPE_N_BYTES + cfg->n_special_tokens) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "BPE vocab_size too small");
    }
    target_vocab = cfg->vocab_size;
    min_frequency = cfg->min_frequency <= 0 ? 2 : cfg->min_frequency;

    status = gd_tokenizer_create_empty(cfg->split_digits, 1, &tok);
    if (status != GD_OK) {
        return status;
    }
    tok->min_frequency = min_frequency;
    status = gd_tokenizer_add_byte_tokens(tok);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_add_special_tokens(tok, cfg->special_tokens, cfg->n_special_tokens);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_word_map_init(&word_map, 4096U);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }

    for (i = 0; i < n_input_paths; ++i) {
        uint8_t *data = NULL;
        size_t len = 0U;
        gd_collect_user user;
        if (input_paths[i] == NULL) {
            gd_word_map_free(&word_map);
            gd_tokenizer_destroy(tok);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "null tokenizer input path");
        }
        status = gd_read_file(input_paths[i], &data, &len);
        if (status != GD_OK) {
            gd_word_map_free(&word_map);
            gd_tokenizer_destroy(tok);
            return status;
        }
        user.words = &word_map;
        status = gd_pretokenize(data,
                                len,
                                cfg->special_tokens,
                                NULL,
                                cfg->n_special_tokens,
                                cfg->split_digits,
                                1,
                                gd_collect_piece_cb,
                                NULL,
                                &user);
        free(data);
        if (status != GD_OK) {
            gd_word_map_free(&word_map);
            gd_tokenizer_destroy(tok);
            return status;
        }
    }

    status = gd_build_words_from_map(&word_map, &words, &n_words);
    gd_word_map_free(&word_map);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }

    while (tok->n_tokens < target_vocab) {
        gd_pair_stats stats = {0};
        int32_t left = -1;
        int32_t right = -1;
        uint64_t count = 0U;
        const gd_bpe_token *lt;
        const gd_bpe_token *rt;
        uint8_t merged_bytes[GD_BPE_MAX_PIECE_BYTES * 2U];
        size_t merged_len;
        int32_t existing;
        int32_t merged_id = -1;

        status = gd_count_word_pairs(words, n_words, &stats);
        if (status != GD_OK) {
            gd_bpe_words_free(words, n_words);
            gd_tokenizer_destroy(tok);
            return status;
        }
        if (gd_choose_best_pair(&stats,
                                (uint64_t)min_frequency,
                                cfg->seed,
                                &left,
                                &right,
                                &count) == 0) {
            gd_pair_stats_free(&stats);
            break;
        }
        gd_pair_stats_free(&stats);

        if (left < 0 || right < 0 || left >= tok->n_tokens || right >= tok->n_tokens) {
            gd_bpe_words_free(words, n_words);
            gd_tokenizer_destroy(tok);
            return _gd_error(GD_ERR_INTERNAL, "BPE trainer selected invalid pair");
        }
        lt = &tok->tokens[left];
        rt = &tok->tokens[right];
        if (lt->len + rt->len > sizeof(merged_bytes)) {
            gd_bpe_words_free(words, n_words);
            gd_tokenizer_destroy(tok);
            return _gd_error(GD_ERR_INTERNAL, "BPE merge exceeded piece limit");
        }
        merged_len = lt->len + rt->len;
        memcpy(merged_bytes, lt->bytes, lt->len);
        memcpy(&merged_bytes[lt->len], rt->bytes, rt->len);
        existing = gd_bytes_map_get(&tok->bytes_map, merged_bytes, merged_len);
        if (existing >= 0) {
            gd_merge_words(words, n_words, left, right, existing);
            continue;
        }
        status = gd_tokenizer_add_token(tok, merged_bytes, merged_len, 0, left, right,
                                        &merged_id);
        if (status != GD_OK) {
            gd_bpe_words_free(words, n_words);
            gd_tokenizer_destroy(tok);
            return status;
        }
        (void)count;
        gd_merge_words(words, n_words, left, right, merged_id);
    }

    gd_bpe_words_free(words, n_words);
    gd_tokenizer_recompute_hash(tok);
    *out = tok;
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

static gd_status gd_encode_piece(gd_tokenizer *tok,
                                 const uint8_t *bytes,
                                 size_t len,
                                 gd_i32_vec *out)
{
    int32_t parts[GD_BPE_MAX_PIECE_BYTES];
    int n_parts;
    int i;
    gd_status status;

    if (len == 0U) {
        return GD_OK;
    }
    if (len > GD_BPE_MAX_PIECE_BYTES) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tokenizer piece too large");
    }
    n_parts = (int)len;
    for (i = 0; i < n_parts; ++i) {
        parts[i] = (int32_t)bytes[i];
    }
    while (n_parts > 1) {
        int best_index = -1;
        int32_t best_rank = INT32_MAX;
        for (i = 0; i + 1 < n_parts; ++i) {
            int32_t rank = gd_pair_map_get(&tok->pair_map, parts[i], parts[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
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
        }
        n_parts -= 1;
    }
    for (i = 0; i < n_parts; ++i) {
        status = gd_i32_vec_push(out, parts[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

typedef struct gd_encode_user {
    gd_tokenizer *tok;
    gd_i32_vec *ids;
} gd_encode_user;

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
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer encode arguments");
    }
    *ids_out = NULL;
    *n_ids_out = 0;
    user.tok = tok;
    user.ids = &ids;
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
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid tokenizer decode arguments");
    }
    *text_out = NULL;
    for (i = 0; i < n_ids; ++i) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->n_tokens) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "token id out of range");
        }
        if (tok->tokens[id].len > SIZE_MAX - total - 1U) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "decoded text too large");
        }
        total += tok->tokens[id].len;
    }
    text = (char *)gd_xmalloc(total + 1U);
    if (text == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "decoded text allocation failed");
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
