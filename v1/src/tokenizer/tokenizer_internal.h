#ifndef GRADIENTS_TOKENIZER_INTERNAL_H
#define GRADIENTS_TOKENIZER_INTERNAL_H

#include "gradients/tokenizer.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

typedef gd_status (*gd_normal_piece_cb)(const uint8_t *bytes, size_t len, void *user);
typedef gd_status (*gd_special_piece_cb)(int32_t id, void *user);

uint64_t gd_hash_mix_u64(uint64_t h, uint64_t v);
uint64_t gd_hash_bytes(const uint8_t *bytes, size_t len);
uint64_t gd_pair_hash(int32_t left, int32_t right);
bool gd_mul_overflows_size(size_t a, size_t b);
size_t gd_next_pow2(size_t n);
void *gd_xmalloc(size_t n);
gd_status gd_memdup_bytes(const uint8_t *src, size_t len, uint8_t **out);
gd_status gd_memdup_cstr(const uint8_t *src, size_t len, char **out);

void gd_pair_map_free(gd_pair_map *map);
gd_status gd_pair_map_init(gd_pair_map *map, size_t requested_cap);
gd_status gd_pair_map_insert(gd_pair_map *map, int32_t left, int32_t right, int32_t id);
int32_t gd_pair_map_get(const gd_pair_map *map, int32_t left, int32_t right);

void gd_bytes_map_free(gd_bytes_map *map);
gd_status gd_bytes_map_init(gd_bytes_map *map, size_t requested_cap);
gd_status gd_bytes_map_insert(gd_bytes_map *map, const uint8_t *bytes, size_t len, int32_t id);
int32_t gd_bytes_map_get(const gd_bytes_map *map, const uint8_t *bytes, size_t len);

void gd_word_map_free(gd_word_map *map);
gd_status gd_word_map_init(gd_word_map *map, size_t requested_cap);
gd_status gd_word_map_add(gd_word_map *map, const uint8_t *bytes, size_t len);

void gd_pair_stats_free(gd_pair_stats *stats);
gd_status gd_pair_stats_init(gd_pair_stats *stats, size_t requested_cap);
gd_status gd_pair_stats_add(gd_pair_stats *stats, int32_t left, int32_t right, uint64_t count);

void gd_tokenizer_recompute_hash(gd_tokenizer *tok);
gd_status gd_tokenizer_create_empty(int split_digits, int allow_special, gd_tokenizer **out);
gd_status gd_tokenizer_add_token(gd_tokenizer *tok,
                                 const uint8_t *bytes,
                                 size_t len,
                                 int is_special,
                                 int32_t left,
                                 int32_t right,
                                 int32_t *id_out);
gd_status gd_tokenizer_add_byte_tokens(gd_tokenizer *tok);

gd_status gd_pretokenize(const uint8_t *data,
                          size_t len,
                          const char **specials,
                          const int32_t *special_ids,
                          int n_specials,
                          int split_digits,
                          int allow_special,
                          gd_normal_piece_cb normal_cb,
                          gd_special_piece_cb special_cb,
                          void *user);

#endif /* GRADIENTS_TOKENIZER_INTERNAL_H */
