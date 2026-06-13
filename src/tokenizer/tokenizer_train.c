#include "tokenizer_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct gd_collect_user {
    gd_word_map *words;
} gd_collect_user;

static gd_status gd_read_file(const char *path, uint8_t **data_out, size_t *len_out)
{
    FILE *f;
    long end;
    size_t nread;
    uint8_t *data;

    if (path == NULL || data_out == NULL || len_out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid file read arguments");
    }
    *data_out = NULL;
    *len_out = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return gd_tokenizer_error(GD_ERR_IO, "failed to open tokenizer input");
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        fclose(f);
        return gd_tokenizer_error(GD_ERR_IO, "failed to seek tokenizer input");
    }
    end = ftell(f);
    if (end < 0) {
        fclose(f);
        return gd_tokenizer_error(GD_ERR_IO, "failed to size tokenizer input");
    }
    if ((unsigned long)end > (unsigned long)SIZE_MAX) {
        fclose(f);
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "tokenizer input too large");
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        fclose(f);
        return gd_tokenizer_error(GD_ERR_IO, "failed to rewind tokenizer input");
    }
    data = (uint8_t *)gd_xmalloc((size_t)end);
    if (data == NULL) {
        fclose(f);
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "tokenizer input allocation failed");
    }
    nread = fread(data, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(data);
        fclose(f);
        return gd_tokenizer_error(GD_ERR_IO, "failed to read tokenizer input");
    }
    if (fclose(f) != 0) {
        free(data);
        return gd_tokenizer_error(GD_ERR_IO, "failed to close tokenizer input");
    }
    *data_out = data;
    *len_out = (size_t)end;
    return GD_OK;
}

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
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid word map conversion");
    }
    *words_out = NULL;
    *n_words_out = 0;
    if (map->size > (size_t)INT_MAX) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "too many unique tokenizer pieces");
    }
    if (gd_mul_overflows_size(map->size, sizeof(gd_bpe_word))) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "word array allocation overflow");
    }
    words = (gd_bpe_word *)calloc(map->size == 0U ? 1U : map->size, sizeof(gd_bpe_word));
    if (words == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "word array allocation failed");
    }
    for (i = 0U; i < map->cap; ++i) {
        const gd_word_map_entry *entry = &map->entries[i];
        size_t j;
        if (entry->used == 0) {
            continue;
        }
        if (entry->len > (size_t)INT_MAX) {
            free(words);
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "tokenizer piece too large");
        }
        words[n_words].ids = (int32_t *)gd_xmalloc(entry->len * sizeof(int32_t));
        if (words[n_words].ids == NULL) {
            int k;
            for (k = 0; k < n_words; ++k) {
                free(words[k].ids);
            }
            free(words);
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "word token allocation failed");
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
            return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid special token");
        }
        for (j = 0; j < i; ++j) {
            if (strcmp(s, special_tokens[j]) == 0) {
                return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "duplicate special token");
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
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "tokenizer output is null");
    }
    *out = NULL;
    if (input_paths == NULL || n_input_paths <= 0 || cfg == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE train arguments");
    }
    if (cfg->n_special_tokens < 0 || cfg->n_special_tokens > GD_BPE_MAX_SPECIALS ||
        (cfg->n_special_tokens > 0 && cfg->special_tokens == NULL)) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid special token config");
    }
    if (cfg->vocab_size < GD_BPE_N_BYTES + cfg->n_special_tokens) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "BPE vocab_size too small");
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
            return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "null tokenizer input path");
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
            return gd_tokenizer_error(GD_ERR_INTERNAL, "BPE trainer selected invalid pair");
        }
        lt = &tok->tokens[left];
        rt = &tok->tokens[right];
        if (lt->len + rt->len > sizeof(merged_bytes)) {
            gd_bpe_words_free(words, n_words);
            gd_tokenizer_destroy(tok);
            return gd_tokenizer_error(GD_ERR_INTERNAL, "BPE merge exceeded piece limit");
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
