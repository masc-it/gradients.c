#include "tokenizer_internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GD_BPE_TRAIN_DEFAULT_BATCH_BYTES (4U * 1024U * 1024U)

struct gd_bpe_trainer {
    int vocab_size;
    int min_frequency;
    int split_digits;
    uint64_t seed;
    char **special_storage;
    const char **special_tokens;
    int n_special_tokens;
    gd_word_map word_map;
    int word_map_ready;
    int finished;
    uint64_t texts;
    uint64_t bytes;
    uint64_t pieces;
};

static void gd_bpe_words_free(gd_bpe_word *words, int n_words);

static uint64_t gd_sat_add_u64(uint64_t a, uint64_t b)
{
    return UINT64_MAX - a < b ? UINT64_MAX : a + b;
}

static gd_status gd_reserve_bytes(uint8_t **buf, size_t *cap, size_t needed)
{
    uint8_t *new_buf;
    size_t new_cap;
    if (buf == NULL || cap == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid byte buffer");
    }
    if (needed <= *cap) {
        return GD_OK;
    }
    new_cap = *cap == 0U ? 4096U : *cap;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2U) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "byte buffer too large");
        }
        new_cap *= 2U;
    }
    new_buf = (uint8_t *)realloc(*buf, new_cap);
    if (new_buf == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "byte buffer allocation failed");
    }
    *buf = new_buf;
    *cap = new_cap;
    return GD_OK;
}

static size_t gd_last_newline_offset(const uint8_t *data, size_t len)
{
    size_t i;
    if (data == NULL) {
        return 0U;
    }
    for (i = len; i > 0U; --i) {
        if (data[i - 1U] == (uint8_t)'\n') {
            return i;
        }
    }
    return 0U;
}

static gd_status gd_validate_train_config(const gd_bpe_train_config *cfg)
{
    int i;
    if (cfg == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "null BPE train config");
    }
    if (cfg->n_special_tokens < 0 || cfg->n_special_tokens > GD_BPE_MAX_SPECIALS ||
        (cfg->n_special_tokens > 0 && cfg->special_tokens == NULL)) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid special token config");
    }
    if (cfg->vocab_size < GD_BPE_N_BYTES + cfg->n_special_tokens) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "BPE vocab_size too small");
    }
    for (i = 0; i < cfg->n_special_tokens; ++i) {
        int j;
        const char *special = cfg->special_tokens[i];
        if (special == NULL || special[0] == '\0') {
            return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid special token");
        }
        for (j = 0; j < i; ++j) {
            if (strcmp(special, cfg->special_tokens[j]) == 0) {
                return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "duplicate special token");
            }
        }
    }
    return GD_OK;
}

static void gd_bpe_trainer_free_specials(gd_bpe_trainer *trainer)
{
    int i;
    if (trainer == NULL) {
        return;
    }
    if (trainer->special_storage != NULL) {
        for (i = 0; i < trainer->n_special_tokens; ++i) {
            free(trainer->special_storage[i]);
        }
    }
    free(trainer->special_storage);
    free(trainer->special_tokens);
    trainer->special_storage = NULL;
    trainer->special_tokens = NULL;
    trainer->n_special_tokens = 0;
}

static gd_status gd_bpe_trainer_copy_specials(gd_bpe_trainer *trainer,
                                              const gd_bpe_train_config *cfg)
{
    int i;
    gd_status status;
    if (cfg->n_special_tokens == 0) {
        return GD_OK;
    }
    trainer->special_storage = (char **)calloc((size_t)cfg->n_special_tokens, sizeof(char *));
    if (trainer->special_storage == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "special token allocation failed");
    }
    trainer->special_tokens = (const char **)calloc((size_t)cfg->n_special_tokens,
                                                    sizeof(const char *));
    if (trainer->special_tokens == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "special token allocation failed");
    }
    trainer->n_special_tokens = cfg->n_special_tokens;
    for (i = 0; i < cfg->n_special_tokens; ++i) {
        const char *special = cfg->special_tokens[i];
        status = gd_memdup_cstr((const uint8_t *)special, strlen(special),
                                &trainer->special_storage[i]);
        if (status != GD_OK) {
            return status;
        }
        trainer->special_tokens[i] = trainer->special_storage[i];
    }
    return GD_OK;
}

static gd_status gd_collect_piece_cb(const uint8_t *bytes, size_t len, void *user)
{
    gd_bpe_trainer *trainer = (gd_bpe_trainer *)user;
    trainer->pieces = gd_sat_add_u64(trainer->pieces, 1U);
    return gd_word_map_add(&trainer->word_map, bytes, len);
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
            gd_bpe_words_free(words, n_words);
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "tokenizer piece too large");
        }
        words[n_words].ids = (int32_t *)gd_xmalloc(entry->len * sizeof(int32_t));
        if (words[n_words].ids == NULL) {
            gd_bpe_words_free(words, n_words);
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
        const char *special = special_tokens[i];
        status = gd_tokenizer_add_token(tok,
                                        (const uint8_t *)special,
                                        strlen(special),
                                        1,
                                        -1,
                                        -1,
                                        NULL);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status gd_run_bpe_merges(gd_tokenizer *tok,
                                   gd_bpe_word *words,
                                   int n_words,
                                   int target_vocab,
                                   int min_frequency,
                                   uint64_t seed)
{
    gd_status status = GD_OK;
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
            return status;
        }
        if (gd_choose_best_pair(&stats,
                                (uint64_t)min_frequency,
                                seed,
                                &left,
                                &right,
                                &count) == 0) {
            gd_pair_stats_free(&stats);
            break;
        }
        gd_pair_stats_free(&stats);

        if (left < 0 || right < 0 || left >= tok->n_tokens || right >= tok->n_tokens) {
            return gd_tokenizer_error(GD_ERR_INTERNAL, "BPE trainer selected invalid pair");
        }
        lt = &tok->tokens[left];
        rt = &tok->tokens[right];
        if (lt->len + rt->len > sizeof(merged_bytes)) {
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
            return status;
        }
        (void)count;
        gd_merge_words(words, n_words, left, right, merged_id);
    }
    return status;
}

gd_status gd_bpe_trainer_create(const gd_bpe_train_config *cfg,
                                gd_bpe_trainer **out)
{
    gd_bpe_trainer *trainer;
    gd_status status;
    if (out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "trainer output is null");
    }
    *out = NULL;
    status = gd_validate_train_config(cfg);
    if (status != GD_OK) {
        return status;
    }
    trainer = (gd_bpe_trainer *)calloc(1U, sizeof(gd_bpe_trainer));
    if (trainer == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "BPE trainer allocation failed");
    }
    trainer->vocab_size = cfg->vocab_size;
    trainer->min_frequency = cfg->min_frequency <= 0 ? 2 : cfg->min_frequency;
    trainer->split_digits = cfg->split_digits != 0 ? 1 : 0;
    trainer->seed = cfg->seed;
    status = gd_bpe_trainer_copy_specials(trainer, cfg);
    if (status != GD_OK) {
        gd_bpe_trainer_destroy(trainer);
        return status;
    }
    status = gd_word_map_init(&trainer->word_map, 4096U);
    if (status != GD_OK) {
        gd_bpe_trainer_destroy(trainer);
        return status;
    }
    trainer->word_map_ready = 1;
    *out = trainer;
    return GD_OK;
}

gd_status gd_bpe_trainer_add_text(gd_bpe_trainer *trainer,
                                  const uint8_t *text,
                                  size_t len)
{
    gd_status status;
    if (trainer == NULL || (text == NULL && len > 0U) || trainer->finished != 0 ||
        trainer->word_map_ready == 0) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE trainer add_text");
    }
    if (len == 0U) {
        trainer->texts = gd_sat_add_u64(trainer->texts, 1U);
        return GD_OK;
    }
    status = gd_pretokenize(text,
                            len,
                            trainer->special_tokens,
                            NULL,
                            trainer->n_special_tokens,
                            trainer->split_digits,
                            1,
                            gd_collect_piece_cb,
                            NULL,
                            trainer);
    if (status != GD_OK) {
        return status;
    }
    trainer->texts = gd_sat_add_u64(trainer->texts, 1U);
    trainer->bytes = gd_sat_add_u64(trainer->bytes, (uint64_t)len);
    return GD_OK;
}

gd_status gd_bpe_trainer_add_file(gd_bpe_trainer *trainer,
                                  const char *input_path,
                                  size_t batch_bytes)
{
    FILE *f;
    uint8_t *read_buf = NULL;
    uint8_t *pending = NULL;
    size_t pending_len = 0U;
    size_t pending_cap = 0U;
    size_t read_bytes;
    gd_status status = GD_OK;
    if (trainer == NULL || input_path == NULL || input_path[0] == '\0') {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE trainer add_file");
    }
    read_bytes = batch_bytes == 0U ? GD_BPE_TRAIN_DEFAULT_BATCH_BYTES : batch_bytes;
    read_buf = (uint8_t *)gd_xmalloc(read_bytes);
    if (read_buf == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "BPE trainer read buffer allocation failed");
    }
    f = fopen(input_path, "rb");
    if (f == NULL) {
        free(read_buf);
        return gd_tokenizer_error(GD_ERR_IO, "failed to open tokenizer input");
    }
    for (;;) {
        size_t nread = fread(read_buf, 1U, read_bytes, f);
        if (nread > 0U) {
            size_t needed;
            size_t flush_len;
            if (pending_len > SIZE_MAX - nread) {
                status = gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "BPE trainer input batch too large");
                break;
            }
            needed = pending_len + nread;
            status = gd_reserve_bytes(&pending, &pending_cap, needed);
            if (status != GD_OK) {
                break;
            }
            memcpy(&pending[pending_len], read_buf, nread);
            pending_len = needed;
            flush_len = gd_last_newline_offset(pending, pending_len);
            if (flush_len == 0U && pending_len >= read_bytes) {
                flush_len = pending_len;
            }
            if (flush_len > 0U) {
                size_t tail_len;
                status = gd_bpe_trainer_add_text(trainer, pending, flush_len);
                if (status != GD_OK) {
                    break;
                }
                tail_len = pending_len - flush_len;
                if (tail_len > 0U) {
                    memmove(pending, &pending[flush_len], tail_len);
                }
                pending_len = tail_len;
            }
        }
        if (nread < read_bytes) {
            if (ferror(f) != 0 && status == GD_OK) {
                status = gd_tokenizer_error(GD_ERR_IO, "failed to read tokenizer input");
            }
            break;
        }
    }
    if (status == GD_OK && pending_len > 0U) {
        status = gd_bpe_trainer_add_text(trainer, pending, pending_len);
    }
    if (fclose(f) != 0 && status == GD_OK) {
        status = gd_tokenizer_error(GD_ERR_IO, "failed to close tokenizer input");
    }
    free(pending);
    free(read_buf);
    return status;
}

gd_status gd_bpe_trainer_get_stats(const gd_bpe_trainer *trainer,
                                   gd_bpe_trainer_stats *out)
{
    if (trainer == NULL || out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE trainer stats");
    }
    out->texts = trainer->texts;
    out->bytes = trainer->bytes;
    out->pieces = trainer->pieces;
    out->unique_pieces = trainer->word_map_ready != 0 ? (uint64_t)trainer->word_map.size : 0U;
    return GD_OK;
}

gd_status gd_bpe_trainer_finish(gd_bpe_trainer *trainer,
                                gd_tokenizer **out)
{
    gd_tokenizer *tok = NULL;
    gd_bpe_word *words = NULL;
    int n_words = 0;
    gd_status status;
    if (out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "tokenizer output is null");
    }
    *out = NULL;
    if (trainer == NULL || trainer->finished != 0 || trainer->word_map_ready == 0) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE trainer finish");
    }

    status = gd_tokenizer_create_empty(trainer->split_digits, 1, &tok);
    if (status != GD_OK) {
        return status;
    }
    tok->min_frequency = trainer->min_frequency;
    status = gd_tokenizer_add_byte_tokens(tok);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    status = gd_add_special_tokens(tok, trainer->special_tokens, trainer->n_special_tokens);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }

    status = gd_build_words_from_map(&trainer->word_map, &words, &n_words);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    gd_word_map_free(&trainer->word_map);
    trainer->word_map_ready = 0;
    trainer->finished = 1;

    status = gd_run_bpe_merges(tok,
                               words,
                               n_words,
                               trainer->vocab_size,
                               trainer->min_frequency,
                               trainer->seed);
    gd_bpe_words_free(words, n_words);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    gd_tokenizer_recompute_hash(tok);
    *out = tok;
    return GD_OK;
}

void gd_bpe_trainer_destroy(gd_bpe_trainer *trainer)
{
    if (trainer == NULL) {
        return;
    }
    if (trainer->word_map_ready != 0) {
        gd_word_map_free(&trainer->word_map);
    }
    gd_bpe_trainer_free_specials(trainer);
    free(trainer);
}

gd_status gd_bpe_tokenizer_train(const char **input_paths,
                                 int n_input_paths,
                                 const gd_bpe_train_config *cfg,
                                 gd_tokenizer **out)
{
    gd_bpe_trainer *trainer = NULL;
    gd_status status;
    int i;
    if (out == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "tokenizer output is null");
    }
    *out = NULL;
    if (input_paths == NULL || n_input_paths <= 0) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "invalid BPE train input paths");
    }
    status = gd_bpe_trainer_create(cfg, &trainer);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < n_input_paths; ++i) {
        if (input_paths[i] == NULL) {
            gd_bpe_trainer_destroy(trainer);
            return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "null tokenizer input path");
        }
        status = gd_bpe_trainer_add_file(trainer, input_paths[i], 0U);
        if (status != GD_OK) {
            gd_bpe_trainer_destroy(trainer);
            return status;
        }
    }
    status = gd_bpe_trainer_finish(trainer, out);
    gd_bpe_trainer_destroy(trainer);
    return status;
}
