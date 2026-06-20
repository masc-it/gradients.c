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

static uint64_t gd_pair_tie_key(uint64_t seed, int32_t left, int32_t right)
{
    uint64_t h = GD_FNV_OFFSET;
    h = gd_hash_mix_u64(h, seed);
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)left);
    h = gd_hash_mix_u64(h, (uint64_t)(uint32_t)right);
    return h;
}

typedef struct gd_train_pair {
    int32_t left;
    int32_t right;
    uint64_t count;
    int occ_head;
    int touched_epoch;
} gd_train_pair;

typedef struct gd_train_pair_slot {
    int used;
    int32_t left;
    int32_t right;
    int pair_index;
} gd_train_pair_slot;

typedef struct gd_train_pair_table {
    gd_train_pair_slot *slots;
    size_t slot_cap;
    size_t slot_size;
    gd_train_pair *pairs;
    int n_pairs;
    int pair_cap;
} gd_train_pair_table;

typedef struct gd_train_occurrence {
    int node;
    int next;
} gd_train_occurrence;

typedef struct gd_train_occ_vec {
    gd_train_occurrence *data;
    int len;
    int cap;
} gd_train_occ_vec;

typedef struct gd_train_heap_item {
    int pair_index;
    uint64_t count;
    uint64_t tie;
    int32_t left;
    int32_t right;
} gd_train_heap_item;

typedef struct gd_train_heap {
    gd_train_heap_item *data;
    int len;
    int cap;
} gd_train_heap;

typedef struct gd_train_touch_vec {
    int *data;
    int len;
    int cap;
    int epoch;
} gd_train_touch_vec;

typedef struct gd_train_node {
    int32_t sym;
    int prev;
    int next;
    int word;
    int alive;
} gd_train_node;

typedef struct gd_train_word {
    int head;
    int len;
    uint64_t count;
} gd_train_word;

typedef struct gd_train_corpus {
    gd_train_word *words;
    int n_words;
    gd_train_node *nodes;
    int n_nodes;
} gd_train_corpus;

static void gd_train_pair_table_free(gd_train_pair_table *table)
{
    if (table == NULL) {
        return;
    }
    free(table->slots);
    free(table->pairs);
    memset(table, 0, sizeof(*table));
}

static void gd_train_occ_vec_free(gd_train_occ_vec *occs)
{
    if (occs == NULL) {
        return;
    }
    free(occs->data);
    memset(occs, 0, sizeof(*occs));
}

static void gd_train_heap_free(gd_train_heap *heap)
{
    if (heap == NULL) {
        return;
    }
    free(heap->data);
    memset(heap, 0, sizeof(*heap));
}

static void gd_train_touch_vec_free(gd_train_touch_vec *touches)
{
    if (touches == NULL) {
        return;
    }
    free(touches->data);
    memset(touches, 0, sizeof(*touches));
}

static void gd_train_corpus_free(gd_train_corpus *corpus)
{
    if (corpus == NULL) {
        return;
    }
    free(corpus->words);
    free(corpus->nodes);
    memset(corpus, 0, sizeof(*corpus));
}

static gd_status gd_train_pair_table_init(gd_train_pair_table *table,
                                          size_t requested_pairs)
{
    size_t slot_cap;
    if (table == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "train pair table is null");
    }
    memset(table, 0, sizeof(*table));
    slot_cap = gd_next_pow2(requested_pairs * 2U + 16U);
    table->slots = (gd_train_pair_slot *)calloc(slot_cap, sizeof(gd_train_pair_slot));
    if (table->slots == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train pair slot allocation failed");
    }
    table->slot_cap = slot_cap;
    table->pair_cap = 1024;
    if ((size_t)table->pair_cap < requested_pairs) {
        while ((size_t)table->pair_cap < requested_pairs) {
            if (table->pair_cap > INT_MAX / 2) {
                gd_train_pair_table_free(table);
                return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "too many train pairs");
            }
            table->pair_cap *= 2;
        }
    }
    table->pairs = (gd_train_pair *)calloc((size_t)table->pair_cap, sizeof(gd_train_pair));
    if (table->pairs == NULL) {
        gd_train_pair_table_free(table);
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train pair allocation failed");
    }
    return GD_OK;
}

static gd_status gd_train_pair_table_rehash(gd_train_pair_table *table,
                                            size_t requested_cap)
{
    gd_train_pair_slot *new_slots;
    size_t new_cap = gd_next_pow2(requested_cap);
    int i;
    if (new_cap < 16U) {
        new_cap = 16U;
    }
    new_slots = (gd_train_pair_slot *)calloc(new_cap, sizeof(gd_train_pair_slot));
    if (new_slots == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train pair rehash failed");
    }
    for (i = 0; i < table->n_pairs; ++i) {
        gd_train_pair *pair = &table->pairs[i];
        size_t mask = new_cap - 1U;
        size_t j = (size_t)gd_pair_hash(pair->left, pair->right) & mask;
        while (new_slots[j].used != 0) {
            j = (j + 1U) & mask;
        }
        new_slots[j].used = 1;
        new_slots[j].left = pair->left;
        new_slots[j].right = pair->right;
        new_slots[j].pair_index = i;
    }
    free(table->slots);
    table->slots = new_slots;
    table->slot_cap = new_cap;
    table->slot_size = (size_t)table->n_pairs;
    return GD_OK;
}

static gd_status gd_train_pair_table_reserve_pair(gd_train_pair_table *table)
{
    gd_train_pair *new_pairs;
    int new_cap;
    if (table->n_pairs < table->pair_cap) {
        return GD_OK;
    }
    if (table->pair_cap > INT_MAX / 2) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "too many train pairs");
    }
    new_cap = table->pair_cap * 2;
    new_pairs = (gd_train_pair *)realloc(table->pairs,
                                         (size_t)new_cap * sizeof(gd_train_pair));
    if (new_pairs == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train pair growth failed");
    }
    memset(&new_pairs[table->pair_cap], 0,
           (size_t)(new_cap - table->pair_cap) * sizeof(gd_train_pair));
    table->pairs = new_pairs;
    table->pair_cap = new_cap;
    return GD_OK;
}

static gd_status gd_train_pair_get_or_create(gd_train_pair_table *table,
                                             int32_t left,
                                             int32_t right,
                                             int *pair_index_out)
{
    uint64_t h;
    size_t mask;
    size_t i;
    gd_status status;
    if ((table->slot_size + 1U) * GD_BPE_HASH_LOAD_DEN >
        table->slot_cap * GD_BPE_HASH_LOAD_NUM) {
        status = gd_train_pair_table_rehash(table, table->slot_cap * 2U);
        if (status != GD_OK) {
            return status;
        }
    }
    h = gd_pair_hash(left, right);
    mask = table->slot_cap - 1U;
    i = (size_t)h & mask;
    while (table->slots[i].used != 0) {
        if (table->slots[i].left == left && table->slots[i].right == right) {
            *pair_index_out = table->slots[i].pair_index;
            return GD_OK;
        }
        i = (i + 1U) & mask;
    }
    status = gd_train_pair_table_reserve_pair(table);
    if (status != GD_OK) {
        return status;
    }
    table->pairs[table->n_pairs].left = left;
    table->pairs[table->n_pairs].right = right;
    table->pairs[table->n_pairs].count = 0U;
    table->pairs[table->n_pairs].occ_head = -1;
    table->slots[i].used = 1;
    table->slots[i].left = left;
    table->slots[i].right = right;
    table->slots[i].pair_index = table->n_pairs;
    table->slot_size += 1U;
    *pair_index_out = table->n_pairs;
    table->n_pairs += 1;
    return GD_OK;
}

static int gd_train_pair_lookup(const gd_train_pair_table *table,
                                int32_t left,
                                int32_t right)
{
    uint64_t h;
    size_t mask;
    size_t i;
    if (table == NULL || table->slot_cap == 0U) {
        return -1;
    }
    h = gd_pair_hash(left, right);
    mask = table->slot_cap - 1U;
    i = (size_t)h & mask;
    while (table->slots[i].used != 0) {
        if (table->slots[i].left == left && table->slots[i].right == right) {
            return table->slots[i].pair_index;
        }
        i = (i + 1U) & mask;
    }
    return -1;
}

static gd_status gd_train_occ_vec_push(gd_train_occ_vec *occs,
                                       int node,
                                       int next,
                                       int *index_out)
{
    gd_train_occurrence *new_data;
    int new_cap;
    if (occs->len == occs->cap) {
        new_cap = occs->cap == 0 ? 4096 : occs->cap;
        while (occs->len >= new_cap) {
            if (new_cap > INT_MAX / 2) {
                return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY,
                                          "too many train pair occurrences");
            }
            new_cap *= 2;
        }
        new_data = (gd_train_occurrence *)realloc(occs->data,
                                                  (size_t)new_cap *
                                                      sizeof(gd_train_occurrence));
        if (new_data == NULL) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY,
                                      "train occurrence growth failed");
        }
        occs->data = new_data;
        occs->cap = new_cap;
    }
    occs->data[occs->len].node = node;
    occs->data[occs->len].next = next;
    *index_out = occs->len;
    occs->len += 1;
    return GD_OK;
}

static int gd_train_heap_item_better(const gd_train_heap_item *a,
                                     const gd_train_heap_item *b)
{
    if (a->count != b->count) {
        return a->count > b->count;
    }
    if (a->tie != b->tie) {
        return a->tie < b->tie;
    }
    if (a->left != b->left) {
        return a->left < b->left;
    }
    return a->right < b->right;
}

static gd_status gd_train_heap_push(gd_train_heap *heap,
                                    int pair_index,
                                    uint64_t count,
                                    uint64_t seed,
                                    int32_t left,
                                    int32_t right)
{
    gd_train_heap_item item;
    gd_train_heap_item *new_data;
    int new_cap;
    int i;
    if (count == 0U) {
        return GD_OK;
    }
    if (heap->len == heap->cap) {
        new_cap = heap->cap == 0 ? 4096 : heap->cap;
        while (heap->len >= new_cap) {
            if (new_cap > INT_MAX / 2) {
                return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train heap too large");
            }
            new_cap *= 2;
        }
        new_data = (gd_train_heap_item *)realloc(heap->data,
                                                 (size_t)new_cap *
                                                     sizeof(gd_train_heap_item));
        if (new_data == NULL) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train heap growth failed");
        }
        heap->data = new_data;
        heap->cap = new_cap;
    }
    item.pair_index = pair_index;
    item.count = count;
    item.tie = gd_pair_tie_key(seed, left, right);
    item.left = left;
    item.right = right;
    i = heap->len;
    heap->len += 1;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!gd_train_heap_item_better(&item, &heap->data[parent])) {
            break;
        }
        heap->data[i] = heap->data[parent];
        i = parent;
    }
    heap->data[i] = item;
    return GD_OK;
}

static int gd_train_heap_pop(gd_train_heap *heap, gd_train_heap_item *out)
{
    gd_train_heap_item item;
    int i = 0;
    if (heap->len == 0) {
        return 0;
    }
    *out = heap->data[0];
    heap->len -= 1;
    if (heap->len == 0) {
        return 1;
    }
    item = heap->data[heap->len];
    for (;;) {
        int left = i * 2 + 1;
        int right = left + 1;
        int child;
        if (left >= heap->len) {
            break;
        }
        child = left;
        if (right < heap->len &&
            gd_train_heap_item_better(&heap->data[right], &heap->data[left])) {
            child = right;
        }
        if (!gd_train_heap_item_better(&heap->data[child], &item)) {
            break;
        }
        heap->data[i] = heap->data[child];
        i = child;
    }
    heap->data[i] = item;
    return 1;
}

static gd_status gd_train_touch_vec_push(gd_train_touch_vec *touches, int pair_index)
{
    int *new_data;
    int new_cap;
    if (touches == NULL) {
        return GD_OK;
    }
    if (touches->len == touches->cap) {
        new_cap = touches->cap == 0 ? 4096 : touches->cap;
        while (touches->len >= new_cap) {
            if (new_cap > INT_MAX / 2) {
                return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "too many touched pairs");
            }
            new_cap *= 2;
        }
        new_data = (int *)realloc(touches->data, (size_t)new_cap * sizeof(int));
        if (new_data == NULL) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "touched pair growth failed");
        }
        touches->data = new_data;
        touches->cap = new_cap;
    }
    touches->data[touches->len] = pair_index;
    touches->len += 1;
    return GD_OK;
}

static gd_status gd_train_mark_touched(gd_train_pair_table *table,
                                       gd_train_touch_vec *touches,
                                       int pair_index)
{
    if (touches == NULL) {
        return GD_OK;
    }
    if (table->pairs[pair_index].touched_epoch == touches->epoch) {
        return GD_OK;
    }
    table->pairs[pair_index].touched_epoch = touches->epoch;
    return gd_train_touch_vec_push(touches, pair_index);
}

static void gd_train_touch_begin(gd_train_pair_table *table, gd_train_touch_vec *touches)
{
    int i;
    if (touches == NULL) {
        return;
    }
    touches->len = 0;
    if (touches->epoch == INT_MAX) {
        for (i = 0; i < table->n_pairs; ++i) {
            table->pairs[i].touched_epoch = 0;
        }
        touches->epoch = 1;
    } else {
        touches->epoch += 1;
    }
}

static gd_status gd_train_heap_push_touched(const gd_train_pair_table *table,
                                            const gd_train_touch_vec *touches,
                                            gd_train_heap *heap,
                                            uint64_t seed)
{
    int i;
    if (touches == NULL) {
        return GD_OK;
    }
    for (i = 0; i < touches->len; ++i) {
        int pair_index = touches->data[i];
        const gd_train_pair *pair = &table->pairs[pair_index];
        gd_status status = gd_train_heap_push(heap,
                                              pair_index,
                                              pair->count,
                                              seed,
                                              pair->left,
                                              pair->right);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status gd_train_pair_increment(gd_train_pair_table *table,
                                         gd_train_occ_vec *occs,
                                         gd_train_touch_vec *touches,
                                         int32_t left,
                                         int32_t right,
                                         uint64_t count,
                                         int left_node)
{
    int pair_index;
    int occ_index;
    gd_train_pair *pair;
    gd_status status;
    if (count == 0U) {
        return GD_OK;
    }
    status = gd_train_pair_get_or_create(table, left, right, &pair_index);
    if (status != GD_OK) {
        return status;
    }
    pair = &table->pairs[pair_index];
    pair->count = gd_sat_add_u64(pair->count, count);
    if (left_node >= 0) {
        status = gd_train_occ_vec_push(occs, left_node, pair->occ_head, &occ_index);
        if (status != GD_OK) {
            return status;
        }
        pair->occ_head = occ_index;
    }
    return gd_train_mark_touched(table, touches, pair_index);
}

static gd_status gd_train_pair_decrement(gd_train_pair_table *table,
                                         gd_train_touch_vec *touches,
                                         int32_t left,
                                         int32_t right,
                                         uint64_t count)
{
    int pair_index;
    gd_train_pair *pair;
    if (count == 0U) {
        return GD_OK;
    }
    pair_index = gd_train_pair_lookup(table, left, right);
    if (pair_index < 0) {
        return GD_OK;
    }
    pair = &table->pairs[pair_index];
    pair->count = pair->count > count ? pair->count - count : 0U;
    return gd_train_mark_touched(table, touches, pair_index);
}

static int gd_train_pair_occurrence_valid(const gd_train_corpus *corpus,
                                          int node_index,
                                          int32_t left,
                                          int32_t right,
                                          int *right_node_out)
{
    const gd_train_node *node;
    int right_node;
    if (node_index < 0 || node_index >= corpus->n_nodes) {
        return 0;
    }
    node = &corpus->nodes[node_index];
    if (node->alive == 0 || node->sym != left) {
        return 0;
    }
    right_node = node->next;
    if (right_node < 0 || right_node >= corpus->n_nodes ||
        corpus->nodes[right_node].alive == 0 || corpus->nodes[right_node].sym != right) {
        return 0;
    }
    *right_node_out = right_node;
    return 1;
}

static gd_status gd_train_corpus_build(const gd_bpe_word *words,
                                       int n_words,
                                       gd_train_corpus *corpus,
                                       gd_train_pair_table *table,
                                       gd_train_occ_vec *occs,
                                       gd_train_heap *heap,
                                       uint64_t seed)
{
    int i;
    int node_cursor = 0;
    int64_t total_symbols = 0;
    gd_status status;
    memset(corpus, 0, sizeof(*corpus));
    for (i = 0; i < n_words; ++i) {
        if (words[i].len < 0 || total_symbols > (int64_t)INT_MAX - words[i].len) {
            return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "too many BPE training symbols");
        }
        total_symbols += words[i].len;
    }
    corpus->words = (gd_train_word *)calloc((size_t)(n_words == 0 ? 1 : n_words),
                                            sizeof(gd_train_word));
    if (corpus->words == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train word allocation failed");
    }
    corpus->nodes = (gd_train_node *)calloc((size_t)(total_symbols == 0 ? 1 : total_symbols),
                                            sizeof(gd_train_node));
    if (corpus->nodes == NULL) {
        gd_train_corpus_free(corpus);
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "train node allocation failed");
    }
    corpus->n_words = n_words;
    corpus->n_nodes = (int)total_symbols;
    status = gd_train_pair_table_init(table, 4096U);
    if (status != GD_OK) {
        gd_train_corpus_free(corpus);
        return status;
    }
    for (i = 0; i < n_words; ++i) {
        int j;
        int head = node_cursor;
        corpus->words[i].head = words[i].len > 0 ? head : -1;
        corpus->words[i].len = words[i].len;
        corpus->words[i].count = words[i].count;
        for (j = 0; j < words[i].len; ++j) {
            int node_index = node_cursor + j;
            corpus->nodes[node_index].sym = words[i].ids[j];
            corpus->nodes[node_index].prev = j == 0 ? -1 : node_index - 1;
            corpus->nodes[node_index].next = j + 1 == words[i].len ? -1 : node_index + 1;
            corpus->nodes[node_index].word = i;
            corpus->nodes[node_index].alive = 1;
        }
        for (j = 0; j + 1 < words[i].len; ++j) {
            int node_index = node_cursor + j;
            status = gd_train_pair_increment(table,
                                             occs,
                                             NULL,
                                             words[i].ids[j],
                                             words[i].ids[j + 1],
                                             words[i].count,
                                             node_index);
            if (status != GD_OK) {
                gd_train_corpus_free(corpus);
                return status;
            }
        }
        node_cursor += words[i].len;
    }
    for (i = 0; i < table->n_pairs; ++i) {
        status = gd_train_heap_push(heap,
                                    i,
                                    table->pairs[i].count,
                                    seed,
                                    table->pairs[i].left,
                                    table->pairs[i].right);
        if (status != GD_OK) {
            gd_train_corpus_free(corpus);
            return status;
        }
    }
    return GD_OK;
}

static int gd_train_heap_best_pair(gd_train_pair_table *table,
                                   gd_train_heap *heap,
                                   uint64_t min_frequency,
                                   int *pair_index_out,
                                   uint64_t *count_out)
{
    gd_train_heap_item item;
    while (gd_train_heap_pop(heap, &item) != 0) {
        gd_train_pair *pair;
        if (item.pair_index < 0 || item.pair_index >= table->n_pairs) {
            continue;
        }
        pair = &table->pairs[item.pair_index];
        if (pair->count != item.count || pair->left != item.left || pair->right != item.right) {
            continue;
        }
        if (pair->count < min_frequency || pair->count == 0U) {
            return 0;
        }
        *pair_index_out = item.pair_index;
        *count_out = pair->count;
        return 1;
    }
    return 0;
}

static gd_status gd_train_merge_pair(gd_train_corpus *corpus,
                                     gd_train_pair_table *table,
                                     gd_train_occ_vec *occs,
                                     gd_train_touch_vec *touches,
                                     gd_train_heap *heap,
                                     uint64_t seed,
                                     int pair_index,
                                     int32_t left,
                                     int32_t right,
                                     int32_t merged,
                                     uint64_t *merged_count_out)
{
    int occ_index;
    uint64_t merged_count = 0U;
    gd_status status;
    if (pair_index < 0 || pair_index >= table->n_pairs) {
        return gd_tokenizer_error(GD_ERR_INTERNAL, "invalid train pair index");
    }
    gd_train_touch_begin(table, touches);
    occ_index = table->pairs[pair_index].occ_head;
    while (occ_index >= 0) {
        int node_index;
        int right_node;
        int prev_node;
        int next_node;
        int word_index;
        uint64_t word_count;
        gd_train_node *node;
        gd_train_node *right_sym;
        if (occ_index >= occs->len) {
            return gd_tokenizer_error(GD_ERR_INTERNAL, "invalid train occurrence index");
        }
        node_index = occs->data[occ_index].node;
        occ_index = occs->data[occ_index].next;
        if (gd_train_pair_occurrence_valid(corpus, node_index, left, right, &right_node) == 0) {
            continue;
        }
        node = &corpus->nodes[node_index];
        right_sym = &corpus->nodes[right_node];
        prev_node = node->prev;
        next_node = right_sym->next;
        word_index = node->word;
        word_count = corpus->words[word_index].count;

        if (prev_node >= 0) {
            status = gd_train_pair_decrement(table,
                                             touches,
                                             corpus->nodes[prev_node].sym,
                                             left,
                                             word_count);
            if (status != GD_OK) {
                return status;
            }
        }
        status = gd_train_pair_decrement(table, touches, left, right, word_count);
        if (status != GD_OK) {
            return status;
        }
        if (next_node >= 0) {
            status = gd_train_pair_decrement(table,
                                             touches,
                                             right,
                                             corpus->nodes[next_node].sym,
                                             word_count);
            if (status != GD_OK) {
                return status;
            }
        }

        node->sym = merged;
        node->next = next_node;
        if (next_node >= 0) {
            corpus->nodes[next_node].prev = node_index;
        }
        right_sym->alive = 0;
        right_sym->prev = -1;
        right_sym->next = -1;
        corpus->words[word_index].len -= 1;
        merged_count = gd_sat_add_u64(merged_count, word_count);

        if (prev_node >= 0) {
            status = gd_train_pair_increment(table,
                                             occs,
                                             touches,
                                             corpus->nodes[prev_node].sym,
                                             merged,
                                             word_count,
                                             prev_node);
            if (status != GD_OK) {
                return status;
            }
        }
        if (next_node >= 0) {
            status = gd_train_pair_increment(table,
                                             occs,
                                             touches,
                                             merged,
                                             corpus->nodes[next_node].sym,
                                             word_count,
                                             node_index);
            if (status != GD_OK) {
                return status;
            }
        }
    }
    table->pairs[pair_index].occ_head = -1;
    status = gd_train_heap_push_touched(table, touches, heap, seed);
    if (status != GD_OK) {
        return status;
    }
    if (merged_count_out != NULL) {
        *merged_count_out = merged_count;
    }
    return GD_OK;
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
    gd_train_corpus corpus = {0};
    gd_train_pair_table table = {0};
    gd_train_occ_vec occs = {0};
    gd_train_heap heap = {0};
    gd_train_touch_vec touches = {0};
    gd_status status;

    status = gd_train_corpus_build(words, n_words, &corpus, &table, &occs, &heap, seed);
    if (status != GD_OK) {
        gd_train_pair_table_free(&table);
        gd_train_occ_vec_free(&occs);
        gd_train_heap_free(&heap);
        gd_train_touch_vec_free(&touches);
        gd_train_corpus_free(&corpus);
        return status;
    }

    while (tok->n_tokens < target_vocab) {
        int pair_index = -1;
        uint64_t count = 0U;
        int32_t left;
        int32_t right;
        const gd_bpe_token *lt;
        const gd_bpe_token *rt;
        uint8_t merged_bytes[GD_BPE_MAX_PIECE_BYTES * 2U];
        size_t merged_len;
        int32_t existing;
        int32_t merged_id = -1;
        uint64_t actual_merges = 0U;

        if (gd_train_heap_best_pair(&table,
                                    &heap,
                                    (uint64_t)min_frequency,
                                    &pair_index,
                                    &count) == 0) {
            break;
        }
        left = table.pairs[pair_index].left;
        right = table.pairs[pair_index].right;
        if (left < 0 || right < 0 || left >= tok->n_tokens || right >= tok->n_tokens) {
            status = gd_tokenizer_error(GD_ERR_INTERNAL, "BPE trainer selected invalid pair");
            break;
        }
        lt = &tok->tokens[left];
        rt = &tok->tokens[right];
        if (lt->len + rt->len > sizeof(merged_bytes)) {
            status = gd_tokenizer_error(GD_ERR_INTERNAL, "BPE merge exceeded piece limit");
            break;
        }
        merged_len = lt->len + rt->len;
        memcpy(merged_bytes, lt->bytes, lt->len);
        memcpy(&merged_bytes[lt->len], rt->bytes, rt->len);
        existing = gd_bytes_map_get(&tok->bytes_map, merged_bytes, merged_len);
        if (existing >= 0) {
            merged_id = existing;
        } else {
            status = gd_tokenizer_add_token(tok, merged_bytes, merged_len, 0, left, right,
                                            &merged_id);
            if (status != GD_OK) {
                break;
            }
        }
        status = gd_train_merge_pair(&corpus,
                                     &table,
                                     &occs,
                                     &touches,
                                     &heap,
                                     seed,
                                     pair_index,
                                     left,
                                     right,
                                     merged_id,
                                     &actual_merges);
        if (status != GD_OK) {
            break;
        }
        if (actual_merges == 0U) {
            table.pairs[pair_index].count = 0U;
        }
        (void)count;
    }

    gd_train_pair_table_free(&table);
    gd_train_occ_vec_free(&occs);
    gd_train_heap_free(&heap);
    gd_train_touch_vec_free(&touches);
    gd_train_corpus_free(&corpus);
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
