#include "tokenizer_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void gd_pair_map_free(gd_pair_map *map)
{
    if (map != NULL) {
        free(map->entries);
        map->entries = NULL;
        map->cap = 0U;
        map->size = 0U;
    }
}

gd_status gd_pair_map_init(gd_pair_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "pair map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_pair_entry *)calloc(cap, sizeof(gd_pair_entry));
    if (map->entries == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "pair map allocation failed");
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

gd_status gd_pair_map_insert(gd_pair_map *map,
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

int32_t gd_pair_map_get(const gd_pair_map *map, int32_t left, int32_t right)
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

void gd_bytes_map_free(gd_bytes_map *map)
{
    if (map != NULL) {
        free(map->entries);
        map->entries = NULL;
        map->cap = 0U;
        map->size = 0U;
    }
}

gd_status gd_bytes_map_init(gd_bytes_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "bytes map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_bytes_entry *)calloc(cap, sizeof(gd_bytes_entry));
    if (map->entries == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "bytes map allocation failed");
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

gd_status gd_bytes_map_insert(gd_bytes_map *map,
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

int32_t gd_bytes_map_get(const gd_bytes_map *map, const uint8_t *bytes, size_t len)
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

void gd_word_map_free(gd_word_map *map)
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

gd_status gd_word_map_init(gd_word_map *map, size_t requested_cap)
{
    size_t cap;
    if (map == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "word map is null");
    }
    cap = gd_next_pow2(requested_cap);
    map->entries = (gd_word_map_entry *)calloc(cap, sizeof(gd_word_map_entry));
    if (map->entries == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "word map allocation failed");
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

gd_status gd_word_map_add(gd_word_map *map, const uint8_t *bytes, size_t len)
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

void gd_pair_stats_free(gd_pair_stats *stats)
{
    if (stats != NULL) {
        free(stats->entries);
        stats->entries = NULL;
        stats->cap = 0U;
        stats->size = 0U;
    }
}

gd_status gd_pair_stats_init(gd_pair_stats *stats, size_t requested_cap)
{
    size_t cap;
    if (stats == NULL) {
        return gd_tokenizer_error(GD_ERR_INVALID_ARGUMENT, "pair stats is null");
    }
    cap = gd_next_pow2(requested_cap);
    stats->entries = (gd_pair_stat_entry *)calloc(cap, sizeof(gd_pair_stat_entry));
    if (stats->entries == NULL) {
        return gd_tokenizer_error(GD_ERR_OUT_OF_MEMORY, "pair stats allocation failed");
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

gd_status gd_pair_stats_add(gd_pair_stats *stats,
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
