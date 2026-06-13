#include "dataloader_internal.h"
#include "dataset_internal.h"

#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_GDDS_IMPL_MAGIC UINT64_C(0x6764647364737631)
#define GD_GDDS_FNV_OFFSET UINT64_C(14695981039346656037)
#define GD_GDDS_FNV_PRIME UINT64_C(1099511628211)

#define GD_GDDS_HEADER_MAGIC_OFFSET 0U
#define GD_GDDS_HEADER_VERSION_OFFSET 8U
#define GD_GDDS_HEADER_SIZE_OFFSET 12U
#define GD_GDDS_HEADER_FIELD_COUNT_OFFSET 16U
#define GD_GDDS_HEADER_N_SAMPLES_OFFSET 24U
#define GD_GDDS_HEADER_SCHEMA_OFFSET 32U
#define GD_GDDS_HEADER_INDEX_OFFSET 40U
#define GD_GDDS_HEADER_DATA_OFFSET 48U
#define GD_GDDS_HEADER_DATA_NBYTES_OFFSET 56U
#define GD_GDDS_HEADER_SCHEMA_HASH_OFFSET 64U

#define GD_GDDS_FIELD_NAME_OFFSET 0U
#define GD_GDDS_FIELD_DTYPE_OFFSET 64U
#define GD_GDDS_FIELD_RANK_OFFSET 68U
#define GD_GDDS_FIELD_SHAPE_OFFSET 72U
#define GD_GDDS_FIELD_FLAGS_OFFSET 136U
#define GD_GDDS_FIELD_PAD_VALUE_OFFSET 144U

#define GD_GDDS_FIELD_FLAGS_COLLATE_MASK UINT64_C(0x00000000000000ff)
#define GD_GDDS_FIELD_FLAGS_GENERATED_MASK UINT64_C(0x000000000000ff00)
#define GD_GDDS_FIELD_FLAGS_RAGGED_DIM_MASK UINT64_C(0x0000000000ff0000)
#define GD_GDDS_FIELD_FLAGS_SOURCE_MASK UINT64_C(0x00000000ff000000)
#define GD_GDDS_FIELD_FLAGS_GENERATED_SHIFT 8U
#define GD_GDDS_FIELD_FLAGS_RAGGED_DIM_SHIFT 16U
#define GD_GDDS_FIELD_FLAGS_SOURCE_SHIFT 24U

#define GD_GDDS_RECORD_MAGIC_OFFSET 0U
#define GD_GDDS_RECORD_FIELD_COUNT_OFFSET 4U
#define GD_GDDS_RECORD_HEADER_NBYTES_OFFSET 8U
#define GD_GDDS_RECORD_PAYLOAD_NBYTES_OFFSET 12U

#define GD_GDDS_RECORD_FIELD_ID_OFFSET 0U
#define GD_GDDS_RECORD_FIELD_RANK_OFFSET 2U
#define GD_GDDS_RECORD_FIELD_FLAGS_OFFSET 4U
#define GD_GDDS_RECORD_FIELD_SHAPE_OFFSET 8U
#define GD_GDDS_RECORD_FIELD_DATA_OFFSET 72U
#define GD_GDDS_RECORD_FIELD_NBYTES_OFFSET 80U

typedef struct gd_gdds_header_internal {
    uint32_t version;
    uint32_t header_size;
    uint32_t field_count;
    uint64_t n_samples;
    uint64_t schema_offset;
    uint64_t index_offset;
    uint64_t data_offset;
    uint64_t data_nbytes;
    uint64_t schema_hash;
} gd_gdds_header_internal;

typedef struct gd_gdds_shard {
    char *path;
    int fd;
    uint8_t *map;
    size_t map_len;
    const uint8_t *index;
    gd_gdds_header_internal header;
    gd_gdds_field_info *fields;
    uint64_t sample_base;
} gd_gdds_shard;

typedef struct gd_gdds_dataset_impl {
    uint64_t magic;
    gd_gdds_shard *shards;
    int n_shards;
    gd_gdds_field_info *storage_fields;
    int n_storage_fields;
    gd_gdds_field_info *fields;
    int n_fields;
    uint64_t n_samples;
    uint64_t schema_hash;
    gd_dataset_transform_fn transform;
    void *transform_user_data;
} gd_gdds_dataset_impl;

typedef struct gd_gdds_record_field_view {
    int field_id;
    int rank;
    int64_t shape[GD_MAX_DIMS];
    const uint8_t *data;
    size_t nbytes;
} gd_gdds_record_field_view;

typedef struct gd_gdds_record_view {
    int n_entries;
    gd_gdds_record_field_view entries[GD_GDDS_MAX_FIELDS];
    const gd_gdds_record_field_view *by_field[GD_GDDS_MAX_FIELDS];
} gd_gdds_record_view;

static uint32_t gd_gdds_get_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U);
}

static uint32_t gd_gdds_get_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) |
           ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}

static uint64_t gd_gdds_get_le64(const uint8_t *p)
{
    return (uint64_t)gd_gdds_get_le32(p) |
           ((uint64_t)gd_gdds_get_le32(p + 4) << 32U);
}

static int64_t gd_gdds_get_i64(const uint8_t *p)
{
    uint64_t u = gd_gdds_get_le64(p);
    int64_t out;
    memcpy(&out, &u, sizeof(out));
    return out;
}

static uint64_t gd_gdds_fnv64_bytes(uint64_t h, const void *data, size_t n)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t i;
    for (i = 0U; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= GD_GDDS_FNV_PRIME;
    }
    return h;
}

static uint64_t gd_gdds_fnv64_u64(uint64_t h, uint64_t v)
{
    uint8_t b[8];
    int i;
    for (i = 0; i < 8; ++i) {
        b[i] = (uint8_t)((v >> (8U * (uint32_t)i)) & UINT64_C(0xff));
    }
    return gd_gdds_fnv64_bytes(h, b, sizeof(b));
}

static uint64_t gd_gdds_fnv64_i64(uint64_t h, int64_t v)
{
    uint64_t u;
    memcpy(&u, &v, sizeof(u));
    return gd_gdds_fnv64_u64(h, u);
}

static uint64_t gd_gdds_fnv64_str(uint64_t h, const char *s)
{
    if (s == NULL) {
        return gd_gdds_fnv64_u64(h, 0U);
    }
    return gd_gdds_fnv64_bytes(h, s, strlen(s) + 1U);
}

static uint64_t gd_gdds_pack_field_flags(const gd_gdds_field_info *field)
{
    uint64_t flags = 0U;
    if (field == NULL) {
        return 0U;
    }
    flags |= (uint64_t)field->collate & UINT64_C(0xff);
    flags |= (((uint64_t)field->generated) & UINT64_C(0xff)) << GD_GDDS_FIELD_FLAGS_GENERATED_SHIFT;
    if (field->ragged_dim >= 0) {
        flags |= (((uint64_t)field->ragged_dim + 1U) & UINT64_C(0xff)) <<
                 GD_GDDS_FIELD_FLAGS_RAGGED_DIM_SHIFT;
    }
    if (field->source_field >= 0) {
        flags |= (((uint64_t)field->source_field + 1U) & UINT64_C(0xff)) <<
                 GD_GDDS_FIELD_FLAGS_SOURCE_SHIFT;
    }
    return flags;
}

static uint64_t gd_gdds_schema_hash(const gd_gdds_field_info *fields, int n_fields)
{
    uint64_t h = GD_GDDS_FNV_OFFSET;
    int has_collate_metadata = 0;
    int i;
    int d;
    h = gd_gdds_fnv64_str(h, "gdds-schema-v1");
    h = gd_gdds_fnv64_u64(h, (uint64_t)n_fields);
    for (i = 0; i < n_fields; ++i) {
        h = gd_gdds_fnv64_str(h, fields[i].name);
        h = gd_gdds_fnv64_u64(h, (uint64_t)fields[i].dtype);
        h = gd_gdds_fnv64_u64(h, (uint64_t)fields[i].rank);
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            h = gd_gdds_fnv64_i64(h, fields[i].shape[d]);
        }
        if (gd_gdds_pack_field_flags(&fields[i]) != 0U || fields[i].pad_value_bits != 0U) {
            has_collate_metadata = 1;
        }
    }
    if (has_collate_metadata != 0) {
        h = gd_gdds_fnv64_str(h, "gdds-collate-v1");
        for (i = 0; i < n_fields; ++i) {
            h = gd_gdds_fnv64_u64(h, gd_gdds_pack_field_flags(&fields[i]));
            h = gd_gdds_fnv64_u64(h, fields[i].pad_value_bits);
        }
    }
    return h != 0U ? h : 1U;
}

static char *gd_gdds_strdup(const char *s)
{
    size_t n;
    char *out;
    if (s == NULL) {
        return NULL;
    }
    n = strlen(s);
    out = (char *)malloc(n + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n + 1U);
    return out;
}

static int gd_gdds_has_suffix(const char *s, const char *suffix)
{
    size_t n;
    size_t m;
    if (s == NULL || suffix == NULL) {
        return 0;
    }
    n = strlen(s);
    m = strlen(suffix);
    if (m > n) {
        return 0;
    }
    return memcmp(s + n - m, suffix, m) == 0 ? 1 : 0;
}

static char *gd_gdds_join_path(const char *dir, const char *name)
{
    size_t dir_len;
    size_t name_len;
    int needs_slash;
    char *out;
    if (dir == NULL || name == NULL) {
        return NULL;
    }
    dir_len = strlen(dir);
    name_len = strlen(name);
    needs_slash = (dir_len > 0U && dir[dir_len - 1U] != '/') ? 1 : 0;
    if (dir_len > SIZE_MAX - name_len - (size_t)needs_slash - 1U) {
        return NULL;
    }
    out = (char *)malloc(dir_len + (size_t)needs_slash + name_len + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, dir, dir_len);
    if (needs_slash != 0) {
        out[dir_len] = '/';
    }
    memcpy(out + dir_len + (size_t)needs_slash, name, name_len + 1U);
    return out;
}

static gd_status gd_gdds_check_add(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (UINT64_MAX - a < b) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = a + b;
    return GD_OK;
}

static gd_status gd_gdds_check_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (a != 0U && b > UINT64_MAX / a) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = a * b;
    return GD_OK;
}

static gd_status gd_gdds_range_in_file(uint64_t offset,
                                       uint64_t nbytes,
                                       size_t file_nbytes)
{
    uint64_t end = 0U;
    gd_status status = gd_gdds_check_add(offset, nbytes, &end);
    if (status != GD_OK) {
        return status;
    }
    if (offset > (uint64_t)file_nbytes || end > (uint64_t)file_nbytes) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_gdds_shape_nbytes(const int64_t *shape,
                                      int rank,
                                      gd_dtype dtype,
                                      size_t *out)
{
    size_t item_size;
    size_t numel = 1U;
    int i;
    if (shape == NULL || out == NULL || rank < 0 || rank > (int)GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(dtype);
    if (item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < rank; ++i) {
        uint64_t dim;
        if (shape[i] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        dim = (uint64_t)shape[i];
        if (dim > (uint64_t)(SIZE_MAX / numel)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        numel *= (size_t)dim;
    }
    if (numel > SIZE_MAX / item_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = numel * item_size;
    return GD_OK;
}

static gd_status gd_gdds_tail_nbytes(const int64_t *shape,
                                     int rank,
                                     gd_dtype dtype,
                                     size_t *out)
{
    return gd_gdds_shape_nbytes(shape, rank, dtype, out);
}

static gd_status gd_gdds_read_header_from_map(const uint8_t *map,
                                              size_t map_len,
                                              gd_gdds_header_internal *out)
{
    if (map == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    if (map_len < GD_GDDS_HEADER_SIZE) {
        return GD_ERR_IO;
    }
    if (memcmp(map + GD_GDDS_HEADER_MAGIC_OFFSET,
               GD_GDDS_MAGIC,
               strlen(GD_GDDS_MAGIC)) != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out->version = gd_gdds_get_le32(map + GD_GDDS_HEADER_VERSION_OFFSET);
    out->header_size = gd_gdds_get_le32(map + GD_GDDS_HEADER_SIZE_OFFSET);
    out->field_count = gd_gdds_get_le32(map + GD_GDDS_HEADER_FIELD_COUNT_OFFSET);
    out->n_samples = gd_gdds_get_le64(map + GD_GDDS_HEADER_N_SAMPLES_OFFSET);
    out->schema_offset = gd_gdds_get_le64(map + GD_GDDS_HEADER_SCHEMA_OFFSET);
    out->index_offset = gd_gdds_get_le64(map + GD_GDDS_HEADER_INDEX_OFFSET);
    out->data_offset = gd_gdds_get_le64(map + GD_GDDS_HEADER_DATA_OFFSET);
    out->data_nbytes = gd_gdds_get_le64(map + GD_GDDS_HEADER_DATA_NBYTES_OFFSET);
    out->schema_hash = gd_gdds_get_le64(map + GD_GDDS_HEADER_SCHEMA_HASH_OFFSET);
    if (out->version != GD_GDDS_VERSION || out->header_size != GD_GDDS_HEADER_SIZE ||
        out->field_count == 0U || out->field_count > GD_GDDS_MAX_FIELDS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_gdds_read_schema(const uint8_t *map,
                                     size_t map_len,
                                     const gd_gdds_header_internal *header,
                                     gd_gdds_field_info **fields_out)
{
    gd_gdds_field_info *fields;
    uint64_t schema_nbytes = 0U;
    uint64_t schema_end = 0U;
    uint64_t computed_hash;
    uint32_t i;
    gd_status status;
    if (map == NULL || header == NULL || fields_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *fields_out = NULL;
    status = gd_gdds_check_mul((uint64_t)header->field_count,
                               GD_GDDS_FIELD_DESC_SIZE,
                               &schema_nbytes);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gdds_check_add(header->schema_offset, schema_nbytes, &schema_end);
    if (status != GD_OK) {
        return status;
    }
    if (header->schema_offset < GD_GDDS_HEADER_SIZE || schema_end > (uint64_t)map_len) {
        return GD_ERR_IO;
    }
    fields = (gd_gdds_field_info *)calloc((size_t)header->field_count, sizeof(*fields));
    if (fields == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < header->field_count; ++i) {
        const uint8_t *src = map + header->schema_offset +
                             (uint64_t)i * GD_GDDS_FIELD_DESC_SIZE;
        uint32_t dtype_u;
        uint32_t rank_u;
        int d;
        uint32_t j;
        if (memchr(src + GD_GDDS_FIELD_NAME_OFFSET, '\0', GD_GDDS_FIELD_NAME_MAX) == NULL) {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        memcpy(fields[i].name, src + GD_GDDS_FIELD_NAME_OFFSET, GD_GDDS_FIELD_NAME_MAX);
        fields[i].name[GD_GDDS_FIELD_NAME_MAX - 1U] = '\0';
        if (fields[i].name[0] == '\0') {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        dtype_u = gd_gdds_get_le32(src + GD_GDDS_FIELD_DTYPE_OFFSET);
        if (dtype_u > (uint32_t)GD_DTYPE_U8 || gd_dtype_size((gd_dtype)dtype_u) == 0U) {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        fields[i].dtype = (gd_dtype)dtype_u;
        rank_u = gd_gdds_get_le32(src + GD_GDDS_FIELD_RANK_OFFSET);
        if (rank_u > GD_MAX_DIMS) {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        fields[i].rank = (int)rank_u;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            fields[i].shape[d] = gd_gdds_get_i64(src + GD_GDDS_FIELD_SHAPE_OFFSET +
                                                 (uint64_t)d * sizeof(uint64_t));
            if (d < fields[i].rank) {
                if (fields[i].shape[d] != -1 && fields[i].shape[d] <= 0) {
                    free(fields);
                    return GD_ERR_INVALID_ARGUMENT;
                }
            } else if (fields[i].shape[d] != 0) {
                free(fields);
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
        {
            const uint64_t flags = gd_gdds_get_le64(src + GD_GDDS_FIELD_FLAGS_OFFSET);
            const uint64_t collate = flags & GD_GDDS_FIELD_FLAGS_COLLATE_MASK;
            const uint64_t generated = (flags & GD_GDDS_FIELD_FLAGS_GENERATED_MASK) >>
                                       GD_GDDS_FIELD_FLAGS_GENERATED_SHIFT;
            const uint64_t ragged = (flags & GD_GDDS_FIELD_FLAGS_RAGGED_DIM_MASK) >>
                                    GD_GDDS_FIELD_FLAGS_RAGGED_DIM_SHIFT;
            const uint64_t source_field = (flags & GD_GDDS_FIELD_FLAGS_SOURCE_MASK) >>
                                          GD_GDDS_FIELD_FLAGS_SOURCE_SHIFT;
            if (collate > (uint64_t)GD_GDDS_COLLATE_GENERATED ||
                generated > (uint64_t)GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS ||
                ragged > (uint64_t)GD_MAX_DIMS ||
                source_field > (uint64_t)GD_GDDS_MAX_FIELDS) {
                free(fields);
                return GD_ERR_INVALID_ARGUMENT;
            }
            fields[i].collate = (gd_gdds_collate_mode)collate;
            fields[i].generated = (gd_gdds_generated_kind)generated;
            fields[i].ragged_dim = ragged == 0U ? -1 : (int)ragged - 1;
            fields[i].source_field = source_field == 0U ? -1 : (int)source_field - 1;
            fields[i].pad_value_bits = gd_gdds_get_le64(src + GD_GDDS_FIELD_PAD_VALUE_OFFSET);
            if (fields[i].collate == GD_GDDS_COLLATE_GENERATED) {
                if (fields[i].generated == GD_GDDS_GENERATED_NONE || fields[i].ragged_dim >= 0 ||
                    fields[i].pad_value_bits != 0U || fields[i].source_field < 0 ||
                    fields[i].source_field >= (int)header->field_count) {
                    free(fields);
                    return GD_ERR_INVALID_ARGUMENT;
                }
            } else {
                if (fields[i].generated != GD_GDDS_GENERATED_NONE || fields[i].source_field >= 0) {
                    free(fields);
                    return GD_ERR_INVALID_ARGUMENT;
                }
                if (fields[i].collate == GD_GDDS_COLLATE_STACK) {
                    if (fields[i].pad_value_bits != 0U) {
                        free(fields);
                        return GD_ERR_INVALID_ARGUMENT;
                    }
                    fields[i].ragged_dim = -1;
                } else if (fields[i].ragged_dim != 0 || fields[i].rank <= 0 || fields[i].shape[0] != -1 ||
                           (fields[i].collate != GD_GDDS_COLLATE_PAD_LONGEST &&
                            fields[i].pad_value_bits != 0U)) {
                    free(fields);
                    return GD_ERR_INVALID_ARGUMENT;
                }
            }
        }
        for (j = 0U; j < i; ++j) {
            if (strcmp(fields[j].name, fields[i].name) == 0) {
                free(fields);
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    for (i = 0U; i < header->field_count; ++i) {
        if (fields[i].collate == GD_GDDS_COLLATE_GENERATED) {
            if (fields[fields[i].source_field].collate == GD_GDDS_COLLATE_GENERATED) {
                free(fields);
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    computed_hash = gd_gdds_schema_hash(fields, (int)header->field_count);
    if (header->schema_hash != 0U && header->schema_hash != computed_hash) {
        free(fields);
        return GD_ERR_INVALID_ARGUMENT;
    }
    *fields_out = fields;
    return GD_OK;
}

static gd_status gd_gdds_validate_index(const gd_gdds_shard *shard)
{
    uint64_t index_nbytes = 0U;
    uint64_t index_end = 0U;
    uint64_t i;
    gd_status status;
    if (shard == NULL || shard->map == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_check_mul(shard->header.n_samples,
                               GD_GDDS_INDEX_ENTRY_SIZE,
                               &index_nbytes);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gdds_check_add(shard->header.index_offset, index_nbytes, &index_end);
    if (status != GD_OK) {
        return status;
    }
    if (shard->header.index_offset < GD_GDDS_HEADER_SIZE ||
        index_end > (uint64_t)shard->map_len ||
        shard->header.data_offset > (uint64_t)shard->map_len ||
        shard->header.data_nbytes > (uint64_t)shard->map_len - shard->header.data_offset) {
        return GD_ERR_IO;
    }
    for (i = 0U; i < shard->header.n_samples; ++i) {
        const uint8_t *entry = shard->map + shard->header.index_offset +
                               i * GD_GDDS_INDEX_ENTRY_SIZE;
        uint64_t offset = gd_gdds_get_le64(entry);
        uint64_t nbytes = gd_gdds_get_le64(entry + 8);
        uint64_t end = 0U;
        status = gd_gdds_check_add(offset, nbytes, &end);
        if (status != GD_OK) {
            return status;
        }
        if (nbytes < GD_GDDS_RECORD_HEADER_SIZE ||
            offset < shard->header.data_offset ||
            end > shard->header.data_offset + shard->header.data_nbytes ||
            end > (uint64_t)shard->map_len) {
            return GD_ERR_IO;
        }
    }
    return GD_OK;
}

static void gd_gdds_shard_close(gd_gdds_shard *shard)
{
    if (shard == NULL) {
        return;
    }
    if (shard->map != NULL) {
        (void)munmap(shard->map, shard->map_len);
    }
    if (shard->fd >= 0) {
        (void)close(shard->fd);
    }
    free(shard->fields);
    free(shard->path);
    memset(shard, 0, sizeof(*shard));
    shard->fd = -1;
}

static gd_status gd_gdds_shard_open(const char *path,
                                    uint64_t sample_base,
                                    gd_gdds_shard *out)
{
    struct stat st;
    gd_status status;
    if (path == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) {
        return GD_ERR_IO;
    }
    if (fstat(out->fd, &st) != 0 || st.st_size <= 0) {
        gd_gdds_shard_close(out);
        return GD_ERR_IO;
    }
    out->map_len = (size_t)st.st_size;
    out->map = (uint8_t *)mmap(NULL, out->map_len, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->map == MAP_FAILED) {
        out->map = NULL;
        gd_gdds_shard_close(out);
        return GD_ERR_IO;
    }
    status = gd_gdds_read_header_from_map(out->map, out->map_len, &out->header);
    if (status == GD_OK) {
        status = gd_gdds_read_schema(out->map, out->map_len, &out->header, &out->fields);
    }
    if (status == GD_OK) {
        status = gd_gdds_validate_index(out);
    }
    if (status != GD_OK) {
        gd_gdds_shard_close(out);
        return status;
    }
    out->index = out->map + out->header.index_offset;
    out->sample_base = sample_base;
    out->path = gd_gdds_strdup(path);
    if (out->path == NULL) {
        gd_gdds_shard_close(out);
        return GD_ERR_OUT_OF_MEMORY;
    }
    return GD_OK;
}

static int gd_gdds_fields_equal(const gd_gdds_field_info *a,
                                const gd_gdds_field_info *b,
                                int n_fields)
{
    int i;
    int d;
    if (a == NULL || b == NULL || n_fields < 0) {
        return 0;
    }
    for (i = 0; i < n_fields; ++i) {
        if (strcmp(a[i].name, b[i].name) != 0 || a[i].dtype != b[i].dtype ||
            a[i].rank != b[i].rank || a[i].collate != b[i].collate ||
            a[i].generated != b[i].generated || a[i].ragged_dim != b[i].ragged_dim ||
            a[i].source_field != b[i].source_field ||
            a[i].pad_value_bits != b[i].pad_value_bits) {
            return 0;
        }
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            if (a[i].shape[d] != b[i].shape[d]) {
                return 0;
            }
        }
    }
    return 1;
}

static void gd_gdds_impl_destroy(void *impl_v)
{
    gd_gdds_dataset_impl *impl = (gd_gdds_dataset_impl *)impl_v;
    int i;
    if (impl == NULL) {
        return;
    }
    for (i = 0; i < impl->n_shards; ++i) {
        gd_gdds_shard_close(&impl->shards[i]);
    }
    free(impl->shards);
    free(impl->storage_fields);
    free(impl->fields);
    free(impl);
}

static uint64_t gd_gdds_num_samples_cb(const void *impl_v)
{
    const gd_gdds_dataset_impl *impl = (const gd_gdds_dataset_impl *)impl_v;
    return impl != NULL ? impl->n_samples : 0U;
}

static gd_status gd_gdds_impl_from_dataset(const gd_dataset *dataset,
                                           const gd_gdds_dataset_impl **out)
{
    const gd_gdds_dataset_impl *impl;
    if (dataset == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (strcmp(gd_dataset_name(dataset), "gdds") != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    impl = (const gd_gdds_dataset_impl *)gd_dataset_const_data(dataset);
    if (impl == NULL || impl->magic != GD_GDDS_IMPL_MAGIC) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = impl;
    return GD_OK;
}

static gd_status gd_gdds_validate_runtime_fields(const gd_gdds_field_info *fields,
                                                 int n_fields)
{
    int i;
    int j;
    if (fields == NULL || n_fields <= 0 || n_fields > (int)GD_GDDS_MAX_FIELDS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < n_fields; ++i) {
        const gd_gdds_field_info *field = &fields[i];
        int d;
        if (field->name[0] == '\0' ||
            memchr(field->name, '\0', GD_GDDS_FIELD_NAME_MAX) == NULL ||
            gd_dtype_size(field->dtype) == 0U || field->rank < 0 ||
            field->rank > (int)GD_MAX_DIMS ||
            field->collate < GD_GDDS_COLLATE_STACK ||
            field->collate > GD_GDDS_COLLATE_GENERATED ||
            field->generated < GD_GDDS_GENERATED_NONE ||
            field->generated > GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            if (d < field->rank) {
                if (field->shape[d] != -1 && field->shape[d] <= 0) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
            } else if (field->shape[d] != 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
        switch (field->collate) {
        case GD_GDDS_COLLATE_STACK:
            if (field->generated != GD_GDDS_GENERATED_NONE || field->source_field >= 0 ||
                field->pad_value_bits != 0U || field->ragged_dim != -1) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            for (d = 0; d < field->rank; ++d) {
                if (field->shape[d] <= 0) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
            }
            break;
        case GD_GDDS_COLLATE_PAD_LONGEST:
        case GD_GDDS_COLLATE_PACKED_SEQUENCE:
            if (field->generated != GD_GDDS_GENERATED_NONE || field->source_field >= 0 ||
                field->ragged_dim != 0 || field->rank <= 0 || field->shape[0] != -1 ||
                (field->collate == GD_GDDS_COLLATE_PACKED_SEQUENCE && field->pad_value_bits != 0U)) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            for (d = 1; d < field->rank; ++d) {
                if (field->shape[d] <= 0) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
            }
            break;
        case GD_GDDS_COLLATE_GENERATED:
            if (field->generated == GD_GDDS_GENERATED_NONE || field->source_field < 0 ||
                field->source_field >= n_fields || field->ragged_dim != -1 ||
                field->pad_value_bits != 0U) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            break;
        default:
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (j = i + 1; j < n_fields; ++j) {
            if (strcmp(field->name, fields[j].name) == 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    for (i = 0; i < n_fields; ++i) {
        const gd_gdds_field_info *field = &fields[i];
        const gd_gdds_field_info *source;
        if (field->collate != GD_GDDS_COLLATE_GENERATED) {
            continue;
        }
        source = &fields[field->source_field];
        if (source->collate == GD_GDDS_COLLATE_GENERATED) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        switch (field->generated) {
        case GD_GDDS_GENERATED_LENGTHS:
            if (field->dtype != GD_DTYPE_I32 ||
                (source->collate != GD_GDDS_COLLATE_PAD_LONGEST &&
                 source->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE)) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            break;
        case GD_GDDS_GENERATED_MASK:
            if ((field->dtype != GD_DTYPE_U8 && field->dtype != GD_DTYPE_I32) ||
                source->collate != GD_GDDS_COLLATE_PAD_LONGEST) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            break;
        case GD_GDDS_GENERATED_CU_SEQLENS:
        case GD_GDDS_GENERATED_POSITIONS:
            if (field->dtype != GD_DTYPE_I32 ||
                source->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            break;
        case GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS:
            if (field->dtype != GD_DTYPE_I32 || source->dtype != GD_DTYPE_I32 ||
                source->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE || source->rank != 1 ||
                source->shape[0] != -1) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            break;
        default:
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_copy_field_specs(const gd_dataset_field_spec *specs,
                                          int n_specs,
                                          gd_gdds_field_info **out)
{
    gd_gdds_field_info *fields;
    int i;
    gd_status status;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (specs == NULL || n_specs <= 0 || n_specs > (int)GD_GDDS_MAX_FIELDS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    fields = (gd_gdds_field_info *)calloc((size_t)n_specs, sizeof(*fields));
    if (fields == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0; i < n_specs; ++i) {
        size_t name_len;
        int d;
        if (specs[i].name == NULL || specs[i].name[0] == '\0') {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        name_len = strlen(specs[i].name);
        if (name_len >= GD_GDDS_FIELD_NAME_MAX) {
            free(fields);
            return GD_ERR_INVALID_ARGUMENT;
        }
        memcpy(fields[i].name, specs[i].name, name_len + 1U);
        fields[i].dtype = specs[i].dtype;
        fields[i].rank = specs[i].rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            fields[i].shape[d] = specs[i].shape[d];
        }
        fields[i].collate = specs[i].collate;
        fields[i].generated = specs[i].generated;
        fields[i].ragged_dim = specs[i].ragged_dim;
        fields[i].source_field = specs[i].source_field;
        fields[i].pad_value_bits = specs[i].pad_value_bits;
        if (fields[i].collate == GD_GDDS_COLLATE_STACK) {
            fields[i].ragged_dim = -1;
            fields[i].source_field = -1;
        } else if (fields[i].collate == GD_GDDS_COLLATE_PAD_LONGEST ||
                   fields[i].collate == GD_GDDS_COLLATE_PACKED_SEQUENCE) {
            fields[i].ragged_dim = 0;
            fields[i].source_field = -1;
        } else if (fields[i].collate == GD_GDDS_COLLATE_GENERATED) {
            fields[i].ragged_dim = -1;
        }
    }
    status = gd_gdds_validate_runtime_fields(fields, n_specs);
    if (status != GD_OK) {
        free(fields);
        return status;
    }
    *out = fields;
    return GD_OK;
}

static const gd_gdds_shard *gd_gdds_find_shard(const gd_gdds_dataset_impl *impl,
                                               uint64_t sample_index,
                                               uint64_t *local_out)
{
    int lo;
    int hi;
    if (impl == NULL || local_out == NULL || sample_index >= impl->n_samples) {
        return NULL;
    }
    lo = 0;
    hi = impl->n_shards - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        const gd_gdds_shard *shard = &impl->shards[mid];
        uint64_t begin = shard->sample_base;
        uint64_t end = begin + shard->header.n_samples;
        if (sample_index < begin) {
            hi = mid - 1;
        } else if (sample_index >= end) {
            lo = mid + 1;
        } else {
            *local_out = sample_index - begin;
            return shard;
        }
    }
    return NULL;
}

static gd_status gd_gdds_parse_record(const gd_gdds_dataset_impl *impl,
                                      uint64_t sample_index,
                                      gd_gdds_record_view *out)
{
    const gd_gdds_shard *shard;
    const uint8_t *record;
    uint64_t local = 0U;
    uint64_t record_offset;
    uint64_t record_nbytes_u64;
    uint32_t record_field_count;
    uint32_t record_header_nbytes;
    uint64_t payload_nbytes;
    size_t record_nbytes;
    uint32_t i;
    gd_status status;
    if (impl == NULL || out == NULL || sample_index >= impl->n_samples) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    shard = gd_gdds_find_shard(impl, sample_index, &local);
    if (shard == NULL) {
        return GD_ERR_INTERNAL;
    }
    record_offset = gd_gdds_get_le64(shard->index + local * GD_GDDS_INDEX_ENTRY_SIZE);
    record_nbytes_u64 = gd_gdds_get_le64(shard->index + local * GD_GDDS_INDEX_ENTRY_SIZE + 8U);
    status = gd_gdds_range_in_file(record_offset, record_nbytes_u64, shard->map_len);
    if (status != GD_OK) {
        return status;
    }
    if (record_nbytes_u64 > (uint64_t)SIZE_MAX) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    record_nbytes = (size_t)record_nbytes_u64;
    record = shard->map + record_offset;
    if (record_nbytes < GD_GDDS_RECORD_HEADER_SIZE ||
        memcmp(record + GD_GDDS_RECORD_MAGIC_OFFSET,
               GD_GDDS_RECORD_MAGIC,
               strlen(GD_GDDS_RECORD_MAGIC)) != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    record_field_count = gd_gdds_get_le16(record + GD_GDDS_RECORD_FIELD_COUNT_OFFSET);
    record_header_nbytes = gd_gdds_get_le32(record + GD_GDDS_RECORD_HEADER_NBYTES_OFFSET);
    payload_nbytes = gd_gdds_get_le64(record + GD_GDDS_RECORD_PAYLOAD_NBYTES_OFFSET);
    if (record_field_count == 0U || record_field_count > GD_GDDS_MAX_FIELDS ||
        record_field_count > (uint32_t)impl->n_storage_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (record_header_nbytes != GD_GDDS_RECORD_HEADER_SIZE +
            record_field_count * GD_GDDS_RECORD_FIELD_DESC_SIZE ||
        record_header_nbytes > record_nbytes ||
        payload_nbytes > (uint64_t)(record_nbytes - (size_t)record_header_nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out->n_entries = (int)record_field_count;
    for (i = 0U; i < record_field_count; ++i) {
        const uint8_t *entry = record + GD_GDDS_RECORD_HEADER_SIZE +
                               i * GD_GDDS_RECORD_FIELD_DESC_SIZE;
        gd_gdds_record_field_view *view = &out->entries[i];
        uint32_t field_id = gd_gdds_get_le16(entry + GD_GDDS_RECORD_FIELD_ID_OFFSET);
        uint32_t rank = gd_gdds_get_le16(entry + GD_GDDS_RECORD_FIELD_RANK_OFFSET);
        uint64_t data_offset = gd_gdds_get_le64(entry + GD_GDDS_RECORD_FIELD_DATA_OFFSET);
        uint64_t data_nbytes = gd_gdds_get_le64(entry + GD_GDDS_RECORD_FIELD_NBYTES_OFFSET);
        size_t expected_nbytes = 0U;
        int d;
        if (field_id >= (uint32_t)impl->n_storage_fields || rank > GD_MAX_DIMS ||
            data_offset > payload_nbytes || data_nbytes > payload_nbytes - data_offset ||
            data_nbytes > (uint64_t)SIZE_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (out->by_field[field_id] != NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((int)rank != impl->storage_fields[field_id].rank) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        view->field_id = (int)field_id;
        view->rank = (int)rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            view->shape[d] = gd_gdds_get_i64(entry + GD_GDDS_RECORD_FIELD_SHAPE_OFFSET +
                                             (uint64_t)d * sizeof(uint64_t));
            if (d < view->rank) {
                if (view->shape[d] <= 0) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
                if (impl->storage_fields[field_id].shape[d] != -1 &&
                    impl->storage_fields[field_id].shape[d] != view->shape[d]) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
            } else if (view->shape[d] != 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
        status = gd_gdds_shape_nbytes(view->shape,
                                      view->rank,
                                      impl->storage_fields[field_id].dtype,
                                      &expected_nbytes);
        if (status != GD_OK) {
            return status;
        }
        if ((uint64_t)expected_nbytes != data_nbytes) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        (void)gd_gdds_get_le32(entry + GD_GDDS_RECORD_FIELD_FLAGS_OFFSET);
        view->data = record + record_header_nbytes + data_offset;
        view->nbytes = (size_t)data_nbytes;
        out->by_field[field_id] = view;
    }
    return GD_OK;
}

gd_status gd_dataset_open_gdds_with_transform(const char **paths,
                                              int n_paths,
                                              const gd_dataset_transform_config *transform,
                                              gd_dataset **out)
{
    static const gd_dataset_ops ops = {
        .name = "gdds",
        .num_samples = gd_gdds_num_samples_cb,
        .destroy = gd_gdds_impl_destroy,
    };
    gd_gdds_dataset_impl *impl;
    uint64_t sample_base = 0U;
    int i;
    gd_status status;
    const int has_transform = (transform != NULL && transform->transform != NULL) ? 1 : 0;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (paths == NULL || n_paths <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (has_transform != 0 &&
        (transform->output_fields == NULL || transform->n_output_fields <= 0)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    impl = (gd_gdds_dataset_impl *)calloc(1U, sizeof(*impl));
    if (impl == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    impl->magic = GD_GDDS_IMPL_MAGIC;
    impl->shards = (gd_gdds_shard *)calloc((size_t)n_paths, sizeof(*impl->shards));
    if (impl->shards == NULL) {
        free(impl);
        return GD_ERR_OUT_OF_MEMORY;
    }
    impl->n_shards = n_paths;
    for (i = 0; i < n_paths; ++i) {
        status = gd_gdds_shard_open(paths[i], sample_base, &impl->shards[i]);
        if (status != GD_OK) {
            gd_gdds_impl_destroy(impl);
            return status;
        }
        if (i == 0) {
            size_t schema_nbytes = (size_t)impl->shards[i].header.field_count *
                                   sizeof(*impl->storage_fields);
            impl->n_storage_fields = (int)impl->shards[i].header.field_count;
            impl->storage_fields = (gd_gdds_field_info *)malloc(schema_nbytes);
            if (impl->storage_fields == NULL) {
                gd_gdds_impl_destroy(impl);
                return GD_ERR_OUT_OF_MEMORY;
            }
            memcpy(impl->storage_fields, impl->shards[i].fields, schema_nbytes);
        } else if ((int)impl->shards[i].header.field_count != impl->n_storage_fields ||
                   !gd_gdds_fields_equal(impl->storage_fields,
                                         impl->shards[i].fields,
                                         impl->n_storage_fields)) {
            gd_gdds_impl_destroy(impl);
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (UINT64_MAX - sample_base < impl->shards[i].header.n_samples) {
            gd_gdds_impl_destroy(impl);
            return GD_ERR_OUT_OF_MEMORY;
        }
        sample_base += impl->shards[i].header.n_samples;
    }
    impl->n_samples = sample_base;
    if (has_transform != 0) {
        status = gd_gdds_copy_field_specs(transform->output_fields,
                                          transform->n_output_fields,
                                          &impl->fields);
        if (status != GD_OK) {
            gd_gdds_impl_destroy(impl);
            return status;
        }
        impl->n_fields = transform->n_output_fields;
        impl->transform = transform->transform;
        impl->transform_user_data = transform->user_data;
    } else {
        size_t schema_nbytes = (size_t)impl->n_storage_fields * sizeof(*impl->fields);
        impl->fields = (gd_gdds_field_info *)malloc(schema_nbytes);
        if (impl->fields == NULL) {
            gd_gdds_impl_destroy(impl);
            return GD_ERR_OUT_OF_MEMORY;
        }
        memcpy(impl->fields, impl->storage_fields, schema_nbytes);
        impl->n_fields = impl->n_storage_fields;
    }
    impl->schema_hash = gd_gdds_schema_hash(impl->fields, impl->n_fields);
    status = gd_dataset_create(&ops, impl, out);
    if (status != GD_OK) {
        gd_gdds_impl_destroy(impl);
    }
    return status;
}

gd_status gd_dataset_open_gdds(const char **paths, int n_paths, gd_dataset **out)
{
    return gd_dataset_open_gdds_with_transform(paths, n_paths, NULL, out);
}

gd_status gd_dataset_open_gdds_file_with_transform(const char *path,
                                                   const gd_dataset_transform_config *transform,
                                                   gd_dataset **out)
{
    const char *paths[1];
    if (path == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    paths[0] = path;
    return gd_dataset_open_gdds_with_transform(paths, 1, transform, out);
}

gd_status gd_dataset_open_gdds_file(const char *path, gd_dataset **out)
{
    return gd_dataset_open_gdds_file_with_transform(path, NULL, out);
}

static int gd_gdds_string_ptr_cmp(const void *a, const void *b)
{
    const char *const *sa = (const char *const *)a;
    const char *const *sb = (const char *const *)b;
    return strcmp(*sa, *sb);
}

static gd_status gd_gdds_push_path(char ***paths,
                                   int *count,
                                   int *capacity,
                                   char *path)
{
    char **new_paths;
    int new_capacity;
    if (paths == NULL || count == NULL || capacity == NULL || path == NULL) {
        free(path);
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (*count >= *capacity) {
        if (*capacity > 32768) {
            free(path);
            return GD_ERR_OUT_OF_MEMORY;
        }
        new_capacity = *capacity == 0 ? 8 : *capacity * 2;
        if (new_capacity > 65536) {
            free(path);
            return GD_ERR_OUT_OF_MEMORY;
        }
        new_paths = (char **)realloc(*paths, (size_t)new_capacity * sizeof((*paths)[0]));
        if (new_paths == NULL) {
            free(path);
            return GD_ERR_OUT_OF_MEMORY;
        }
        *paths = new_paths;
        *capacity = new_capacity;
    }
    (*paths)[*count] = path;
    *count += 1;
    return GD_OK;
}

gd_status gd_dataset_open_gdds_split_with_transform(const char *dir,
                                                    const char *split,
                                                    const gd_dataset_transform_config *transform,
                                                    gd_dataset **out)
{
    DIR *dp;
    struct dirent *ent;
    char **paths = NULL;
    int count = 0;
    int capacity = 0;
    size_t split_len;
    gd_status status = GD_OK;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (dir == NULL || split == NULL || split[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    split_len = strlen(split);
    dp = opendir(dir);
    if (dp == NULL) {
        return GD_ERR_IO;
    }
    while ((ent = readdir(dp)) != NULL) {
        const char *name = ent->d_name;
        size_t name_len = strlen(name);
        int match_single;
        int match_shard;
        char *joined;
        if (!gd_gdds_has_suffix(name, ".gdds")) {
            continue;
        }
        match_single = (name_len == split_len + 5U &&
                        memcmp(name, split, split_len) == 0 &&
                        memcmp(name + split_len, ".gdds", 5U) == 0) ? 1 : 0;
        match_shard = (name_len > split_len + 6U &&
                       memcmp(name, split, split_len) == 0 &&
                       name[split_len] == '-') ? 1 : 0;
        if (match_single == 0 && match_shard == 0) {
            continue;
        }
        joined = gd_gdds_join_path(dir, name);
        if (joined == NULL) {
            status = GD_ERR_OUT_OF_MEMORY;
            break;
        }
        status = gd_gdds_push_path(&paths, &count, &capacity, joined);
        if (status != GD_OK) {
            break;
        }
    }
    if (closedir(dp) != 0 && status == GD_OK) {
        status = GD_ERR_IO;
    }
    if (status == GD_OK && count <= 0) {
        status = GD_ERR_IO;
    }
    if (status == GD_OK) {
        qsort(paths, (size_t)count, sizeof(paths[0]), gd_gdds_string_ptr_cmp);
        status = gd_dataset_open_gdds_with_transform((const char **)paths, count, transform, out);
    }
    while (count > 0) {
        count -= 1;
        free(paths[count]);
    }
    free(paths);
    return status;
}

gd_status gd_dataset_open_gdds_split(const char *dir,
                                     const char *split,
                                     gd_dataset **out)
{
    return gd_dataset_open_gdds_split_with_transform(dir, split, NULL, out);
}

int gd_gdds_dataset_field_count(const gd_dataset *dataset)
{
    const gd_gdds_dataset_impl *impl;
    if (gd_gdds_impl_from_dataset(dataset, &impl) != GD_OK) {
        return 0;
    }
    return impl->n_fields;
}

int gd_gdds_dataset_field_index(const gd_dataset *dataset, const char *name)
{
    const gd_gdds_dataset_impl *impl;
    int i;
    if (name == NULL || gd_gdds_impl_from_dataset(dataset, &impl) != GD_OK) {
        return -1;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        if (strcmp(impl->fields[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

gd_status gd_gdds_dataset_field_info(const gd_dataset *dataset,
                                     int field_index,
                                     gd_gdds_field_info *out)
{
    const gd_gdds_dataset_impl *impl;
    gd_status status;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    if (field_index < 0 || field_index >= impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = impl->fields[field_index];
    return GD_OK;
}

gd_status gd_gdds_dataset_read_field(const gd_dataset *dataset,
                                     uint64_t sample_index,
                                     int field_index,
                                     gd_gdds_sample_field *out)
{
    const gd_gdds_dataset_impl *impl;
    gd_gdds_record_view record;
    const gd_gdds_record_field_view *view;
    gd_status status;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    if (impl->transform != NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    if (field_index < 0 || field_index >= impl->n_storage_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_parse_record(impl, sample_index, &record);
    if (status != GD_OK) {
        return status;
    }
    view = record.by_field[field_index];
    if (view == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memcpy(out->name, impl->storage_fields[field_index].name, sizeof(out->name));
    out->name[sizeof(out->name) - 1U] = '\0';
    out->dtype = impl->storage_fields[field_index].dtype;
    out->rank = view->rank;
    memcpy(out->shape, view->shape, sizeof(out->shape));
    out->data = view->data;
    out->nbytes = view->nbytes;
    return GD_OK;
}


static gd_status gd_gdds_require_source_field(const gd_gdds_dataset_impl *impl,
                                              int source_field,
                                              gd_gdds_collate_mode expected)
{
    if (impl == NULL || source_field < 0 || source_field >= impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (impl->fields[source_field].collate != expected) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_gdds_validate_varlen_source(const gd_gdds_field_info *info)
{
    int d;
    if (info == NULL || info->rank <= 0 || info->rank > (int)GD_MAX_DIMS ||
        info->ragged_dim != 0 || info->shape[0] != -1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 1; d < info->rank; ++d) {
        if (info->shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_init_one_batch_field(const gd_gdds_dataset_impl *impl,
                                              int field_index,
                                              int batch_size,
                                              gd_batch_field_desc *out)
{
    const gd_gdds_field_info *info;
    int d;
    if (impl == NULL || out == NULL || field_index < 0 || field_index >= impl->n_fields ||
        batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    info = &impl->fields[field_index];
    memset(out, 0, sizeof(*out));
    out->name = info->name;
    out->dtype = info->dtype;
    switch (info->collate) {
    case GD_GDDS_COLLATE_STACK:
        if (info->rank + 1 > (int)GD_BATCH_MAX_RANK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        out->rank = info->rank + 1;
        out->sizes[0] = (int64_t)batch_size;
        for (d = 0; d < info->rank; ++d) {
            if (info->shape[d] <= 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->sizes[d + 1] = info->shape[d];
        }
        break;
    case GD_GDDS_COLLATE_PAD_LONGEST:
        if (gd_gdds_validate_varlen_source(info) != GD_OK ||
            info->rank + 1 > (int)GD_BATCH_MAX_RANK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        out->rank = info->rank + 1;
        out->sizes[0] = (int64_t)batch_size;
        out->sizes[1] = -1;
        for (d = 1; d < info->rank; ++d) {
            out->sizes[d + 1] = info->shape[d];
        }
        break;
    case GD_GDDS_COLLATE_PACKED_SEQUENCE:
        if (gd_gdds_validate_varlen_source(info) != GD_OK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        out->rank = info->rank;
        out->sizes[0] = -1;
        for (d = 1; d < info->rank; ++d) {
            out->sizes[d] = info->shape[d];
        }
        break;
    case GD_GDDS_COLLATE_GENERATED:
        switch (info->generated) {
        case GD_GDDS_GENERATED_LENGTHS:
            if (info->dtype != GD_DTYPE_I32) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            if (gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PAD_LONGEST) != GD_OK &&
                gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PACKED_SEQUENCE) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->rank = 1;
            out->sizes[0] = (int64_t)batch_size;
            break;
        case GD_GDDS_GENERATED_MASK:
            if (info->dtype != GD_DTYPE_U8 && info->dtype != GD_DTYPE_I32) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            if (gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PAD_LONGEST) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->rank = 2;
            out->sizes[0] = (int64_t)batch_size;
            out->sizes[1] = -1;
            break;
        case GD_GDDS_GENERATED_CU_SEQLENS:
            if (info->dtype != GD_DTYPE_I32 ||
                gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PACKED_SEQUENCE) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->rank = 1;
            out->sizes[0] = (int64_t)batch_size + 1;
            break;
        case GD_GDDS_GENERATED_POSITIONS:
            if (info->dtype != GD_DTYPE_I32 ||
                gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PACKED_SEQUENCE) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->rank = 1;
            out->sizes[0] = -1;
            break;
        case GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS:
            if (info->dtype != GD_DTYPE_I32 || impl->fields[info->source_field].dtype != GD_DTYPE_I32 ||
                gd_gdds_require_source_field(impl,
                                             info->source_field,
                                             GD_GDDS_COLLATE_PACKED_SEQUENCE) != GD_OK) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            out->rank = 1;
            out->sizes[0] = -1;
            break;
        default:
            return GD_ERR_INVALID_ARGUMENT;
        }
        break;
    default:
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

gd_status _gd_gdds_init_batch_fields(const gd_dataset *dataset,
                                      int batch_size,
                                      gd_batch_field_desc **fields_out,
                                      int *n_fields_out)
{
    const gd_gdds_dataset_impl *impl;
    gd_batch_field_desc *fields;
    int i;
    gd_status status;
    if (fields_out == NULL || n_fields_out == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *fields_out = NULL;
    *n_fields_out = 0;
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    fields = (gd_batch_field_desc *)calloc((size_t)impl->n_fields, sizeof(*fields));
    if (fields == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        status = gd_gdds_init_one_batch_field(impl, i, batch_size, &fields[i]);
        if (status != GD_OK) {
            free(fields);
            return status;
        }
    }
    *fields_out = fields;
    *n_fields_out = impl->n_fields;
    return GD_OK;
}

static gd_status gd_gdds_field_shape_nbytes(const gd_gdds_field_info *info,
                                            size_t *out)
{
    if (info == NULL || info->collate != GD_GDDS_COLLATE_STACK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_gdds_shape_nbytes(info->shape, info->rank, info->dtype, out);
}

static gd_status gd_gdds_varlen_tail_nbytes(const gd_gdds_field_info *info,
                                            size_t *out)
{
    if (gd_gdds_validate_varlen_source(info) != GD_OK || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_gdds_tail_nbytes(&info->shape[1], info->rank - 1, info->dtype, out);
}

static gd_status gd_gdds_validate_record_field_shape(const gd_gdds_field_info *info,
                                                     const gd_gdds_record_field_view *view)
{
    int d;
    if (info == NULL || view == NULL || view->rank != info->rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 0; d < info->rank; ++d) {
        if (view->shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (info->shape[d] != -1 && info->shape[d] != view->shape[d]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_fill_scalar(void *dst,
                                     size_t nbytes,
                                     gd_dtype dtype,
                                     uint64_t value_bits)
{
    size_t item_size;
    size_t count;
    uint8_t scalar[8];
    size_t i;
    if (dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(dtype);
    if (item_size == 0U || item_size > sizeof(scalar) || nbytes % item_size != 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (value_bits == 0U) {
        memset(dst, 0, nbytes);
        return GD_OK;
    }
    for (i = 0U; i < sizeof(scalar); ++i) {
        scalar[i] = (uint8_t)((value_bits >> (8U * (uint32_t)i)) & UINT64_C(0xff));
    }
    count = nbytes / item_size;
    for (i = 0U; i < count; ++i) {
        memcpy((uint8_t *)dst + i * item_size, scalar, item_size);
    }
    return GD_OK;
}

static gd_status gd_gdds_strides(const int64_t *shape,
                                 int rank,
                                 size_t item_size,
                                 size_t *strides)
{
    int i;
    if (shape == NULL || strides == NULL || rank < 0 || rank > (int)GD_MAX_DIMS ||
        item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (rank == 0) {
        return GD_OK;
    }
    i = rank - 1;
    strides[i] = item_size;
    while (i > 0) {
        uint64_t dim;
        if (shape[i] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        dim = (uint64_t)shape[i];
        if (dim > (uint64_t)(SIZE_MAX / strides[i])) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        strides[i - 1] = strides[i] * (size_t)dim;
        i -= 1;
    }
    return GD_OK;
}

static void gd_gdds_copy_rect(uint8_t *dst,
                              const uint8_t *src,
                              const size_t *dst_strides,
                              const size_t *src_strides,
                              const int64_t *copy_shape,
                              int rank,
                              int dim,
                              size_t item_size)
{
    int64_t i;
    if (dim == rank - 1) {
        (void)memcpy(dst, src, (size_t)copy_shape[dim] * item_size);
        return;
    }
    for (i = 0; i < copy_shape[dim]; ++i) {
        gd_gdds_copy_rect(dst + (size_t)i * dst_strides[dim],
                          src + (size_t)i * src_strides[dim],
                          dst_strides,
                          src_strides,
                          copy_shape,
                          rank,
                          dim + 1,
                          item_size);
    }
}

static gd_status gd_gdds_copy_value(uint8_t *dst,
                                    const int64_t *dst_shape,
                                    const gd_gdds_record_field_view *src_view,
                                    gd_dtype dtype,
                                    int truncate,
                                    int zero_pad)
{
    size_t item_size;
    size_t src_nbytes = 0U;
    int64_t copy_shape[GD_MAX_DIMS];
    size_t dst_strides[GD_MAX_DIMS];
    size_t src_strides[GD_MAX_DIMS];
    int exact = 1;
    int partial = 0;
    int d;
    gd_status status;
    if (dst == NULL || dst_shape == NULL || src_view == NULL || src_view->data == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(dtype);
    if (item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_shape_nbytes(src_view->shape, src_view->rank, dtype, &src_nbytes);
    if (status != GD_OK) {
        return status;
    }
    if (src_nbytes != src_view->nbytes) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (src_view->rank == 0) {
        (void)memcpy(dst, src_view->data, item_size);
        return GD_OK;
    }
    for (d = 0; d < src_view->rank; ++d) {
        if (dst_shape[d] <= 0 || src_view->shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (src_view->shape[d] > dst_shape[d]) {
            if (truncate == 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            copy_shape[d] = dst_shape[d];
            exact = 0;
            partial = 1;
        } else {
            copy_shape[d] = src_view->shape[d];
            if (src_view->shape[d] != dst_shape[d]) {
                exact = 0;
                partial = 1;
            }
        }
    }
    if (exact != 0) {
        (void)memcpy(dst, src_view->data, src_view->nbytes);
        return GD_OK;
    }
    if (partial != 0 && zero_pad == 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_strides(dst_shape, src_view->rank, item_size, dst_strides);
    if (status == GD_OK) {
        status = gd_gdds_strides(src_view->shape, src_view->rank, item_size, src_strides);
    }
    if (status != GD_OK) {
        return status;
    }
    gd_gdds_copy_rect(dst,
                      src_view->data,
                      dst_strides,
                      src_strides,
                      copy_shape,
                      src_view->rank,
                      0,
                      item_size);
    return GD_OK;
}

static gd_status gd_gdds_batch_field_sizes(const gd_gdds_dataset_impl *impl,
                                           int field_index,
                                           int batch_size,
                                           const int64_t *max_lens,
                                           const int64_t *total_lens,
                                           int64_t sizes[GD_BATCH_MAX_RANK],
                                           int *rank_out)
{
    const gd_gdds_field_info *info;
    const gd_gdds_field_info *source;
    int d;
    if (impl == NULL || max_lens == NULL || total_lens == NULL || sizes == NULL ||
        rank_out == NULL || field_index < 0 || field_index >= impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    info = &impl->fields[field_index];
    memset(sizes, 0, (size_t)GD_BATCH_MAX_RANK * sizeof(sizes[0]));
    switch (info->collate) {
    case GD_GDDS_COLLATE_STACK:
        sizes[0] = (int64_t)batch_size;
        for (d = 0; d < info->rank; ++d) {
            sizes[d + 1] = info->shape[d];
        }
        *rank_out = info->rank + 1;
        return GD_OK;
    case GD_GDDS_COLLATE_PAD_LONGEST:
        sizes[0] = (int64_t)batch_size;
        sizes[1] = max_lens[field_index];
        for (d = 1; d < info->rank; ++d) {
            sizes[d + 1] = info->shape[d];
        }
        *rank_out = info->rank + 1;
        return GD_OK;
    case GD_GDDS_COLLATE_PACKED_SEQUENCE:
        sizes[0] = total_lens[field_index];
        for (d = 1; d < info->rank; ++d) {
            sizes[d] = info->shape[d];
        }
        *rank_out = info->rank;
        return GD_OK;
    case GD_GDDS_COLLATE_GENERATED:
        source = &impl->fields[info->source_field];
        (void)source;
        switch (info->generated) {
        case GD_GDDS_GENERATED_LENGTHS:
            sizes[0] = (int64_t)batch_size;
            *rank_out = 1;
            return GD_OK;
        case GD_GDDS_GENERATED_MASK:
            sizes[0] = (int64_t)batch_size;
            sizes[1] = max_lens[info->source_field];
            *rank_out = 2;
            return GD_OK;
        case GD_GDDS_GENERATED_CU_SEQLENS:
            sizes[0] = (int64_t)batch_size + 1;
            *rank_out = 1;
            return GD_OK;
        case GD_GDDS_GENERATED_POSITIONS:
            sizes[0] = total_lens[info->source_field];
            *rank_out = 1;
            return GD_OK;
        case GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS:
            sizes[0] = total_lens[info->source_field] + 1;
            *rank_out = 1;
            return GD_OK;
        default:
            return GD_ERR_INVALID_ARGUMENT;
        }
    default:
        return GD_ERR_INVALID_ARGUMENT;
    }
}

static gd_status gd_gdds_prepare_batch_storage(gd_batch *batch,
                                               const gd_gdds_dataset_impl *impl,
                                               int batch_size,
                                               const int64_t *max_lens,
                                               const int64_t *total_lens)
{
    int i;
    gd_status status;
    if (batch == NULL || impl == NULL || max_lens == NULL || total_lens == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        int64_t sizes[GD_BATCH_MAX_RANK];
        int rank = 0;
        status = gd_gdds_batch_field_sizes(impl,
                                           i,
                                           batch_size,
                                           max_lens,
                                           total_lens,
                                           sizes,
                                           &rank);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_batch_resize_field(batch, i, info->dtype, rank, sizes, 0);
        if (status != GD_OK) {
            return status;
        }
        if (info->collate == GD_GDDS_COLLATE_PAD_LONGEST) {
            status = gd_gdds_fill_scalar(gd_batch_host_data(batch, i),
                                         gd_batch_field_nbytes(batch, i),
                                         info->dtype,
                                         info->pad_value_bits);
        } else {
            status = gd_gdds_fill_scalar(gd_batch_host_data(batch, i),
                                         gd_batch_field_nbytes(batch, i),
                                         info->dtype,
                                         0U);
        }
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_compute_batch_lengths(const gd_gdds_dataset_impl *impl,
                                               const uint64_t *sample_ids,
                                               int batch_size,
                                               int64_t *lengths,
                                               int64_t *max_lens,
                                               int64_t *total_lens,
                                               int64_t *offsets)
{
    int b;
    int i;
    gd_status status;
    if (impl == NULL || sample_ids == NULL || lengths == NULL || max_lens == NULL ||
        total_lens == NULL || offsets == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdds_record_view record;
        status = gd_gdds_parse_record(impl, sample_ids[b], &record);
        if (status != GD_OK) {
            return status;
        }
        for (i = 0; i < impl->n_fields; ++i) {
            const gd_gdds_field_info *info = &impl->fields[i];
            const gd_gdds_record_field_view *view;
            if (info->collate == GD_GDDS_COLLATE_GENERATED) {
                continue;
            }
            view = record.by_field[i];
            if (view == NULL) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            status = gd_gdds_validate_record_field_shape(info, view);
            if (status != GD_OK) {
                return status;
            }
            if (info->collate == GD_GDDS_COLLATE_STACK) {
                size_t expected_nbytes = 0U;
                status = gd_gdds_field_shape_nbytes(info, &expected_nbytes);
                if (status != GD_OK || expected_nbytes != view->nbytes) {
                    return status != GD_OK ? status : GD_ERR_INVALID_ARGUMENT;
                }
            } else {
                const int64_t len = view->shape[0];
                if (len <= 0 || len > (int64_t)INT32_MAX) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
                lengths[(size_t)i * (size_t)batch_size + (size_t)b] = len;
                if (len > max_lens[i]) {
                    max_lens[i] = len;
                }
                if (INT64_MAX - total_lens[i] < len) {
                    return GD_ERR_OUT_OF_MEMORY;
                }
                total_lens[i] += len;
            }
        }
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        int64_t total = 0;
        offsets[(size_t)i * (size_t)(batch_size + 1)] = 0;
        if (info->collate != GD_GDDS_COLLATE_PAD_LONGEST &&
            info->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE) {
            continue;
        }
        for (b = 0; b < batch_size; ++b) {
            const int64_t len = lengths[(size_t)i * (size_t)batch_size + (size_t)b];
            total += len;
            offsets[(size_t)i * (size_t)(batch_size + 1) + (size_t)b + 1U] = total;
        }
        if (total != total_lens[i] || total > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_copy_stack_field(gd_batch *batch,
                                          int field_index,
                                          int batch_row,
                                          const gd_gdds_field_info *info,
                                          const gd_gdds_record_field_view *view)
{
    size_t sample_nbytes = 0U;
    uint8_t *dst;
    gd_status status;
    if (batch == NULL || info == NULL || view == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_field_shape_nbytes(info, &sample_nbytes);
    if (status != GD_OK) {
        return status;
    }
    if (view->nbytes != sample_nbytes) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (uint8_t *)gd_batch_host_data(batch, field_index) +
          (size_t)batch_row * sample_nbytes;
    memcpy(dst, view->data, sample_nbytes);
    return GD_OK;
}

static gd_status gd_gdds_copy_pad_field(gd_batch *batch,
                                        int field_index,
                                        int batch_row,
                                        const gd_gdds_field_info *info,
                                        const gd_gdds_record_field_view *view)
{
    int64_t dst_shape[GD_MAX_DIMS];
    size_t row_nbytes = 0U;
    uint8_t *dst;
    int d;
    gd_status status;
    if (batch == NULL || info == NULL || view == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst_shape[0] = gd_batch_field_dim(batch, field_index, 1);
    for (d = 1; d < info->rank; ++d) {
        dst_shape[d] = info->shape[d];
    }
    for (; d < (int)GD_MAX_DIMS; ++d) {
        dst_shape[d] = 0;
    }
    status = gd_gdds_shape_nbytes(dst_shape, info->rank, info->dtype, &row_nbytes);
    if (status != GD_OK) {
        return status;
    }
    dst = (uint8_t *)gd_batch_host_data(batch, field_index) +
          (size_t)batch_row * row_nbytes;
    return gd_gdds_copy_value(dst, dst_shape, view, info->dtype, 0, 1);
}

static gd_status gd_gdds_copy_packed_field(gd_batch *batch,
                                           int field_index,
                                           int batch_row,
                                           const gd_gdds_field_info *info,
                                           const gd_gdds_record_field_view *view,
                                           const int64_t *offsets,
                                           int batch_size)
{
    size_t token_nbytes = 0U;
    uint8_t *dst;
    int64_t offset;
    gd_status status;
    if (batch == NULL || info == NULL || view == NULL || offsets == NULL || batch_row < 0 ||
        batch_row >= batch_size) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_varlen_tail_nbytes(info, &token_nbytes);
    if (status != GD_OK) {
        return status;
    }
    offset = offsets[(size_t)field_index * (size_t)(batch_size + 1) + (size_t)batch_row];
    if (offset < 0 || (uint64_t)offset > (uint64_t)(SIZE_MAX / token_nbytes)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    dst = (uint8_t *)gd_batch_host_data(batch, field_index) + (size_t)offset * token_nbytes;
    memcpy(dst, view->data, view->nbytes);
    return GD_OK;
}

static gd_status gd_gdds_fill_lengths(gd_batch *batch,
                                      int field_index,
                                      int source_field,
                                      int batch_size,
                                      const int64_t *lengths)
{
    int32_t *dst;
    int b;
    if (batch == NULL || lengths == NULL || source_field < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (int32_t *)gd_batch_host_data(batch, field_index);
    if (dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (b = 0; b < batch_size; ++b) {
        const int64_t len = lengths[(size_t)source_field * (size_t)batch_size + (size_t)b];
        if (len < 0 || len > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        dst[b] = (int32_t)len;
    }
    return GD_OK;
}

static gd_status gd_gdds_fill_mask(gd_batch *batch,
                                   int field_index,
                                   int source_field,
                                   int batch_size,
                                   const int64_t *lengths)
{
    const gd_dtype dtype = gd_batch_field_dtype(batch, field_index);
    const int64_t width = gd_batch_field_dim(batch, field_index, 1);
    int b;
    if (batch == NULL || lengths == NULL || source_field < 0 || width <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (dtype == GD_DTYPE_U8) {
        uint8_t *dst = (uint8_t *)gd_batch_host_data(batch, field_index);
        if (dst == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (b = 0; b < batch_size; ++b) {
            const int64_t len = lengths[(size_t)source_field * (size_t)batch_size + (size_t)b];
            if (len < 0 || len > width) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            memset(dst + (size_t)b * (size_t)width, 1, (size_t)len);
        }
        return GD_OK;
    }
    if (dtype == GD_DTYPE_I32) {
        int32_t *dst = (int32_t *)gd_batch_host_data(batch, field_index);
        int64_t j;
        if (dst == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (b = 0; b < batch_size; ++b) {
            const int64_t len = lengths[(size_t)source_field * (size_t)batch_size + (size_t)b];
            if (len < 0 || len > width) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            for (j = 0; j < len; ++j) {
                dst[(size_t)b * (size_t)width + (size_t)j] = 1;
            }
        }
        return GD_OK;
    }
    return GD_ERR_INVALID_ARGUMENT;
}

static gd_status gd_gdds_fill_cu_seqlens(gd_batch *batch,
                                         int field_index,
                                         int source_field,
                                         int batch_size,
                                         const int64_t *offsets)
{
    int32_t *dst;
    int b;
    if (batch == NULL || offsets == NULL || source_field < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (int32_t *)gd_batch_host_data(batch, field_index);
    if (dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (b = 0; b <= batch_size; ++b) {
        const int64_t offset = offsets[(size_t)source_field * (size_t)(batch_size + 1) + (size_t)b];
        if (offset < 0 || offset > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        dst[b] = (int32_t)offset;
    }
    return GD_OK;
}

static gd_status gd_gdds_fill_cu_seqlens_from_lengths(gd_batch *batch,
                                                      int field_index,
                                                      int source_field)
{
    const int32_t *lengths;
    int32_t *dst;
    int64_t source_count;
    int64_t out_count;
    int64_t offset = 0;
    int64_t i;
    if (batch == NULL || source_field < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    source_count = gd_batch_field_dim(batch, source_field, 0);
    out_count = gd_batch_field_dim(batch, field_index, 0);
    if (source_count <= 0 || out_count != source_count + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    lengths = (const int32_t *)gd_batch_host_data(batch, source_field);
    dst = (int32_t *)gd_batch_host_data(batch, field_index);
    if (lengths == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst[0] = 0;
    for (i = 0; i < source_count; ++i) {
        const int32_t len = lengths[i];
        if (len <= 0 || offset > (int64_t)INT32_MAX - (int64_t)len) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        offset += (int64_t)len;
        dst[i + 1] = (int32_t)offset;
    }
    return GD_OK;
}

static gd_status gd_gdds_fill_positions(gd_batch *batch,
                                        int field_index,
                                        int source_field,
                                        int batch_size,
                                        const int64_t *offsets)
{
    int32_t *dst;
    int b;
    if (batch == NULL || offsets == NULL || source_field < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (int32_t *)gd_batch_host_data(batch, field_index);
    if (dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (b = 0; b < batch_size; ++b) {
        const int64_t begin = offsets[(size_t)source_field * (size_t)(batch_size + 1) + (size_t)b];
        const int64_t end = offsets[(size_t)source_field * (size_t)(batch_size + 1) + (size_t)b + 1U];
        int64_t p;
        if (begin < 0 || end < begin || end > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (p = begin; p < end; ++p) {
            dst[p] = (int32_t)(p - begin);
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_fill_generated_field(gd_batch *batch,
                                              const gd_gdds_field_info *info,
                                              int field_index,
                                              int batch_size,
                                              const int64_t *lengths,
                                              const int64_t *offsets)
{
    if (batch == NULL || info == NULL || info->collate != GD_GDDS_COLLATE_GENERATED) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    switch (info->generated) {
    case GD_GDDS_GENERATED_LENGTHS:
        return gd_gdds_fill_lengths(batch, field_index, info->source_field, batch_size, lengths);
    case GD_GDDS_GENERATED_MASK:
        return gd_gdds_fill_mask(batch, field_index, info->source_field, batch_size, lengths);
    case GD_GDDS_GENERATED_CU_SEQLENS:
        return gd_gdds_fill_cu_seqlens(batch, field_index, info->source_field, batch_size, offsets);
    case GD_GDDS_GENERATED_POSITIONS:
        return gd_gdds_fill_positions(batch, field_index, info->source_field, batch_size, offsets);
    case GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS:
        (void)lengths;
        (void)batch_size;
        (void)offsets;
        return gd_gdds_fill_cu_seqlens_from_lengths(batch, field_index, info->source_field);
    default:
        return GD_ERR_INVALID_ARGUMENT;
    }
}

static int gd_gdds_transform_fast_stack_ok(const gd_gdds_dataset_impl *impl)
{
    int i;
    if (impl == NULL || impl->transform == NULL) {
        return 0;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        int d;
        if (info->collate != GD_GDDS_COLLATE_STACK) {
            return 0;
        }
        for (d = 0; d < info->rank; ++d) {
            if (info->shape[d] <= 0) {
                return 0;
            }
        }
    }
    return 1;
}

static gd_status gd_gdds_validate_batch_schema(const gd_gdds_dataset_impl *impl,
                                               gd_batch *batch)
{
    int i;
    if (impl == NULL || batch == NULL || gd_batch_field_count(batch) != impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const char *name = gd_batch_field_name(batch, i);
        if (name == NULL || strcmp(name, impl->fields[i].name) != 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_set_source_sample_from_record(const gd_gdds_dataset_impl *impl,
                                                       const gd_gdds_record_view *record,
                                                       gd_sample *sample)
{
    int i;
    if (impl == NULL || record == NULL || sample == NULL ||
        sample->n_fields != impl->n_storage_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_storage_fields; ++i) {
        const gd_gdds_field_info *info = &impl->storage_fields[i];
        gd_sample_field *field = &sample->fields[i];
        const gd_gdds_record_field_view *view;
        int d;
        field->dtype = info->dtype;
        field->rank = info->rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            field->shape[d] = info->shape[d];
        }
        field->data = NULL;
        field->nbytes = 0U;
        field->capacity_nbytes = 0U;
        field->writable = 0;
        if (info->collate == GD_GDDS_COLLATE_GENERATED) {
            continue;
        }
        view = record->by_field[i];
        if (view == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (gd_gdds_validate_record_field_shape(info, view) != GD_OK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        field->rank = view->rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            field->shape[d] = view->shape[d];
        }
        field->data = view->data;
        field->nbytes = view->nbytes;
    }
    return GD_OK;
}

static gd_status gd_gdds_validate_sample_field(const gd_gdds_field_info *info,
                                               const gd_sample_field *field)
{
    size_t expected_nbytes = 0U;
    int d;
    gd_status status;
    if (info == NULL || field == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (info->collate == GD_GDDS_COLLATE_GENERATED) {
        return GD_OK;
    }
    if (field->dtype != info->dtype || field->rank != info->rank ||
        (field->data == NULL && field->nbytes > 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 0; d < info->rank; ++d) {
        if (field->shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (info->shape[d] != -1 && info->shape[d] != field->shape[d]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    for (; d < (int)GD_MAX_DIMS; ++d) {
        if (field->shape[d] != 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    status = gd_gdds_shape_nbytes(field->shape, field->rank, field->dtype, &expected_nbytes);
    if (status != GD_OK) {
        return status;
    }
    if (expected_nbytes != field->nbytes) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (info->collate == GD_GDDS_COLLATE_STACK) {
        for (d = 0; d < info->rank; ++d) {
            if (info->shape[d] != field->shape[d]) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_validate_transformed_sample(const gd_gdds_dataset_impl *impl,
                                                     const gd_sample *sample)
{
    int i;
    if (impl == NULL || sample == NULL || sample->n_fields != impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        gd_status status = gd_gdds_validate_sample_field(&impl->fields[i], &sample->fields[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static gd_status gd_gdds_update_lengths_from_sample(const gd_gdds_dataset_impl *impl,
                                                    const gd_sample *sample,
                                                    int batch_row,
                                                    int batch_size,
                                                    int64_t *lengths,
                                                    int64_t *max_lens,
                                                    int64_t *total_lens)
{
    int i;
    if (impl == NULL || sample == NULL || lengths == NULL || max_lens == NULL ||
        total_lens == NULL || batch_row < 0 || batch_row >= batch_size) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        const gd_sample_field *field = &sample->fields[i];
        int64_t len;
        if (info->collate != GD_GDDS_COLLATE_PAD_LONGEST &&
            info->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE) {
            continue;
        }
        len = field->shape[0];
        if (len <= 0 || len > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        lengths[(size_t)i * (size_t)batch_size + (size_t)batch_row] = len;
        if (len > max_lens[i]) {
            max_lens[i] = len;
        }
        if (INT64_MAX - total_lens[i] < len) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        total_lens[i] += len;
    }
    return GD_OK;
}

static gd_status gd_gdds_compute_offsets_from_lengths(const gd_gdds_dataset_impl *impl,
                                                      int batch_size,
                                                      const int64_t *lengths,
                                                      const int64_t *total_lens,
                                                      int64_t *offsets)
{
    int i;
    int b;
    if (impl == NULL || lengths == NULL || total_lens == NULL || offsets == NULL ||
        batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        int64_t total = 0;
        offsets[(size_t)i * (size_t)(batch_size + 1)] = 0;
        if (info->collate != GD_GDDS_COLLATE_PAD_LONGEST &&
            info->collate != GD_GDDS_COLLATE_PACKED_SEQUENCE) {
            continue;
        }
        for (b = 0; b < batch_size; ++b) {
            const int64_t len = lengths[(size_t)i * (size_t)batch_size + (size_t)b];
            total += len;
            offsets[(size_t)i * (size_t)(batch_size + 1) + (size_t)b + 1U] = total;
        }
        if (total != total_lens[i] || total > (int64_t)INT32_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    return GD_OK;
}

static void gd_gdds_sample_field_as_view(const gd_sample_field *field,
                                         int field_index,
                                         gd_gdds_record_field_view *view)
{
    int d;
    memset(view, 0, sizeof(*view));
    view->field_id = field_index;
    view->rank = field->rank;
    for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
        view->shape[d] = field->shape[d];
    }
    view->data = (const uint8_t *)field->data;
    view->nbytes = field->nbytes;
}

static gd_status gd_gdds_copy_sample_field_to_batch(gd_batch *batch,
                                                    int field_index,
                                                    int batch_row,
                                                    int batch_size,
                                                    const gd_gdds_field_info *info,
                                                    const gd_sample_field *field,
                                                    const int64_t *offsets)
{
    gd_gdds_record_field_view view;
    if (batch == NULL || info == NULL || field == NULL || offsets == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_gdds_sample_field_as_view(field, field_index, &view);
    switch (info->collate) {
    case GD_GDDS_COLLATE_STACK:
        return gd_gdds_copy_stack_field(batch, field_index, batch_row, info, &view);
    case GD_GDDS_COLLATE_PAD_LONGEST:
        return gd_gdds_copy_pad_field(batch, field_index, batch_row, info, &view);
    case GD_GDDS_COLLATE_PACKED_SEQUENCE:
        return gd_gdds_copy_packed_field(batch,
                                         field_index,
                                         batch_row,
                                         info,
                                         &view,
                                         offsets,
                                         batch_size);
    default:
        return GD_ERR_INVALID_ARGUMENT;
    }
}

static gd_status gd_gdds_alias_stack_sample_to_batch(gd_sample *sample,
                                                     gd_batch *batch,
                                                     const gd_gdds_dataset_impl *impl,
                                                     int batch_row)
{
    int i;
    if (sample == NULL || batch == NULL || impl == NULL || sample->n_fields != impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_sample_reset_from_gdds_fields(sample, impl->fields, impl->n_fields);
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        gd_sample_field *field = &sample->fields[i];
        size_t sample_nbytes = 0U;
        uint8_t *base;
        gd_status status = gd_gdds_field_shape_nbytes(info, &sample_nbytes);
        if (status != GD_OK) {
            return status;
        }
        base = (uint8_t *)gd_batch_host_data(batch, i);
        if (base == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        field->data = base + (size_t)batch_row * sample_nbytes;
        field->nbytes = sample_nbytes;
        field->capacity_nbytes = sample_nbytes;
        field->writable = 1;
    }
    return GD_OK;
}

static gd_status gd_gdds_collate_transformed_fast_stack(const gd_gdds_dataset_impl *impl,
                                                        const uint64_t *sample_ids,
                                                        int batch_size,
                                                        gd_batch *batch)
{
    gd_sample src;
    gd_sample dst;
    int64_t *max_lens = NULL;
    int64_t *total_lens = NULL;
    int b;
    gd_status status;
    memset(&src, 0, sizeof(src));
    memset(&dst, 0, sizeof(dst));
    max_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*max_lens));
    total_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*total_lens));
    if (max_lens == NULL || total_lens == NULL) {
        status = GD_ERR_OUT_OF_MEMORY;
        goto done;
    }
    status = gd_sample_init_from_gdds_fields(&src, impl->storage_fields, impl->n_storage_fields, 0);
    if (status == GD_OK) {
        status = gd_sample_init_from_gdds_fields(&dst, impl->fields, impl->n_fields, 0);
    }
    if (status != GD_OK) {
        goto done;
    }
    status = gd_gdds_prepare_batch_storage(batch, impl, batch_size, max_lens, total_lens);
    if (status != GD_OK) {
        goto done;
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdds_record_view record;
        status = gd_gdds_parse_record(impl, sample_ids[b], &record);
        if (status == GD_OK) {
            status = gd_gdds_set_source_sample_from_record(impl, &record, &src);
        }
        if (status == GD_OK) {
            status = gd_gdds_alias_stack_sample_to_batch(&dst, batch, impl, b);
        }
        if (status == GD_OK) {
            status = impl->transform(&src, &dst, impl->transform_user_data);
        }
        if (status == GD_OK) {
            status = gd_gdds_validate_transformed_sample(impl, &dst);
        }
        if (status != GD_OK) {
            goto done;
        }
    }

done:
    gd_sample_deinit(&dst);
    gd_sample_deinit(&src);
    free(total_lens);
    free(max_lens);
    return status;
}

static gd_status gd_gdds_collate_transformed_general(const gd_gdds_dataset_impl *impl,
                                                     const uint64_t *sample_ids,
                                                     int batch_size,
                                                     gd_batch *batch)
{
    gd_sample src;
    gd_sample *samples = NULL;
    int64_t *lengths = NULL;
    int64_t *max_lens = NULL;
    int64_t *total_lens = NULL;
    int64_t *offsets = NULL;
    int b;
    int i;
    gd_status status;
    memset(&src, 0, sizeof(src));
    status = gd_sample_init_from_gdds_fields(&src, impl->storage_fields, impl->n_storage_fields, 0);
    if (status != GD_OK) {
        return status;
    }
    samples = (gd_sample *)calloc((size_t)batch_size, sizeof(*samples));
    lengths = (int64_t *)calloc((size_t)impl->n_fields * (size_t)batch_size, sizeof(*lengths));
    max_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*max_lens));
    total_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*total_lens));
    offsets = (int64_t *)calloc((size_t)impl->n_fields * ((size_t)batch_size + 1U), sizeof(*offsets));
    if (samples == NULL || lengths == NULL || max_lens == NULL || total_lens == NULL ||
        offsets == NULL) {
        status = GD_ERR_OUT_OF_MEMORY;
        goto done;
    }
    for (b = 0; b < batch_size; ++b) {
        status = gd_sample_init_from_gdds_fields(&samples[b], impl->fields, impl->n_fields, 1);
        if (status != GD_OK) {
            goto done;
        }
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdds_record_view record;
        gd_sample_reset_from_gdds_fields(&samples[b], impl->fields, impl->n_fields);
        status = gd_gdds_parse_record(impl, sample_ids[b], &record);
        if (status == GD_OK) {
            status = gd_gdds_set_source_sample_from_record(impl, &record, &src);
        }
        if (status == GD_OK) {
            status = impl->transform(&src, &samples[b], impl->transform_user_data);
        }
        if (status == GD_OK) {
            status = gd_gdds_validate_transformed_sample(impl, &samples[b]);
        }
        if (status == GD_OK) {
            status = gd_gdds_update_lengths_from_sample(impl,
                                                        &samples[b],
                                                        b,
                                                        batch_size,
                                                        lengths,
                                                        max_lens,
                                                        total_lens);
        }
        if (status != GD_OK) {
            goto done;
        }
    }
    status = gd_gdds_compute_offsets_from_lengths(impl,
                                                  batch_size,
                                                  lengths,
                                                  total_lens,
                                                  offsets);
    if (status != GD_OK) {
        goto done;
    }
    status = gd_gdds_prepare_batch_storage(batch, impl, batch_size, max_lens, total_lens);
    if (status != GD_OK) {
        goto done;
    }
    for (b = 0; b < batch_size; ++b) {
        for (i = 0; i < impl->n_fields; ++i) {
            const gd_gdds_field_info *info = &impl->fields[i];
            if (info->collate == GD_GDDS_COLLATE_GENERATED) {
                continue;
            }
            status = gd_gdds_copy_sample_field_to_batch(batch,
                                                        i,
                                                        b,
                                                        batch_size,
                                                        info,
                                                        &samples[b].fields[i],
                                                        offsets);
            if (status != GD_OK) {
                goto done;
            }
        }
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        if (info->collate != GD_GDDS_COLLATE_GENERATED) {
            continue;
        }
        status = gd_gdds_fill_generated_field(batch, info, i, batch_size, lengths, offsets);
        if (status != GD_OK) {
            goto done;
        }
    }

done:
    if (samples != NULL) {
        for (i = 0; i < batch_size; ++i) {
            gd_sample_deinit(&samples[i]);
        }
    }
    free(offsets);
    free(total_lens);
    free(max_lens);
    free(lengths);
    free(samples);
    gd_sample_deinit(&src);
    return status;
}

static gd_status gd_gdds_collate_transformed(const gd_gdds_dataset_impl *impl,
                                             const uint64_t *sample_ids,
                                             int batch_size,
                                             gd_batch *batch)
{
    if (gd_gdds_transform_fast_stack_ok(impl) != 0) {
        return gd_gdds_collate_transformed_fast_stack(impl, sample_ids, batch_size, batch);
    }
    return gd_gdds_collate_transformed_general(impl, sample_ids, batch_size, batch);
}

gd_status _gd_collate_gdds(gd_dataset *dataset,
                            const uint64_t *sample_ids,
                            int batch_size,
                            gd_batch *batch,
                            void *user_data)
{
    const gd_gdds_dataset_impl *impl;
    int64_t *lengths = NULL;
    int64_t *max_lens = NULL;
    int64_t *total_lens = NULL;
    int64_t *offsets = NULL;
    int b;
    int i;
    gd_status status;
    (void)user_data;
    if (dataset == NULL || sample_ids == NULL || batch == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gdds_validate_batch_schema(impl, batch);
    if (status != GD_OK) {
        return status;
    }
    if (impl->transform != NULL) {
        return gd_gdds_collate_transformed(impl, sample_ids, batch_size, batch);
    }
    lengths = (int64_t *)calloc((size_t)impl->n_fields * (size_t)batch_size, sizeof(*lengths));
    max_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*max_lens));
    total_lens = (int64_t *)calloc((size_t)impl->n_fields, sizeof(*total_lens));
    offsets = (int64_t *)calloc((size_t)impl->n_fields * ((size_t)batch_size + 1U), sizeof(*offsets));
    if (lengths == NULL || max_lens == NULL || total_lens == NULL || offsets == NULL) {
        status = GD_ERR_OUT_OF_MEMORY;
        goto done;
    }
    status = gd_gdds_compute_batch_lengths(impl,
                                           sample_ids,
                                           batch_size,
                                           lengths,
                                           max_lens,
                                           total_lens,
                                           offsets);
    if (status != GD_OK) {
        goto done;
    }
    status = gd_gdds_prepare_batch_storage(batch, impl, batch_size, max_lens, total_lens);
    if (status != GD_OK) {
        goto done;
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdds_record_view record;
        status = gd_gdds_parse_record(impl, sample_ids[b], &record);
        if (status != GD_OK) {
            goto done;
        }
        for (i = 0; i < impl->n_fields; ++i) {
            const gd_gdds_field_info *info = &impl->fields[i];
            const gd_gdds_record_field_view *view;
            if (info->collate == GD_GDDS_COLLATE_GENERATED) {
                continue;
            }
            view = record.by_field[i];
            if (view == NULL) {
                status = GD_ERR_INVALID_ARGUMENT;
                goto done;
            }
            switch (info->collate) {
            case GD_GDDS_COLLATE_STACK:
                status = gd_gdds_copy_stack_field(batch, i, b, info, view);
                break;
            case GD_GDDS_COLLATE_PAD_LONGEST:
                status = gd_gdds_copy_pad_field(batch, i, b, info, view);
                break;
            case GD_GDDS_COLLATE_PACKED_SEQUENCE:
                status = gd_gdds_copy_packed_field(batch,
                                                   i,
                                                   b,
                                                   info,
                                                   view,
                                                   offsets,
                                                   batch_size);
                break;
            default:
                status = GD_ERR_INVALID_ARGUMENT;
                break;
            }
            if (status != GD_OK) {
                goto done;
            }
        }
    }
    for (i = 0; i < impl->n_fields; ++i) {
        const gd_gdds_field_info *info = &impl->fields[i];
        if (info->collate != GD_GDDS_COLLATE_GENERATED) {
            continue;
        }
        status = gd_gdds_fill_generated_field(batch, info, i, batch_size, lengths, offsets);
        if (status != GD_OK) {
            goto done;
        }
    }

done:
    free(offsets);
    free(total_lens);
    free(max_lens);
    free(lengths);
    return status;
}
