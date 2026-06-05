#include <gradients/dataloader.h>
#include <gradients/dataset.h>

#include <dirent.h>
#include <fcntl.h>
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
#define GD_GDDS_HEADER_DATA_HASH_OFFSET 72U

#define GD_GDDS_FIELD_NAME_OFFSET 0U
#define GD_GDDS_FIELD_DTYPE_OFFSET 64U
#define GD_GDDS_FIELD_RANK_OFFSET 68U
#define GD_GDDS_FIELD_SHAPE_OFFSET 72U
#define GD_GDDS_FIELD_FLAGS_OFFSET 136U

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
    uint64_t data_hash;
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
    gd_gdds_field_info *fields;
    int n_fields;
    uint64_t n_samples;
    uint64_t schema_hash;
    uint64_t fingerprint;
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

static uint64_t gd_gdds_schema_hash(const gd_gdds_field_info *fields, int n_fields)
{
    uint64_t h = GD_GDDS_FNV_OFFSET;
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
    out->data_hash = gd_gdds_get_le64(map + GD_GDDS_HEADER_DATA_HASH_OFFSET);
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
        (void)gd_gdds_get_le64(src + GD_GDDS_FIELD_FLAGS_OFFSET);
        for (j = 0U; j < i; ++j) {
            if (strcmp(fields[j].name, fields[i].name) == 0) {
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
            a[i].rank != b[i].rank) {
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
    free(impl->fields);
    free(impl);
}

static uint64_t gd_gdds_num_samples_cb(const void *impl_v)
{
    const gd_gdds_dataset_impl *impl = (const gd_gdds_dataset_impl *)impl_v;
    return impl != NULL ? impl->n_samples : 0U;
}

static uint64_t gd_gdds_fingerprint_cb(const void *impl_v)
{
    const gd_gdds_dataset_impl *impl = (const gd_gdds_dataset_impl *)impl_v;
    return impl != NULL ? impl->fingerprint : 0U;
}

static gd_status gd_gdds_get_u64_cb(const void *impl_v, const char *key, uint64_t *out)
{
    const gd_gdds_dataset_impl *impl = (const gd_gdds_dataset_impl *)impl_v;
    if (impl == NULL || key == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (strcmp(key, "n_samples") == 0) {
        *out = impl->n_samples;
    } else if (strcmp(key, "n_shards") == 0) {
        *out = (uint64_t)impl->n_shards;
    } else if (strcmp(key, "field_count") == 0) {
        *out = (uint64_t)impl->n_fields;
    } else if (strcmp(key, "schema_hash") == 0) {
        *out = impl->schema_hash;
    } else {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
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
        record_field_count > (uint32_t)impl->n_fields) {
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
        if (field_id >= (uint32_t)impl->n_fields || rank > GD_MAX_DIMS ||
            data_offset > payload_nbytes || data_nbytes > payload_nbytes - data_offset ||
            data_nbytes > (uint64_t)SIZE_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (out->by_field[field_id] != NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((int)rank != impl->fields[field_id].rank) {
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
                if (impl->fields[field_id].shape[d] != -1 &&
                    impl->fields[field_id].shape[d] != view->shape[d]) {
                    return GD_ERR_INVALID_ARGUMENT;
                }
            } else if (view->shape[d] != 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
        status = gd_gdds_shape_nbytes(view->shape,
                                      view->rank,
                                      impl->fields[field_id].dtype,
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

static gd_status gd_gdds_compute_fingerprint(gd_gdds_dataset_impl *impl)
{
    uint64_t h = GD_GDDS_FNV_OFFSET;
    int i;
    if (impl == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    h = gd_gdds_fnv64_str(h, "gdds-dataset-v1");
    h = gd_gdds_fnv64_u64(h, (uint64_t)impl->n_shards);
    h = gd_gdds_fnv64_u64(h, impl->n_samples);
    h = gd_gdds_fnv64_u64(h, impl->schema_hash);
    for (i = 0; i < impl->n_shards; ++i) {
        h = gd_gdds_fnv64_u64(h, impl->shards[i].header.n_samples);
        h = gd_gdds_fnv64_u64(h, impl->shards[i].header.data_nbytes);
        h = gd_gdds_fnv64_u64(h, impl->shards[i].header.data_hash);
    }
    impl->fingerprint = h != 0U ? h : 1U;
    return GD_OK;
}

gd_status gd_dataset_open_gdds(const char **paths, int n_paths, gd_dataset **out)
{
    static const gd_dataset_ops ops = {
        .name = "gdds",
        .num_samples = gd_gdds_num_samples_cb,
        .fingerprint = gd_gdds_fingerprint_cb,
        .get_u64 = gd_gdds_get_u64_cb,
        .destroy = gd_gdds_impl_destroy,
    };
    gd_gdds_dataset_impl *impl;
    uint64_t sample_base = 0U;
    int i;
    gd_status status;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (paths == NULL || n_paths <= 0) {
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
                                   sizeof(*impl->fields);
            impl->n_fields = (int)impl->shards[i].header.field_count;
            impl->schema_hash = gd_gdds_schema_hash(impl->shards[i].fields, impl->n_fields);
            impl->fields = (gd_gdds_field_info *)malloc(schema_nbytes);
            if (impl->fields == NULL) {
                gd_gdds_impl_destroy(impl);
                return GD_ERR_OUT_OF_MEMORY;
            }
            memcpy(impl->fields, impl->shards[i].fields, schema_nbytes);
        } else if ((int)impl->shards[i].header.field_count != impl->n_fields ||
                   !gd_gdds_fields_equal(impl->fields,
                                         impl->shards[i].fields,
                                         impl->n_fields)) {
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
    status = gd_gdds_compute_fingerprint(impl);
    if (status == GD_OK) {
        status = gd_dataset_create(&ops, impl, out);
    }
    if (status != GD_OK) {
        gd_gdds_impl_destroy(impl);
    }
    return status;
}

gd_status gd_dataset_open_gdds_file(const char *path, gd_dataset **out)
{
    const char *paths[1];
    if (path == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    paths[0] = path;
    return gd_dataset_open_gdds(paths, 1, out);
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

gd_status gd_dataset_open_gdds_split(const char *dir,
                                     const char *split,
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
        status = gd_dataset_open_gdds((const char **)paths, count, out);
    }
    while (count > 0) {
        count -= 1;
        free(paths[count]);
    }
    free(paths);
    return status;
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
    if (field_index < 0 || field_index >= impl->n_fields) {
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
    memcpy(out->name, impl->fields[field_index].name, sizeof(out->name));
    out->name[sizeof(out->name) - 1U] = '\0';
    out->dtype = impl->fields[field_index].dtype;
    out->rank = view->rank;
    memcpy(out->shape, view->shape, sizeof(out->shape));
    out->data = view->data;
    out->nbytes = view->nbytes;
    return GD_OK;
}

gd_status gd_gdds_init_batch_fields(const gd_dataset *dataset,
                                    int batch_size,
                                    gd_batch_field_desc *fields,
                                    int field_cap,
                                    int *n_fields_out)
{
    const gd_gdds_dataset_impl *impl;
    int i;
    int d;
    gd_status status;
    if (fields == NULL || n_fields_out == NULL || batch_size <= 0 || field_cap < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *n_fields_out = 0;
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    if (field_cap < impl->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < impl->n_fields; ++i) {
        if (impl->fields[i].rank + 1 > (int)GD_BATCH_MAX_RANK) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (d = 0; d < impl->fields[i].rank; ++d) {
            if (impl->fields[i].shape[d] < 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    for (i = 0; i < impl->n_fields; ++i) {
        memset(&fields[i], 0, sizeof(fields[i]));
        fields[i].name = impl->fields[i].name;
        fields[i].dtype = impl->fields[i].dtype;
        fields[i].rank = impl->fields[i].rank + 1;
        fields[i].sizes[0] = (int64_t)batch_size;
        for (d = 0; d < impl->fields[i].rank; ++d) {
            fields[i].sizes[d + 1] = impl->fields[i].shape[d];
        }
    }
    *n_fields_out = impl->n_fields;
    return GD_OK;
}

static gd_status gd_gdds_batch_tail_shape(const gd_batch *batch,
                                          int field_index,
                                          int batch_size,
                                          int64_t *tail_shape,
                                          int *tail_rank_out,
                                          size_t *item_nbytes_out)
{
    int rank;
    gd_dtype dtype;
    int d;
    if (batch == NULL || tail_shape == NULL || tail_rank_out == NULL ||
        item_nbytes_out == NULL || field_index < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    rank = gd_batch_field_rank(batch, field_index);
    dtype = gd_batch_field_dtype(batch, field_index);
    if (rank <= 0 || rank > (int)GD_BATCH_MAX_RANK ||
        gd_batch_field_dim(batch, field_index, 0) != (int64_t)batch_size) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 0; d < rank - 1; ++d) {
        tail_shape[d] = gd_batch_field_dim(batch, field_index, d + 1);
        if (tail_shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    for (; d < (int)GD_MAX_DIMS; ++d) {
        tail_shape[d] = 0;
    }
    *tail_rank_out = rank - 1;
    return gd_gdds_tail_nbytes(tail_shape, rank - 1, dtype, item_nbytes_out);
}

static int gd_gdds_schema_exact_to_batch(const gd_gdds_field_info *info,
                                         const int64_t *tail_shape,
                                         int tail_rank)
{
    int d;
    if (info == NULL || tail_shape == NULL || info->rank != tail_rank) {
        return 0;
    }
    for (d = 0; d < tail_rank; ++d) {
        if (info->shape[d] < 0 || info->shape[d] != tail_shape[d]) {
            return 0;
        }
    }
    return 1;
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

gd_status gd_collate_gdds(gd_dataset *dataset,
                          const uint64_t *sample_ids,
                          int batch_size,
                          gd_batch *batch,
                          void *user_data)
{
    const gd_gdds_collate_config *cfg = (const gd_gdds_collate_config *)user_data;
    const gd_gdds_dataset_impl *impl;
    int out_count;
    int ds_index[GD_GDDS_MAX_FIELDS];
    int exact[GD_GDDS_MAX_FIELDS];
    int tail_rank[GD_GDDS_MAX_FIELDS];
    int64_t tail_shape[GD_GDDS_MAX_FIELDS][GD_MAX_DIMS];
    size_t item_nbytes[GD_GDDS_MAX_FIELDS];
    int zero_pad = cfg == NULL ? 1 : (cfg->zero_pad != 0 ? 1 : 0);
    int truncate = cfg == NULL ? 0 : (cfg->truncate != 0 ? 1 : 0);
    int i;
    int b;
    gd_status status;
    if (dataset == NULL || sample_ids == NULL || batch == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_gdds_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    out_count = gd_batch_field_count(batch);
    if (out_count <= 0 || out_count > (int)GD_GDDS_MAX_FIELDS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < out_count; ++i) {
        const char *name = gd_batch_field_name(batch, i);
        gd_gdds_field_info info;
        void *host = gd_batch_host_data(batch, i);
        size_t field_nbytes;
        ds_index[i] = gd_gdds_dataset_field_index(dataset, name);
        if (ds_index[i] < 0 || host == NULL) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        status = gd_gdds_dataset_field_info(dataset, ds_index[i], &info);
        if (status != GD_OK) {
            return status;
        }
        if (gd_batch_field_dtype(batch, i) != info.dtype) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        status = gd_gdds_batch_tail_shape(batch,
                                          i,
                                          batch_size,
                                          tail_shape[i],
                                          &tail_rank[i],
                                          &item_nbytes[i]);
        if (status != GD_OK) {
            return status;
        }
        if (tail_rank[i] != info.rank) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        exact[i] = gd_gdds_schema_exact_to_batch(&info, tail_shape[i], tail_rank[i]);
        field_nbytes = gd_batch_field_nbytes(batch, i);
        if ((uint64_t)item_nbytes[i] > (uint64_t)(SIZE_MAX / (size_t)batch_size) ||
            field_nbytes != item_nbytes[i] * (size_t)batch_size) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (exact[i] == 0 && zero_pad != 0) {
            (void)memset(host, 0, field_nbytes);
        }
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdds_record_view record;
        status = gd_gdds_parse_record(impl, sample_ids[b], &record);
        if (status != GD_OK) {
            return status;
        }
        for (i = 0; i < out_count; ++i) {
            uint8_t *row = (uint8_t *)gd_batch_host_data(batch, i) +
                           (size_t)b * item_nbytes[i];
            const gd_gdds_record_field_view *view = record.by_field[ds_index[i]];
            if (view == NULL || view->rank != tail_rank[i]) {
                return GD_ERR_INVALID_ARGUMENT;
            }
            status = gd_gdds_copy_value(row,
                                        tail_shape[i],
                                        view,
                                        gd_batch_field_dtype(batch, i),
                                        truncate,
                                        zero_pad);
            if (status != GD_OK) {
                return status;
            }
        }
    }
    return GD_OK;
}
