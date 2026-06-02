#include "gradients/dataset_vlm.h"

#include "../core/internal.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_GDVLM_IMPL_MAGIC UINT64_C(0x6764766c6d647331)
#define GD_GDVLM_MAX_PATH 4096U

typedef struct gd_gdvlm_idx_entry {
    uint32_t shard_idx;
    uint32_t sample_idx;
    uint64_t body_offset;
    uint32_t record_nbytes;
    uint32_t token_len;
    uint16_t label_id;
    uint16_t flags;
    uint32_t raw_pos;
} gd_gdvlm_idx_entry;

typedef struct gd_gdvlm_shard {
    char *path;
    int fd;
    uint8_t *map;
    size_t map_len;
    const uint8_t *payload;
    size_t payload_len;
    gd_gdvlm_header header;
} gd_gdvlm_shard;

typedef struct gd_gdvlm_dataset_impl {
    uint64_t magic;
    char *index_path;
    gd_gdvlm_shard *shards;
    int n_shards;
    gd_gdvlm_idx_entry *entries;
    uint64_t n_samples;
    uint32_t max_text_len;
    size_t patch_nbytes;
    gd_gdvlm_header meta;
    uint64_t fingerprint;
} gd_gdvlm_dataset_impl;

static char *gd_gdvlm_strdup(const char *s)
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

static uint32_t gd_gdvlm_get_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U);
}

static uint32_t gd_gdvlm_get_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
           ((uint32_t)p[3] << 24U);
}

static uint64_t gd_gdvlm_get_le64(const uint8_t *p)
{
    return (uint64_t)gd_gdvlm_get_le32(p) |
           ((uint64_t)gd_gdvlm_get_le32(p + 4) << 32U);
}

static size_t gd_gdvlm_token_dtype_size(gd_gdvlm_token_dtype dtype)
{
    if (dtype == GD_GDVLM_TOKEN_DTYPE_U16) {
        return 2U;
    }
    if (dtype == GD_GDVLM_TOKEN_DTYPE_U32) {
        return 4U;
    }
    return 0U;
}

static gd_status gd_gdvlm_patch_nbytes(const gd_gdvlm_header *h, size_t *out)
{
    uint64_t nbytes;
    if (h == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm patch size args");
    }
    if (h->patch_dtype != GD_GDVLM_PATCH_DTYPE_F16 || h->num_patches == 0U ||
        h->patch_dim == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm patch metadata");
    }
    nbytes = (uint64_t)h->num_patches * (uint64_t)h->patch_dim * 2U;
    if (nbytes > (uint64_t)SIZE_MAX) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm patch payload too large");
    }
    *out = (size_t)nbytes;
    return GD_OK;
}

static gd_status gd_gdvlm_validate_header(const gd_gdvlm_header *h)
{
    if (h == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm header is null");
    }
    if (h->version != GD_GDVLM_VERSION || h->header_size != GD_GDVLM_HEADER_SIZE ||
        h->payload_offset < GD_GDVLM_HEADER_SIZE || h->n_samples == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm header");
    }
    if (h->height == 0U || h->width == 0U || h->channels == 0U ||
        h->patch_size == 0U || h->num_patches == 0U || h->patch_dim == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm image metadata");
    }
    if (gd_gdvlm_token_dtype_size(h->token_dtype) == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm token dtype");
    }
    if (h->patch_dtype != GD_GDVLM_PATCH_DTYPE_F16) {
        return _gd_error(GD_ERR_UNSUPPORTED, "unsupported gdvlm patch dtype");
    }
    return GD_OK;
}

gd_status gd_gdvlm_read_header(const char *path, gd_gdvlm_header *out)
{
    FILE *f;
    uint8_t b[GD_GDVLM_HEADER_SIZE];
    if (path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm header args");
    }
    memset(out, 0, sizeof(*out));
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open gdvlm shard");
    }
    if (fread(b, 1U, sizeof(b), f) != sizeof(b)) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read gdvlm header");
    }
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close gdvlm shard");
    }
    if (memcmp(b, "GDVLMv1\0", 8U) != 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm magic");
    }
    out->version = gd_gdvlm_get_le32(b + 8);
    out->header_size = gd_gdvlm_get_le32(b + 12);
    out->height = gd_gdvlm_get_le32(b + 16);
    out->width = gd_gdvlm_get_le32(b + 20);
    out->channels = gd_gdvlm_get_le32(b + 24);
    out->patch_size = gd_gdvlm_get_le32(b + 28);
    out->num_patches = gd_gdvlm_get_le32(b + 32);
    out->patch_dim = gd_gdvlm_get_le32(b + 36);
    out->patch_dtype = (gd_gdvlm_patch_dtype)gd_gdvlm_get_le32(b + 40);
    out->token_dtype = (gd_gdvlm_token_dtype)gd_gdvlm_get_le32(b + 44);
    out->vocab_size = gd_gdvlm_get_le32(b + 48);
    out->shard_idx = gd_gdvlm_get_le32(b + 52);
    out->num_shards = gd_gdvlm_get_le32(b + 56);
    out->payload_offset = gd_gdvlm_get_le32(b + 60);
    out->n_samples = gd_gdvlm_get_le64(b + 64);
    out->tokenizer_hash = gd_gdvlm_get_le64(b + 72);
    return gd_gdvlm_validate_header(out);
}

static void gd_gdvlm_shard_close(gd_gdvlm_shard *shard)
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
    free(shard->path);
    memset(shard, 0, sizeof(*shard));
    shard->fd = -1;
}

static gd_status gd_gdvlm_shard_open(const char *path,
                                     uint32_t expected_idx,
                                     gd_gdvlm_shard *out)
{
    struct stat st;
    gd_status status;
    if (path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm shard open args");
    }
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    status = gd_gdvlm_read_header(path, &out->header);
    if (status != GD_OK) {
        return status;
    }
    if (out->header.shard_idx != expected_idx) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm shard index mismatch");
    }
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) {
        return _gd_error(GD_ERR_IO, "failed to open gdvlm shard");
    }
    if (fstat(out->fd, &st) != 0) {
        gd_gdvlm_shard_close(out);
        return _gd_error(GD_ERR_IO, "failed to stat gdvlm shard");
    }
    if (st.st_size < 0 || (uint64_t)st.st_size < (uint64_t)out->header.payload_offset) {
        gd_gdvlm_shard_close(out);
        return _gd_error(GD_ERR_IO, "gdvlm shard truncated");
    }
    out->map_len = (size_t)st.st_size;
    out->map = (uint8_t *)mmap(NULL, out->map_len, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->map == MAP_FAILED) {
        out->map = NULL;
        gd_gdvlm_shard_close(out);
        return _gd_error(GD_ERR_IO, "failed to mmap gdvlm shard");
    }
    out->payload = out->map + out->header.payload_offset;
    out->payload_len = out->map_len - (size_t)out->header.payload_offset;
    out->path = gd_gdvlm_strdup(path);
    if (out->path == NULL) {
        gd_gdvlm_shard_close(out);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm shard path allocation failed");
    }
    return GD_OK;
}

static int gd_gdvlm_headers_match(const gd_gdvlm_header *a,
                                  const gd_gdvlm_header *b)
{
    return a->height == b->height && a->width == b->width &&
           a->channels == b->channels && a->patch_size == b->patch_size &&
           a->num_patches == b->num_patches && a->patch_dim == b->patch_dim &&
           a->patch_dtype == b->patch_dtype && a->token_dtype == b->token_dtype &&
           a->vocab_size == b->vocab_size && a->num_shards == b->num_shards &&
           a->tokenizer_hash == b->tokenizer_hash;
}

static void gd_gdvlm_impl_destroy(void *impl_v)
{
    gd_gdvlm_dataset_impl *impl = (gd_gdvlm_dataset_impl *)impl_v;
    int i;
    if (impl == NULL) {
        return;
    }
    for (i = 0; i < impl->n_shards; ++i) {
        gd_gdvlm_shard_close(&impl->shards[i]);
    }
    free(impl->shards);
    free(impl->entries);
    free(impl->index_path);
    free(impl);
}

static gd_status gd_gdvlm_read_combined_idx(const char *path,
                                            gd_gdvlm_idx_entry **entries_out,
                                            uint64_t *n_entries_out,
                                            uint32_t *max_shard_out,
                                            uint32_t *max_text_out)
{
    FILE *f;
    uint8_t header[GD_GDVLM_IDX_HEADER_SIZE];
    gd_gdvlm_idx_entry *entries = NULL;
    uint64_t n_entries;
    uint64_t i;
    uint32_t max_shard = 0U;
    uint32_t max_text = 0U;

    if (path == NULL || entries_out == NULL || n_entries_out == NULL ||
        max_shard_out == NULL || max_text_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm idx read args");
    }
    *entries_out = NULL;
    *n_entries_out = 0U;
    *max_shard_out = 0U;
    *max_text_out = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open gdvlm idx");
    }
    if (fread(header, 1U, sizeof(header), f) != sizeof(header)) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read gdvlm idx header");
    }
    if (memcmp(header, "GDVLMIDX", 8U) != 0 ||
        gd_gdvlm_get_le32(header + 8) != GD_GDVLM_IDX_VERSION ||
        gd_gdvlm_get_le32(header + 12) != GD_GDVLM_IDX_ENTRY_SIZE) {
        (void)fclose(f);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm idx header");
    }
    n_entries = gd_gdvlm_get_le64(header + 16);
    if (n_entries == 0U || n_entries > (uint64_t)(SIZE_MAX / sizeof(*entries))) {
        (void)fclose(f);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm idx entry count");
    }
    entries = (gd_gdvlm_idx_entry *)calloc((size_t)n_entries, sizeof(*entries));
    if (entries == NULL) {
        (void)fclose(f);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm idx allocation failed");
    }
    for (i = 0U; i < n_entries; ++i) {
        uint8_t b[GD_GDVLM_IDX_ENTRY_SIZE];
        if (fread(b, 1U, sizeof(b), f) != sizeof(b)) {
            free(entries);
            (void)fclose(f);
            return _gd_error(GD_ERR_IO, "failed to read gdvlm idx entry");
        }
        entries[i].shard_idx = gd_gdvlm_get_le32(b);
        entries[i].sample_idx = gd_gdvlm_get_le32(b + 4);
        entries[i].body_offset = gd_gdvlm_get_le64(b + 8);
        entries[i].record_nbytes = gd_gdvlm_get_le32(b + 16);
        entries[i].token_len = gd_gdvlm_get_le32(b + 20);
        entries[i].label_id = (uint16_t)gd_gdvlm_get_le16(b + 24);
        entries[i].flags = (uint16_t)gd_gdvlm_get_le16(b + 26);
        entries[i].raw_pos = gd_gdvlm_get_le32(b + 28);
        if (entries[i].shard_idx > max_shard) {
            max_shard = entries[i].shard_idx;
        }
        if (entries[i].token_len > max_text) {
            max_text = entries[i].token_len;
        }
    }
    if (fclose(f) != 0) {
        free(entries);
        return _gd_error(GD_ERR_IO, "failed to close gdvlm idx");
    }
    *entries_out = entries;
    *n_entries_out = n_entries;
    *max_shard_out = max_shard;
    *max_text_out = max_text;
    return GD_OK;
}

static uint64_t gd_gdvlm_fnv64_bytes(uint64_t h, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0U; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t gd_gdvlm_fnv64_u64(uint64_t h, uint64_t v)
{
    return gd_gdvlm_fnv64_bytes(h, &v, sizeof(v));
}

static uint64_t gd_gdvlm_fnv64_str(uint64_t h, const char *s)
{
    if (s == NULL) {
        return gd_gdvlm_fnv64_u64(h, 0U);
    }
    return gd_gdvlm_fnv64_bytes(h, s, strlen(s) + 1U);
}

static uint64_t gd_gdvlm_compute_fingerprint(const gd_gdvlm_dataset_impl *impl)
{
    uint64_t h = UINT64_C(1469598103934665603);
    uint64_t i;
    int s;
    h = gd_gdvlm_fnv64_str(h, "gdvlm-v1");
    h = gd_gdvlm_fnv64_str(h, impl->index_path);
    h = gd_gdvlm_fnv64_u64(h, impl->n_samples);
    h = gd_gdvlm_fnv64_u64(h, impl->max_text_len);
    h = gd_gdvlm_fnv64_u64(h, impl->patch_nbytes);
    h = gd_gdvlm_fnv64_bytes(h, &impl->meta, sizeof(impl->meta));
    for (s = 0; s < impl->n_shards; ++s) {
        h = gd_gdvlm_fnv64_str(h, impl->shards[s].path);
        h = gd_gdvlm_fnv64_u64(h, impl->shards[s].header.n_samples);
    }
    for (i = 0U; i < impl->n_samples; ++i) {
        h = gd_gdvlm_fnv64_bytes(h, &impl->entries[i], sizeof(impl->entries[i]));
    }
    return h != 0U ? h : 1U;
}

static gd_status gd_gdvlm_validate_entries(gd_gdvlm_dataset_impl *impl)
{
    uint64_t i;
    size_t token_size = gd_gdvlm_token_dtype_size(impl->meta.token_dtype);
    for (i = 0U; i < impl->n_samples; ++i) {
        const gd_gdvlm_idx_entry *e = &impl->entries[i];
        const gd_gdvlm_shard *shard;
        uint64_t need;
        if (e->shard_idx >= (uint32_t)impl->n_shards) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm idx shard out of range");
        }
        shard = &impl->shards[e->shard_idx];
        if (e->sample_idx >= shard->header.n_samples) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm idx sample out of range");
        }
        need = 8U + (uint64_t)e->token_len * (uint64_t)token_size +
               (uint64_t)impl->patch_nbytes;
        if (need > UINT32_MAX || e->record_nbytes != (uint32_t)need) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm idx record size mismatch");
        }
        if (e->body_offset > (uint64_t)shard->payload_len ||
            (uint64_t)e->record_nbytes > (uint64_t)shard->payload_len - e->body_offset) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm idx record out of shard bounds");
        }
    }
    return GD_OK;
}

static uint64_t gd_gdvlm_impl_num_samples(const void *impl_v)
{
    const gd_gdvlm_dataset_impl *impl = (const gd_gdvlm_dataset_impl *)impl_v;
    return impl != NULL ? impl->n_samples : 0U;
}

static uint64_t gd_gdvlm_impl_fingerprint(const void *impl_v)
{
    const gd_gdvlm_dataset_impl *impl = (const gd_gdvlm_dataset_impl *)impl_v;
    return impl != NULL ? impl->fingerprint : 0U;
}

static gd_status gd_gdvlm_impl_get_u64(const void *impl_v,
                                       const char *key,
                                       uint64_t *out)
{
    const gd_gdvlm_dataset_impl *impl = (const gd_gdvlm_dataset_impl *)impl_v;
    if (impl == NULL || key == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm metadata args");
    }
    if (strcmp(key, "height") == 0) {
        *out = impl->meta.height;
    } else if (strcmp(key, "width") == 0) {
        *out = impl->meta.width;
    } else if (strcmp(key, "channels") == 0) {
        *out = impl->meta.channels;
    } else if (strcmp(key, "patch_size") == 0) {
        *out = impl->meta.patch_size;
    } else if (strcmp(key, "num_patches") == 0 ||
               strcmp(key, "image_prefix_tokens") == 0) {
        *out = impl->meta.num_patches;
    } else if (strcmp(key, "patch_dim") == 0) {
        *out = impl->meta.patch_dim;
    } else if (strcmp(key, "patch_nbytes") == 0) {
        *out = (uint64_t)impl->patch_nbytes;
    } else if (strcmp(key, "patch_dtype") == 0) {
        *out = (uint64_t)impl->meta.patch_dtype;
    } else if (strcmp(key, "token_dtype") == 0) {
        *out = (uint64_t)impl->meta.token_dtype;
    } else if (strcmp(key, "vocab_size") == 0) {
        *out = impl->meta.vocab_size;
    } else if (strcmp(key, "tokenizer_hash") == 0) {
        *out = impl->meta.tokenizer_hash;
    } else if (strcmp(key, "n_samples") == 0) {
        *out = impl->n_samples;
    } else if (strcmp(key, "max_text_len") == 0) {
        *out = impl->max_text_len;
    } else {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown gdvlm metadata key");
    }
    return GD_OK;
}

gd_status gd_dataset_open_gdvlm(const char **shard_paths,
                                int n_shards,
                                const char *index_path,
                                gd_dataset **out)
{
    static const gd_dataset_ops ops = {
        "gdvlm",
        gd_gdvlm_impl_num_samples,
        gd_gdvlm_impl_fingerprint,
        gd_gdvlm_impl_get_u64,
        gd_gdvlm_impl_destroy
    };
    gd_gdvlm_dataset_impl *impl;
    uint32_t max_shard = 0U;
    gd_status status;
    int i;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm dataset output is null");
    }
    *out = NULL;
    if (shard_paths == NULL || n_shards <= 0 || index_path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm dataset args");
    }
    impl = (gd_gdvlm_dataset_impl *)calloc(1U, sizeof(*impl));
    if (impl == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm dataset allocation failed");
    }
    impl->magic = GD_GDVLM_IMPL_MAGIC;
    impl->index_path = gd_gdvlm_strdup(index_path);
    impl->shards = (gd_gdvlm_shard *)calloc((size_t)n_shards, sizeof(gd_gdvlm_shard));
    if (impl->index_path == NULL || impl->shards == NULL) {
        gd_gdvlm_impl_destroy(impl);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm dataset allocation failed");
    }
    impl->n_shards = n_shards;
    status = gd_gdvlm_read_combined_idx(index_path, &impl->entries, &impl->n_samples,
                                        &max_shard, &impl->max_text_len);
    if (status != GD_OK) {
        gd_gdvlm_impl_destroy(impl);
        return status;
    }
    if (max_shard >= (uint32_t)n_shards) {
        gd_gdvlm_impl_destroy(impl);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm idx references missing shard");
    }
    for (i = 0; i < n_shards; ++i) {
        status = gd_gdvlm_shard_open(shard_paths[i], (uint32_t)i, &impl->shards[i]);
        if (status != GD_OK) {
            gd_gdvlm_impl_destroy(impl);
            return status;
        }
        if (i == 0) {
            impl->meta = impl->shards[i].header;
            status = gd_gdvlm_patch_nbytes(&impl->meta, &impl->patch_nbytes);
            if (status != GD_OK) {
                gd_gdvlm_impl_destroy(impl);
                return status;
            }
        } else if (gd_gdvlm_headers_match(&impl->meta, &impl->shards[i].header) == 0) {
            gd_gdvlm_impl_destroy(impl);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm shard metadata mismatch");
        }
        if (impl->shards[i].header.num_shards != (uint32_t)n_shards) {
            gd_gdvlm_impl_destroy(impl);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm shard count mismatch");
        }
    }
    status = gd_gdvlm_validate_entries(impl);
    if (status != GD_OK) {
        gd_gdvlm_impl_destroy(impl);
        return status;
    }
    impl->fingerprint = gd_gdvlm_compute_fingerprint(impl);
    status = gd_dataset_create(&ops, impl, out);
    if (status != GD_OK) {
        gd_gdvlm_impl_destroy(impl);
        return status;
    }
    return GD_OK;
}

static gd_status gd_gdvlm_idx_max_shard(const char *index_path,
                                        uint32_t *max_shard_out)
{
    gd_gdvlm_idx_entry *entries = NULL;
    uint64_t n_entries = 0U;
    uint32_t max_shard = 0U;
    uint32_t max_text = 0U;
    gd_status status;
    status = gd_gdvlm_read_combined_idx(index_path, &entries, &n_entries,
                                        &max_shard, &max_text);
    free(entries);
    if (status != GD_OK) {
        return status;
    }
    *max_shard_out = max_shard;
    return GD_OK;
}

gd_status gd_dataset_open_gdvlm_split(const char *dir,
                                      const char *split_tag,
                                      gd_dataset **out)
{
    char index_path[GD_GDVLM_MAX_PATH];
    char **paths = NULL;
    const char **path_ptrs = NULL;
    uint32_t max_shard = 0U;
    uint32_t n_shards;
    int n;
    uint32_t i;
    gd_status status;

    if (dir == NULL || split_tag == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm split open args");
    }
    n = snprintf(index_path, sizeof(index_path), "%s/%s.idx", dir, split_tag);
    if (n < 0 || (size_t)n >= sizeof(index_path)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm index path too long");
    }
    status = gd_gdvlm_idx_max_shard(index_path, &max_shard);
    if (status != GD_OK) {
        return status;
    }
    n_shards = max_shard + 1U;
    paths = (char **)calloc((size_t)n_shards, sizeof(char *));
    path_ptrs = (const char **)calloc((size_t)n_shards, sizeof(char *));
    if (paths == NULL || path_ptrs == NULL) {
        free(paths);
        free(path_ptrs);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm split path allocation failed");
    }
    for (i = 0U; i < n_shards; ++i) {
        paths[i] = (char *)calloc(GD_GDVLM_MAX_PATH, 1U);
        if (paths[i] == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "gdvlm shard path allocation failed");
            break;
        }
        n = snprintf(paths[i], GD_GDVLM_MAX_PATH, "%s/%s-%05u-of-%05u.gdvlm",
                     dir, split_tag, i + 1U, n_shards);
        if (n < 0 || (size_t)n >= GD_GDVLM_MAX_PATH) {
            status = _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm shard path too long");
            break;
        }
        path_ptrs[i] = paths[i];
    }
    if (status == GD_OK) {
        status = gd_dataset_open_gdvlm(path_ptrs, (int)n_shards, index_path, out);
    }
    for (i = 0U; i < n_shards; ++i) {
        free(paths[i]);
    }
    free(paths);
    free(path_ptrs);
    return status;
}

static gd_status gd_gdvlm_impl_from_dataset(const gd_dataset *dataset,
                                            const gd_gdvlm_dataset_impl **out)
{
    const gd_gdvlm_dataset_impl *impl;
    if (dataset == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm dataset args");
    }
    if (strcmp(gd_dataset_name(dataset), "gdvlm") != 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset is not gdvlm");
    }
    impl = (const gd_gdvlm_dataset_impl *)gd_dataset_const_data(dataset);
    if (impl == NULL || impl->magic != GD_GDVLM_IMPL_MAGIC) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset is not gdvlm");
    }
    *out = impl;
    return GD_OK;
}

gd_status gd_gdvlm_dataset_sample_info(const gd_dataset *dataset,
                                       uint64_t sample_index,
                                       gd_gdvlm_sample_info *out)
{
    const gd_gdvlm_dataset_impl *impl;
    const gd_gdvlm_idx_entry *e;
    gd_status status;
    status = gd_gdvlm_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    if (out == NULL || sample_index >= impl->n_samples) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm sample info args");
    }
    e = &impl->entries[sample_index];
    out->shard_idx = e->shard_idx;
    out->sample_idx = e->sample_idx;
    out->body_offset = e->body_offset;
    out->record_nbytes = e->record_nbytes;
    out->token_len = e->token_len;
    out->tokens_copied = 0U;
    out->label_id = (int32_t)e->label_id;
    out->raw_pos = e->raw_pos;
    return GD_OK;
}

gd_status gd_gdvlm_dataset_read_sample(const gd_dataset *dataset,
                                       uint64_t sample_index,
                                       gd_gdvlm_sample_info *info_out,
                                       int32_t *tokens,
                                       int token_capacity,
                                       void *patches,
                                       size_t patch_nbytes)
{
    const gd_gdvlm_dataset_impl *impl;
    const gd_gdvlm_idx_entry *e;
    const gd_gdvlm_shard *shard;
    const uint8_t *record;
    const uint8_t *token_src;
    const uint8_t *patch_src;
    uint32_t n_copy;
    uint32_t label;
    uint32_t token_len;
    uint32_t j;
    size_t token_size;
    gd_status status;

    status = gd_gdvlm_impl_from_dataset(dataset, &impl);
    if (status != GD_OK) {
        return status;
    }
    if (sample_index >= impl->n_samples || token_capacity < 0 ||
        (token_capacity > 0 && tokens == NULL)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm sample read args");
    }
    if (patches != NULL && patch_nbytes != impl->patch_nbytes) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdvlm patch output size");
    }
    e = &impl->entries[sample_index];
    shard = &impl->shards[e->shard_idx];
    record = shard->payload + e->body_offset;
    label = gd_gdvlm_get_le32(record);
    token_len = gd_gdvlm_get_le32(record + 4);
    if (label != (uint32_t)e->label_id || token_len != e->token_len) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdvlm record metadata mismatch");
    }
    token_src = record + 8;
    token_size = gd_gdvlm_token_dtype_size(impl->meta.token_dtype);
    patch_src = token_src + (uint64_t)e->token_len * token_size;
    n_copy = e->token_len < (uint32_t)token_capacity ? e->token_len : (uint32_t)token_capacity;
    for (j = 0U; j < n_copy; ++j) {
        const uint8_t *p = token_src + (uint64_t)j * token_size;
        tokens[j] = impl->meta.token_dtype == GD_GDVLM_TOKEN_DTYPE_U16 ?
            (int32_t)gd_gdvlm_get_le16(p) : (int32_t)gd_gdvlm_get_le32(p);
    }
    if (patches != NULL) {
        memcpy(patches, patch_src, impl->patch_nbytes);
    }
    if (info_out != NULL) {
        info_out->shard_idx = e->shard_idx;
        info_out->sample_idx = e->sample_idx;
        info_out->body_offset = e->body_offset;
        info_out->record_nbytes = e->record_nbytes;
        info_out->token_len = e->token_len;
        info_out->tokens_copied = n_copy;
        info_out->label_id = (int32_t)e->label_id;
        info_out->raw_pos = e->raw_pos;
    }
    return GD_OK;
}
