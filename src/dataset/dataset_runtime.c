#include "gradients/dataset.h"

#include "../core/internal.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_GDTOK_IMPL_MAGIC UINT64_C(0x6764746f6b647331)

struct gd_dataset {
    gd_dataset_ops ops;
    void *impl;
};

static char *gd_dsrt_strdup(const char *s)
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

gd_status gd_dataset_create(const gd_dataset_ops *ops,
                            void *impl,
                            gd_dataset **out)
{
    gd_dataset *dataset;
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset output is null");
    }
    *out = NULL;
    if (ops == NULL || ops->name == NULL || ops->num_samples == NULL ||
        ops->fingerprint == NULL || impl == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset create args");
    }
    dataset = (gd_dataset *)calloc(1U, sizeof(*dataset));
    if (dataset == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset allocation failed");
    }
    dataset->ops = *ops;
    dataset->ops.name = gd_dsrt_strdup(ops->name);
    if (dataset->ops.name == NULL) {
        free(dataset);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset name allocation failed");
    }
    dataset->impl = impl;
    *out = dataset;
    return GD_OK;
}

void gd_dataset_destroy(gd_dataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    if (dataset->ops.destroy != NULL) {
        dataset->ops.destroy(dataset->impl);
    }
    free((char *)dataset->ops.name);
    free(dataset);
}

const char *gd_dataset_name(const gd_dataset *dataset)
{
    if (dataset == NULL || dataset->ops.name == NULL) {
        return "";
    }
    return dataset->ops.name;
}

void *gd_dataset_data(gd_dataset *dataset)
{
    return dataset != NULL ? dataset->impl : NULL;
}

const void *gd_dataset_const_data(const gd_dataset *dataset)
{
    return dataset != NULL ? dataset->impl : NULL;
}

uint64_t gd_dataset_num_samples(const gd_dataset *dataset)
{
    if (dataset == NULL || dataset->ops.num_samples == NULL) {
        return 0U;
    }
    return dataset->ops.num_samples(dataset->impl);
}

uint64_t gd_dataset_fingerprint(const gd_dataset *dataset)
{
    if (dataset == NULL || dataset->ops.fingerprint == NULL) {
        return 0U;
    }
    return dataset->ops.fingerprint(dataset->impl);
}

gd_status gd_dataset_get_u64(const gd_dataset *dataset,
                             const char *key,
                             uint64_t *out)
{
    if (dataset == NULL || key == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset metadata args");
    }
    if (dataset->ops.get_u64 == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "dataset has no uint64 metadata");
    }
    return dataset->ops.get_u64(dataset->impl, key, out);
}

typedef struct gd_gdtok_shard {
    char *path;
    int fd;
    uint8_t *map;
    size_t map_len;
    const uint8_t *payload;
    gd_gdtok_header header;
    uint64_t sample_base;
} gd_gdtok_shard;

typedef struct gd_gdtok_dataset_impl {
    uint64_t magic;
    gd_gdtok_shard *shards;
    int n_shards;
    uint32_t block_len;
    uint32_t vocab_size;
    gd_gdtok_dtype dtype;
    uint64_t tokenizer_hash;
    uint64_t n_samples;
    uint64_t n_tokens;
} gd_gdtok_dataset_impl;

static uint32_t gd_gdtok_get_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U);
}

static uint32_t gd_gdtok_get_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
           ((uint32_t)p[3] << 24U);
}

static size_t gd_gdtok_dtype_size(gd_gdtok_dtype dtype)
{
    if (dtype == GD_GDTOK_DTYPE_U16) {
        return 2U;
    }
    if (dtype == GD_GDTOK_DTYPE_U32) {
        return 4U;
    }
    return 0U;
}

static gd_status gd_gdtok_checked_payload_nbytes(const gd_gdtok_header *h,
                                                 size_t *out)
{
    size_t dtype_size;
    if (h == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok payload size args");
    }
    dtype_size = gd_gdtok_dtype_size(h->dtype);
    if (dtype_size == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok dtype");
    }
    if (h->n_tokens > (uint64_t)(SIZE_MAX / dtype_size)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok payload too large");
    }
    *out = (size_t)h->n_tokens * dtype_size;
    return GD_OK;
}

static gd_status gd_gdtok_shard_open(const char *path,
                                     uint64_t sample_base,
                                     gd_gdtok_shard *out)
{
    struct stat st;
    size_t payload_nbytes = 0U;
    size_t need = 0U;
    gd_status status;

    if (path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok shard open args");
    }
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    status = gd_gdtok_read_header(path, &out->header);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gdtok_checked_payload_nbytes(&out->header, &payload_nbytes);
    if (status != GD_OK) {
        return status;
    }
    if (out->header.payload_offset > (uint64_t)(SIZE_MAX - payload_nbytes)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok mapping too large");
    }
    need = (size_t)out->header.payload_offset + payload_nbytes;
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) {
        return _gd_error(GD_ERR_IO, "failed to open gdtok shard");
    }
    if (fstat(out->fd, &st) != 0) {
        (void)close(out->fd);
        out->fd = -1;
        return _gd_error(GD_ERR_IO, "failed to stat gdtok shard");
    }
    if (st.st_size < 0 || (uint64_t)st.st_size < (uint64_t)need) {
        (void)close(out->fd);
        out->fd = -1;
        return _gd_error(GD_ERR_IO, "gdtok shard truncated");
    }
    out->map_len = (size_t)st.st_size;
    out->map = (uint8_t *)mmap(NULL, out->map_len, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->map == MAP_FAILED) {
        (void)close(out->fd);
        out->fd = -1;
        out->map = NULL;
        return _gd_error(GD_ERR_IO, "failed to mmap gdtok shard");
    }
    out->payload = out->map + out->header.payload_offset;
    out->sample_base = sample_base;
    out->path = gd_dsrt_strdup(path);
    if (out->path == NULL) {
        (void)munmap(out->map, out->map_len);
        (void)close(out->fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok shard path allocation failed");
    }
    return GD_OK;
}

static void gd_gdtok_shard_close(gd_gdtok_shard *shard)
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

static void gd_gdtok_impl_destroy(void *impl_v)
{
    gd_gdtok_dataset_impl *impl = (gd_gdtok_dataset_impl *)impl_v;
    int i;
    if (impl == NULL) {
        return;
    }
    for (i = 0; i < impl->n_shards; ++i) {
        gd_gdtok_shard_close(&impl->shards[i]);
    }
    free(impl->shards);
    free(impl);
}

static uint64_t gd_dsrt_fnv64_bytes(uint64_t h, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0U; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t gd_dsrt_fnv64_u64(uint64_t h, uint64_t v)
{
    return gd_dsrt_fnv64_bytes(h, &v, sizeof(v));
}

static uint64_t gd_dsrt_fnv64_str(uint64_t h, const char *s)
{
    if (s == NULL) {
        return gd_dsrt_fnv64_u64(h, 0U);
    }
    return gd_dsrt_fnv64_bytes(h, s, strlen(s) + 1U);
}

static uint64_t gd_gdtok_impl_num_samples(const void *impl_v)
{
    const gd_gdtok_dataset_impl *impl = (const gd_gdtok_dataset_impl *)impl_v;
    return impl != NULL ? impl->n_samples : 0U;
}

static uint64_t gd_gdtok_impl_fingerprint(const void *impl_v)
{
    const gd_gdtok_dataset_impl *impl = (const gd_gdtok_dataset_impl *)impl_v;
    uint64_t h = UINT64_C(1469598103934665603);
    int i;
    if (impl == NULL) {
        return 0U;
    }
    h = gd_dsrt_fnv64_str(h, "gdtok-v1");
    h = gd_dsrt_fnv64_u64(h, (uint64_t)impl->n_shards);
    h = gd_dsrt_fnv64_u64(h, (uint64_t)impl->block_len);
    h = gd_dsrt_fnv64_u64(h, (uint64_t)impl->vocab_size);
    h = gd_dsrt_fnv64_u64(h, (uint64_t)impl->dtype);
    h = gd_dsrt_fnv64_u64(h, impl->tokenizer_hash);
    h = gd_dsrt_fnv64_u64(h, impl->n_samples);
    h = gd_dsrt_fnv64_u64(h, impl->n_tokens);
    for (i = 0; i < impl->n_shards; ++i) {
        const gd_gdtok_shard *shard = &impl->shards[i];
        h = gd_dsrt_fnv64_str(h, shard->path);
        h = gd_dsrt_fnv64_u64(h, shard->header.n_tokens);
        h = gd_dsrt_fnv64_u64(h, shard->header.n_samples);
        h = gd_dsrt_fnv64_u64(h, shard->sample_base);
    }
    return h != 0U ? h : 1U;
}

static gd_status gd_gdtok_impl_get_u64(const void *impl_v,
                                       const char *key,
                                       uint64_t *out)
{
    const gd_gdtok_dataset_impl *impl = (const gd_gdtok_dataset_impl *)impl_v;
    if (impl == NULL || key == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok metadata args");
    }
    if (strcmp(key, "block_len") == 0) {
        *out = (uint64_t)impl->block_len;
        return GD_OK;
    }
    if (strcmp(key, "vocab_size") == 0) {
        *out = (uint64_t)impl->vocab_size;
        return GD_OK;
    }
    if (strcmp(key, "tokenizer_hash") == 0) {
        *out = impl->tokenizer_hash;
        return GD_OK;
    }
    if (strcmp(key, "n_samples") == 0) {
        *out = impl->n_samples;
        return GD_OK;
    }
    if (strcmp(key, "n_tokens") == 0) {
        *out = impl->n_tokens;
        return GD_OK;
    }
    return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown gdtok metadata key");
}

gd_status gd_dataset_open_gdtok(const char **paths,
                                int n_paths,
                                gd_dataset **out)
{
    static const gd_dataset_ops ops = {
        "gdtok",
        gd_gdtok_impl_num_samples,
        gd_gdtok_impl_fingerprint,
        gd_gdtok_impl_get_u64,
        gd_gdtok_impl_destroy
    };
    gd_gdtok_dataset_impl *impl;
    uint64_t sample_base = 0U;
    uint64_t n_tokens = 0U;
    int i;
    gd_status status;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok dataset output is null");
    }
    *out = NULL;
    if (paths == NULL || n_paths <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok dataset paths");
    }
    impl = (gd_gdtok_dataset_impl *)calloc(1U, sizeof(*impl));
    if (impl == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok dataset allocation failed");
    }
    impl->magic = GD_GDTOK_IMPL_MAGIC;
    impl->shards = (gd_gdtok_shard *)calloc((size_t)n_paths, sizeof(gd_gdtok_shard));
    if (impl->shards == NULL) {
        free(impl);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok shard array allocation failed");
    }
    impl->n_shards = n_paths;
    for (i = 0; i < n_paths; ++i) {
        gd_gdtok_shard *shard = &impl->shards[i];
        status = gd_gdtok_shard_open(paths[i], sample_base, shard);
        if (status != GD_OK) {
            gd_gdtok_impl_destroy(impl);
            return status;
        }
        if (shard->header.n_samples == 0U) {
            gd_gdtok_impl_destroy(impl);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard has zero samples");
        }
        if (i == 0) {
            impl->block_len = shard->header.block_len;
            impl->vocab_size = shard->header.vocab_size;
            impl->dtype = shard->header.dtype;
            impl->tokenizer_hash = shard->header.tokenizer_hash;
        } else if (impl->block_len != shard->header.block_len ||
                   impl->vocab_size != shard->header.vocab_size ||
                   impl->dtype != shard->header.dtype ||
                   impl->tokenizer_hash != shard->header.tokenizer_hash) {
            gd_gdtok_impl_destroy(impl);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard metadata mismatch");
        }
        if (shard->header.block_len == 0U ||
            shard->header.n_samples > (UINT64_MAX - 1U) / shard->header.block_len ||
            shard->header.n_tokens != shard->header.n_samples * shard->header.block_len + 1U) {
            gd_gdtok_impl_destroy(impl);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard token/sample mismatch");
        }
        if (UINT64_MAX - sample_base < shard->header.n_samples ||
            UINT64_MAX - n_tokens < shard->header.n_tokens) {
            gd_gdtok_impl_destroy(impl);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok dataset size overflow");
        }
        sample_base += shard->header.n_samples;
        n_tokens += shard->header.n_tokens;
    }
    impl->n_samples = sample_base;
    impl->n_tokens = n_tokens;
    status = gd_dataset_create(&ops, impl, out);
    if (status != GD_OK) {
        gd_gdtok_impl_destroy(impl);
        return status;
    }
    return GD_OK;
}

static const gd_gdtok_shard *gd_gdtok_impl_find_shard(const gd_gdtok_dataset_impl *impl,
                                                      uint64_t sample_index,
                                                      uint64_t *local_out)
{
    int i;
    if (impl == NULL || local_out == NULL || sample_index >= impl->n_samples) {
        return NULL;
    }
    for (i = 0; i < impl->n_shards; ++i) {
        const gd_gdtok_shard *shard = &impl->shards[i];
        if (sample_index >= shard->sample_base &&
            sample_index < shard->sample_base + shard->header.n_samples) {
            *local_out = sample_index - shard->sample_base;
            return shard;
        }
    }
    return NULL;
}

static int32_t gd_gdtok_shard_read_id(const gd_gdtok_shard *shard,
                                      uint64_t token_index)
{
    const uint8_t *p;
    if (shard == NULL || token_index >= shard->header.n_tokens) {
        return -1;
    }
    p = shard->payload + token_index * gd_gdtok_dtype_size(shard->header.dtype);
    if (shard->header.dtype == GD_GDTOK_DTYPE_U16) {
        return (int32_t)gd_gdtok_get_le16(p);
    }
    return (int32_t)gd_gdtok_get_le32(p);
}

gd_status gd_gdtok_dataset_read_lm_sample(const gd_dataset *dataset,
                                          uint64_t sample_index,
                                          int32_t *tokens,
                                          int32_t *targets)
{
    const gd_gdtok_dataset_impl *impl;
    const gd_gdtok_shard *shard;
    uint64_t local = 0U;
    uint64_t start;
    uint32_t j;

    if (dataset == NULL || tokens == NULL || targets == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok sample read args");
    }
    if (strcmp(gd_dataset_name(dataset), "gdtok") != 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset is not gdtok");
    }
    impl = (const gd_gdtok_dataset_impl *)gd_dataset_const_data(dataset);
    if (impl == NULL || impl->magic != GD_GDTOK_IMPL_MAGIC) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset is not gdtok");
    }
    if (sample_index >= impl->n_samples) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok sample index out of range");
    }
    shard = gd_gdtok_impl_find_shard(impl, sample_index, &local);
    if (shard == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "gdtok sample shard lookup failed");
    }
    start = local * (uint64_t)impl->block_len;
    for (j = 0U; j < impl->block_len; ++j) {
        int32_t tok = gd_gdtok_shard_read_id(shard, start + (uint64_t)j);
        int32_t tgt = gd_gdtok_shard_read_id(shard, start + (uint64_t)j + 1U);
        if (tok < 0 || tgt < 0) {
            return _gd_error(GD_ERR_INTERNAL, "gdtok sample read out of range");
        }
        tokens[j] = tok;
        targets[j] = tgt;
    }
    return GD_OK;
}
