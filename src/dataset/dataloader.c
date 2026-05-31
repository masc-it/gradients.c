#include "gradients/dataloader.h"

#include "../core/internal.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_DL_N_SLOTS 2

typedef struct gd_token_shard {
    char *path;
    int fd;
    uint8_t *map;
    size_t map_len;
    const uint8_t *payload;
    gd_gdtok_header header;
    uint64_t sample_base;
} gd_token_shard;

struct gd_token_dataset {
    gd_token_shard *shards;
    int n_shards;
    uint32_t block_len;
    uint32_t vocab_size;
    gd_gdtok_dtype dtype;
    uint64_t tokenizer_hash;
    uint64_t n_samples;
    uint64_t n_tokens;
};

typedef struct gd_loader_slot {
    gd_batch_slot pub;
    int32_t *host_tokens;
    int32_t *host_targets;
    int32_t *host_positions;
} gd_loader_slot;

struct gd_dataloader {
    gd_context *ctx;
    gd_token_dataset *ds;
    gd_dataloader_config cfg;
    gd_loader_slot slots[GD_DL_N_SLOTS];
    uint64_t rng_state;
    uint64_t cursor;
    gd_dataloader_metrics metrics;
};

static char *gd_dl_strdup(const char *s)
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

static uint32_t gd_dl_get_le16(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U);
}

static uint32_t gd_dl_get_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
           ((uint32_t)p[3] << 24U);
}

static uint64_t gd_dl_splitmix64(uint64_t *state)
{
    uint64_t z;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static size_t gd_dl_dtype_size(gd_gdtok_dtype dtype)
{
    if (dtype == GD_GDTOK_DTYPE_U16) {
        return 2U;
    }
    if (dtype == GD_GDTOK_DTYPE_U32) {
        return 4U;
    }
    return 0U;
}

static gd_status gd_dl_checked_payload_nbytes(const gd_gdtok_header *h, size_t *out)
{
    size_t dtype_size;
    if (h == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok payload size args");
    }
    dtype_size = gd_dl_dtype_size(h->dtype);
    if (dtype_size == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok dtype");
    }
    if (h->n_tokens > (uint64_t)(SIZE_MAX / dtype_size)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "gdtok payload too large");
    }
    *out = (size_t)h->n_tokens * dtype_size;
    return GD_OK;
}

static gd_status gd_token_shard_open(const char *path,
                                     uint64_t sample_base,
                                     gd_token_shard *out)
{
    struct stat st;
    size_t payload_nbytes = 0U;
    size_t need = 0U;
    gd_status status;

    if (path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid token shard open args");
    }
    memset(out, 0, sizeof(*out));
    out->fd = -1;
    status = gd_gdtok_read_header(path, &out->header);
    if (status != GD_OK) {
        return status;
    }
    status = gd_dl_checked_payload_nbytes(&out->header, &payload_nbytes);
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
        close(out->fd);
        out->fd = -1;
        return _gd_error(GD_ERR_IO, "failed to stat gdtok shard");
    }
    if (st.st_size < 0 || (uint64_t)st.st_size < (uint64_t)need) {
        close(out->fd);
        out->fd = -1;
        return _gd_error(GD_ERR_IO, "gdtok shard truncated");
    }
    out->map_len = (size_t)st.st_size;
    out->map = (uint8_t *)mmap(NULL, out->map_len, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->map == MAP_FAILED) {
        close(out->fd);
        out->fd = -1;
        out->map = NULL;
        return _gd_error(GD_ERR_IO, "failed to mmap gdtok shard");
    }
    out->payload = out->map + out->header.payload_offset;
    out->sample_base = sample_base;
    out->path = gd_dl_strdup(path);
    if (out->path == NULL) {
        munmap(out->map, out->map_len);
        close(out->fd);
        memset(out, 0, sizeof(*out));
        out->fd = -1;
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "token shard path allocation failed");
    }
    return GD_OK;
}

static void gd_token_shard_close(gd_token_shard *shard)
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

gd_status gd_token_dataset_open(const char **paths,
                                int n_paths,
                                gd_token_dataset **out)
{
    gd_token_dataset *ds;
    uint64_t sample_base = 0U;
    uint64_t n_tokens = 0U;
    int i;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "token dataset output is null");
    }
    *out = NULL;
    if (paths == NULL || n_paths <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid token dataset paths");
    }
    ds = (gd_token_dataset *)calloc(1U, sizeof(*ds));
    if (ds == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "token dataset allocation failed");
    }
    ds->shards = (gd_token_shard *)calloc((size_t)n_paths, sizeof(gd_token_shard));
    if (ds->shards == NULL) {
        free(ds);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "token shard array allocation failed");
    }
    ds->n_shards = n_paths;
    for (i = 0; i < n_paths; ++i) {
        gd_status status;
        gd_token_shard *shard = &ds->shards[i];
        status = gd_token_shard_open(paths[i], sample_base, shard);
        if (status != GD_OK) {
            gd_token_dataset_close(ds);
            return status;
        }
        if (shard->header.n_samples == 0U) {
            gd_token_dataset_close(ds);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard has zero samples");
        }
        if (i == 0) {
            ds->block_len = shard->header.block_len;
            ds->vocab_size = shard->header.vocab_size;
            ds->dtype = shard->header.dtype;
            ds->tokenizer_hash = shard->header.tokenizer_hash;
        } else if (ds->block_len != shard->header.block_len ||
                   ds->vocab_size != shard->header.vocab_size ||
                   ds->dtype != shard->header.dtype ||
                   ds->tokenizer_hash != shard->header.tokenizer_hash) {
            gd_token_dataset_close(ds);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard metadata mismatch");
        }
        if (shard->header.block_len == 0U ||
            shard->header.n_samples > (UINT64_MAX - 1U) / shard->header.block_len ||
            shard->header.n_tokens != shard->header.n_samples * shard->header.block_len + 1U) {
            gd_token_dataset_close(ds);
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gdtok shard token/sample mismatch");
        }
        if (UINT64_MAX - sample_base < shard->header.n_samples ||
            UINT64_MAX - n_tokens < shard->header.n_tokens) {
            gd_token_dataset_close(ds);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "token dataset size overflow");
        }
        sample_base += shard->header.n_samples;
        n_tokens += shard->header.n_tokens;
    }
    ds->n_samples = sample_base;
    ds->n_tokens = n_tokens;
    *out = ds;
    return GD_OK;
}

void gd_token_dataset_close(gd_token_dataset *ds)
{
    int i;
    if (ds == NULL) {
        return;
    }
    for (i = 0; i < ds->n_shards; ++i) {
        gd_token_shard_close(&ds->shards[i]);
    }
    free(ds->shards);
    free(ds);
}

uint64_t gd_token_dataset_num_samples(const gd_token_dataset *ds)
{
    return ds != NULL ? ds->n_samples : 0U;
}

uint32_t gd_token_dataset_block_len(const gd_token_dataset *ds)
{
    return ds != NULL ? ds->block_len : 0U;
}

uint32_t gd_token_dataset_vocab_size(const gd_token_dataset *ds)
{
    return ds != NULL ? ds->vocab_size : 0U;
}

uint64_t gd_token_dataset_tokenizer_hash(const gd_token_dataset *ds)
{
    return ds != NULL ? ds->tokenizer_hash : 0U;
}

static const gd_token_shard *gd_token_dataset_find_shard(const gd_token_dataset *ds,
                                                         uint64_t sample_index,
                                                         uint64_t *local_out)
{
    int i;
    if (ds == NULL || local_out == NULL || sample_index >= ds->n_samples) {
        return NULL;
    }
    for (i = 0; i < ds->n_shards; ++i) {
        const gd_token_shard *shard = &ds->shards[i];
        if (sample_index >= shard->sample_base &&
            sample_index < shard->sample_base + shard->header.n_samples) {
            *local_out = sample_index - shard->sample_base;
            return shard;
        }
    }
    return NULL;
}

static int32_t gd_token_shard_read_id(const gd_token_shard *shard, uint64_t token_index)
{
    const uint8_t *p;
    if (shard == NULL || token_index >= shard->header.n_tokens) {
        return -1;
    }
    p = shard->payload + token_index * gd_dl_dtype_size(shard->header.dtype);
    if (shard->header.dtype == GD_GDTOK_DTYPE_U16) {
        return (int32_t)gd_dl_get_le16(p);
    }
    return (int32_t)gd_dl_get_le32(p);
}

static gd_status gd_token_dataset_read_sample(const gd_token_dataset *ds,
                                              uint64_t sample_index,
                                              int32_t *tokens,
                                              int32_t *targets)
{
    const gd_token_shard *shard;
    uint64_t local = 0U;
    uint64_t start;
    uint32_t j;

    if (ds == NULL || tokens == NULL || targets == NULL || sample_index >= ds->n_samples) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid token sample read");
    }
    shard = gd_token_dataset_find_shard(ds, sample_index, &local);
    if (shard == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "token sample shard lookup failed");
    }
    start = local * (uint64_t)ds->block_len;
    for (j = 0U; j < ds->block_len; ++j) {
        int32_t tok = gd_token_shard_read_id(shard, start + (uint64_t)j);
        int32_t tgt = gd_token_shard_read_id(shard, start + (uint64_t)j + 1U);
        if (tok < 0 || tgt < 0) {
            return _gd_error(GD_ERR_INTERNAL, "token sample read out of range");
        }
        tokens[j] = tok;
        targets[j] = tgt;
    }
    return GD_OK;
}

static void gd_loader_slot_clear(gd_loader_slot *slot)
{
    if (slot == NULL) {
        return;
    }
    gd_tensor_release(slot->pub.tokens);
    gd_tensor_release(slot->pub.targets);
    gd_tensor_release(slot->pub.positions);
    free(slot->host_tokens);
    free(slot->host_targets);
    free(slot->host_positions);
    memset(slot, 0, sizeof(*slot));
}

static gd_status gd_loader_slot_init(gd_context *ctx,
                                     const gd_dataloader_config *cfg,
                                     int index,
                                     gd_loader_slot *slot)
{
    gd_tensor_desc desc;
    int64_t sizes[2];
    size_t n_elem;
    size_t nbytes;
    gd_status status;

    if (ctx == NULL || cfg == NULL || slot == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader slot init args");
    }
    memset(slot, 0, sizeof(*slot));
    slot->pub.index = index;
    slot->pub.state = GD_BATCH_SLOT_FREE;
    n_elem = (size_t)cfg->batch_size * (size_t)cfg->block_len;
    if (cfg->batch_size <= 0 || cfg->block_len <= 0 ||
        n_elem > SIZE_MAX / sizeof(int32_t)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader slot shape");
    }
    nbytes = n_elem * sizeof(int32_t);
    slot->host_tokens = (int32_t *)malloc(nbytes);
    slot->host_targets = (int32_t *)malloc(nbytes);
    slot->host_positions = (int32_t *)malloc(nbytes);
    if (slot->host_tokens == NULL || slot->host_targets == NULL ||
        slot->host_positions == NULL) {
        gd_loader_slot_clear(slot);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "loader host buffer allocation failed");
    }
    sizes[0] = cfg->batch_size;
    sizes[1] = cfg->block_len;
    status = gd_tensor_desc_contiguous(GD_DTYPE_I32, cfg->device, 2, sizes, &desc);
    if (status == GD_OK) {
        status = gd_tensor_empty(ctx, &desc, &slot->pub.tokens);
    }
    if (status == GD_OK) {
        status = gd_tensor_empty(ctx, &desc, &slot->pub.targets);
    }
    if (status == GD_OK) {
        status = gd_tensor_empty(ctx, &desc, &slot->pub.positions);
    }
    if (status != GD_OK) {
        gd_loader_slot_clear(slot);
        return status;
    }
    return GD_OK;
}

static uint64_t gd_dataloader_next_sample(gd_dataloader *dl)
{
    if (dl->cfg.mode == GD_DATALOADER_SEQUENTIAL || dl->cfg.shuffle == 0) {
        uint64_t sample = dl->cursor % dl->ds->n_samples;
        dl->cursor += 1U;
        return sample;
    }
    return gd_dl_splitmix64(&dl->rng_state) % dl->ds->n_samples;
}

static gd_status gd_dataloader_fill_slot(gd_dataloader *dl, gd_loader_slot *slot)
{
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;
    size_t n_elem;
    size_t nbytes;
    int b;
    int j;
    gd_status status;

    if (dl == NULL || slot == NULL || slot->pub.state != GD_BATCH_SLOT_FREE) {
        return _gd_error(GD_ERR_INVALID_STATE, "loader slot is not free");
    }
    slot->pub.state = GD_BATCH_SLOT_FILLING;
    n_elem = (size_t)dl->cfg.batch_size * (size_t)dl->cfg.block_len;
    nbytes = n_elem * sizeof(int32_t);
    t0 = _gd_profile_now_ns();
    for (b = 0; b < dl->cfg.batch_size; ++b) {
        uint64_t sample = gd_dataloader_next_sample(dl);
        int32_t *tokens = &slot->host_tokens[(size_t)b * (size_t)dl->cfg.block_len];
        int32_t *targets = &slot->host_targets[(size_t)b * (size_t)dl->cfg.block_len];
        int32_t *positions = &slot->host_positions[(size_t)b * (size_t)dl->cfg.block_len];
        status = gd_token_dataset_read_sample(dl->ds, sample, tokens, targets);
        if (status != GD_OK) {
            slot->pub.state = GD_BATCH_SLOT_FREE;
            return status;
        }
        for (j = 0; j < dl->cfg.block_len; ++j) {
            positions[j] = (int32_t)j;
        }
    }
    t1 = _gd_profile_now_ns();
    status = gd_tensor_copy_from_cpu(dl->ctx, slot->pub.tokens, slot->host_tokens, nbytes);
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(dl->ctx, slot->pub.targets, slot->host_targets, nbytes);
    }
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(dl->ctx, slot->pub.positions, slot->host_positions, nbytes);
    }
    t2 = _gd_profile_now_ns();
    if (status != GD_OK) {
        slot->pub.state = GD_BATCH_SLOT_FREE;
        return status;
    }
    dl->metrics.host_fill_ns += t1 - t0;
    dl->metrics.host_to_device_copy_ns += t2 - t1;
    dl->metrics.batches_prepared += 1U;
    dl->metrics.samples_prepared += (uint64_t)dl->cfg.batch_size;
    slot->pub.state = GD_BATCH_SLOT_READY;
    return GD_OK;
}

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_token_dataset *ds,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out)
{
    gd_dataloader *dl;
    int i;
    gd_status status;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader output is null");
    }
    *out = NULL;
    if (ctx == NULL || ds == NULL || cfg == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader create args");
    }
    if (cfg->batch_size <= 0 || cfg->block_len <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader shape");
    }
    if (cfg->block_len != (int)ds->block_len) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader block_len mismatch");
    }
    if (cfg->double_buffer == 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader double_buffer must be true in v1");
    }
    if (cfg->expected_tokenizer_hash != 0U &&
        cfg->expected_tokenizer_hash != ds->tokenizer_hash) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader tokenizer hash mismatch");
    }
    dl = (gd_dataloader *)calloc(1U, sizeof(*dl));
    if (dl == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader allocation failed");
    }
    dl->ctx = ctx;
    dl->ds = ds;
    dl->cfg = *cfg;
    dl->rng_state = cfg->seed;
    dl->cursor = 0U;
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        status = gd_loader_slot_init(ctx, cfg, i, &dl->slots[i]);
        if (status != GD_OK) {
            gd_dataloader_destroy(dl);
            return status;
        }
    }
    *out = dl;
    return GD_OK;
}

void gd_dataloader_destroy(gd_dataloader *dl)
{
    int i;
    if (dl == NULL) {
        return;
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        gd_loader_slot_clear(&dl->slots[i]);
    }
    free(dl);
}

gd_status gd_dataloader_prefetch(gd_dataloader *dl)
{
    int i;
    int ready_count = 0;
    if (dl == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader is null");
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_READY) {
            ready_count += 1;
        }
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_FREE) {
            return gd_dataloader_fill_slot(dl, &dl->slots[i]);
        }
    }
    if (ready_count > 0) {
        return GD_OK;
    }
    return _gd_error(GD_ERR_INVALID_STATE, "no free dataloader slot");
}

gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch_slot **slot_out)
{
    uint64_t t0;
    int i;
    gd_status status;

    if (dl == NULL || slot_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader next args");
    }
    *slot_out = NULL;
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_READY) {
            dl->slots[i].pub.state = GD_BATCH_SLOT_IN_USE;
            dl->metrics.batches_returned += 1U;
            *slot_out = &dl->slots[i].pub;
            return GD_OK;
        }
    }
    t0 = _gd_profile_now_ns();
    status = gd_dataloader_prefetch(dl);
    dl->metrics.wait_for_batch_ns += _gd_profile_now_ns() - t0;
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_READY) {
            dl->slots[i].pub.state = GD_BATCH_SLOT_IN_USE;
            dl->metrics.batches_returned += 1U;
            *slot_out = &dl->slots[i].pub;
            return GD_OK;
        }
    }
    return _gd_error(GD_ERR_INVALID_STATE, "dataloader has no ready slot");
}

gd_status gd_dataloader_release_slot(gd_dataloader *dl, gd_batch_slot *slot)
{
    int i;
    if (dl == NULL || slot == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader release args");
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (&dl->slots[i].pub == slot) {
            if (dl->slots[i].pub.state != GD_BATCH_SLOT_IN_USE) {
                return _gd_error(GD_ERR_INVALID_STATE, "dataloader slot is not in use");
            }
            dl->slots[i].pub.state = GD_BATCH_SLOT_FREE;
            return GD_OK;
        }
    }
    return _gd_error(GD_ERR_INVALID_ARGUMENT, "slot does not belong to dataloader");
}

static int gd_dataloader_has_live_slot(const gd_dataloader *dl)
{
    int i;
    if (dl == NULL) {
        return 0;
    }
    for (i = 0; i < GD_DL_N_SLOTS; ++i) {
        if (dl->slots[i].pub.state != GD_BATCH_SLOT_FREE) {
            return 1;
        }
    }
    return 0;
}

gd_status gd_dataloader_state_save(gd_dataloader *dl, const char *path)
{
    FILE *f;
    if (dl == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state save args");
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataloader state output");
    }
    fprintf(f,
            "{\n"
            "  \"format\": \"gd-dataloader-v1\",\n"
            "  \"cursor\": %" PRIu64 ",\n"
            "  \"rng_state\": %" PRIu64 ",\n"
            "  \"batches_prepared\": %" PRIu64 ",\n"
            "  \"batches_returned\": %" PRIu64 "\n"
            "}\n",
            dl->cursor,
            dl->rng_state,
            dl->metrics.batches_prepared,
            dl->metrics.batches_returned);
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close dataloader state output");
    }
    return GD_OK;
}

static gd_status gd_dl_read_file(const char *path, char **text_out)
{
    FILE *f;
    long end;
    char *text;
    size_t nread;
    if (path == NULL || text_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state read args");
    }
    *text_out = NULL;
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataloader state");
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to seek dataloader state");
    }
    end = ftell(f);
    if (end < 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to size dataloader state");
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to rewind dataloader state");
    }
    text = (char *)malloc((size_t)end + 1U);
    if (text == NULL) {
        (void)fclose(f);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader state allocation failed");
    }
    nread = fread(text, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(text);
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read dataloader state");
    }
    if (fclose(f) != 0) {
        free(text);
        return _gd_error(GD_ERR_IO, "failed to close dataloader state");
    }
    text[(size_t)end] = '\0';
    *text_out = text;
    return GD_OK;
}

static gd_status gd_dl_parse_u64_field(const char *text,
                                       const char *field,
                                       uint64_t *out)
{
    const char *p;
    char *end = NULL;
    unsigned long long v;
    if (text == NULL || field == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state field args");
    }
    p = strstr(text, field);
    if (p == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing dataloader state field");
    }
    p += strlen(field);
    errno = 0;
    v = strtoull(p, &end, 10);
    if (errno != 0 || end == p) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state integer");
    }
    *out = (uint64_t)v;
    return GD_OK;
}

gd_status gd_dataloader_state_load(gd_dataloader *dl, const char *path)
{
    char *text = NULL;
    uint64_t cursor = 0U;
    uint64_t rng_state = 0U;
    gd_status status;
    if (dl == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state load args");
    }
    if (gd_dataloader_has_live_slot(dl) != 0) {
        return _gd_error(GD_ERR_INVALID_STATE, "cannot load dataloader state with live slots");
    }
    status = gd_dl_read_file(path, &text);
    if (status != GD_OK) {
        return status;
    }
    if (strstr(text, "gd-dataloader-v1") == NULL) {
        free(text);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unsupported dataloader state format");
    }
    status = gd_dl_parse_u64_field(text, "\"cursor\":", &cursor);
    if (status == GD_OK) {
        status = gd_dl_parse_u64_field(text, "\"rng_state\":", &rng_state);
    }
    free(text);
    if (status != GD_OK) {
        return status;
    }
    dl->cursor = cursor;
    dl->rng_state = rng_state;
    memset(&dl->metrics, 0, sizeof(dl->metrics));
    return GD_OK;
}

void gd_dataloader_metrics_get(const gd_dataloader *dl,
                               gd_dataloader_metrics *metrics_out)
{
    if (metrics_out == NULL) {
        return;
    }
    memset(metrics_out, 0, sizeof(*metrics_out));
    if (dl == NULL) {
        return;
    }
    *metrics_out = dl->metrics;
}
