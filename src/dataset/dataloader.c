#include "dataloader_internal.h"

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

static void *gd_dl_alloc_host_buffer(size_t nbytes)
{
    void *ptr = NULL;
    if (posix_memalign(&ptr, 64U, nbytes) != 0) {
        return NULL;
    }
    return ptr;
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
    free(slot->sample_ids);
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
    if (cfg->batch_size <= 0 || cfg->block_len <= 0 ||
        (size_t)cfg->batch_size > SIZE_MAX / (size_t)cfg->block_len) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader slot shape");
    }
    n_elem = (size_t)cfg->batch_size * (size_t)cfg->block_len;
    if (n_elem > SIZE_MAX / sizeof(int32_t)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader slot byte size");
    }
    nbytes = n_elem * sizeof(int32_t);
    slot->host_tokens = (int32_t *)gd_dl_alloc_host_buffer(nbytes);
    slot->host_targets = (int32_t *)gd_dl_alloc_host_buffer(nbytes);
    slot->host_positions = (int32_t *)gd_dl_alloc_host_buffer(nbytes);
    slot->sample_ids = (uint64_t *)calloc((size_t)cfg->batch_size, sizeof(uint64_t));
    if (slot->host_tokens == NULL || slot->host_targets == NULL ||
        slot->host_positions == NULL || slot->sample_ids == NULL) {
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

static uint64_t gd_dataloader_next_sample_locked(gd_dataloader *dl)
{
    if (dl->cfg.mode == GD_DATALOADER_SEQUENTIAL || dl->cfg.shuffle == 0) {
        uint64_t sample = dl->cursor % dl->ds->n_samples;
        dl->cursor += 1U;
        return sample;
    }
    return gd_dl_splitmix64(&dl->rng_state) % dl->ds->n_samples;
}

gd_status _gd_dataloader_fill_slot_from_samples(gd_dataloader *dl,
                                                gd_loader_slot *slot,
                                                gd_dataloader_fill_stats *stats)
{
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;
    size_t n_elem;
    size_t nbytes;
    int b;
    int j;
    gd_status status;

    if (dl == NULL || slot == NULL || stats == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader fill args");
    }
    memset(stats, 0, sizeof(*stats));
    n_elem = (size_t)dl->cfg.batch_size * (size_t)dl->cfg.block_len;
    nbytes = n_elem * sizeof(int32_t);
    t0 = _gd_profile_now_ns();
    for (b = 0; b < dl->cfg.batch_size; ++b) {
        int32_t *tokens = &slot->host_tokens[(size_t)b * (size_t)dl->cfg.block_len];
        int32_t *targets = &slot->host_targets[(size_t)b * (size_t)dl->cfg.block_len];
        int32_t *positions = &slot->host_positions[(size_t)b * (size_t)dl->cfg.block_len];
        status = gd_token_dataset_read_sample(dl->ds, slot->sample_ids[b], tokens, targets);
        if (status != GD_OK) {
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
        return status;
    }
    stats->host_fill_ns = t1 - t0;
    stats->host_to_device_copy_ns = t2 - t1;
    stats->samples_prepared = (uint64_t)dl->cfg.batch_size;
    return GD_OK;
}

static int gd_dataloader_ready_count_locked(const gd_dataloader *dl)
{
    int ready_depth = 0;
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_READY) {
            ready_depth += 1;
        }
    }
    return ready_depth;
}

void _gd_dataloader_add_fill_metrics_locked(gd_dataloader *dl,
                                            const gd_dataloader_fill_stats *stats)
{
    int ready_depth;
    if (dl == NULL || stats == NULL) {
        return;
    }
    dl->metrics.host_fill_ns += stats->host_fill_ns;
    dl->metrics.host_to_device_copy_ns += stats->host_to_device_copy_ns;
    dl->metrics.batches_prepared += 1U;
    dl->metrics.samples_prepared += stats->samples_prepared;
    ready_depth = gd_dataloader_ready_count_locked(dl);
    if ((uint64_t)ready_depth > dl->metrics.max_ready_depth) {
        dl->metrics.max_ready_depth = (uint64_t)ready_depth;
    }
}

static gd_loader_slot *gd_dataloader_find_free_slot_locked(gd_dataloader *dl)
{
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_FREE) {
            return &dl->slots[i];
        }
    }
    return NULL;
}

static gd_loader_slot *gd_dataloader_find_ready_seq_locked(gd_dataloader *dl,
                                                           uint64_t seq)
{
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_SLOT_READY && dl->slots[i].seq == seq) {
            return &dl->slots[i];
        }
    }
    return NULL;
}

static uint64_t gd_dataloader_queued_locked(const gd_dataloader *dl)
{
    uint64_t queued;
    int i;
    queued = dl->requested;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state != GD_BATCH_SLOT_FREE) {
            queued += 1U;
        }
    }
    return queued;
}

gd_status _gd_dataloader_worker_status_locked(const gd_dataloader *dl)
{
    if (dl == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader is null");
    }
    if (dl->worker_status != GD_OK) {
        const char *msg = dl->worker_error[0] != '\0' ? dl->worker_error :
                          "dataloader worker failed";
        return _gd_error(dl->worker_status, msg);
    }
    return GD_OK;
}

static void gd_dataloader_set_worker_error_locked(gd_dataloader *dl, gd_status status)
{
    const char *err;
    if (dl->worker_status != GD_OK) {
        return;
    }
    dl->worker_status = status != GD_OK ? status : GD_ERR_INTERNAL;
    err = gd_last_error();
    if (err != NULL && err[0] != '\0') {
        (void)snprintf(dl->worker_error, sizeof(dl->worker_error), "%s", err);
    } else {
        (void)snprintf(dl->worker_error, sizeof(dl->worker_error),
                       "%s", gd_status_message(dl->worker_status));
    }
}

static gd_status gd_dataloader_request_locked(gd_dataloader *dl)
{
    gd_status status = _gd_dataloader_worker_status_locked(dl);
    if (status != GD_OK) {
        return status;
    }
    if (gd_dataloader_queued_locked(dl) >= (uint64_t)dl->n_slots) {
        if (gd_dataloader_ready_count_locked(dl) > 0) {
            return GD_OK;
        }
        return _gd_error(GD_ERR_INVALID_STATE, "no free dataloader slot");
    }
    dl->requested += 1U;
    dl->metrics.prefetch_requests += 1U;
    pthread_cond_signal(&dl->work_cv);
    return GD_OK;
}

static void *gd_dataloader_worker_main(void *arg)
{
    gd_dataloader *dl = (gd_dataloader *)arg;
    for (;;) {
        gd_loader_slot *slot = NULL;
        gd_dataloader_fill_stats stats;
        gd_status status = GD_OK;
        int b;

        pthread_mutex_lock(&dl->mutex);
        while (dl->stop == 0) {
            slot = gd_dataloader_find_free_slot_locked(dl);
            if (dl->paused == 0 && dl->requested > 0U && slot != NULL &&
                dl->worker_status == GD_OK) {
                break;
            }
            pthread_cond_wait(&dl->work_cv, &dl->mutex);
        }
        if (dl->stop != 0) {
            pthread_mutex_unlock(&dl->mutex);
            break;
        }
        slot->pub.state = GD_BATCH_SLOT_FILLING;
        slot->seq = dl->next_seq;
        dl->next_seq += 1U;
        dl->requested -= 1U;
        for (b = 0; b < dl->cfg.batch_size; ++b) {
            slot->sample_ids[b] = gd_dataloader_next_sample_locked(dl);
        }
        dl->filling_count += 1;
        pthread_mutex_unlock(&dl->mutex);

        status = _gd_dataloader_fill_slot_from_samples(dl, slot, &stats);

        pthread_mutex_lock(&dl->mutex);
        dl->filling_count -= 1;
        if (status == GD_OK) {
            slot->pub.state = GD_BATCH_SLOT_READY;
            _gd_dataloader_add_fill_metrics_locked(dl, &stats);
        } else {
            slot->pub.state = GD_BATCH_SLOT_FREE;
            gd_dataloader_set_worker_error_locked(dl, status);
        }
        pthread_cond_broadcast(&dl->ready_cv);
        pthread_cond_broadcast(&dl->work_cv);
        pthread_cond_broadcast(&dl->state_cv);
        pthread_mutex_unlock(&dl->mutex);
    }
    return NULL;
}

static gd_status gd_dataloader_init_sync(gd_dataloader *dl)
{
    if (pthread_mutex_init(&dl->mutex, NULL) != 0) {
        return _gd_error(GD_ERR_INTERNAL, "failed to init dataloader mutex");
    }
    if (pthread_cond_init(&dl->work_cv, NULL) != 0) {
        pthread_mutex_destroy(&dl->mutex);
        return _gd_error(GD_ERR_INTERNAL, "failed to init dataloader work cond");
    }
    if (pthread_cond_init(&dl->ready_cv, NULL) != 0) {
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
        return _gd_error(GD_ERR_INTERNAL, "failed to init dataloader ready cond");
    }
    if (pthread_cond_init(&dl->state_cv, NULL) != 0) {
        pthread_cond_destroy(&dl->ready_cv);
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
        return _gd_error(GD_ERR_INTERNAL, "failed to init dataloader state cond");
    }
    dl->sync_ready = 1;
    return GD_OK;
}

static gd_status gd_dataloader_start_workers(gd_dataloader *dl)
{
    int i;
    for (i = 0; i < dl->n_workers; ++i) {
        if (pthread_create(&dl->workers[i], NULL, gd_dataloader_worker_main, dl) != 0) {
            return _gd_error(GD_ERR_INTERNAL, "failed to start dataloader worker");
        }
        dl->workers_started += 1;
    }
    return GD_OK;
}

static gd_status gd_dataloader_normalize_config(const gd_dataloader_config *cfg,
                                                int *workers_out,
                                                int *prefetch_out,
                                                int *slots_out)
{
    int workers;
    int prefetch;
    uint64_t slots;

    if (cfg->num_workers < 0 || cfg->prefetch_factor < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader worker config");
    }
    workers = cfg->num_workers > 0 ? cfg->num_workers : GD_DL_DEFAULT_WORKERS;
    prefetch = cfg->prefetch_factor > 0 ? cfg->prefetch_factor : GD_DL_DEFAULT_PREFETCH_FACTOR;
    if (workers <= 0 || workers > GD_DL_MAX_WORKERS ||
        prefetch <= 0 || prefetch > GD_DL_MAX_PREFETCH_FACTOR) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader worker config out of range");
    }
    slots = (uint64_t)workers * (uint64_t)prefetch;
    if (slots < 2U) {
        slots = 2U;
    }
    if (slots > GD_DL_MAX_SLOTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader slot count out of range");
    }
    *workers_out = workers;
    *prefetch_out = prefetch;
    *slots_out = (int)slots;
    return GD_OK;
}

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_token_dataset *ds,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out)
{
    gd_dataloader *dl;
    int i;
    int n_workers = 0;
    int prefetch_factor = 0;
    int n_slots = 0;
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
    status = gd_dataloader_normalize_config(cfg, &n_workers, &prefetch_factor, &n_slots);
    if (status != GD_OK) {
        return status;
    }
    dl = (gd_dataloader *)calloc(1U, sizeof(*dl));
    if (dl == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader allocation failed");
    }
    dl->ctx = ctx;
    dl->ds = ds;
    dl->cfg = *cfg;
    dl->cfg.num_workers = n_workers;
    dl->cfg.prefetch_factor = prefetch_factor;
    dl->n_workers = n_workers;
    dl->n_slots = n_slots;
    dl->rng_state = cfg->seed;
    dl->cursor = 0U;
    dl->worker_status = GD_OK;
    dl->slots = (gd_loader_slot *)calloc((size_t)n_slots, sizeof(gd_loader_slot));
    dl->workers = (pthread_t *)calloc((size_t)n_workers, sizeof(pthread_t));
    if (dl->slots == NULL || dl->workers == NULL) {
        gd_dataloader_destroy(dl);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader queue allocation failed");
    }
    status = gd_dataloader_init_sync(dl);
    if (status != GD_OK) {
        gd_dataloader_destroy(dl);
        return status;
    }
    for (i = 0; i < dl->n_slots; ++i) {
        status = gd_loader_slot_init(ctx, &dl->cfg, i, &dl->slots[i]);
        if (status != GD_OK) {
            gd_dataloader_destroy(dl);
            return status;
        }
    }
    status = gd_dataloader_start_workers(dl);
    if (status != GD_OK) {
        gd_dataloader_destroy(dl);
        return status;
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
    if (dl->sync_ready != 0) {
        pthread_mutex_lock(&dl->mutex);
        dl->stop = 1;
        pthread_cond_broadcast(&dl->work_cv);
        pthread_mutex_unlock(&dl->mutex);
    }
    for (i = 0; i < dl->workers_started; ++i) {
        (void)pthread_join(dl->workers[i], NULL);
    }
    if (dl->slots != NULL) {
        for (i = 0; i < dl->n_slots; ++i) {
            gd_loader_slot_clear(&dl->slots[i]);
        }
    }
    if (dl->sync_ready != 0) {
        pthread_cond_destroy(&dl->state_cv);
        pthread_cond_destroy(&dl->ready_cv);
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
    }
    free(dl->workers);
    free(dl->slots);
    free(dl);
}

gd_status gd_dataloader_prefetch(gd_dataloader *dl)
{
    gd_status status;
    if (dl == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader is null");
    }
    pthread_mutex_lock(&dl->mutex);
    status = gd_dataloader_request_locked(dl);
    pthread_mutex_unlock(&dl->mutex);
    return status;
}

gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch_slot **slot_out)
{
    uint64_t t0;
    int waited = 0;
    gd_loader_slot *slot;
    gd_status status;

    if (dl == NULL || slot_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader next args");
    }
    *slot_out = NULL;
    t0 = _gd_profile_now_ns();
    pthread_mutex_lock(&dl->mutex);
    status = _gd_dataloader_worker_status_locked(dl);
    if (status != GD_OK) {
        pthread_mutex_unlock(&dl->mutex);
        return status;
    }
    slot = gd_dataloader_find_ready_seq_locked(dl, dl->deliver_seq);
    while (slot == NULL) {
        if (gd_dataloader_queued_locked(dl) == 0U) {
            status = gd_dataloader_request_locked(dl);
            if (status != GD_OK) {
                pthread_mutex_unlock(&dl->mutex);
                return status;
            }
        } else if (dl->requested == 0U && dl->filling_count == 0) {
            pthread_mutex_unlock(&dl->mutex);
            return _gd_error(GD_ERR_INTERNAL, "dataloader ordered batch is missing");
        }
        waited = 1;
        pthread_cond_wait(&dl->ready_cv, &dl->mutex);
        status = _gd_dataloader_worker_status_locked(dl);
        if (status != GD_OK) {
            pthread_mutex_unlock(&dl->mutex);
            return status;
        }
        slot = gd_dataloader_find_ready_seq_locked(dl, dl->deliver_seq);
    }
    if (waited != 0) {
        dl->metrics.wait_for_batch_ns += _gd_profile_now_ns() - t0;
    }
    slot->pub.state = GD_BATCH_SLOT_IN_USE;
    dl->deliver_seq += 1U;
    dl->metrics.batches_returned += 1U;
    *slot_out = &slot->pub;
    pthread_mutex_unlock(&dl->mutex);
    return GD_OK;
}

gd_status gd_dataloader_release_slot(gd_dataloader *dl, gd_batch_slot *slot)
{
    int i;
    if (dl == NULL || slot == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader release args");
    }
    pthread_mutex_lock(&dl->mutex);
    for (i = 0; i < dl->n_slots; ++i) {
        if (&dl->slots[i].pub == slot) {
            if (dl->slots[i].pub.state != GD_BATCH_SLOT_IN_USE) {
                pthread_mutex_unlock(&dl->mutex);
                return _gd_error(GD_ERR_INVALID_STATE, "dataloader slot is not in use");
            }
            dl->slots[i].pub.state = GD_BATCH_SLOT_FREE;
            dl->slots[i].seq = 0U;
            pthread_cond_signal(&dl->work_cv);
            pthread_mutex_unlock(&dl->mutex);
            return GD_OK;
        }
    }
    pthread_mutex_unlock(&dl->mutex);
    return _gd_error(GD_ERR_INVALID_ARGUMENT, "slot does not belong to dataloader");
}

int gd_dataloader_slot_count(const gd_dataloader *dl)
{
    return dl != NULL ? dl->n_slots : 0;
}

int _gd_dataloader_has_live_slot_locked(const gd_dataloader *dl)
{
    int i;
    if (dl == NULL) {
        return 0;
    }
    if (dl->requested != 0U) {
        return 1;
    }
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state != GD_BATCH_SLOT_FREE) {
            return 1;
        }
    }
    return 0;
}

gd_status _gd_dataloader_lock_quiesced(gd_dataloader *dl)
{
    gd_status status;
    if (dl == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader is null");
    }
    pthread_mutex_lock(&dl->mutex);
    status = _gd_dataloader_worker_status_locked(dl);
    if (status != GD_OK) {
        pthread_mutex_unlock(&dl->mutex);
        return status;
    }
    dl->paused = 1;
    pthread_cond_broadcast(&dl->work_cv);
    while (dl->filling_count > 0 && dl->worker_status == GD_OK) {
        pthread_cond_wait(&dl->state_cv, &dl->mutex);
    }
    status = _gd_dataloader_worker_status_locked(dl);
    if (status != GD_OK) {
        dl->paused = 0;
        pthread_cond_broadcast(&dl->work_cv);
        pthread_mutex_unlock(&dl->mutex);
        return status;
    }
    return GD_OK;
}

void _gd_dataloader_unlock_resume(gd_dataloader *dl)
{
    if (dl == NULL) {
        return;
    }
    dl->paused = 0;
    pthread_cond_broadcast(&dl->work_cv);
    pthread_mutex_unlock(&dl->mutex);
}

void gd_dataloader_metrics_get(const gd_dataloader *dl,
                               gd_dataloader_metrics *metrics_out)
{
    gd_dataloader *mut;
    if (metrics_out == NULL) {
        return;
    }
    memset(metrics_out, 0, sizeof(*metrics_out));
    if (dl == NULL) {
        return;
    }
    mut = (gd_dataloader *)dl;
    pthread_mutex_lock(&mut->mutex);
    *metrics_out = mut->metrics;
    pthread_mutex_unlock(&mut->mutex);
}
