#include "dataloader_internal.h"

#include "../core/internal.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static uint64_t gd_dl_splitmix64(uint64_t *state)
{
    uint64_t z;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static uint64_t gd_dl_fnv64_bytes(uint64_t h, const void *data, size_t n)
{
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0U; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t gd_dl_fnv64_u64(uint64_t h, uint64_t v)
{
    return gd_dl_fnv64_bytes(h, &v, sizeof(v));
}

static uint64_t gd_dl_fnv64_str(uint64_t h, const char *s)
{
    if (s == NULL) {
        return gd_dl_fnv64_u64(h, 0U);
    }
    return gd_dl_fnv64_bytes(h, s, strlen(s) + 1U);
}

static gd_status gd_dl_field_nbytes(const gd_batch_field_desc *desc,
                                    size_t *out)
{
    size_t item_size;
    size_t numel = 1U;
    int i;
    if (desc == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid batch field size args");
    }
    if (desc->name == NULL || desc->name[0] == '\0') {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "batch field name is empty");
    }
    if (desc->rank < 0 || desc->rank > GD_BATCH_MAX_RANK) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "batch field rank out of range");
    }
    item_size = gd_dtype_sizeof(desc->dtype);
    if (item_size == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "batch field dtype is invalid");
    }
    for (i = 0; i < desc->rank; ++i) {
        if (desc->sizes[i] <= 0) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "batch field dimension is invalid");
        }
        if ((uint64_t)desc->sizes[i] > (uint64_t)(SIZE_MAX / numel)) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "batch field numel overflow");
        }
        numel *= (size_t)desc->sizes[i];
    }
    if (numel > SIZE_MAX / item_size) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "batch field byte size overflow");
    }
    *out = numel * item_size;
    return GD_OK;
}

static gd_status gd_dl_validate_fields(const gd_batch_field_desc *fields,
                                       int n_fields)
{
    int i;
    int j;
    if (fields == NULL || n_fields <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader fields");
    }
    for (i = 0; i < n_fields; ++i) {
        size_t nbytes = 0U;
        gd_status status = gd_dl_field_nbytes(&fields[i], &nbytes);
        if (status != GD_OK) {
            return status;
        }
        for (j = i + 1; j < n_fields; ++j) {
            if (fields[i].name != NULL && fields[j].name != NULL &&
                strcmp(fields[i].name, fields[j].name) == 0) {
                return _gd_error(GD_ERR_INVALID_ARGUMENT, "duplicate batch field name");
            }
        }
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
    int i;
    if (slot == NULL) {
        return;
    }
    if (slot->pub.fields != NULL) {
        for (i = 0; i < slot->pub.n_fields; ++i) {
            gd_tensor_release(slot->pub.fields[i].tensor);
            free(slot->pub.fields[i].host_data);
            free(slot->pub.fields[i].name);
        }
    }
    free(slot->pub.fields);
    free(slot->pub.sample_ids);
    memset(slot, 0, sizeof(*slot));
}

static gd_status gd_loader_slot_init(gd_dataloader *dl,
                                     int index,
                                     gd_loader_slot *slot)
{
    int i;
    if (dl == NULL || slot == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader slot init args");
    }
    memset(slot, 0, sizeof(*slot));
    slot->pub.index = index;
    slot->pub.state = GD_BATCH_FREE;
    slot->pub.batch_size = dl->cfg.batch_size;
    slot->pub.n_fields = dl->n_fields;
    slot->pub.fields = (gd_batch_field *)calloc((size_t)dl->n_fields,
                                                sizeof(gd_batch_field));
    slot->pub.sample_ids = (uint64_t *)calloc((size_t)dl->cfg.batch_size,
                                             sizeof(uint64_t));
    if (slot->pub.fields == NULL || slot->pub.sample_ids == NULL) {
        gd_loader_slot_clear(slot);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "loader slot allocation failed");
    }
    for (i = 0; i < dl->n_fields; ++i) {
        const gd_batch_field_desc *src = &dl->field_descs[i];
        gd_batch_field *dst = &slot->pub.fields[i];
        gd_tensor_desc tensor_desc;
        gd_status status;
        size_t nbytes = 0U;
        int d;
        status = gd_dl_field_nbytes(src, &nbytes);
        if (status != GD_OK) {
            gd_loader_slot_clear(slot);
            return status;
        }
        dst->name = gd_dl_strdup(src->name);
        dst->dtype = src->dtype;
        dst->rank = src->rank;
        dst->nbytes = nbytes;
        for (d = 0; d < GD_BATCH_MAX_RANK; ++d) {
            dst->sizes[d] = src->sizes[d];
        }
        dst->host_data = gd_dl_alloc_host_buffer(nbytes);
        if (dst->name == NULL || dst->host_data == NULL) {
            gd_loader_slot_clear(slot);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "loader field allocation failed");
        }
        status = gd_tensor_desc_contiguous(dst->dtype, dl->cfg.device, dst->rank,
                                           dst->sizes, &tensor_desc);
        if (status == GD_OK) {
            status = gd_tensor_empty(dl->ctx, &tensor_desc, &dst->tensor);
        }
        if (status != GD_OK) {
            gd_loader_slot_clear(slot);
            return status;
        }
    }
    return GD_OK;
}

static uint64_t gd_dataloader_next_sample_locked(gd_dataloader *dl)
{
    uint64_t n_samples = gd_dataset_num_samples(dl->dataset);
    if (dl->cfg.sampler == GD_SAMPLER_SEQUENTIAL) {
        uint64_t sample = dl->cursor % n_samples;
        dl->cursor += 1U;
        return sample;
    }
    return gd_dl_splitmix64(&dl->rng_state) % n_samples;
}

gd_status _gd_dataloader_fill_slot_from_samples(gd_dataloader *dl,
                                                gd_loader_slot *slot,
                                                gd_dataloader_fill_stats *stats)
{
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;
    int i;
    gd_status status;

    if (dl == NULL || slot == NULL || stats == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid loader fill args");
    }
    memset(stats, 0, sizeof(*stats));
    t0 = _gd_profile_now_ns();
    status = dl->collate(dl->dataset, slot->pub.sample_ids, dl->cfg.batch_size,
                         &slot->pub, dl->collate_data);
    t1 = _gd_profile_now_ns();
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < slot->pub.n_fields; ++i) {
        gd_batch_field *field = &slot->pub.fields[i];
        status = gd_tensor_copy_from_cpu(dl->ctx, field->tensor,
                                         field->host_data, field->nbytes);
        if (status != GD_OK) {
            return status;
        }
    }
    t2 = _gd_profile_now_ns();
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
        if (dl->slots[i].pub.state == GD_BATCH_READY) {
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
        if (dl->slots[i].pub.state == GD_BATCH_FREE) {
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
        if (dl->slots[i].pub.state == GD_BATCH_READY && dl->slots[i].seq == seq) {
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
        if (dl->slots[i].pub.state != GD_BATCH_FREE) {
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

static void gd_dataloader_set_worker_error_locked(gd_dataloader *dl,
                                                  gd_status status)
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
        dl->requested -= 1U;
        slot->pub.state = GD_BATCH_FILLING;
        slot->seq = dl->next_seq;
        dl->next_seq += 1U;
        for (b = 0; b < dl->cfg.batch_size; ++b) {
            slot->pub.sample_ids[b] = gd_dataloader_next_sample_locked(dl);
        }
        dl->filling_count += 1;
        pthread_mutex_unlock(&dl->mutex);

        status = _gd_dataloader_fill_slot_from_samples(dl, slot, &stats);

        pthread_mutex_lock(&dl->mutex);
        dl->filling_count -= 1;
        if (status == GD_OK) {
            slot->pub.state = GD_BATCH_READY;
            _gd_dataloader_add_fill_metrics_locked(dl, &stats);
            pthread_cond_broadcast(&dl->ready_cv);
        } else {
            slot->pub.state = GD_BATCH_FREE;
            gd_dataloader_set_worker_error_locked(dl, status);
            pthread_cond_broadcast(&dl->ready_cv);
        }
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

static gd_status gd_dataloader_copy_fields(gd_dataloader *dl,
                                           const gd_batch_field_desc *fields)
{
    int i;
    dl->field_descs = (gd_batch_field_desc *)calloc((size_t)dl->n_fields,
                                                    sizeof(gd_batch_field_desc));
    if (dl->field_descs == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader field descriptor allocation failed");
    }
    for (i = 0; i < dl->n_fields; ++i) {
        dl->field_descs[i] = fields[i];
        dl->field_descs[i].name = gd_dl_strdup(fields[i].name);
        if (dl->field_descs[i].name == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader field name allocation failed");
        }
    }
    return GD_OK;
}

uint64_t _gd_dataloader_schema_hash(const gd_dataloader *dl)
{
    uint64_t h = UINT64_C(1469598103934665603);
    int i;
    int d;
    if (dl == NULL) {
        return 0U;
    }
    h = gd_dl_fnv64_str(h, "gd-batch-schema-v1");
    h = gd_dl_fnv64_u64(h, (uint64_t)dl->n_fields);
    for (i = 0; i < dl->n_fields; ++i) {
        const gd_batch_field_desc *field = &dl->field_descs[i];
        h = gd_dl_fnv64_str(h, field->name);
        h = gd_dl_fnv64_u64(h, (uint64_t)field->dtype);
        h = gd_dl_fnv64_u64(h, (uint64_t)field->rank);
        for (d = 0; d < field->rank; ++d) {
            h = gd_dl_fnv64_u64(h, (uint64_t)field->sizes[d]);
        }
    }
    return h != 0U ? h : 1U;
}

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_dataset *dataset,
                               const gd_dataloader_config *cfg,
                               const gd_batch_field_desc *fields,
                               int n_fields,
                               gd_collate_fn collate,
                               void *collate_data,
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
    if (ctx == NULL || dataset == NULL || cfg == NULL || collate == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader create args");
    }
    if (cfg->batch_size <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader batch size");
    }
    if (cfg->sampler != GD_SAMPLER_RANDOM_REPLACEMENT &&
        cfg->sampler != GD_SAMPLER_SEQUENTIAL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader sampler");
    }
    if (gd_dataset_num_samples(dataset) == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader dataset is empty");
    }
    if (cfg->expected_dataset_fingerprint != 0U &&
        cfg->expected_dataset_fingerprint != gd_dataset_fingerprint(dataset)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader dataset fingerprint mismatch");
    }
    status = gd_dl_validate_fields(fields, n_fields);
    if (status != GD_OK) {
        return status;
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
    dl->dataset = dataset;
    dl->cfg = *cfg;
    dl->cfg.num_workers = n_workers;
    dl->cfg.prefetch_factor = prefetch_factor;
    dl->n_workers = n_workers;
    dl->n_slots = n_slots;
    dl->n_fields = n_fields;
    dl->collate = collate;
    dl->collate_data = collate_data;
    dl->rng_state = cfg->seed;
    dl->cursor = 0U;
    dl->worker_status = GD_OK;
    status = gd_dataloader_copy_fields(dl, fields);
    if (status != GD_OK) {
        gd_dataloader_destroy(dl);
        return status;
    }
    dl->schema_hash = _gd_dataloader_schema_hash(dl);
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
        status = gd_loader_slot_init(dl, i, &dl->slots[i]);
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
    if (dl->field_descs != NULL) {
        for (i = 0; i < dl->n_fields; ++i) {
            free((char *)dl->field_descs[i].name);
        }
    }
    if (dl->sync_ready != 0) {
        pthread_cond_destroy(&dl->state_cv);
        pthread_cond_destroy(&dl->ready_cv);
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
    }
    free(dl->field_descs);
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

gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch **batch_out)
{
    uint64_t t0;
    int waited = 0;
    gd_loader_slot *slot;
    gd_status status;

    if (dl == NULL || batch_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader next args");
    }
    *batch_out = NULL;
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
    slot->pub.state = GD_BATCH_IN_USE;
    dl->deliver_seq += 1U;
    dl->metrics.batches_returned += 1U;
    *batch_out = &slot->pub;
    pthread_mutex_unlock(&dl->mutex);
    return GD_OK;
}

gd_status gd_dataloader_release(gd_dataloader *dl, gd_batch *batch)
{
    int i;
    if (dl == NULL || batch == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader release args");
    }
    pthread_mutex_lock(&dl->mutex);
    for (i = 0; i < dl->n_slots; ++i) {
        if (&dl->slots[i].pub == batch) {
            if (dl->slots[i].pub.state != GD_BATCH_IN_USE) {
                pthread_mutex_unlock(&dl->mutex);
                return _gd_error(GD_ERR_INVALID_STATE, "dataloader batch is not in use");
            }
            dl->slots[i].pub.state = GD_BATCH_FREE;
            dl->slots[i].seq = 0U;
            pthread_cond_signal(&dl->work_cv);
            pthread_mutex_unlock(&dl->mutex);
            return GD_OK;
        }
    }
    pthread_mutex_unlock(&dl->mutex);
    return _gd_error(GD_ERR_INVALID_ARGUMENT, "batch does not belong to dataloader");
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
        if (dl->slots[i].pub.state != GD_BATCH_FREE) {
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

