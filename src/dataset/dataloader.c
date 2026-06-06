#include "dataloader_internal.h"

#include "../core/memory_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

gd_dataloader_config gd_dataloader_config_default(int batch_size)
{
    gd_dataloader_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.batch_size = batch_size;
    return cfg;
}

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

static uint64_t gd_dl_splitmix64_value(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31U);
}

static uint32_t gd_sampler_ceil_log2_u64(uint64_t value)
{
    uint32_t bits = 0U;
    uint64_t x;
    if (value <= 1U) {
        return 0U;
    }
    x = value - 1U;
    while (x != 0U) {
        bits += 1U;
        x >>= 1U;
    }
    return bits;
}

static uint64_t gd_sampler_feistel_round(uint64_t right,
                                         uint64_t key,
                                         uint32_t round,
                                         uint64_t half_mask)
{
    uint64_t x = right;
    x ^= key + UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(round + 1U);
    x ^= (uint64_t)round << 48U;
    return gd_dl_splitmix64_value(x) & half_mask;
}

static uint64_t gd_sampler_feistel_permute(uint64_t x,
                                           uint64_t key,
                                           uint32_t half_bits)
{
    const uint32_t rounds = 6U;
    const uint64_t half_mask = (UINT64_C(1) << half_bits) - 1U;
    uint64_t left = x >> half_bits;
    uint64_t right = x & half_mask;
    uint32_t round;
    for (round = 0U; round < rounds; ++round) {
        const uint64_t new_left = right;
        const uint64_t f = gd_sampler_feistel_round(right, key, round, half_mask);
        const uint64_t new_right = (left ^ f) & half_mask;
        left = new_left;
        right = new_right;
    }
    return (left << half_bits) | right;
}

static uint64_t gd_sampler_random_index(const gd_sampler *sampler,
                                        uint64_t epoch,
                                        uint64_t position)
{
    uint64_t x;
    uint64_t key;
    uint32_t bits;
    uint32_t half_bits;
    if (sampler == NULL || sampler->n_samples == 0U) {
        return 0U;
    }
    if (sampler->n_samples == 1U) {
        return 0U;
    }
    bits = gd_sampler_ceil_log2_u64(sampler->n_samples);
    half_bits = (bits + 1U) / 2U;
    if (half_bits > 32U) {
        half_bits = 32U;
    }
    key = gd_dl_splitmix64_value(sampler->seed ^ gd_dl_splitmix64_value(epoch));
    x = position;
    do {
        x = gd_sampler_feistel_permute(x, key, half_bits);
    } while (x >= sampler->n_samples);
    return x;
}

gd_status gd_sampler_create_random(const gd_dataset *dataset,
                                   uint64_t seed,
                                   gd_sampler **out)
{
    gd_sampler *sampler;
    uint64_t n_samples;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (dataset == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_samples = gd_dataset_num_samples(dataset);
    if (n_samples == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    sampler = (gd_sampler *)calloc(1U, sizeof(*sampler));
    if (sampler == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    sampler->kind = GD_SAMPLER_KIND_RANDOM;
    sampler->n_samples = n_samples;
    sampler->seed = seed;
    *out = sampler;
    return GD_OK;
}

void gd_sampler_destroy(gd_sampler *sampler)
{
    free(sampler);
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

static gd_status gd_dl_field_nbytes(const gd_batch_field_desc *desc, size_t *out)
{
    size_t item_size;
    size_t numel = 1U;
    int i;
    if (desc == NULL || out == NULL || desc->name == NULL || desc->name[0] == '\0' ||
        desc->rank < 0 || desc->rank > GD_BATCH_MAX_RANK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(desc->dtype);
    if (item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < desc->rank; ++i) {
        if (desc->sizes[i] < 0) {
            *out = 0U;
            return GD_OK;
        }
        if (desc->sizes[i] == 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((uint64_t)desc->sizes[i] > (uint64_t)(SIZE_MAX / numel)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        numel *= (size_t)desc->sizes[i];
    }
    if (numel > SIZE_MAX / item_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = numel * item_size;
    return GD_OK;
}

static gd_status gd_dl_validate_fields(const gd_batch_field_desc *fields, int n_fields)
{
    int i;
    int j;
    if (fields == NULL || n_fields <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < n_fields; ++i) {
        size_t nbytes = 0U;
        gd_status status = gd_dl_field_nbytes(&fields[i], &nbytes);
        if (status != GD_OK) {
            return status;
        }
        (void)nbytes;
        for (j = i + 1; j < n_fields; ++j) {
            if (fields[i].name != NULL && fields[j].name != NULL &&
                strcmp(fields[i].name, fields[j].name) == 0) {
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
    }
    return GD_OK;
}

static void gd_batch_clear_data_binding(gd_batch *slot)
{
    int i;
    if (slot == NULL) {
        return;
    }
    if (slot->ctx != NULL && slot->data_slot >= 0 &&
        (slot->state == GD_BATCH_FILLING || slot->state == GD_BATCH_READY ||
         slot->state == GD_BATCH_IN_USE)) {
        (void)gd_context_data_slot_abort(slot->ctx, slot->data_slot, slot->data_generation);
    }
    slot->data_slot = -1;
    slot->data_generation = 0U;
    slot->last_fence = 0U;
    for (i = 0; i < slot->n_fields; ++i) {
        memset(&slot->fields[i].tensor, 0, sizeof(slot->fields[i].tensor));
        slot->fields[i].has_tensor = false;
    }
}

static void gd_batch_destroy_slot(gd_batch *slot)
{
    int i;
    if (slot == NULL) {
        return;
    }
    gd_batch_clear_data_binding(slot);
    if (slot->fields != NULL) {
        for (i = 0; i < slot->n_fields; ++i) {
            free(slot->fields[i].host_data);
            free(slot->fields[i].name);
        }
    }
    free(slot->fields);
    free(slot->sample_ids);
    memset(slot, 0, sizeof(*slot));
    slot->data_slot = -1;
}

static gd_status gd_batch_init_slot(gd_dataloader *dl, int index, gd_batch *slot)
{
    int i;
    if (dl == NULL || slot == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(slot, 0, sizeof(*slot));
    slot->index = index;
    slot->state = GD_BATCH_FREE;
    slot->batch_size = dl->cfg.batch_size;
    slot->n_fields = dl->n_fields;
    slot->ctx = dl->ctx;
    slot->data_slot = -1;
    slot->fields = (gd_batch_field *)calloc((size_t)dl->n_fields, sizeof(*slot->fields));
    slot->sample_ids = (uint64_t *)calloc((size_t)dl->cfg.batch_size,
                                          sizeof(*slot->sample_ids));
    if (slot->fields == NULL || slot->sample_ids == NULL) {
        gd_batch_destroy_slot(slot);
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0; i < dl->n_fields; ++i) {
        const gd_batch_field_desc *src = &dl->field_descs[i];
        gd_batch_field *dst = &slot->fields[i];
        size_t nbytes = 0U;
        gd_status status = gd_dl_field_nbytes(src, &nbytes);
        int d;
        if (status != GD_OK) {
            gd_batch_destroy_slot(slot);
            return status;
        }
        dst->name = gd_dl_strdup(src->name);
        dst->dtype = src->dtype;
        dst->rank = src->rank;
        dst->nbytes = nbytes;
        dst->host_capacity_nbytes = nbytes;
        for (d = 0; d < GD_BATCH_MAX_RANK; ++d) {
            dst->sizes[d] = src->sizes[d];
        }
        dst->host_data = nbytes > 0U ? malloc(nbytes) : NULL;
        if (dst->name == NULL || (nbytes > 0U && dst->host_data == NULL)) {
            gd_batch_destroy_slot(slot);
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    return GD_OK;
}

static uint64_t gd_dataloader_sample_at_locked(const gd_dataloader *dl,
                                                uint64_t position)
{
    if (dl->sampler != NULL) {
        return gd_sampler_random_index(dl->sampler, dl->epoch, position);
    }
    return position;
}

static void gd_dataloader_assign_batch_locked(gd_dataloader *dl, gd_batch *slot)
{
    int b;
    if (dl->samples_in_epoch + (uint64_t)dl->cfg.batch_size > dl->samples_per_epoch) {
        dl->epoch += 1U;
        dl->samples_in_epoch = 0U;
    }
    for (b = 0; b < dl->cfg.batch_size; ++b) {
        const uint64_t position = dl->samples_in_epoch + (uint64_t)b;
        slot->sample_ids[b] = gd_dataloader_sample_at_locked(dl, position);
    }
    dl->samples_in_epoch += (uint64_t)dl->cfg.batch_size;
}

gd_status gd_dl_fill_slot(gd_dataloader *dl,
                          gd_batch *slot,
                          gd_dataloader_fill_stats *stats)
{
    uint64_t t0;
    uint64_t t1;
    uint64_t t2;
    int32_t data_slot = -1;
    uint64_t data_generation = 0U;
    int i;
    gd_status status;
    if (dl == NULL || slot == NULL || stats == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(stats, 0, sizeof(*stats));
    gd_batch_clear_data_binding(slot);
    t0 = gd_context_now_ns();
    status = dl->collate(dl->dataset, slot->sample_ids, dl->cfg.batch_size,
                         slot, dl->collate_data);
    t1 = gd_context_now_ns();
    if (status != GD_OK) {
        return status;
    }
    status = gd_context_data_slot_acquire(dl->ctx, &data_slot, &data_generation);
    if (status != GD_OK) {
        return status;
    }
    slot->data_slot = data_slot;
    slot->data_generation = data_generation;
    for (i = 0; i < slot->n_fields; ++i) {
        gd_batch_field *field = &slot->fields[i];
        gd_shape shape;
        int d;
        if (field->rank < 0 || field->rank > GD_BATCH_MAX_RANK ||
            field->nbytes == 0U || field->host_data == NULL) {
            (void)gd_context_data_slot_abort(dl->ctx, data_slot, data_generation);
            gd_batch_clear_data_binding(slot);
            return GD_ERR_INVALID_ARGUMENT;
        }
        for (d = 0; d < field->rank; ++d) {
            if (field->sizes[d] <= 0) {
                (void)gd_context_data_slot_abort(dl->ctx, data_slot, data_generation);
                gd_batch_clear_data_binding(slot);
                return GD_ERR_INVALID_ARGUMENT;
            }
        }
        shape = gd_shape_make((uint32_t)field->rank, field->sizes);
        status = gd_context_data_slot_tensor(dl->ctx,
                                             data_slot,
                                             data_generation,
                                             field->dtype,
                                             shape,
                                             256U,
                                             &field->tensor);
        if (status != GD_OK) {
            (void)gd_context_data_slot_abort(dl->ctx, data_slot, data_generation);
            gd_batch_clear_data_binding(slot);
            return status;
        }
        if (field->tensor.storage.host_ptr == NULL) {
            (void)gd_context_data_slot_abort(dl->ctx, data_slot, data_generation);
            gd_batch_clear_data_binding(slot);
            return GD_ERR_UNSUPPORTED;
        }
        memcpy(field->tensor.storage.host_ptr, field->host_data, field->nbytes);
        field->tensor.version += 1U;
        if (field->tensor.version == 0U) {
            field->tensor.version = 1U;
        }
        field->has_tensor = true;
    }
    status = gd_context_data_slot_publish(dl->ctx, data_slot, data_generation);
    if (status != GD_OK) {
        (void)gd_context_data_slot_abort(dl->ctx, data_slot, data_generation);
        gd_batch_clear_data_binding(slot);
        return status;
    }
    t2 = gd_context_now_ns();
    stats->host_fill_ns = t1 - t0;
    stats->host_to_data_copy_ns = t2 - t1;
    stats->samples_prepared = (uint64_t)dl->cfg.batch_size;
    return GD_OK;
}

static int gd_ready_count_locked(const gd_dataloader *dl)
{
    int ready_depth = 0;
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].state == GD_BATCH_READY) {
            ready_depth += 1;
        }
    }
    return ready_depth;
}

void gd_dl_add_fill_metrics_locked(gd_dataloader *dl,
                                   const gd_dataloader_fill_stats *stats)
{
    int ready_depth;
    if (dl == NULL || stats == NULL) {
        return;
    }
    dl->metrics.host_fill_ns += stats->host_fill_ns;
    dl->metrics.host_to_data_copy_ns += stats->host_to_data_copy_ns;
    dl->metrics.batches_prepared += 1U;
    dl->metrics.samples_prepared += stats->samples_prepared;
    ready_depth = gd_ready_count_locked(dl);
    if ((uint64_t)ready_depth > dl->metrics.max_ready_depth) {
        dl->metrics.max_ready_depth = (uint64_t)ready_depth;
    }
}

static gd_batch *gd_find_free_slot_locked(gd_dataloader *dl)
{
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].state == GD_BATCH_FREE) {
            return &dl->slots[i];
        }
    }
    return NULL;
}

static gd_batch *gd_find_ready_seq_locked(gd_dataloader *dl, uint64_t seq)
{
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].state == GD_BATCH_READY && dl->slots[i].seq == seq) {
            return &dl->slots[i];
        }
    }
    return NULL;
}

static uint64_t gd_queued_locked(const gd_dataloader *dl)
{
    uint64_t queued = dl->requested;
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].state != GD_BATCH_FREE) {
            queued += 1U;
        }
    }
    return queued;
}

gd_status gd_dl_worker_status_locked(const gd_dataloader *dl)
{
    if (dl == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return dl->worker_status;
}

static void gd_set_worker_error_locked(gd_dataloader *dl, gd_status status)
{
    if (dl->worker_status != GD_OK) {
        return;
    }
    dl->worker_status = status != GD_OK ? status : GD_ERR_INTERNAL;
    (void)snprintf(dl->worker_error, sizeof(dl->worker_error), "%s",
                   gd_status_string(dl->worker_status));
}

static gd_status gd_request_locked(gd_dataloader *dl)
{
    gd_status status = gd_dl_worker_status_locked(dl);
    if (status != GD_OK) {
        return status;
    }
    if (gd_queued_locked(dl) >= (uint64_t)dl->n_slots) {
        if (gd_ready_count_locked(dl) > 0) {
            return GD_OK;
        }
        return GD_ERR_BAD_STATE;
    }
    dl->requested += 1U;
    dl->metrics.prefetch_requests += 1U;
    pthread_cond_signal(&dl->work_cv);
    return GD_OK;
}

static void *gd_worker_main(void *arg)
{
    gd_dataloader *dl = (gd_dataloader *)arg;
    for (;;) {
        gd_batch *slot = NULL;
        gd_dataloader_fill_stats stats;
        gd_status status;
        pthread_mutex_lock(&dl->mutex);
        while (dl->stop == 0) {
            slot = gd_find_free_slot_locked(dl);
            if (dl->requested > 0U && slot != NULL && dl->worker_status == GD_OK) {
                break;
            }
            pthread_cond_wait(&dl->work_cv, &dl->mutex);
        }
        if (dl->stop != 0) {
            pthread_mutex_unlock(&dl->mutex);
            break;
        }
        dl->requested -= 1U;
        slot->state = GD_BATCH_FILLING;
        slot->seq = dl->next_seq;
        dl->next_seq += 1U;
        gd_dataloader_assign_batch_locked(dl, slot);
        dl->filling_count += 1;
        pthread_mutex_unlock(&dl->mutex);

        status = gd_dl_fill_slot(dl, slot, &stats);

        pthread_mutex_lock(&dl->mutex);
        dl->filling_count -= 1;
        if (status == GD_OK) {
            slot->state = GD_BATCH_READY;
            gd_dl_add_fill_metrics_locked(dl, &stats);
            pthread_cond_broadcast(&dl->ready_cv);
        } else {
            gd_batch_clear_data_binding(slot);
            slot->state = GD_BATCH_FREE;
            gd_set_worker_error_locked(dl, status);
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
        return GD_ERR_INTERNAL;
    }
    if (pthread_cond_init(&dl->work_cv, NULL) != 0) {
        pthread_mutex_destroy(&dl->mutex);
        return GD_ERR_INTERNAL;
    }
    if (pthread_cond_init(&dl->ready_cv, NULL) != 0) {
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
        return GD_ERR_INTERNAL;
    }
    if (pthread_cond_init(&dl->state_cv, NULL) != 0) {
        pthread_cond_destroy(&dl->ready_cv);
        pthread_cond_destroy(&dl->work_cv);
        pthread_mutex_destroy(&dl->mutex);
        return GD_ERR_INTERNAL;
    }
    dl->sync_ready = 1;
    return GD_OK;
}

static gd_status gd_start_workers(gd_dataloader *dl)
{
    int i;
    for (i = 0; i < dl->n_workers; ++i) {
        if (pthread_create(&dl->workers[i], NULL, gd_worker_main, dl) != 0) {
            return GD_ERR_INTERNAL;
        }
        dl->workers_started += 1;
    }
    return GD_OK;
}

static gd_status gd_normalize_config(const gd_dataloader_config *cfg,
                                     int *workers_out,
                                     int *prefetch_out,
                                     int *slots_out)
{
    int workers;
    int prefetch;
    uint64_t slots;
    if (cfg == NULL || workers_out == NULL || prefetch_out == NULL || slots_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cfg->num_workers < 0 || cfg->prefetch_factor < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    workers = cfg->num_workers > 0 ? cfg->num_workers : GD_DL_DEFAULT_WORKERS;
    prefetch = cfg->prefetch_factor > 0 ? cfg->prefetch_factor : GD_DL_DEFAULT_PREFETCH_FACTOR;
    if (workers <= 0 || workers > GD_DL_MAX_WORKERS ||
        prefetch <= 0 || prefetch > GD_DL_MAX_PREFETCH_FACTOR) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    slots = (uint64_t)workers * (uint64_t)prefetch;
    if (slots < 2U) {
        slots = 2U;
    }
    if (slots > GD_DL_MAX_SLOTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *workers_out = workers;
    *prefetch_out = prefetch;
    *slots_out = (int)slots;
    return GD_OK;
}

static gd_status gd_copy_fields(gd_dataloader *dl, const gd_batch_field_desc *fields)
{
    int i;
    dl->field_descs = (gd_batch_field_desc *)calloc((size_t)dl->n_fields,
                                                    sizeof(*dl->field_descs));
    if (dl->field_descs == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0; i < dl->n_fields; ++i) {
        dl->field_descs[i] = fields[i];
        dl->field_descs[i].name = gd_dl_strdup(fields[i].name);
        if (dl->field_descs[i].name == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    return GD_OK;
}

uint64_t gd_dl_schema_hash(const gd_dataloader *dl)
{
    uint64_t h = UINT64_C(1469598103934665603);
    int i;
    int d;
    if (dl == NULL) {
        return 0U;
    }
    h = gd_dl_fnv64_str(h, "gd-batch-schema-v2");
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
                               gd_sampler *sampler,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out)
{
    gd_dataloader *dl;
    gd_batch_field_desc *fields = NULL;
    int n_fields = 0;
    int n_workers = 0;
    int prefetch_factor = 0;
    int n_slots = 0;
    int i;
    gd_status status;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (ctx == NULL || dataset == NULL || cfg == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cfg->batch_size <= 0 || gd_dataset_num_samples(dataset) == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (sampler != NULL &&
        (sampler->kind != GD_SAMPLER_KIND_RANDOM ||
         sampler->n_samples != gd_dataset_num_samples(dataset))) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (gd_dataset_num_samples(dataset) / (uint64_t)cfg->batch_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = _gd_gdds_init_batch_fields(dataset, cfg->batch_size, &fields, &n_fields);
    if (status != GD_OK) {
        return status;
    }
    status = gd_dl_validate_fields(fields, n_fields);
    if (status != GD_OK) {
        free(fields);
        return status;
    }
    status = gd_normalize_config(cfg, &n_workers, &prefetch_factor, &n_slots);
    if (status != GD_OK) {
        free(fields);
        return status;
    }
    {
        gd_memory_stats mem_stats;
        status = gd_memory_stats_query(ctx, &mem_stats);
        if (status != GD_OK) {
            free(fields);
            return status;
        }
        if (mem_stats.data.slots == 0U) {
            free(fields);
            return GD_ERR_BAD_STATE;
        }
        if ((uint32_t)n_slots > mem_stats.data.slots) {
            n_slots = (int)mem_stats.data.slots;
        }
    }
    dl = (gd_dataloader *)calloc(1U, sizeof(*dl));
    if (dl == NULL) {
        free(fields);
        return GD_ERR_OUT_OF_MEMORY;
    }
    dl->ctx = ctx;
    dl->dataset = dataset;
    dl->sampler = sampler;
    dl->cfg = *cfg;
    dl->cfg.num_workers = n_workers;
    dl->cfg.prefetch_factor = prefetch_factor;
    dl->n_workers = n_workers;
    dl->n_slots = n_slots;
    dl->n_fields = n_fields;
    dl->collate = _gd_collate_gdds;
    dl->collate_data = NULL;
    dl->steps_per_epoch = gd_dataset_num_samples(dataset) / (uint64_t)cfg->batch_size;
    dl->samples_per_epoch = dl->steps_per_epoch * (uint64_t)cfg->batch_size;
    dl->worker_status = GD_OK;
    status = gd_copy_fields(dl, fields);
    free(fields);
    if (status != GD_OK) {
        gd_dataloader_destroy(dl);
        return status;
    }
    dl->schema_hash = gd_dl_schema_hash(dl);
    dl->slots = (gd_batch *)calloc((size_t)n_slots, sizeof(*dl->slots));
    dl->workers = (pthread_t *)calloc((size_t)n_workers, sizeof(*dl->workers));
    if (dl->slots == NULL || dl->workers == NULL) {
        gd_dataloader_destroy(dl);
        return GD_ERR_OUT_OF_MEMORY;
    }
    status = gd_dataloader_init_sync(dl);
    if (status != GD_OK) {
        gd_dataloader_destroy(dl);
        return status;
    }
    for (i = 0; i < dl->n_slots; ++i) {
        status = gd_batch_init_slot(dl, i, &dl->slots[i]);
        if (status != GD_OK) {
            gd_dataloader_destroy(dl);
            return status;
        }
    }
    status = gd_start_workers(dl);
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
            gd_batch_destroy_slot(&dl->slots[i]);
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
        return GD_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&dl->mutex);
    status = gd_request_locked(dl);
    pthread_mutex_unlock(&dl->mutex);
    return status;
}

gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch **batch_out)
{
    uint64_t t0;
    int waited = 0;
    gd_batch *slot;
    gd_status status;
    if (dl == NULL || batch_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *batch_out = NULL;
    t0 = gd_context_now_ns();
    pthread_mutex_lock(&dl->mutex);
    status = gd_dl_worker_status_locked(dl);
    if (status != GD_OK) {
        pthread_mutex_unlock(&dl->mutex);
        return status;
    }
    slot = gd_find_ready_seq_locked(dl, dl->deliver_seq);
    while (slot == NULL) {
        if (gd_queued_locked(dl) == 0U) {
            status = gd_request_locked(dl);
            if (status != GD_OK) {
                pthread_mutex_unlock(&dl->mutex);
                return status;
            }
        } else if (dl->requested == 0U && dl->filling_count == 0) {
            pthread_mutex_unlock(&dl->mutex);
            return GD_ERR_INTERNAL;
        }
        waited = 1;
        pthread_cond_wait(&dl->ready_cv, &dl->mutex);
        status = gd_dl_worker_status_locked(dl);
        if (status != GD_OK) {
            pthread_mutex_unlock(&dl->mutex);
            return status;
        }
        slot = gd_find_ready_seq_locked(dl, dl->deliver_seq);
    }
    if (waited != 0) {
        dl->metrics.wait_for_batch_ns += gd_context_now_ns() - t0;
    }
    slot->state = GD_BATCH_IN_USE;
    dl->deliver_seq += 1U;
    dl->metrics.batches_returned += 1U;
    *batch_out = slot;
    pthread_mutex_unlock(&dl->mutex);
    return GD_OK;
}

gd_status gd_dataloader_release(gd_dataloader *dl, gd_batch *batch)
{
    int i;
    if (dl == NULL || batch == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&dl->mutex);
    for (i = 0; i < dl->n_slots; ++i) {
        if (&dl->slots[i] == batch) {
            if (batch->state == GD_BATCH_IN_STEP) {
                pthread_mutex_unlock(&dl->mutex);
                return GD_ERR_BUSY;
            }
            if (batch->state != GD_BATCH_IN_USE && batch->state != GD_BATCH_RETIRED) {
                pthread_mutex_unlock(&dl->mutex);
                return GD_ERR_BAD_STATE;
            }
            gd_batch_clear_data_binding(batch);
            batch->state = GD_BATCH_FREE;
            batch->seq = 0U;
            pthread_cond_signal(&dl->work_cv);
            pthread_mutex_unlock(&dl->mutex);
            return GD_OK;
        }
    }
    pthread_mutex_unlock(&dl->mutex);
    return GD_ERR_INVALID_ARGUMENT;
}

int gd_dataloader_slot_count(const gd_dataloader *dl)
{
    return dl != NULL ? dl->n_slots : 0;
}

uint64_t gd_dataloader_steps_per_epoch(const gd_dataloader *dl)
{
    return dl != NULL ? dl->steps_per_epoch : 0U;
}

uint64_t gd_dataloader_samples_per_epoch(const gd_dataloader *dl)
{
    return dl != NULL ? dl->samples_per_epoch : 0U;
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
