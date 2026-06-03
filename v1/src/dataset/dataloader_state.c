#include "dataloader_internal.h"

#include "../core/internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char gd_dl_state_magic[8] = {'G', 'D', 'L', 'D', 'S', 'T', '3', '\0'};

typedef struct gd_dl_state_ready_batch {
    uint64_t seq;
    uint64_t *samples;
} gd_dl_state_ready_batch;

typedef struct gd_dl_state_file {
    uint32_t batch_size;
    uint32_t sampler;
    uint32_t slot_count;
    uint32_t ready_count;
    uint32_t n_fields;
    uint64_t seed;
    uint64_t dataset_fingerprint;
    uint64_t dataset_samples;
    uint64_t schema_hash;
    uint64_t cursor;
    uint64_t rng_state;
    uint64_t next_seq;
    uint64_t deliver_seq;
    uint64_t requested;
    gd_dataloader_metrics metrics;
    gd_dl_state_ready_batch *ready;
} gd_dl_state_file;

static int gd_dl_write_bytes(FILE *f, const void *p, size_t n)
{
    return fwrite(p, 1U, n, f) == n ? 0 : 1;
}

static int gd_dl_read_bytes(FILE *f, void *p, size_t n)
{
    return fread(p, 1U, n, f) == n ? 0 : 1;
}

static int gd_dl_write_u32(FILE *f, uint32_t v)
{
    unsigned char b[4];
    int i;
    for (i = 0; i < 4; ++i) {
        b[i] = (unsigned char)(v & 0xffU);
        v >>= 8U;
    }
    return gd_dl_write_bytes(f, b, sizeof(b));
}

static int gd_dl_write_u64(FILE *f, uint64_t v)
{
    unsigned char b[8];
    int i;
    for (i = 0; i < 8; ++i) {
        b[i] = (unsigned char)(v & UINT64_C(0xff));
        v >>= 8U;
    }
    return gd_dl_write_bytes(f, b, sizeof(b));
}

static int gd_dl_read_u32(FILE *f, uint32_t *out)
{
    unsigned char b[4];
    uint32_t v = 0U;
    int i;
    if (gd_dl_read_bytes(f, b, sizeof(b)) != 0) {
        return 1;
    }
    for (i = 3; i >= 0; --i) {
        v = (v << 8U) | (uint32_t)b[i];
    }
    *out = v;
    return 0;
}

static int gd_dl_read_u64(FILE *f, uint64_t *out)
{
    unsigned char b[8];
    uint64_t v = 0U;
    int i;
    if (gd_dl_read_bytes(f, b, sizeof(b)) != 0) {
        return 1;
    }
    for (i = 7; i >= 0; --i) {
        v = (v << 8U) | (uint64_t)b[i];
    }
    *out = v;
    return 0;
}

static int gd_dl_write_metrics(FILE *f, const gd_dataloader_metrics *m)
{
    return gd_dl_write_u64(f, m->batches_prepared) != 0 ||
           gd_dl_write_u64(f, m->batches_returned) != 0 ||
           gd_dl_write_u64(f, m->samples_prepared) != 0 ||
           gd_dl_write_u64(f, m->host_fill_ns) != 0 ||
           gd_dl_write_u64(f, m->host_to_device_copy_ns) != 0 ||
           gd_dl_write_u64(f, m->wait_for_batch_ns) != 0 ||
           gd_dl_write_u64(f, m->prefetch_requests) != 0 ||
           gd_dl_write_u64(f, m->max_ready_depth) != 0;
}

static int gd_dl_read_metrics(FILE *f, gd_dataloader_metrics *m)
{
    return gd_dl_read_u64(f, &m->batches_prepared) != 0 ||
           gd_dl_read_u64(f, &m->batches_returned) != 0 ||
           gd_dl_read_u64(f, &m->samples_prepared) != 0 ||
           gd_dl_read_u64(f, &m->host_fill_ns) != 0 ||
           gd_dl_read_u64(f, &m->host_to_device_copy_ns) != 0 ||
           gd_dl_read_u64(f, &m->wait_for_batch_ns) != 0 ||
           gd_dl_read_u64(f, &m->prefetch_requests) != 0 ||
           gd_dl_read_u64(f, &m->max_ready_depth) != 0;
}

static gd_loader_slot *gd_dl_find_ready_seq(gd_dataloader *dl, uint64_t seq)
{
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_READY && dl->slots[i].seq == seq) {
            return &dl->slots[i];
        }
    }
    return NULL;
}

static uint32_t gd_dl_ready_count(const gd_dataloader *dl)
{
    uint32_t n = 0U;
    int i;
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_READY) {
            n += 1U;
        }
    }
    return n;
}

static gd_status gd_dl_write_state_locked(gd_dataloader *dl, const char *path)
{
    FILE *f;
    uint32_t ready_count;
    uint64_t seq;
    uint64_t end_seq;
    int b;

    ready_count = gd_dl_ready_count(dl);
    f = fopen(path, "wb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataloader state output");
    }
    if (gd_dl_write_bytes(f, gd_dl_state_magic, sizeof(gd_dl_state_magic)) != 0 ||
        gd_dl_write_u32(f, 3U) != 0 ||
        gd_dl_write_u32(f, (uint32_t)dl->cfg.batch_size) != 0 ||
        gd_dl_write_u32(f, (uint32_t)dl->cfg.sampler) != 0 ||
        gd_dl_write_u32(f, (uint32_t)dl->n_slots) != 0 ||
        gd_dl_write_u32(f, ready_count) != 0 ||
        gd_dl_write_u32(f, (uint32_t)dl->n_fields) != 0 ||
        gd_dl_write_u64(f, dl->cfg.seed) != 0 ||
        gd_dl_write_u64(f, gd_dataset_fingerprint(dl->dataset)) != 0 ||
        gd_dl_write_u64(f, gd_dataset_num_samples(dl->dataset)) != 0 ||
        gd_dl_write_u64(f, dl->schema_hash) != 0 ||
        gd_dl_write_u64(f, dl->cursor) != 0 ||
        gd_dl_write_u64(f, dl->rng_state) != 0 ||
        gd_dl_write_u64(f, dl->next_seq) != 0 ||
        gd_dl_write_u64(f, dl->deliver_seq) != 0 ||
        gd_dl_write_u64(f, dl->requested) != 0 ||
        gd_dl_write_metrics(f, &dl->metrics) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to write dataloader state header");
    }
    end_seq = dl->next_seq;
    for (seq = dl->deliver_seq; seq < end_seq; ++seq) {
        gd_loader_slot *slot = gd_dl_find_ready_seq(dl, seq);
        if (slot == NULL) {
            continue;
        }
        if (gd_dl_write_u64(f, slot->seq) != 0) {
            (void)fclose(f);
            return _gd_error(GD_ERR_IO, "failed to write dataloader state batch");
        }
        for (b = 0; b < dl->cfg.batch_size; ++b) {
            if (gd_dl_write_u64(f, slot->pub.sample_ids[b]) != 0) {
                (void)fclose(f);
                return _gd_error(GD_ERR_IO, "failed to write dataloader state samples");
            }
        }
    }
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close dataloader state output");
    }
    return GD_OK;
}

gd_status gd_dataloader_state_save(gd_dataloader *dl, const char *path)
{
    gd_status status;
    int i;

    if (dl == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state save args");
    }
    status = _gd_dataloader_lock_quiesced(dl);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < dl->n_slots; ++i) {
        if (dl->slots[i].pub.state == GD_BATCH_IN_USE) {
            _gd_dataloader_unlock_resume(dl);
            return _gd_error(GD_ERR_INVALID_STATE,
                             "cannot save dataloader state with in-use batches");
        }
    }
    status = gd_dl_write_state_locked(dl, path);
    _gd_dataloader_unlock_resume(dl);
    return status;
}

static void gd_dl_state_file_clear(gd_dl_state_file *s)
{
    uint32_t i;
    if (s == NULL) {
        return;
    }
    if (s->ready != NULL) {
        for (i = 0U; i < s->ready_count; ++i) {
            free(s->ready[i].samples);
        }
    }
    free(s->ready);
    memset(s, 0, sizeof(*s));
}

static gd_status gd_dl_read_state_file(const char *path, gd_dl_state_file *s)
{
    FILE *f;
    unsigned char magic[8];
    uint32_t version = 0U;
    uint32_t i;

    memset(s, 0, sizeof(*s));
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataloader state");
    }
    if (gd_dl_read_bytes(f, magic, sizeof(magic)) != 0 ||
        gd_dl_read_u32(f, &version) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read dataloader state magic");
    }
    if (memcmp(magic, gd_dl_state_magic, sizeof(magic)) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state magic");
    }
    if (version != 3U) {
        (void)fclose(f);
        return _gd_error(GD_ERR_UNSUPPORTED, "unsupported dataloader state version");
    }
    if (gd_dl_read_u32(f, &s->batch_size) != 0 ||
        gd_dl_read_u32(f, &s->sampler) != 0 ||
        gd_dl_read_u32(f, &s->slot_count) != 0 ||
        gd_dl_read_u32(f, &s->ready_count) != 0 ||
        gd_dl_read_u32(f, &s->n_fields) != 0 ||
        gd_dl_read_u64(f, &s->seed) != 0 ||
        gd_dl_read_u64(f, &s->dataset_fingerprint) != 0 ||
        gd_dl_read_u64(f, &s->dataset_samples) != 0 ||
        gd_dl_read_u64(f, &s->schema_hash) != 0 ||
        gd_dl_read_u64(f, &s->cursor) != 0 ||
        gd_dl_read_u64(f, &s->rng_state) != 0 ||
        gd_dl_read_u64(f, &s->next_seq) != 0 ||
        gd_dl_read_u64(f, &s->deliver_seq) != 0 ||
        gd_dl_read_u64(f, &s->requested) != 0 ||
        gd_dl_read_metrics(f, &s->metrics) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read dataloader state header");
    }
    if (s->batch_size == 0U || s->ready_count > GD_DL_MAX_SLOTS ||
        s->slot_count > GD_DL_MAX_SLOTS) {
        (void)fclose(f);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state counts");
    }
    if (s->ready_count > 0U) {
        s->ready = (gd_dl_state_ready_batch *)calloc((size_t)s->ready_count,
                                                     sizeof(gd_dl_state_ready_batch));
        if (s->ready == NULL) {
            (void)fclose(f);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader state allocation failed");
        }
    }
    for (i = 0U; i < s->ready_count; ++i) {
        uint32_t b;
        if (gd_dl_read_u64(f, &s->ready[i].seq) != 0) {
            (void)fclose(f);
            gd_dl_state_file_clear(s);
            return _gd_error(GD_ERR_IO, "failed to read dataloader state batch");
        }
        s->ready[i].samples = (uint64_t *)calloc((size_t)s->batch_size, sizeof(uint64_t));
        if (s->ready[i].samples == NULL) {
            (void)fclose(f);
            gd_dl_state_file_clear(s);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataloader state sample allocation failed");
        }
        for (b = 0U; b < s->batch_size; ++b) {
            if (gd_dl_read_u64(f, &s->ready[i].samples[b]) != 0) {
                (void)fclose(f);
                gd_dl_state_file_clear(s);
                return _gd_error(GD_ERR_IO, "failed to read dataloader state samples");
            }
        }
    }
    if (fclose(f) != 0) {
        gd_dl_state_file_clear(s);
        return _gd_error(GD_ERR_IO, "failed to close dataloader state");
    }
    return GD_OK;
}

static gd_status gd_dl_validate_state(const gd_dataloader *dl, const gd_dl_state_file *s)
{
    if (s->batch_size != (uint32_t)dl->cfg.batch_size ||
        s->sampler != (uint32_t)dl->cfg.sampler ||
        s->n_fields != (uint32_t)dl->n_fields ||
        s->slot_count > (uint32_t)dl->n_slots ||
        s->dataset_samples != gd_dataset_num_samples(dl->dataset) ||
        s->schema_hash != dl->schema_hash) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader state/config mismatch");
    }
    if (s->dataset_fingerprint != 0U &&
        s->dataset_fingerprint != gd_dataset_fingerprint(dl->dataset)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataloader state dataset mismatch");
    }
    if (s->ready_count > (uint32_t)dl->n_slots ||
        s->requested > (uint64_t)dl->n_slots - (uint64_t)s->ready_count) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state queue depth");
    }
    return GD_OK;
}

gd_status gd_dataloader_state_load(gd_dataloader *dl, const char *path)
{
    gd_dl_state_file s;
    gd_status status;
    uint32_t i;

    if (dl == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataloader state load args");
    }
    status = gd_dl_read_state_file(path, &s);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_dataloader_lock_quiesced(dl);
    if (status != GD_OK) {
        gd_dl_state_file_clear(&s);
        return status;
    }
    if (_gd_dataloader_has_live_slot_locked(dl) != 0) {
        _gd_dataloader_unlock_resume(dl);
        gd_dl_state_file_clear(&s);
        return _gd_error(GD_ERR_INVALID_STATE, "cannot load dataloader state with live batches");
    }
    status = gd_dl_validate_state(dl, &s);
    if (status == GD_OK) {
        dl->cursor = s.cursor;
        dl->rng_state = s.rng_state;
        dl->next_seq = s.next_seq;
        dl->deliver_seq = s.deliver_seq;
        dl->requested = s.requested;
        dl->metrics = s.metrics;
        for (i = 0U; i < s.ready_count && status == GD_OK; ++i) {
            gd_dataloader_fill_stats stats;
            gd_loader_slot *slot = &dl->slots[i];
            int b;
            slot->pub.state = GD_BATCH_FILLING;
            slot->seq = s.ready[i].seq;
            for (b = 0; b < dl->cfg.batch_size; ++b) {
                slot->pub.sample_ids[b] = s.ready[i].samples[b];
            }
            status = _gd_dataloader_fill_slot_from_samples(dl, slot, &stats);
            if (status == GD_OK) {
                slot->pub.state = GD_BATCH_READY;
            } else {
                slot->pub.state = GD_BATCH_FREE;
            }
        }
    }
    _gd_dataloader_unlock_resume(dl);
    gd_dl_state_file_clear(&s);
    return status;
}
