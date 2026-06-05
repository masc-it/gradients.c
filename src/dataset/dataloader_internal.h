#ifndef GD_DATALOADER_INTERNAL_H
#define GD_DATALOADER_INTERNAL_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/dataloader.h>

#define GD_DL_DEFAULT_WORKERS 1
#define GD_DL_DEFAULT_PREFETCH_FACTOR 2
#define GD_DL_MAX_WORKERS 64
#define GD_DL_MAX_PREFETCH_FACTOR 16
#define GD_DL_MAX_SLOTS 1024

typedef struct gd_batch_field {
    char *name;
    gd_dtype dtype;
    int rank;
    int64_t sizes[GD_BATCH_MAX_RANK];
    size_t nbytes;
    void *host_data;
    gd_tensor tensor;
    bool has_tensor;
} gd_batch_field;

struct gd_batch {
    int index;
    gd_batch_state state;
    int batch_size;
    int n_fields;
    gd_batch_field *fields;
    uint64_t *sample_ids;
    gd_context *ctx;
    int32_t data_slot;
    uint64_t data_generation;
    uint64_t seq;
    uint64_t last_fence;
    bool is_empty;
};

typedef struct gd_dataloader_fill_stats {
    uint64_t host_fill_ns;
    uint64_t host_to_data_copy_ns;
    uint64_t samples_prepared;
} gd_dataloader_fill_stats;

struct gd_dataloader {
    gd_context *ctx;
    gd_dataset *dataset;
    gd_dataloader_config cfg;
    gd_batch_field_desc *field_descs;
    int n_fields;
    uint64_t schema_hash;
    gd_collate_fn collate;
    void *collate_data;
    gd_batch *slots;
    int n_slots;
    int n_workers;
    pthread_t *workers;
    int workers_started;
    pthread_mutex_t mutex;
    pthread_cond_t work_cv;
    pthread_cond_t ready_cv;
    pthread_cond_t state_cv;
    int sync_ready;
    int stop;
    int filling_count;
    uint64_t requested;
    uint64_t rng_state;
    uint64_t cursor;
    uint64_t next_seq;
    uint64_t deliver_seq;
    gd_status worker_status;
    char worker_error[256];
    gd_dataloader_metrics metrics;
};

gd_status gd_dl_fill_slot(gd_dataloader *dl,
                          gd_batch *slot,
                          gd_dataloader_fill_stats *stats);
void gd_dl_add_fill_metrics_locked(gd_dataloader *dl,
                                   const gd_dataloader_fill_stats *stats);
gd_status gd_dl_worker_status_locked(const gd_dataloader *dl);
uint64_t gd_dl_schema_hash(const gd_dataloader *dl);

#endif /* GD_DATALOADER_INTERNAL_H */
