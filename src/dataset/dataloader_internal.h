#ifndef GD_DATALOADER_INTERNAL_H
#define GD_DATALOADER_INTERNAL_H

#include "gradients/dataloader.h"

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>

#define GD_DL_DEFAULT_WORKERS 1
#define GD_DL_DEFAULT_PREFETCH_FACTOR 2
#define GD_DL_MAX_WORKERS 64
#define GD_DL_MAX_PREFETCH_FACTOR 16
#define GD_DL_MAX_SLOTS 1024

typedef struct gd_loader_slot {
    gd_batch_slot pub;
    int32_t *host_tokens;
    int32_t *host_targets;
    int32_t *host_positions;
    uint64_t *sample_ids;
    uint64_t seq;
} gd_loader_slot;

typedef struct gd_dataloader_fill_stats {
    uint64_t host_fill_ns;
    uint64_t host_to_device_copy_ns;
    uint64_t samples_prepared;
} gd_dataloader_fill_stats;

struct gd_dataloader {
    gd_context *ctx;
    gd_token_dataset *ds;
    gd_dataloader_config cfg;
    gd_loader_slot *slots;
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
    int paused;
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

gd_status _gd_dataloader_fill_slot_from_samples(gd_dataloader *dl,
                                                gd_loader_slot *slot,
                                                gd_dataloader_fill_stats *stats);
void _gd_dataloader_add_fill_metrics_locked(gd_dataloader *dl,
                                            const gd_dataloader_fill_stats *stats);
gd_status _gd_dataloader_lock_quiesced(gd_dataloader *dl);
void _gd_dataloader_unlock_resume(gd_dataloader *dl);
int _gd_dataloader_has_live_slot_locked(const gd_dataloader *dl);
gd_status _gd_dataloader_worker_status_locked(const gd_dataloader *dl);

#endif /* GD_DATALOADER_INTERNAL_H */
