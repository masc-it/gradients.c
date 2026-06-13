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

typedef struct gd_batch_field_desc {
    const char *name;
    gd_dtype dtype;
    int rank;
    int64_t sizes[GD_BATCH_MAX_RANK];
} gd_batch_field_desc;

typedef gd_status (*gd_collate_fn)(gd_dataset *dataset,
                                   const uint64_t *sample_ids,
                                   int batch_size,
                                   gd_batch *batch,
                                   void *user_data);

typedef struct gd_batch_field {
    char *name;
    gd_dtype dtype;
    int rank;
    int64_t sizes[GD_BATCH_MAX_RANK];
    size_t nbytes;
    size_t host_capacity_nbytes;
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

typedef enum gd_sampler_kind {
    GD_SAMPLER_KIND_RANDOM = 1,
} gd_sampler_kind;

struct gd_sampler {
    gd_sampler_kind kind;
    uint64_t n_samples;
    uint64_t seed;
};

struct gd_dataloader {
    gd_context *ctx;
    gd_dataset *dataset;
    gd_sampler *sampler;
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
    uint64_t steps_per_epoch;
    uint64_t samples_per_epoch;
    uint64_t epoch;
    uint64_t samples_in_epoch;
    uint64_t next_seq;
    uint64_t deliver_seq;
    gd_status worker_status;
    char worker_error[256];
    gd_dataloader_metrics metrics;
};

gd_status _gd_batch_resize_field(gd_batch *batch,
                                  int field_index,
                                  gd_dtype dtype,
                                  int rank,
                                  const int64_t *sizes,
                                  int zero_fill);

gd_status _gd_gdds_init_batch_fields(const gd_dataset *dataset,
                                      int batch_size,
                                      gd_batch_field_desc **fields_out,
                                      int *n_fields_out);
gd_status _gd_collate_gdds(gd_dataset *dataset,
                            const uint64_t *sample_ids,
                            int batch_size,
                            gd_batch *batch,
                            void *user_data);

gd_status gd_dl_fill_slot(gd_dataloader *dl,
                          gd_batch *slot,
                          gd_dataloader_fill_stats *stats);
void gd_dl_add_fill_metrics_locked(gd_dataloader *dl,
                                   const gd_dataloader_fill_stats *stats);
gd_status gd_dl_worker_status_locked(const gd_dataloader *dl);
uint64_t gd_dl_schema_hash(const gd_dataloader *dl);

#endif /* GD_DATALOADER_INTERNAL_H */
