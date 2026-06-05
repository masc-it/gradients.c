#ifndef GRADIENTS_DATALOADER_H
#define GRADIENTS_DATALOADER_H

#include <stddef.h>
#include <stdint.h>

#include <gradients/dataset.h>
#include <gradients/memory.h>
#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_BATCH_MAX_RANK 8

typedef struct gd_dataloader gd_dataloader;
typedef struct gd_batch gd_batch;

typedef enum gd_sampler_mode {
    GD_SAMPLER_RANDOM_REPLACEMENT = 0,
    GD_SAMPLER_SEQUENTIAL = 1,
} gd_sampler_mode;

typedef enum gd_batch_state {
    GD_BATCH_FREE = 0,
    GD_BATCH_FILLING = 1,
    GD_BATCH_READY = 2,
    GD_BATCH_IN_USE = 3,
    GD_BATCH_IN_STEP = 4,
    GD_BATCH_RETIRED = 5,
} gd_batch_state;

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

typedef struct gd_dataloader_config {
    int batch_size;
    uint64_t seed;
    gd_sampler_mode sampler;
    uint64_t expected_dataset_fingerprint; /* 0 disables check. */
    int num_workers;                       /* 0 => one background worker. */
    int prefetch_factor;                   /* 0 => two slots per worker. */
} gd_dataloader_config;

typedef struct gd_dataloader_metrics {
    uint64_t batches_prepared;
    uint64_t batches_returned;
    uint64_t samples_prepared;
    uint64_t host_fill_ns;
    uint64_t host_to_data_copy_ns;
    uint64_t wait_for_batch_ns;
    uint64_t prefetch_requests;
    uint64_t max_ready_depth;
} gd_dataloader_metrics;

/* Defaults: sequential sampling, seed 0, fingerprint check disabled, and
   implementation defaults for worker/prefetch counts. */
gd_dataloader_config gd_dataloader_config_default(int batch_size);

/* Convenience builder: defaults plus sampler/seed and dataset fingerprint check
   when dataset is non-NULL. */
gd_dataloader_config gd_dataloader_config_build(const gd_dataset *dataset,
                                                int batch_size,
                                                gd_sampler_mode sampler,
                                                uint64_t seed);

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_dataset *dataset,
                               const gd_dataloader_config *cfg,
                               const gd_batch_field_desc *fields,
                               int n_fields,
                               gd_collate_fn collate,
                               void *collate_data,
                               gd_dataloader **out);
void gd_dataloader_destroy(gd_dataloader *dl);

gd_status gd_dataloader_prefetch(gd_dataloader *dl);
gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch **batch_out);
gd_status gd_dataloader_release(gd_dataloader *dl, gd_batch *batch);
int gd_dataloader_slot_count(const gd_dataloader *dl);

void gd_dataloader_metrics_get(const gd_dataloader *dl,
                               gd_dataloader_metrics *metrics_out);

gd_batch *gd_batch_empty(void);
int gd_batch_index(const gd_batch *batch);
gd_batch_state gd_batch_get_state(const gd_batch *batch);
int gd_batch_field_count(const gd_batch *batch);
int gd_batch_field_index(const gd_batch *batch, const char *name);
const char *gd_batch_field_name(const gd_batch *batch, int field_index);
gd_dtype gd_batch_field_dtype(const gd_batch *batch, int field_index);
int gd_batch_field_rank(const gd_batch *batch, int field_index);
int64_t gd_batch_field_dim(const gd_batch *batch, int field_index, int dim_index);
size_t gd_batch_field_nbytes(const gd_batch *batch, int field_index);
void *gd_batch_host_data(gd_batch *batch, int field_index);
gd_tensor *gd_batch_tensor_at(gd_batch *batch, int field_index);
gd_tensor *gd_batch_tensor(gd_batch *batch, const char *name);
const uint64_t *gd_batch_sample_ids(const gd_batch *batch);

/* Built-in collate for fixed-block GDTOK language-model batches.
   Requires fields named: tokens, targets, positions. All int32 [B,T]. */
gd_status gd_collate_gdtok_lm(gd_dataset *dataset,
                              const uint64_t *sample_ids,
                              int batch_size,
                              gd_batch *batch,
                              void *user_data);

typedef struct gd_gdds_collate_config {
    int zero_pad; /* 0 disables padding; default when config is NULL is enabled. */
    int truncate; /* 0 fails when a sample is larger than batch capacity. */
} gd_gdds_collate_config;

gd_status gd_gdds_init_batch_fields(const gd_dataset *dataset,
                                    int batch_size,
                                    gd_batch_field_desc *fields,
                                    int field_cap,
                                    int *n_fields_out);

gd_status gd_collate_gdds(gd_dataset *dataset,
                          const uint64_t *sample_ids,
                          int batch_size,
                          gd_batch *batch,
                          void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATALOADER_H */
