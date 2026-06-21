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
typedef struct gd_sampler gd_sampler;

/* RandomSampler equivalent: yields a deterministic no-replacement random order
   over each dataloader epoch/pass. The implementation is O(1) memory; it does
   not materialize a full permutation. */
gd_status gd_sampler_create_random(const gd_dataset *dataset,
                                   uint64_t seed,
                                   gd_sampler **out);
/* Randomize samples independently within each GDDS shard while keeping shard
   order fixed. This improves locality versus global random sampling for
   mmap-backed datasets on seek-sensitive storage. */
gd_status gd_sampler_create_gdds_shard_random(const gd_dataset *dataset,
                                              uint64_t seed,
                                              gd_sampler **out);
void gd_sampler_destroy(gd_sampler *sampler);

typedef enum gd_batch_state {
    GD_BATCH_FREE = 0,
    GD_BATCH_FILLING = 1,
    GD_BATCH_READY = 2,
    GD_BATCH_IN_USE = 3,
    GD_BATCH_IN_STEP = 4,
    GD_BATCH_RETIRED = 5,
} gd_batch_state;

typedef struct gd_dataloader_config {
    int batch_size;
    int num_workers;     /* 0 => one background worker. */
    int prefetch_factor; /* 0 => two slots per worker. */
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

/* Defaults: sequential sampling when sampler is NULL and implementation
   defaults for worker/prefetch counts. */
gd_dataloader_config gd_dataloader_config_default(int batch_size);

/* `dataset` must be GDDS. Batch fields and collation are inferred from GDDS
   metadata written during dataset prep. `sampler` may be NULL for deterministic
   sequential sampling. Non-NULL samplers are borrowed by the dataloader and must
   outlive it. The fixed-batch dataloader drops the last incomplete batch. */
gd_status gd_dataloader_create(gd_context *ctx,
                               gd_dataset *dataset,
                               gd_sampler *sampler,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out);
void gd_dataloader_destroy(gd_dataloader *dl);

gd_status gd_dataloader_prefetch(gd_dataloader *dl);
gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch **batch_out);
gd_status gd_dataloader_release(gd_dataloader *dl, gd_batch *batch);
int gd_dataloader_slot_count(const gd_dataloader *dl);
uint64_t gd_dataloader_steps_per_epoch(const gd_dataloader *dl);
uint64_t gd_dataloader_samples_per_epoch(const gd_dataloader *dl);

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

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATALOADER_H */
