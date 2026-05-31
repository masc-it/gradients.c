#ifndef GRADIENTS_DATALOADER_H
#define GRADIENTS_DATALOADER_H

#include "gradients/dataset.h"
#include "gradients/device.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_token_dataset gd_token_dataset;
typedef struct gd_dataloader gd_dataloader;

typedef enum gd_dataloader_mode {
    GD_DATALOADER_RANDOM = 0,
    GD_DATALOADER_SEQUENTIAL = 1
} gd_dataloader_mode;

typedef enum gd_batch_slot_state {
    GD_BATCH_SLOT_FREE = 0,
    GD_BATCH_SLOT_FILLING = 1,
    GD_BATCH_SLOT_READY = 2,
    GD_BATCH_SLOT_IN_USE = 3
} gd_batch_slot_state;

typedef struct gd_batch_slot {
    int index;
    gd_batch_slot_state state;
    gd_tensor *tokens;    /* int32 [B,T] */
    gd_tensor *targets;   /* int32 [B,T] */
    gd_tensor *positions; /* int32 [B,T] */
} gd_batch_slot;

typedef struct gd_dataloader_config {
    int batch_size;
    int block_len;
    uint64_t seed;
    int shuffle;
    int double_buffer;
    gd_device device;
    gd_dataloader_mode mode;
    uint64_t expected_tokenizer_hash; /* 0 disables check */
} gd_dataloader_config;

typedef struct gd_dataloader_metrics {
    uint64_t batches_prepared;
    uint64_t batches_returned;
    uint64_t samples_prepared;
    uint64_t host_fill_ns;
    uint64_t host_to_device_copy_ns;
    uint64_t wait_for_batch_ns;
} gd_dataloader_metrics;

gd_status gd_token_dataset_open(const char **paths,
                                int n_paths,
                                gd_token_dataset **out);
void gd_token_dataset_close(gd_token_dataset *ds);

uint64_t gd_token_dataset_num_samples(const gd_token_dataset *ds);
uint32_t gd_token_dataset_block_len(const gd_token_dataset *ds);
uint32_t gd_token_dataset_vocab_size(const gd_token_dataset *ds);
uint64_t gd_token_dataset_tokenizer_hash(const gd_token_dataset *ds);

gd_status gd_dataloader_create(gd_context *ctx,
                               gd_token_dataset *ds,
                               const gd_dataloader_config *cfg,
                               gd_dataloader **out);
void gd_dataloader_destroy(gd_dataloader *dl);

gd_status gd_dataloader_prefetch(gd_dataloader *dl);
gd_status gd_dataloader_next(gd_dataloader *dl, gd_batch_slot **slot_out);
gd_status gd_dataloader_release_slot(gd_dataloader *dl, gd_batch_slot *slot);

gd_status gd_dataloader_state_save(gd_dataloader *dl, const char *path);
gd_status gd_dataloader_state_load(gd_dataloader *dl, const char *path);

void gd_dataloader_metrics_get(const gd_dataloader *dl,
                               gd_dataloader_metrics *metrics_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATALOADER_H */
