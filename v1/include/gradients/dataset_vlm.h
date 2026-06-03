#ifndef GRADIENTS_DATASET_VLM_H
#define GRADIENTS_DATASET_VLM_H

#include "gradients/dataloader.h"
#include "gradients/dataset.h"
#include "gradients/status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_GDVLM_MAGIC "GDVLMv1"
#define GD_GDVLM_VERSION 1U
#define GD_GDVLM_HEADER_SIZE 128U
#define GD_GDVLM_IDX_MAGIC "GDVLMIDX"
#define GD_GDVLM_IDX_VERSION 1U
#define GD_GDVLM_IDX_HEADER_SIZE 24U
#define GD_GDVLM_IDX_ENTRY_SIZE 32U
#define GD_VLM_BATCH_FIELD_COUNT 8

typedef enum gd_gdvlm_patch_dtype {
    GD_GDVLM_PATCH_DTYPE_F16 = 1
} gd_gdvlm_patch_dtype;

typedef enum gd_gdvlm_token_dtype {
    GD_GDVLM_TOKEN_DTYPE_U16 = 1,
    GD_GDVLM_TOKEN_DTYPE_U32 = 2
} gd_gdvlm_token_dtype;

typedef struct gd_gdvlm_header {
    uint32_t version;
    uint32_t header_size;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint32_t patch_size;
    uint32_t num_patches;
    uint32_t patch_dim;
    gd_gdvlm_patch_dtype patch_dtype;
    gd_gdvlm_token_dtype token_dtype;
    uint32_t vocab_size;
    uint32_t shard_idx;
    uint32_t num_shards;
    uint32_t payload_offset;
    uint64_t n_samples;
    uint64_t tokenizer_hash;
} gd_gdvlm_header;

typedef struct gd_gdvlm_sample_info {
    uint32_t shard_idx;
    uint32_t sample_idx;
    uint64_t body_offset;
    uint32_t record_nbytes;
    uint32_t token_len;
    uint32_t tokens_copied;
    int32_t label_id;
    uint32_t raw_pos;
} gd_gdvlm_sample_info;

typedef enum gd_vlm_text_target_mode {
    /* tokens[text k] -> targets[text k + 1]; last text token masked out. */
    GD_VLM_TEXT_TARGET_NEXT = 0,
    /* tokens[text 0] = pad, tokens[text k] = ids[k - 1]; targets = ids. */
    GD_VLM_TEXT_TARGET_SHIFT_RIGHT = 1,
    /* targets = tokens for non-autoregressive/probing use cases. */
    GD_VLM_TEXT_TARGET_SELF = 2
} gd_vlm_text_target_mode;

typedef struct gd_vlm_collate_config {
    int max_text_len;       /* <= 0 uses dataset max_text_len */
    int sequence_len;       /* <= 0 uses num_patches + max_text_len */
    int pad_token_id;
    int target_pad_id;
    int truncate_text;      /* 0 => fail if token_len exceeds capacity */
    gd_vlm_text_target_mode target_mode;
} gd_vlm_collate_config;

gd_status gd_gdvlm_read_header(const char *path, gd_gdvlm_header *out);

gd_status gd_dataset_open_gdvlm(const char **shard_paths,
                                int n_shards,
                                const char *index_path,
                                gd_dataset **out);

gd_status gd_dataset_open_gdvlm_split(const char *dir,
                                      const char *split_tag,
                                      gd_dataset **out);

gd_status gd_gdvlm_dataset_sample_info(const gd_dataset *dataset,
                                       uint64_t sample_index,
                                       gd_gdvlm_sample_info *out);

/* Reads one sample. Token copy is bounded by token_capacity; info_out->token_len
 * remains original length and info_out->tokens_copied reports copied count. */
gd_status gd_gdvlm_dataset_read_sample(const gd_dataset *dataset,
                                       uint64_t sample_index,
                                       gd_gdvlm_sample_info *info_out,
                                       int32_t *tokens,
                                       int token_capacity,
                                       void *patches,
                                       size_t patch_nbytes);

gd_status gd_vlm_init_batch_fields(const gd_dataset *dataset,
                                   const gd_vlm_collate_config *cfg,
                                   int batch_size,
                                   gd_batch_field_desc *fields,
                                   int field_cap,
                                   int *n_fields_out);

gd_status gd_collate_gdvlm(gd_dataset *dataset,
                           const uint64_t *sample_ids,
                           int batch_size,
                           gd_batch *batch,
                           void *user_data);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATASET_VLM_H */
