#include "gradients/dataset_vlm.h"


#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static gd_status gd_vlm_u64_to_i64_dim(uint64_t value, int64_t *out)
{
    if (out == NULL || value == 0U || value > (uint64_t)INT64_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = (int64_t)value;
    return GD_OK;
}

static gd_status gd_vlm_resolve_lengths(const gd_dataset *dataset,
                                        const gd_vlm_collate_config *cfg,
                                        int *prefix_out,
                                        int *max_text_out,
                                        int *seq_out)
{
    uint64_t prefix_u64 = 0U;
    uint64_t max_text_u64 = 0U;
    int prefix;
    int max_text;
    int seq;
    gd_status status;

    if (dataset == NULL || prefix_out == NULL || max_text_out == NULL || seq_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_dataset_get_u64(dataset, "image_prefix_tokens", &prefix_u64);
    if (status == GD_OK) {
        status = gd_dataset_get_u64(dataset, "max_text_len", &max_text_u64);
    }
    if (status != GD_OK) {
        return status;
    }
    if (prefix_u64 == 0U || prefix_u64 > (uint64_t)INT_MAX ||
        max_text_u64 == 0U || max_text_u64 > (uint64_t)INT_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    prefix = (int)prefix_u64;
    max_text = cfg != NULL && cfg->max_text_len > 0 ? cfg->max_text_len : (int)max_text_u64;
    if (max_text <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cfg != NULL && cfg->sequence_len > 0) {
        seq = cfg->sequence_len;
        if (seq < prefix + max_text) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    } else {
        if (prefix > INT_MAX - max_text) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        seq = prefix + max_text;
    }
    *prefix_out = prefix;
    *max_text_out = max_text;
    *seq_out = seq;
    return GD_OK;
}

gd_status gd_vlm_init_batch_fields(const gd_dataset *dataset,
                                   const gd_vlm_collate_config *cfg,
                                   int batch_size,
                                   gd_batch_field_desc *fields,
                                   int field_cap,
                                   int *n_fields_out)
{
    uint64_t num_patches = 0U;
    uint64_t patch_dim = 0U;
    int64_t patches_dim0;
    int64_t patches_dim1;
    int64_t patches_dim2;
    int prefix = 0;
    int max_text = 0;
    int seq = 0;
    gd_status status;

    if (dataset == NULL || fields == NULL || n_fields_out == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (field_cap < GD_VLM_BATCH_FIELD_COUNT) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_dataset_get_u64(dataset, "num_patches", &num_patches);
    if (status == GD_OK) {
        status = gd_dataset_get_u64(dataset, "patch_dim", &patch_dim);
    }
    if (status == GD_OK) {
        status = gd_vlm_resolve_lengths(dataset, cfg, &prefix, &max_text, &seq);
    }
    if (status != GD_OK) {
        return status;
    }
    (void)prefix;
    (void)max_text;
    status = gd_vlm_u64_to_i64_dim((uint64_t)batch_size, &patches_dim0);
    if (status == GD_OK) {
        status = gd_vlm_u64_to_i64_dim(num_patches, &patches_dim1);
    }
    if (status == GD_OK) {
        status = gd_vlm_u64_to_i64_dim(patch_dim, &patches_dim2);
    }
    if (status != GD_OK) {
        return status;
    }
    memset(fields, 0, (size_t)field_cap * sizeof(fields[0]));
    fields[0].name = "patches";
    fields[0].dtype = GD_DTYPE_F16;
    fields[0].rank = 3;
    fields[0].sizes[0] = patches_dim0;
    fields[0].sizes[1] = patches_dim1;
    fields[0].sizes[2] = patches_dim2;
    fields[1].name = "tokens";
    fields[1].dtype = GD_DTYPE_I32;
    fields[1].rank = 2;
    fields[1].sizes[0] = batch_size;
    fields[1].sizes[1] = seq;
    fields[2].name = "targets";
    fields[2].dtype = GD_DTYPE_I32;
    fields[2].rank = 2;
    fields[2].sizes[0] = batch_size;
    fields[2].sizes[1] = seq;
    fields[3].name = "positions";
    fields[3].dtype = GD_DTYPE_I32;
    fields[3].rank = 2;
    fields[3].sizes[0] = batch_size;
    fields[3].sizes[1] = seq;
    fields[4].name = "loss_mask";
    fields[4].dtype = GD_DTYPE_F32;
    fields[4].rank = 2;
    fields[4].sizes[0] = batch_size;
    fields[4].sizes[1] = seq;
    fields[5].name = "prefix_len";
    fields[5].dtype = GD_DTYPE_I32;
    fields[5].rank = 1;
    fields[5].sizes[0] = batch_size;
    fields[6].name = "text_len";
    fields[6].dtype = GD_DTYPE_I32;
    fields[6].rank = 1;
    fields[6].sizes[0] = batch_size;
    fields[7].name = "label_id";
    fields[7].dtype = GD_DTYPE_I32;
    fields[7].rank = 1;
    fields[7].sizes[0] = batch_size;
    *n_fields_out = GD_VLM_BATCH_FIELD_COUNT;
    return GD_OK;
}

static gd_status gd_vlm_require_field(gd_batch *batch,
                                      const char *name,
                                      gd_dtype dtype,
                                      int rank,
                                      int64_t dim0,
                                      int64_t dim1,
                                      int64_t dim2,
                                      int *idx_out)
{
    int idx;
    if (batch == NULL || name == NULL || idx_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    idx = gd_batch_field_index(batch, name);
    if (idx < 0 || gd_batch_field_dtype(batch, idx) != dtype ||
        gd_batch_field_rank(batch, idx) != rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (rank > 0 && gd_batch_field_dim(batch, idx, 0) != dim0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (rank > 1 && gd_batch_field_dim(batch, idx, 1) != dim1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (rank > 2 && gd_batch_field_dim(batch, idx, 2) != dim2) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *idx_out = idx;
    return GD_OK;
}

static gd_status gd_vlm_validate_collate_config(const gd_vlm_collate_config *cfg)
{
    if (cfg == NULL) {
        return GD_OK;
    }
    if (cfg->target_mode != GD_VLM_TEXT_TARGET_NEXT &&
        cfg->target_mode != GD_VLM_TEXT_TARGET_SHIFT_RIGHT &&
        cfg->target_mode != GD_VLM_TEXT_TARGET_SELF) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_vlm_fill_positions(int32_t *row, int seq)
{
    int j;
    for (j = 0; j < seq; ++j) {
        row[j] = (int32_t)j;
    }
}

static void gd_vlm_apply_next_targets(int32_t *tokens,
                                      int32_t *targets,
                                      float *mask,
                                      int prefix,
                                      int text_len,
                                      int target_pad_id)
{
    int j;
    (void)target_pad_id;
    for (j = 0; j + 1 < text_len; ++j) {
        targets[prefix + j] = tokens[prefix + j + 1];
        mask[prefix + j] = 1.0F;
    }
}

static void gd_vlm_apply_shift_right_targets(int32_t *tokens,
                                             int32_t *targets,
                                             float *mask,
                                             int prefix,
                                             int text_len,
                                             int pad_token_id)
{
    int j;
    for (j = text_len - 1; j >= 0; --j) {
        int32_t original = tokens[prefix + j];
        targets[prefix + j] = original;
        mask[prefix + j] = 1.0F;
        tokens[prefix + j] = j == 0 ? (int32_t)pad_token_id : tokens[prefix + j - 1];
    }
}

static void gd_vlm_apply_self_targets(int32_t *tokens,
                                      int32_t *targets,
                                      float *mask,
                                      int prefix,
                                      int text_len)
{
    int j;
    for (j = 0; j < text_len; ++j) {
        targets[prefix + j] = tokens[prefix + j];
        mask[prefix + j] = 1.0F;
    }
}

gd_status gd_collate_gdvlm(gd_dataset *dataset,
                           const uint64_t *sample_ids,
                           int batch_size,
                           gd_batch *batch,
                           void *user_data)
{
    const gd_vlm_collate_config *cfg = (const gd_vlm_collate_config *)user_data;
    gd_vlm_text_target_mode target_mode = GD_VLM_TEXT_TARGET_NEXT;
    int pad_token_id = 0;
    int target_pad_id = -100;
    int truncate_text = 0;
    int prefix = 0;
    int max_text = 0;
    int seq = 0;
    uint64_t num_patches = 0U;
    uint64_t patch_dim = 0U;
    uint64_t patch_nbytes_u64 = 0U;
    size_t patch_nbytes;
    int patches_idx;
    int tokens_idx;
    int targets_idx;
    int positions_idx;
    int mask_idx;
    int prefix_idx;
    int text_idx;
    int label_idx;
    uint16_t *patches;
    int32_t *tokens;
    int32_t *targets;
    int32_t *positions;
    float *loss_mask;
    int32_t *prefix_len;
    int32_t *text_len;
    int32_t *label_id;
    int b;
    gd_status status;

    if (dataset == NULL || sample_ids == NULL || batch == NULL || batch_size <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_vlm_validate_collate_config(cfg);
    if (status == GD_OK) {
        status = gd_vlm_resolve_lengths(dataset, cfg, &prefix, &max_text, &seq);
    }
    if (status == GD_OK) {
        status = gd_dataset_get_u64(dataset, "num_patches", &num_patches);
    }
    if (status == GD_OK) {
        status = gd_dataset_get_u64(dataset, "patch_dim", &patch_dim);
    }
    if (status == GD_OK) {
        status = gd_dataset_get_u64(dataset, "patch_nbytes", &patch_nbytes_u64);
    }
    if (status != GD_OK) {
        return status;
    }
    if (patch_nbytes_u64 > (uint64_t)SIZE_MAX) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    patch_nbytes = (size_t)patch_nbytes_u64;
    if (cfg != NULL) {
        pad_token_id = cfg->pad_token_id;
        target_pad_id = cfg->target_pad_id;
        truncate_text = cfg->truncate_text;
        target_mode = cfg->target_mode;
    }
    status = gd_vlm_require_field(batch, "patches", GD_DTYPE_F16, 3, batch_size,
                                  (int64_t)num_patches, (int64_t)patch_dim, &patches_idx);
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "tokens", GD_DTYPE_I32, 2, batch_size,
                                      seq, 0, &tokens_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "targets", GD_DTYPE_I32, 2, batch_size,
                                      seq, 0, &targets_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "positions", GD_DTYPE_I32, 2, batch_size,
                                      seq, 0, &positions_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "loss_mask", GD_DTYPE_F32, 2, batch_size,
                                      seq, 0, &mask_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "prefix_len", GD_DTYPE_I32, 1, batch_size,
                                      0, 0, &prefix_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "text_len", GD_DTYPE_I32, 1, batch_size,
                                      0, 0, &text_idx);
    }
    if (status == GD_OK) {
        status = gd_vlm_require_field(batch, "label_id", GD_DTYPE_I32, 1, batch_size,
                                      0, 0, &label_idx);
    }
    if (status != GD_OK) {
        return status;
    }
    patches = (uint16_t *)gd_batch_host_data(batch, patches_idx);
    tokens = (int32_t *)gd_batch_host_data(batch, tokens_idx);
    targets = (int32_t *)gd_batch_host_data(batch, targets_idx);
    positions = (int32_t *)gd_batch_host_data(batch, positions_idx);
    loss_mask = (float *)gd_batch_host_data(batch, mask_idx);
    prefix_len = (int32_t *)gd_batch_host_data(batch, prefix_idx);
    text_len = (int32_t *)gd_batch_host_data(batch, text_idx);
    label_id = (int32_t *)gd_batch_host_data(batch, label_idx);
    if (patches == NULL || tokens == NULL || targets == NULL || positions == NULL ||
        loss_mask == NULL || prefix_len == NULL || text_len == NULL || label_id == NULL) {
        return GD_ERR_INTERNAL;
    }
    for (b = 0; b < batch_size; ++b) {
        gd_gdvlm_sample_info info;
        int32_t *tok_row = &tokens[(size_t)b * (size_t)seq];
        int32_t *tgt_row = &targets[(size_t)b * (size_t)seq];
        int32_t *pos_row = &positions[(size_t)b * (size_t)seq];
        float *mask_row = &loss_mask[(size_t)b * (size_t)seq];
        void *patch_row = (void *)((uint8_t *)patches + (size_t)b * patch_nbytes);
        int effective_text;
        int j;

        for (j = 0; j < seq; ++j) {
            tok_row[j] = (int32_t)pad_token_id;
            tgt_row[j] = (int32_t)target_pad_id;
            mask_row[j] = 0.0F;
        }
        gd_vlm_fill_positions(pos_row, seq);
        status = gd_gdvlm_dataset_read_sample(dataset, sample_ids[b], &info,
                                              tok_row + prefix, max_text,
                                              patch_row, patch_nbytes);
        if (status != GD_OK) {
            return status;
        }
        if (info.token_len > (uint32_t)max_text && truncate_text == 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        effective_text = (int)info.tokens_copied;
        prefix_len[b] = (int32_t)prefix;
        text_len[b] = (int32_t)effective_text;
        label_id[b] = info.label_id;
        if (target_mode == GD_VLM_TEXT_TARGET_SHIFT_RIGHT) {
            gd_vlm_apply_shift_right_targets(tok_row, tgt_row, mask_row, prefix,
                                             effective_text, pad_token_id);
        } else if (target_mode == GD_VLM_TEXT_TARGET_SELF) {
            gd_vlm_apply_self_targets(tok_row, tgt_row, mask_row, prefix, effective_text);
        } else {
            gd_vlm_apply_next_targets(tok_row, tgt_row, mask_row, prefix,
                                      effective_text, target_pad_id);
        }
    }
    return GD_OK;
}
