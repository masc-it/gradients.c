#include "dataloader_internal.h"

#include "../core/internal.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

int gd_batch_index(const gd_batch *batch)
{
    return batch != NULL ? batch->index : -1;
}

gd_batch_state gd_batch_get_state(const gd_batch *batch)
{
    return batch != NULL ? batch->state : GD_BATCH_FREE;
}

int gd_batch_field_count(const gd_batch *batch)
{
    return batch != NULL ? batch->n_fields : 0;
}

int gd_batch_field_index(const gd_batch *batch, const char *name)
{
    int i;
    if (batch == NULL || name == NULL) {
        return -1;
    }
    for (i = 0; i < batch->n_fields; ++i) {
        if (batch->fields[i].name != NULL && strcmp(batch->fields[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

const char *gd_batch_field_name(const gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return NULL;
    }
    return batch->fields[field_index].name;
}

gd_dtype gd_batch_field_dtype(const gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return GD_DTYPE_INVALID;
    }
    return batch->fields[field_index].dtype;
}

int gd_batch_field_rank(const gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return -1;
    }
    return batch->fields[field_index].rank;
}

int64_t gd_batch_field_dim(const gd_batch *batch, int field_index, int dim_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields ||
        dim_index < 0 || dim_index >= batch->fields[field_index].rank) {
        return -1;
    }
    return batch->fields[field_index].sizes[dim_index];
}

size_t gd_batch_field_nbytes(const gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return 0U;
    }
    return batch->fields[field_index].nbytes;
}

void *gd_batch_host_data(gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return NULL;
    }
    return batch->fields[field_index].host_data;
}

gd_tensor *gd_batch_tensor_at(gd_batch *batch, int field_index)
{
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return NULL;
    }
    return batch->fields[field_index].tensor;
}

gd_tensor *gd_batch_tensor(gd_batch *batch, const char *name)
{
    int idx = gd_batch_field_index(batch, name);
    return gd_batch_tensor_at(batch, idx);
}

const uint64_t *gd_batch_sample_ids(const gd_batch *batch)
{
    return batch != NULL ? batch->sample_ids : NULL;
}

static gd_status gd_collate_require_i32_bt(gd_batch *batch,
                                           int field_index,
                                           const char *name,
                                           int batch_size,
                                           int block_len)
{
    if (field_index < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing LM batch field");
    }
    if (gd_batch_field_dtype(batch, field_index) != GD_DTYPE_I32 ||
        gd_batch_field_rank(batch, field_index) != 2 ||
        gd_batch_field_dim(batch, field_index, 0) != (int64_t)batch_size ||
        gd_batch_field_dim(batch, field_index, 1) != (int64_t)block_len) {
        (void)name;
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid LM batch field schema");
    }
    return GD_OK;
}

gd_status gd_collate_gdtok_lm(gd_dataset *dataset,
                              const uint64_t *sample_ids,
                              int batch_size,
                              gd_batch *batch,
                              void *user_data)
{
    uint64_t block_len_u64 = 0U;
    int block_len;
    int tokens_idx;
    int targets_idx;
    int positions_idx;
    int32_t *tokens;
    int32_t *targets;
    int32_t *positions;
    int b;
    gd_status status;

    if (dataset == NULL || sample_ids == NULL || batch == NULL || batch_size <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid LM collate args");
    }
    status = gd_dataset_get_u64(dataset, "block_len", &block_len_u64);
    if (status != GD_OK) {
        return status;
    }
    if (block_len_u64 == 0U || block_len_u64 > (uint64_t)INT_MAX) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid LM block_len");
    }
    block_len = (int)block_len_u64;
    tokens_idx = gd_batch_field_index(batch, "tokens");
    targets_idx = gd_batch_field_index(batch, "targets");
    positions_idx = gd_batch_field_index(batch, "positions");
    status = gd_collate_require_i32_bt(batch, tokens_idx, "tokens", batch_size, block_len);
    if (status == GD_OK) {
        status = gd_collate_require_i32_bt(batch, targets_idx, "targets", batch_size,
                                           block_len);
    }
    if (status == GD_OK) {
        status = gd_collate_require_i32_bt(batch, positions_idx, "positions", batch_size,
                                           block_len);
    }
    if (status != GD_OK) {
        return status;
    }
    tokens = (int32_t *)gd_batch_host_data(batch, tokens_idx);
    targets = (int32_t *)gd_batch_host_data(batch, targets_idx);
    positions = (int32_t *)gd_batch_host_data(batch, positions_idx);
    if (tokens == NULL || targets == NULL || positions == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "LM batch host field is null");
    }
    for (b = 0; b < batch_size; ++b) {
        int32_t *tok_row = &tokens[(size_t)b * (size_t)block_len];
        int32_t *tgt_row = &targets[(size_t)b * (size_t)block_len];
        int32_t *pos_row = &positions[(size_t)b * (size_t)block_len];
        int j;
        status = gd_gdtok_dataset_read_lm_sample(dataset, sample_ids[b], tok_row, tgt_row);
        if (status != GD_OK) {
            return status;
        }
        for (j = 0; j < block_len; ++j) {
            pos_row[j] = (int32_t)j;
        }
    }
    return GD_OK;
}
