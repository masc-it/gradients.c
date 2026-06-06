#include "dataloader_internal.h"
#include "../core/memory_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static gd_batch gd_empty_batch = {
    -1,
    GD_BATCH_READY,
    0,
    0,
    NULL,
    NULL,
    NULL,
    -1,
    0U,
    0U,
    0U,
    true,
};

gd_batch *gd_batch_empty(void)
{
    return &gd_empty_batch;
}

int _gd_batch_is_empty(const gd_batch *batch)
{
    return batch == &gd_empty_batch || (batch != NULL && batch->is_empty);
}

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
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields ||
        !batch->fields[field_index].has_tensor) {
        return NULL;
    }
    return &batch->fields[field_index].tensor;
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

static gd_status gd_batch_shape_nbytes(gd_dtype dtype,
                                       int rank,
                                       const int64_t *sizes,
                                       size_t *out)
{
    size_t item_size;
    size_t numel = 1U;
    int d;
    if (sizes == NULL || out == NULL || rank < 0 || rank > GD_BATCH_MAX_RANK) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(dtype);
    if (item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 0; d < rank; ++d) {
        if (sizes[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((uint64_t)sizes[d] > (uint64_t)(SIZE_MAX / numel)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        numel *= (size_t)sizes[d];
    }
    if (numel > SIZE_MAX / item_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = numel * item_size;
    return GD_OK;
}

gd_status _gd_batch_resize_field(gd_batch *batch,
                                  int field_index,
                                  gd_dtype dtype,
                                  int rank,
                                  const int64_t *sizes,
                                  int zero_fill)
{
    gd_batch_field *field;
    size_t nbytes = 0U;
    gd_status status;
    int d;
    if (batch == NULL || field_index < 0 || field_index >= batch->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_batch_shape_nbytes(dtype, rank, sizes, &nbytes);
    if (status != GD_OK) {
        return status;
    }
    field = &batch->fields[field_index];
    if (field->has_tensor) {
        return GD_ERR_BAD_STATE;
    }
    if (nbytes > field->host_capacity_nbytes) {
        void *new_data = realloc(field->host_data, nbytes);
        if (new_data == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        field->host_data = new_data;
        field->host_capacity_nbytes = nbytes;
    }
    field->dtype = dtype;
    field->rank = rank;
    for (d = 0; d < GD_BATCH_MAX_RANK; ++d) {
        field->sizes[d] = d < rank ? sizes[d] : 0;
    }
    field->nbytes = nbytes;
    if (zero_fill != 0) {
        memset(field->host_data, 0, nbytes);
    }
    return GD_OK;
}

gd_status _gd_batch_begin_step(gd_batch *batch,
                               gd_context *ctx,
                               int32_t *out_slot,
                               uint64_t *out_generation)
{
    if (out_slot != NULL) {
        *out_slot = -1;
    }
    if (out_generation != NULL) {
        *out_generation = 0U;
    }
    if (batch == NULL || ctx == NULL || out_slot == NULL || out_generation == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (_gd_batch_is_empty(batch) != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (batch->ctx != ctx || batch->state != GD_BATCH_IN_USE || batch->data_slot < 0 ||
        batch->data_generation == 0U) {
        return GD_ERR_BAD_STATE;
    }
    batch->state = GD_BATCH_IN_STEP;
    *out_slot = batch->data_slot;
    *out_generation = batch->data_generation;
    return GD_OK;
}

void _gd_batch_abort_begin_step(gd_batch *batch)
{
    if (batch != NULL && batch->state == GD_BATCH_IN_STEP) {
        batch->state = GD_BATCH_IN_USE;
    }
}

void _gd_batch_end_step(gd_batch *batch, uint64_t fence)
{
    if (batch != NULL && batch->state == GD_BATCH_IN_STEP) {
        batch->last_fence = fence;
        batch->state = GD_BATCH_RETIRED;
    }
}
