#include "dataset_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct gd_dataset {
    gd_dataset_ops ops;
    void *impl;
};

static char *gd_ds_strdup(const char *s)
{
    size_t n;
    char *out;
    if (s == NULL) {
        return NULL;
    }
    n = strlen(s);
    out = (char *)malloc(n + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, n + 1U);
    return out;
}

gd_status gd_dataset_create(const gd_dataset_ops *ops,
                            void *impl,
                            gd_dataset **out)
{
    gd_dataset *dataset;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (ops == NULL || ops->name == NULL || ops->num_samples == NULL || impl == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dataset = (gd_dataset *)calloc(1U, sizeof(*dataset));
    if (dataset == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    dataset->ops = *ops;
    dataset->ops.name = gd_ds_strdup(ops->name);
    if (dataset->ops.name == NULL) {
        free(dataset);
        return GD_ERR_OUT_OF_MEMORY;
    }
    dataset->impl = impl;
    *out = dataset;
    return GD_OK;
}

void gd_dataset_destroy(gd_dataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    if (dataset->ops.destroy != NULL) {
        dataset->ops.destroy(dataset->impl);
    }
    free((char *)dataset->ops.name);
    free(dataset);
}

const char *gd_dataset_name(const gd_dataset *dataset)
{
    if (dataset == NULL || dataset->ops.name == NULL) {
        return "";
    }
    return dataset->ops.name;
}

const void *gd_dataset_const_data(const gd_dataset *dataset)
{
    return dataset != NULL ? dataset->impl : NULL;
}

uint64_t gd_dataset_num_samples(const gd_dataset *dataset)
{
    if (dataset == NULL || dataset->ops.num_samples == NULL) {
        return 0U;
    }
    return dataset->ops.num_samples(dataset->impl);
}

static gd_status gd_sample_shape_nbytes(gd_dtype dtype,
                                        int rank,
                                        const int64_t *shape,
                                        size_t *out)
{
    size_t item_size;
    size_t numel = 1U;
    int d;
    if (shape == NULL || out == NULL || rank < 0 || rank > (int)GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    item_size = gd_dtype_size(dtype);
    if (item_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (d = 0; d < rank; ++d) {
        if (shape[d] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if ((uint64_t)shape[d] > (uint64_t)(SIZE_MAX / numel)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        numel *= (size_t)shape[d];
    }
    if (numel > SIZE_MAX / item_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out = numel * item_size;
    return GD_OK;
}

static int gd_sample_fixed_shape(const gd_gdds_field_info *field)
{
    int d;
    if (field == NULL || field->collate == GD_GDDS_COLLATE_GENERATED) {
        return 0;
    }
    for (d = 0; d < field->rank; ++d) {
        if (field->shape[d] <= 0) {
            return 0;
        }
    }
    return 1;
}

gd_status gd_sample_init_from_gdds_fields(gd_sample *sample,
                                          const gd_gdds_field_info *fields,
                                          int n_fields,
                                          int allocate_fixed)
{
    int i;
    if (sample == NULL || fields == NULL || n_fields <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(sample, 0, sizeof(*sample));
    sample->fields = (gd_sample_field *)calloc((size_t)n_fields, sizeof(*sample->fields));
    if (sample->fields == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    sample->n_fields = n_fields;
    for (i = 0; i < n_fields; ++i) {
        gd_sample_field *dst = &sample->fields[i];
        int d;
        strncpy(dst->name, fields[i].name, sizeof(dst->name) - 1U);
        dst->name[sizeof(dst->name) - 1U] = '\0';
        dst->dtype = fields[i].dtype;
        dst->rank = fields[i].rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            dst->shape[d] = fields[i].shape[d];
        }
        if (allocate_fixed != 0 && gd_sample_fixed_shape(&fields[i]) != 0) {
            size_t nbytes = 0U;
            gd_status status = gd_sample_shape_nbytes(dst->dtype,
                                                      dst->rank,
                                                      dst->shape,
                                                      &nbytes);
            if (status != GD_OK) {
                gd_sample_deinit(sample);
                return status;
            }
            dst->owned_data = malloc(nbytes > 0U ? nbytes : 1U);
            if (dst->owned_data == NULL) {
                gd_sample_deinit(sample);
                return GD_ERR_OUT_OF_MEMORY;
            }
            dst->data = dst->owned_data;
            dst->nbytes = nbytes;
            dst->capacity_nbytes = nbytes;
            dst->writable = 1;
        } else if (allocate_fixed != 0 && fields[i].collate != GD_GDDS_COLLATE_GENERATED) {
            dst->writable = 1;
        }
    }
    return GD_OK;
}

void gd_sample_reset_from_gdds_fields(gd_sample *sample,
                                      const gd_gdds_field_info *fields,
                                      int n_fields)
{
    int i;
    if (sample == NULL || fields == NULL || sample->n_fields != n_fields) {
        return;
    }
    for (i = 0; i < n_fields; ++i) {
        gd_sample_field *dst = &sample->fields[i];
        int d;
        dst->dtype = fields[i].dtype;
        dst->rank = fields[i].rank;
        for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
            dst->shape[d] = fields[i].shape[d];
        }
        if (fields[i].collate == GD_GDDS_COLLATE_GENERATED) {
            dst->data = NULL;
            dst->nbytes = 0U;
            dst->writable = dst->owned_data != NULL ? 1 : 0;
        } else if (dst->owned_data != NULL && gd_sample_fixed_shape(&fields[i]) != 0) {
            size_t nbytes = 0U;
            if (gd_sample_shape_nbytes(dst->dtype, dst->rank, dst->shape, &nbytes) == GD_OK &&
                nbytes <= dst->capacity_nbytes) {
                dst->data = dst->owned_data;
                dst->nbytes = nbytes;
                dst->writable = 1;
            }
        } else {
            dst->data = NULL;
            dst->nbytes = 0U;
            if (fields[i].collate != GD_GDDS_COLLATE_GENERATED && dst->writable != 0) {
                dst->writable = 1;
            } else {
                dst->writable = dst->owned_data != NULL ? 1 : 0;
            }
        }
    }
}

void gd_sample_deinit(gd_sample *sample)
{
    int i;
    if (sample == NULL) {
        return;
    }
    if (sample->fields != NULL) {
        for (i = 0; i < sample->n_fields; ++i) {
            free(sample->fields[i].owned_data);
        }
    }
    free(sample->fields);
    memset(sample, 0, sizeof(*sample));
}

int gd_sample_field_count(const gd_sample *sample)
{
    return sample != NULL ? sample->n_fields : 0;
}

int gd_sample_field_index(const gd_sample *sample, const char *name)
{
    int i;
    if (sample == NULL || name == NULL) {
        return -1;
    }
    for (i = 0; i < sample->n_fields; ++i) {
        if (strcmp(sample->fields[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

const char *gd_sample_field_name(const gd_sample *sample, int field_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return NULL;
    }
    return sample->fields[field_index].name;
}

gd_dtype gd_sample_field_dtype(const gd_sample *sample, int field_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return GD_DTYPE_INVALID;
    }
    return sample->fields[field_index].dtype;
}

int gd_sample_field_rank(const gd_sample *sample, int field_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return -1;
    }
    return sample->fields[field_index].rank;
}

int64_t gd_sample_field_dim(const gd_sample *sample, int field_index, int dim_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields ||
        dim_index < 0 || dim_index >= sample->fields[field_index].rank) {
        return -1;
    }
    return sample->fields[field_index].shape[dim_index];
}

size_t gd_sample_field_nbytes(const gd_sample *sample, int field_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return 0U;
    }
    return sample->fields[field_index].nbytes;
}

const void *gd_sample_field_data(const gd_sample *sample, int field_index)
{
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return NULL;
    }
    return sample->fields[field_index].data;
}

void *gd_sample_mutable_field_data(gd_sample *sample, int field_index)
{
    gd_sample_field *field;
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return NULL;
    }
    field = &sample->fields[field_index];
    return field->writable != 0 ? (void *)field->data : NULL;
}

gd_status gd_sample_resize_field(gd_sample *sample,
                                 int field_index,
                                 gd_dtype dtype,
                                 int rank,
                                 const int64_t *shape)
{
    gd_sample_field *field;
    size_t nbytes = 0U;
    gd_status status;
    int d;
    if (sample == NULL || field_index < 0 || field_index >= sample->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_sample_shape_nbytes(dtype, rank, shape, &nbytes);
    if (status != GD_OK) {
        return status;
    }
    field = &sample->fields[field_index];
    if (field->writable == 0) {
        return GD_ERR_BAD_STATE;
    }
    if (nbytes > field->capacity_nbytes) {
        void *new_data;
        if (field->owned_data == NULL) {
            if (field->data != NULL || field->capacity_nbytes != 0U) {
                return GD_ERR_OUT_OF_MEMORY;
            }
            new_data = malloc(nbytes);
        } else {
            new_data = realloc(field->owned_data, nbytes);
        }
        if (new_data == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        field->owned_data = new_data;
        field->data = new_data;
        field->capacity_nbytes = nbytes;
    } else if (field->owned_data != NULL) {
        field->data = field->owned_data;
    }
    field->dtype = dtype;
    field->rank = rank;
    for (d = 0; d < (int)GD_MAX_DIMS; ++d) {
        field->shape[d] = d < rank ? shape[d] : 0;
    }
    field->nbytes = nbytes;
    return GD_OK;
}

gd_status gd_sample_copy_field(gd_sample *dst,
                               int dst_field_index,
                               const gd_sample *src,
                               int src_field_index)
{
    const gd_sample_field *src_field;
    void *dst_data;
    gd_status status;
    if (dst == NULL || src == NULL || dst_field_index < 0 || dst_field_index >= dst->n_fields ||
        src_field_index < 0 || src_field_index >= src->n_fields) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    src_field = &src->fields[src_field_index];
    if (src_field->data == NULL && src_field->nbytes > 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    status = gd_sample_resize_field(dst,
                                    dst_field_index,
                                    src_field->dtype,
                                    src_field->rank,
                                    src_field->shape);
    if (status != GD_OK) {
        return status;
    }
    dst_data = gd_sample_mutable_field_data(dst, dst_field_index);
    if (dst_data == NULL && src_field->nbytes > 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (src_field->nbytes > 0U) {
        memcpy(dst_data, src_field->data, src_field->nbytes);
    }
    return GD_OK;
}

