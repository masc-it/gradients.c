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

