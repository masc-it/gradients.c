#ifndef GD_DATASET_INTERNAL_H
#define GD_DATASET_INTERNAL_H

#include <gradients/dataset.h>

#include <stdint.h>

typedef uint64_t (*gd_dataset_num_samples_fn)(const void *impl);
typedef void (*gd_dataset_destroy_fn)(void *impl);

typedef struct gd_dataset_ops {
    const char *name;
    gd_dataset_num_samples_fn num_samples;
    gd_dataset_destroy_fn destroy;
} gd_dataset_ops;

typedef struct gd_sample_field {
    char name[GD_GDDS_FIELD_NAME_MAX];
    gd_dtype dtype;
    int rank;
    int64_t shape[GD_MAX_DIMS];
    const void *data;
    size_t nbytes;
    size_t capacity_nbytes;
    void *owned_data;
    int writable;
} gd_sample_field;

struct gd_sample {
    int n_fields;
    gd_sample_field *fields;
};

gd_status gd_dataset_create(const gd_dataset_ops *ops,
                            void *impl,
                            gd_dataset **out);
const void *gd_dataset_const_data(const gd_dataset *dataset);

gd_status gd_sample_init_from_gdds_fields(gd_sample *sample,
                                          const gd_gdds_field_info *fields,
                                          int n_fields,
                                          int allocate_fixed);
void gd_sample_reset_from_gdds_fields(gd_sample *sample,
                                      const gd_gdds_field_info *fields,
                                      int n_fields);
void gd_sample_deinit(gd_sample *sample);

#endif /* GD_DATASET_INTERNAL_H */
