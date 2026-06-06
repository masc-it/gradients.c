#ifndef GD_DATASET_INTERNAL_H
#define GD_DATASET_INTERNAL_H

#include <gradients/dataset.h>

#include <stdint.h>

typedef uint64_t (*gd_dataset_num_samples_fn)(const void *impl);
typedef uint64_t (*gd_dataset_fingerprint_fn)(const void *impl);
typedef void (*gd_dataset_destroy_fn)(void *impl);

typedef struct gd_dataset_ops {
    const char *name;
    gd_dataset_num_samples_fn num_samples;
    gd_dataset_fingerprint_fn fingerprint;
    gd_dataset_destroy_fn destroy;
} gd_dataset_ops;

gd_status gd_dataset_create(const gd_dataset_ops *ops,
                            void *impl,
                            gd_dataset **out);
const void *gd_dataset_const_data(const gd_dataset *dataset);

#endif /* GD_DATASET_INTERNAL_H */
