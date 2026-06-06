#ifndef GRADIENTS_DATASET_H
#define GRADIENTS_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_GDDS_MAGIC "GDDSv1"
#define GD_GDDS_RECORD_MAGIC "GDDR"
#define GD_GDDS_VERSION 1U
#define GD_GDDS_HEADER_SIZE 128U
#define GD_GDDS_FIELD_NAME_MAX 64U
#define GD_GDDS_FIELD_DESC_SIZE 160U
#define GD_GDDS_INDEX_ENTRY_SIZE 16U
#define GD_GDDS_RECORD_HEADER_SIZE 20U
#define GD_GDDS_RECORD_FIELD_DESC_SIZE 88U
#define GD_GDDS_MAX_FIELDS 256U

typedef struct gd_dataset gd_dataset;

void gd_dataset_destroy(gd_dataset *dataset);

const char *gd_dataset_name(const gd_dataset *dataset);
uint64_t gd_dataset_num_samples(const gd_dataset *dataset);

typedef enum gd_gdds_collate_mode {
    GD_GDDS_COLLATE_STACK = 0,
    GD_GDDS_COLLATE_PAD_LONGEST = 1,
    GD_GDDS_COLLATE_PACKED_SEQUENCE = 2,
    GD_GDDS_COLLATE_GENERATED = 3,
} gd_gdds_collate_mode;

typedef enum gd_gdds_generated_kind {
    GD_GDDS_GENERATED_NONE = 0,
    GD_GDDS_GENERATED_LENGTHS = 1,
    GD_GDDS_GENERATED_MASK = 2,
    GD_GDDS_GENERATED_CU_SEQLENS = 3,
    GD_GDDS_GENERATED_POSITIONS = 4,
} gd_gdds_generated_kind;

typedef struct gd_gdds_field_info {
    char name[GD_GDDS_FIELD_NAME_MAX];
    gd_dtype dtype;
    int rank;
    int64_t shape[GD_MAX_DIMS]; /* -1 in the on-disk schema means variable. */
    gd_gdds_collate_mode collate;
    gd_gdds_generated_kind generated;
    int ragged_dim;    /* -1 for fixed-shape/generated fields; currently 0 for varlen. */
    int source_field;  /* source schema field for generated outputs, else -1. */
    uint64_t pad_value_bits; /* little-endian scalar bytes for pad_longest. */
} gd_gdds_field_info;

typedef struct gd_gdds_sample_field {
    char name[GD_GDDS_FIELD_NAME_MAX];
    gd_dtype dtype;
    int rank;
    int64_t shape[GD_MAX_DIMS];
    const void *data; /* mmap-backed; valid until dataset is destroyed. */
    size_t nbytes;
} gd_gdds_sample_field;

gd_status gd_dataset_open_gdds(const char **paths,
                               int n_paths,
                               gd_dataset **out);
gd_status gd_dataset_open_gdds_file(const char *path, gd_dataset **out);
gd_status gd_dataset_open_gdds_split(const char *dir,
                                     const char *split,
                                     gd_dataset **out);

int gd_gdds_dataset_field_count(const gd_dataset *dataset);
int gd_gdds_dataset_field_index(const gd_dataset *dataset, const char *name);
gd_status gd_gdds_dataset_field_info(const gd_dataset *dataset,
                                     int field_index,
                                     gd_gdds_field_info *out);
gd_status gd_gdds_dataset_read_field(const gd_dataset *dataset,
                                     uint64_t sample_index,
                                     int field_index,
                                     gd_gdds_sample_field *out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATASET_H */
