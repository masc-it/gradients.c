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
typedef struct gd_sample gd_sample;

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
    GD_GDDS_GENERATED_CU_SEQLENS_FROM_LENGTHS = 5,
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

typedef struct gd_gdds_shard_info {
    const char *path; /* Borrowed from dataset; valid until dataset is destroyed. */
    uint64_t sample_base;
    uint64_t samples;
    uint64_t bytes;
} gd_gdds_shard_info;

/* Sample-level field schema exposed by a transformed dataset. Shapes do not
   include the dataloader batch dimension; use -1 for the leading ragged
   dimension with pad_longest/packed_sequence. */
typedef struct gd_dataset_field_spec {
    const char *name;
    gd_dtype dtype;
    int rank;
    int64_t shape[GD_MAX_DIMS];
    gd_gdds_collate_mode collate;
    gd_gdds_generated_kind generated;
    int ragged_dim;
    int source_field;
    uint64_t pad_value_bits;
} gd_dataset_field_spec;

/* PyTorch-style dataset transform: called as part of sample fetching before
   dataloader collation. Callbacks run on dataloader worker threads, so user_data
   must remain valid for the dataset lifetime and be thread-safe for the chosen
   worker count. */
typedef gd_status (*gd_dataset_transform_fn)(const gd_sample *src,
                                             gd_sample *dst,
                                             void *user_data);

typedef struct gd_dataset_transform_config {
    gd_dataset_transform_fn transform; /* NULL => identity/raw dataset. */
    void *user_data;
    const gd_dataset_field_spec *output_fields; /* Required when transform != NULL. */
    int n_output_fields;
} gd_dataset_transform_config;

gd_status gd_dataset_open_gdds(const char **paths,
                               int n_paths,
                               gd_dataset **out);
gd_status gd_dataset_open_gdds_with_transform(const char **paths,
                                              int n_paths,
                                              const gd_dataset_transform_config *transform,
                                              gd_dataset **out);
gd_status gd_dataset_open_gdds_file(const char *path, gd_dataset **out);
gd_status gd_dataset_open_gdds_file_with_transform(const char *path,
                                                   const gd_dataset_transform_config *transform,
                                                   gd_dataset **out);
gd_status gd_dataset_open_gdds_split(const char *dir,
                                     const char *split,
                                     gd_dataset **out);
gd_status gd_dataset_open_gdds_split_with_transform(const char *dir,
                                                    const char *split,
                                                    const gd_dataset_transform_config *transform,
                                                    gd_dataset **out);

int gd_gdds_dataset_field_count(const gd_dataset *dataset);
int gd_gdds_dataset_field_index(const gd_dataset *dataset, const char *name);
int gd_gdds_dataset_shard_count(const gd_dataset *dataset);
gd_status gd_gdds_dataset_shard_info(const gd_dataset *dataset,
                                      int shard_index,
                                      gd_gdds_shard_info *out);
gd_status gd_gdds_dataset_field_info(const gd_dataset *dataset,
                                     int field_index,
                                     gd_gdds_field_info *out);
gd_status gd_gdds_dataset_read_field(const gd_dataset *dataset,
                                     uint64_t sample_index,
                                     int field_index,
                                     gd_gdds_sample_field *out);

int gd_sample_field_count(const gd_sample *sample);
int gd_sample_field_index(const gd_sample *sample, const char *name);
const char *gd_sample_field_name(const gd_sample *sample, int field_index);
gd_dtype gd_sample_field_dtype(const gd_sample *sample, int field_index);
int gd_sample_field_rank(const gd_sample *sample, int field_index);
int64_t gd_sample_field_dim(const gd_sample *sample, int field_index, int dim_index);
size_t gd_sample_field_nbytes(const gd_sample *sample, int field_index);
const void *gd_sample_field_data(const gd_sample *sample, int field_index);
void *gd_sample_mutable_field_data(gd_sample *sample, int field_index);
gd_status gd_sample_resize_field(gd_sample *sample,
                                 int field_index,
                                 gd_dtype dtype,
                                 int rank,
                                 const int64_t *shape);
gd_status gd_sample_copy_field(gd_sample *dst,
                               int dst_field_index,
                               const gd_sample *src,
                               int src_field_index);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATASET_H */
