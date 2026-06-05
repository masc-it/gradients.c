#ifndef GRADIENTS_DATASET_H
#define GRADIENTS_DATASET_H

#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_GDTOK_MAGIC "GDTOKv1"
#define GD_GDTOK_VERSION 1U
#define GD_GDTOK_HEADER_SIZE 64U

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

typedef uint64_t (*gd_dataset_num_samples_fn)(const void *impl);
typedef uint64_t (*gd_dataset_fingerprint_fn)(const void *impl);
typedef gd_status (*gd_dataset_get_u64_fn)(const void *impl,
                                           const char *key,
                                           uint64_t *out);
typedef void (*gd_dataset_destroy_fn)(void *impl);

typedef struct gd_dataset_ops {
    const char *name;
    gd_dataset_num_samples_fn num_samples;
    gd_dataset_fingerprint_fn fingerprint;
    gd_dataset_get_u64_fn get_u64;
    gd_dataset_destroy_fn destroy;
} gd_dataset_ops;

gd_status gd_dataset_create(const gd_dataset_ops *ops,
                            void *impl,
                            gd_dataset **out);
void gd_dataset_destroy(gd_dataset *dataset);

const char *gd_dataset_name(const gd_dataset *dataset);
void *gd_dataset_data(gd_dataset *dataset);
const void *gd_dataset_const_data(const gd_dataset *dataset);
uint64_t gd_dataset_num_samples(const gd_dataset *dataset);
uint64_t gd_dataset_fingerprint(const gd_dataset *dataset);
gd_status gd_dataset_get_u64(const gd_dataset *dataset,
                             const char *key,
                             uint64_t *out);

typedef struct gd_gdds_field_info {
    char name[GD_GDDS_FIELD_NAME_MAX];
    gd_dtype dtype;
    int rank;
    int64_t shape[GD_MAX_DIMS]; /* -1 in the on-disk schema means variable. */
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

typedef enum gd_gdtok_dtype {
    GD_GDTOK_DTYPE_U16 = 1,
    GD_GDTOK_DTYPE_U32 = 2,
} gd_gdtok_dtype;

typedef struct gd_gdtok_header {
    uint32_t version;
    uint32_t header_size;
    gd_gdtok_dtype dtype;
    uint32_t vocab_size;
    uint32_t block_len;
    uint64_t n_tokens;
    uint64_t n_samples;
    uint64_t tokenizer_hash;
    uint64_t payload_offset;
} gd_gdtok_header;

gd_status gd_gdtok_read_header(const char *path, gd_gdtok_header *out);

gd_status gd_dataset_open_gdtok(const char **paths,
                                int n_paths,
                                gd_dataset **out);

gd_status gd_gdtok_dataset_read_lm_sample(const gd_dataset *dataset,
                                          uint64_t sample_index,
                                          int32_t *tokens,
                                          int32_t *targets);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DATASET_H */
