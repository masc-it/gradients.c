#ifndef GRADIENTS_DATASET_H
#define GRADIENTS_DATASET_H

#include "gradients/status.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_GDTOK_MAGIC "GDTOKv1"
#define GD_GDTOK_VERSION 1U
#define GD_GDTOK_HEADER_SIZE 64U

typedef struct gd_dataset gd_dataset;

typedef gd_status (*gd_dataset_get_u64_fn)(const void *impl,
                                           const char *key,
                                           uint64_t *out);
typedef uint64_t (*gd_dataset_num_samples_fn)(const void *impl);
typedef uint64_t (*gd_dataset_fingerprint_fn)(const void *impl);
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

typedef enum gd_gdtok_dtype {
    GD_GDTOK_DTYPE_U16 = 1,
    GD_GDTOK_DTYPE_U32 = 2
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

typedef struct gd_dataset_build_config {
    const char *tokenizer_path;
    const char **input_paths;
    int n_input_paths;
    const char *output_dir;
    int block_len;
    double train_ratio;
    double val_ratio;
    uint64_t seed;
    int wrap_plain_text;
    int no_shuffle_split;
    const char *im_start;
    const char *im_end;
} gd_dataset_build_config;

typedef struct gd_dataset_split_result {
    char *shard_path;
    char *metadata_path;
    int *record_indices;
    int n_record_indices;
    uint64_t n_tokens_total;
    uint64_t n_tokens_written;
    uint64_t n_samples;
    uint64_t dropped_tail_tokens;
} gd_dataset_split_result;

typedef struct gd_dataset_build_result {
    int n_records_total;
    gd_dataset_split_result train;
    gd_dataset_split_result val;
} gd_dataset_build_result;

gd_status gd_dataset_build(const gd_dataset_build_config *cfg,
                           gd_dataset_build_result *result_out);

void gd_dataset_build_result_clear(gd_dataset_build_result *result);

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
