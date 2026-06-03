#ifndef GRADIENTS_DATASET_INTERNAL_H
#define GRADIENTS_DATASET_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "gradients/dataset.h"

#define GD_DS_MAX_PATH 4096U

typedef struct gd_ds_i32_vec {
    int32_t *data;
    uint64_t len;
    uint64_t cap;
} gd_ds_i32_vec;

gd_status gd_ds_read_file(const char *path, char **text_out, size_t *len_out);
gd_status gd_ds_mkdir_p(const char *path);
gd_status gd_ds_join_path(const char *dir, const char *name, char **out);
gd_status gd_ds_write_header(FILE *f,
                              gd_gdtok_dtype dtype,
                              uint32_t vocab_size,
                              uint32_t block_len,
                              uint64_t n_tokens,
                              uint64_t n_samples,
                              uint64_t tokenizer_hash);
gd_status gd_ds_write_payload(FILE *f,
                               const gd_ds_i32_vec *tokens,
                               uint64_t n_tokens,
                               gd_gdtok_dtype dtype);

#endif /* GRADIENTS_DATASET_INTERNAL_H */
