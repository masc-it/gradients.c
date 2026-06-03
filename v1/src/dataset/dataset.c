#include "gradients/dataset.h"

#include "gradients/tokenizer.h"

#include "../core/internal.h"
#include "dataset_internal.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define GD_DS_MAX_PATH 4096U

typedef struct gd_ds_record {
    char *text;
} gd_ds_record;

typedef struct gd_ds_records {
    gd_ds_record *data;
    int len;
    int cap;
} gd_ds_records;

typedef struct gd_ds_char_vec {
    char *data;
    size_t len;
    size_t cap;
} gd_ds_char_vec;

static char *gd_ds_strdup_len(const char *src, size_t len)
{
    char *dst;
    if (len >= SIZE_MAX) {
        return NULL;
    }
    dst = (char *)malloc(len + 1U);
    if (dst == NULL) {
        return NULL;
    }
    if (len > 0U) {
        memcpy(dst, src, len);
    }
    dst[len] = '\0';
    return dst;
}

static gd_status gd_ds_char_push(gd_ds_char_vec *vec, const char *data, size_t len)
{
    size_t needed;
    char *new_data;
    size_t new_cap;

    if (vec == NULL || (len > 0U && data == NULL)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset char vector append");
    }
    if (len == 0U) {
        return GD_OK;
    }
    if (len > SIZE_MAX - vec->len - 1U) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset text buffer too large");
    }
    needed = vec->len + len + 1U;
    if (needed > vec->cap) {
        new_cap = vec->cap == 0U ? 256U : vec->cap;
        while (new_cap < needed) {
            if (new_cap > SIZE_MAX / 2U) {
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset text buffer too large");
            }
            new_cap *= 2U;
        }
        new_data = (char *)realloc(vec->data, new_cap);
        if (new_data == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset text buffer allocation failed");
        }
        vec->data = new_data;
        vec->cap = new_cap;
    }
    memcpy(&vec->data[vec->len], data, len);
    vec->len += len;
    vec->data[vec->len] = '\0';
    return GD_OK;
}

static void gd_ds_char_clear(gd_ds_char_vec *vec)
{
    if (vec != NULL) {
        vec->len = 0U;
        if (vec->data != NULL) {
            vec->data[0] = '\0';
        }
    }
}

static void gd_ds_char_free(gd_ds_char_vec *vec)
{
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0U;
        vec->cap = 0U;
    }
}

static void gd_ds_records_free(gd_ds_records *records)
{
    int i;
    if (records == NULL) {
        return;
    }
    for (i = 0; i < records->len; ++i) {
        free(records->data[i].text);
    }
    free(records->data);
    records->data = NULL;
    records->len = 0;
    records->cap = 0;
}

static gd_status gd_ds_records_push_owned(gd_ds_records *records, char *text)
{
    gd_ds_record *new_data;
    int new_cap;
    if (records == NULL || text == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset record append");
    }
    if (records->len == records->cap) {
        new_cap = records->cap == 0 ? 128 : records->cap;
        while (new_cap <= records->len) {
            if (new_cap > INT_MAX / 2) {
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "too many dataset records");
            }
            new_cap *= 2;
        }
        new_data = (gd_ds_record *)realloc(records->data,
                                           (size_t)new_cap * sizeof(gd_ds_record));
        if (new_data == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset record allocation failed");
        }
        records->data = new_data;
        records->cap = new_cap;
    }
    records->data[records->len].text = text;
    records->len += 1;
    return GD_OK;
}

static gd_status gd_ds_i32_push(gd_ds_i32_vec *vec, int32_t value)
{
    int32_t *new_data;
    uint64_t new_cap;
    if (vec == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset token vector");
    }
    if (vec->len == vec->cap) {
        new_cap = vec->cap == 0U ? 4096U : vec->cap;
        while (new_cap <= vec->len) {
            if (new_cap > UINT64_MAX / 2U) {
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset token vector too large");
            }
            new_cap *= 2U;
        }
        if (new_cap > (uint64_t)(SIZE_MAX / sizeof(int32_t))) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset token allocation overflow");
        }
        new_data = (int32_t *)realloc(vec->data, (size_t)new_cap * sizeof(int32_t));
        if (new_data == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset token allocation failed");
        }
        vec->data = new_data;
        vec->cap = new_cap;
    }
    vec->data[vec->len] = value;
    vec->len += 1U;
    return GD_OK;
}

static gd_status gd_ds_i32_append(gd_ds_i32_vec *vec, const int32_t *ids, int n_ids)
{
    int i;
    if (vec == NULL || n_ids < 0 || (n_ids > 0 && ids == NULL)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset token append");
    }
    for (i = 0; i < n_ids; ++i) {
        gd_status status = gd_ds_i32_push(vec, ids[i]);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static void gd_ds_i32_free(gd_ds_i32_vec *vec)
{
    if (vec != NULL) {
        free(vec->data);
        vec->data = NULL;
        vec->len = 0U;
        vec->cap = 0U;
    }
}

static int gd_ds_starts_with(const char *s, const char *prefix)
{
    size_t n;
    if (s == NULL || prefix == NULL) {
        return 0;
    }
    n = strlen(prefix);
    return strncmp(s, prefix, n) == 0 ? 1 : 0;
}

static int gd_ds_ends_with(const char *s, const char *suffix)
{
    size_t slen;
    size_t tlen;
    if (s == NULL || suffix == NULL) {
        return 0;
    }
    slen = strlen(s);
    tlen = strlen(suffix);
    if (slen < tlen) {
        return 0;
    }
    return memcmp(&s[slen - tlen], suffix, tlen) == 0 ? 1 : 0;
}

static gd_status gd_ds_validate_record(const char *text,
                                       const char *im_start,
                                       const char *im_end)
{
    const char *inside_start;
    const char *last_end;
    size_t start_len;
    size_t end_len;
    size_t text_len;

    if (text == NULL || im_start == NULL || im_end == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset record validation");
    }
    if (text[0] == '\0') {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "empty dataset record");
    }
    if (gd_ds_starts_with(text, im_start) == 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset record missing <|im_start|>");
    }
    if (gd_ds_ends_with(text, im_end) == 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset record missing <|im_end|>");
    }
    start_len = strlen(im_start);
    end_len = strlen(im_end);
    text_len = strlen(text);
    if (text_len <= start_len + end_len) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "empty dataset record body");
    }
    inside_start = strstr(&text[start_len], im_start);
    if (inside_start != NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "malformed nested <|im_start|> span");
    }
    last_end = strstr(text, im_end);
    if (last_end == NULL || last_end != &text[text_len - end_len]) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "malformed <|im_end|> span");
    }
    return GD_OK;
}

static int gd_ds_line_is_blank(const char *line, size_t len)
{
    size_t i;
    for (i = 0U; i < len; ++i) {
        unsigned char c = (unsigned char)line[i];
        if (c != (unsigned char)' ' && c != (unsigned char)'\t' &&
            c != (unsigned char)'\r' && c != (unsigned char)'\n' &&
            c != (unsigned char)'\v' && c != (unsigned char)'\f') {
            return 0;
        }
    }
    return 1;
}

static gd_status gd_ds_add_wrapped_paragraph(gd_ds_records *records,
                                             const gd_ds_char_vec *paragraph,
                                             const char *im_start,
                                             const char *im_end)
{
    gd_ds_char_vec wrapped = {0};
    char *owned;
    gd_status status;

    if (paragraph == NULL || paragraph->len == 0U) {
        return GD_OK;
    }
    status = gd_ds_char_push(&wrapped, im_start, strlen(im_start));
    if (status == GD_OK) {
        status = gd_ds_char_push(&wrapped, paragraph->data, paragraph->len);
    }
    if (status == GD_OK) {
        status = gd_ds_char_push(&wrapped, im_end, strlen(im_end));
    }
    if (status != GD_OK) {
        gd_ds_char_free(&wrapped);
        return status;
    }
    owned = wrapped.data;
    wrapped.data = NULL;
    wrapped.len = 0U;
    wrapped.cap = 0U;
    status = gd_ds_validate_record(owned, im_start, im_end);
    if (status != GD_OK) {
        free(owned);
        return status;
    }
    status = gd_ds_records_push_owned(records, owned);
    if (status != GD_OK) {
        free(owned);
    }
    return status;
}

static gd_status gd_ds_load_wrapped_file(const char *path,
                                         gd_ds_records *records,
                                         const char *im_start,
                                         const char *im_end)
{
    char *file = NULL;
    size_t file_len = 0U;
    size_t pos = 0U;
    gd_ds_char_vec paragraph = {0};
    gd_status status;

    status = gd_ds_read_file(path, &file, &file_len);
    if (status != GD_OK) {
        return status;
    }
    while (pos < file_len) {
        size_t line_start = pos;
        size_t line_len;
        while (pos < file_len && file[pos] != '\n') {
            pos += 1U;
        }
        line_len = pos - line_start;
        if (pos < file_len && file[pos] == '\n') {
            pos += 1U;
        }
        while (line_len > 0U &&
               (file[line_start + line_len - 1U] == '\r' ||
                file[line_start + line_len - 1U] == '\n')) {
            line_len -= 1U;
        }
        if (gd_ds_line_is_blank(&file[line_start], line_len) != 0) {
            status = gd_ds_add_wrapped_paragraph(records, &paragraph, im_start, im_end);
            if (status != GD_OK) {
                gd_ds_char_free(&paragraph);
                free(file);
                return status;
            }
            gd_ds_char_clear(&paragraph);
        } else {
            if (paragraph.len > 0U) {
                status = gd_ds_char_push(&paragraph, "\n", 1U);
                if (status != GD_OK) {
                    gd_ds_char_free(&paragraph);
                    free(file);
                    return status;
                }
            }
            status = gd_ds_char_push(&paragraph, &file[line_start], line_len);
            if (status != GD_OK) {
                gd_ds_char_free(&paragraph);
                free(file);
                return status;
            }
        }
    }
    status = gd_ds_add_wrapped_paragraph(records, &paragraph, im_start, im_end);
    gd_ds_char_free(&paragraph);
    free(file);
    return status;
}

static gd_status gd_ds_load_formatted_file(const char *path,
                                           gd_ds_records *records,
                                           const char *im_start,
                                           const char *im_end)
{
    char *file = NULL;
    size_t file_len = 0U;
    size_t pos = 0U;
    gd_status status;

    status = gd_ds_read_file(path, &file, &file_len);
    if (status != GD_OK) {
        return status;
    }
    while (pos < file_len) {
        size_t line_start = pos;
        size_t line_len;
        char *owned;
        while (pos < file_len && file[pos] != '\n') {
            pos += 1U;
        }
        line_len = pos - line_start;
        if (pos < file_len && file[pos] == '\n') {
            pos += 1U;
        }
        while (line_len > 0U &&
               (file[line_start + line_len - 1U] == '\r' ||
                file[line_start + line_len - 1U] == '\n')) {
            line_len -= 1U;
        }
        if (line_len == 0U) {
            continue;
        }
        owned = gd_ds_strdup_len(&file[line_start], line_len);
        if (owned == NULL) {
            free(file);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset record allocation failed");
        }
        status = gd_ds_validate_record(owned, im_start, im_end);
        if (status != GD_OK) {
            free(owned);
            free(file);
            return status;
        }
        status = gd_ds_records_push_owned(records, owned);
        if (status != GD_OK) {
            free(owned);
            free(file);
            return status;
        }
    }
    free(file);
    return GD_OK;
}

static uint64_t gd_ds_splitmix64(uint64_t *state)
{
    uint64_t z;
    *state += UINT64_C(0x9e3779b97f4a7c15);
    z = *state;
    z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31U);
}

static void gd_ds_shuffle_indices(int *indices, int n, uint64_t seed)
{
    uint64_t state = seed;
    int i;
    if (indices == NULL || n <= 1) {
        return;
    }
    for (i = n - 1; i > 0; --i) {
        uint64_t r = gd_ds_splitmix64(&state);
        int j = (int)(r % (uint64_t)(i + 1));
        int tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

static gd_status gd_ds_make_split_indices(int n_records,
                                          double train_ratio,
                                          double val_ratio,
                                          uint64_t seed,
                                          int no_shuffle,
                                          int **train_out,
                                          int *n_train_out,
                                          int **val_out,
                                          int *n_val_out)
{
    int *indices;
    int *train_indices;
    int *val_indices;
    int n_val;
    int n_train;
    int i;
    double ratio_sum;

    if (train_out == NULL || n_train_out == NULL || val_out == NULL || n_val_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset split outputs");
    }
    *train_out = NULL;
    *n_train_out = 0;
    *val_out = NULL;
    *n_val_out = 0;
    if (n_records <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset has no records");
    }
    if (!(train_ratio > 0.0) || val_ratio < 0.0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid train/val ratio");
    }
    ratio_sum = train_ratio + val_ratio;
    if (!(ratio_sum > 0.0)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid train/val ratio sum");
    }
    indices = (int *)malloc((size_t)n_records * sizeof(int));
    if (indices == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset split allocation failed");
    }
    for (i = 0; i < n_records; ++i) {
        indices[i] = i;
    }
    if (no_shuffle == 0) {
        gd_ds_shuffle_indices(indices, n_records, seed);
    }

    n_val = (int)(((double)n_records * val_ratio / ratio_sum) + 0.5);
    if (val_ratio > 0.0 && n_records > 1 && n_val == 0) {
        n_val = 1;
    }
    if (n_val >= n_records && train_ratio > 0.0 && n_records > 1) {
        n_val = n_records - 1;
    }
    n_train = n_records - n_val;
    if (n_train <= 0) {
        free(indices);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset split produced no train records");
    }

    train_indices = (int *)malloc((size_t)n_train * sizeof(int));
    val_indices = n_val > 0 ? (int *)malloc((size_t)n_val * sizeof(int)) : NULL;
    if (train_indices == NULL || (n_val > 0 && val_indices == NULL)) {
        free(train_indices);
        free(val_indices);
        free(indices);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset split allocation failed");
    }
    for (i = 0; i < n_train; ++i) {
        train_indices[i] = indices[i];
    }
    for (i = 0; i < n_val; ++i) {
        val_indices[i] = indices[n_train + i];
    }
    free(indices);
    *train_out = train_indices;
    *n_train_out = n_train;
    *val_out = val_indices;
    *n_val_out = n_val;
    return GD_OK;
}

static void gd_ds_json_string(FILE *f, const char *s)
{
    const unsigned char *p = (const unsigned char *)s;
    fputc('"', f);
    while (*p != (unsigned char)'\0') {
        unsigned char c = *p;
        if (c == (unsigned char)'"' || c == (unsigned char)'\\') {
            fputc('\\', f);
            fputc((int)c, f);
        } else if (c == (unsigned char)'\n') {
            fputs("\\n", f);
        } else if (c == (unsigned char)'\r') {
            fputs("\\r", f);
        } else if (c == (unsigned char)'\t') {
            fputs("\\t", f);
        } else if (c < 0x20U) {
            fprintf(f, "\\u%04x", (unsigned int)c);
        } else {
            fputc((int)c, f);
        }
        ++p;
    }
    fputc('"', f);
}

static gd_status gd_ds_write_metadata(const char *path,
                                      const char *split,
                                      const gd_dataset_build_config *cfg,
                                      const gd_dataset_split_result *result,
                                      int n_records_total,
                                      gd_gdtok_dtype dtype,
                                      int vocab_size,
                                      uint64_t tokenizer_hash)
{
    FILE *f;
    int i;
    f = fopen(path, "wb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataset metadata output");
    }
    fprintf(f, "{\n");
    fprintf(f, "  \"format\": \"gdtok-v1\",\n");
    fprintf(f, "  \"split\": ");
    gd_ds_json_string(f, split);
    fprintf(f, ",\n");
    fprintf(f, "  \"block_len\": %d,\n", cfg->block_len);
    fprintf(f, "  \"dtype\": \"%s\",\n", dtype == GD_GDTOK_DTYPE_U16 ? "u16" : "u32");
    fprintf(f, "  \"vocab_size\": %d,\n", vocab_size);
    fprintf(f, "  \"tokenizer_hash\": \"%016" PRIx64 "\",\n", tokenizer_hash);
    fprintf(f, "  \"train_ratio\": %.9g,\n", cfg->train_ratio);
    fprintf(f, "  \"val_ratio\": %.9g,\n", cfg->val_ratio);
    fprintf(f, "  \"split_seed\": %" PRIu64 ",\n", cfg->seed);
    fprintf(f, "  \"wrap_plain_text\": %s,\n", cfg->wrap_plain_text != 0 ? "true" : "false");
    fprintf(f, "  \"source_files\": [");
    for (i = 0; i < cfg->n_input_paths; ++i) {
        if (i > 0) {
            fprintf(f, ", ");
        }
        gd_ds_json_string(f, cfg->input_paths[i]);
    }
    fprintf(f, "],\n");
    fprintf(f, "  \"n_records_total\": %d,\n", n_records_total);
    fprintf(f, "  \"n_records_split\": %d,\n", result->n_record_indices);
    fprintf(f, "  \"n_tokens_total\": %" PRIu64 ",\n", result->n_tokens_total);
    fprintf(f, "  \"n_tokens\": %" PRIu64 ",\n", result->n_tokens_written);
    fprintf(f, "  \"n_samples\": %" PRIu64 ",\n", result->n_samples);
    fprintf(f, "  \"dropped_tail_tokens\": %" PRIu64 "\n", result->dropped_tail_tokens);
    fprintf(f, "}\n");
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close dataset metadata output");
    }
    return GD_OK;
}

static gd_status gd_ds_build_split(const char *split,
                                   const gd_dataset_build_config *cfg,
                                   gd_tokenizer *tok,
                                   const gd_ds_records *records,
                                   const int *record_indices,
                                   int n_record_indices,
                                   gd_gdtok_dtype dtype,
                                   int vocab_size,
                                   gd_dataset_split_result *result)
{
    gd_ds_i32_vec stream = {0};
    char shard_name[64];
    char json_name[64];
    FILE *f = NULL;
    int i;
    uint64_t n_samples;
    uint64_t n_keep;
    gd_status status;

    if (split == NULL || cfg == NULL || tok == NULL || records == NULL || result == NULL ||
        n_record_indices < 0 || (n_record_indices > 0 && record_indices == NULL)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset split build arguments");
    }
    (void)snprintf(shard_name, sizeof(shard_name), "%s-00000.gdtok", split);
    (void)snprintf(json_name, sizeof(json_name), "%s-00000.json", split);
    status = gd_ds_join_path(cfg->output_dir, shard_name, &result->shard_path);
    if (status == GD_OK) {
        status = gd_ds_join_path(cfg->output_dir, json_name, &result->metadata_path);
    }
    if (status != GD_OK) {
        return status;
    }
    result->record_indices = (int *)malloc((size_t)n_record_indices * sizeof(int));
    if (n_record_indices > 0 && result->record_indices == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset split result allocation failed");
    }
    result->n_record_indices = n_record_indices;
    for (i = 0; i < n_record_indices; ++i) {
        int idx = record_indices[i];
        int32_t *ids = NULL;
        int n_ids = 0;
        if (idx < 0 || idx >= records->len) {
            gd_ds_i32_free(&stream);
            return _gd_error(GD_ERR_INTERNAL, "dataset split index out of range");
        }
        result->record_indices[i] = idx;
        status = gd_tokenizer_encode(tok, records->data[idx].text, &ids, &n_ids);
        if (status != GD_OK) {
            gd_ds_i32_free(&stream);
            return status;
        }
        status = gd_ds_i32_append(&stream, ids, n_ids);
        gd_tokenizer_free(ids);
        if (status != GD_OK) {
            gd_ds_i32_free(&stream);
            return status;
        }
    }

    result->n_tokens_total = stream.len;
    if (stream.len >= (uint64_t)cfg->block_len + 1U) {
        n_samples = (stream.len - 1U) / (uint64_t)cfg->block_len;
        n_keep = n_samples * (uint64_t)cfg->block_len + 1U;
    } else {
        n_samples = 0U;
        n_keep = 0U;
    }
    result->n_samples = n_samples;
    result->n_tokens_written = n_keep;
    result->dropped_tail_tokens = stream.len - n_keep;

    f = fopen(result->shard_path, "wb");
    if (f == NULL) {
        gd_ds_i32_free(&stream);
        return _gd_error(GD_ERR_IO, "failed to open gdtok shard output");
    }
    status = gd_ds_write_header(f,
                                dtype,
                                (uint32_t)vocab_size,
                                (uint32_t)cfg->block_len,
                                n_keep,
                                n_samples,
                                gd_tokenizer_hash(tok));
    if (status == GD_OK) {
        status = gd_ds_write_payload(f, &stream, n_keep, dtype);
    }
    gd_ds_i32_free(&stream);
    if (status != GD_OK) {
        (void)fclose(f);
        return status;
    }
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close gdtok shard output");
    }
    status = gd_ds_write_metadata(result->metadata_path,
                                  split,
                                  cfg,
                                  result,
                                  records->len,
                                  dtype,
                                  vocab_size,
                                  gd_tokenizer_hash(tok));
    return status;
}

void gd_dataset_build_result_clear(gd_dataset_build_result *result)
{
    if (result == NULL) {
        return;
    }
    free(result->train.shard_path);
    free(result->train.metadata_path);
    free(result->train.record_indices);
    free(result->val.shard_path);
    free(result->val.metadata_path);
    free(result->val.record_indices);
    memset(result, 0, sizeof(*result));
}

gd_status gd_dataset_build(const gd_dataset_build_config *cfg,
                           gd_dataset_build_result *result_out)
{
    gd_dataset_build_result result;
    gd_dataset_build_config effective;
    gd_ds_records records = {0};
    gd_tokenizer *tok = NULL;
    gd_tokenizer_config tok_cfg;
    int *train_indices = NULL;
    int *val_indices = NULL;
    int n_train = 0;
    int n_val = 0;
    int vocab_size;
    gd_gdtok_dtype dtype;
    int i;
    gd_status status;

    if (cfg == NULL || result_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset build arguments");
    }
    memset(&result, 0, sizeof(result));
    memset(result_out, 0, sizeof(*result_out));
    if (cfg->tokenizer_path == NULL || cfg->input_paths == NULL || cfg->n_input_paths <= 0 ||
        cfg->output_dir == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "missing dataset build path config");
    }
    if (cfg->block_len <= 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset block_len must be positive");
    }
    if (cfg->block_len > INT32_MAX - 1) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset block_len too large");
    }
    effective = *cfg;
    if (effective.im_start == NULL) {
        effective.im_start = "<|im_start|>";
    }
    if (effective.im_end == NULL) {
        effective.im_end = "<|im_end|>";
    }
    if (effective.train_ratio == 0.0 && effective.val_ratio == 0.0) {
        effective.train_ratio = 0.9;
        effective.val_ratio = 0.1;
    }

    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    status = gd_bpe_tokenizer_load(effective.tokenizer_path, &tok_cfg, &tok);
    if (status != GD_OK) {
        return status;
    }
    status = gd_ds_mkdir_p(effective.output_dir);
    if (status != GD_OK) {
        gd_tokenizer_destroy(tok);
        return status;
    }
    for (i = 0; i < effective.n_input_paths; ++i) {
        if (effective.wrap_plain_text != 0) {
            status = gd_ds_load_wrapped_file(effective.input_paths[i],
                                             &records,
                                             effective.im_start,
                                             effective.im_end);
        } else {
            status = gd_ds_load_formatted_file(effective.input_paths[i],
                                               &records,
                                               effective.im_start,
                                               effective.im_end);
        }
        if (status != GD_OK) {
            gd_ds_records_free(&records);
            gd_tokenizer_destroy(tok);
            return status;
        }
    }
    if (records.len == 0) {
        gd_ds_records_free(&records);
        gd_tokenizer_destroy(tok);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset input produced no records");
    }
    status = gd_ds_make_split_indices(records.len,
                                      effective.train_ratio,
                                      effective.val_ratio,
                                      effective.seed,
                                      effective.no_shuffle_split,
                                      &train_indices,
                                      &n_train,
                                      &val_indices,
                                      &n_val);
    if (status != GD_OK) {
        gd_ds_records_free(&records);
        gd_tokenizer_destroy(tok);
        return status;
    }

    vocab_size = gd_tokenizer_vocab_size(tok);
    dtype = vocab_size <= UINT16_MAX ? GD_GDTOK_DTYPE_U16 : GD_GDTOK_DTYPE_U32;
    result.n_records_total = records.len;
    status = gd_ds_build_split("train",
                               &effective,
                               tok,
                               &records,
                               train_indices,
                               n_train,
                               dtype,
                               vocab_size,
                               &result.train);
    if (status == GD_OK) {
        status = gd_ds_build_split("val",
                                   &effective,
                                   tok,
                                   &records,
                                   val_indices,
                                   n_val,
                                   dtype,
                                   vocab_size,
                                   &result.val);
    }
    free(train_indices);
    free(val_indices);
    gd_ds_records_free(&records);
    gd_tokenizer_destroy(tok);
    if (status != GD_OK) {
        gd_dataset_build_result_clear(&result);
        return status;
    }
    *result_out = result;
    return GD_OK;
}
