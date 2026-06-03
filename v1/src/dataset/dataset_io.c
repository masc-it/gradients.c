#include "dataset_internal.h"

#include "../core/internal.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

gd_status gd_ds_read_file(const char *path, char **text_out, size_t *len_out)
{
    FILE *f;
    long end;
    char *text;
    size_t nread;

    if (path == NULL || text_out == NULL || len_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset file read arguments");
    }
    *text_out = NULL;
    *len_out = 0U;
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open dataset input");
    }
    if (fseek(f, 0L, SEEK_END) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to seek dataset input");
    }
    end = ftell(f);
    if (end < 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to size dataset input");
    }
    if (fseek(f, 0L, SEEK_SET) != 0) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to rewind dataset input");
    }
    text = (char *)malloc((size_t)end + 1U);
    if (text == NULL) {
        (void)fclose(f);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset input allocation failed");
    }
    nread = fread(text, 1U, (size_t)end, f);
    if (nread != (size_t)end) {
        free(text);
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read dataset input");
    }
    if (fclose(f) != 0) {
        free(text);
        return _gd_error(GD_ERR_IO, "failed to close dataset input");
    }
    text[(size_t)end] = '\0';
    *text_out = text;
    *len_out = (size_t)end;
    return GD_OK;
}

gd_status gd_ds_mkdir_p(const char *path)
{
    char tmp[GD_DS_MAX_PATH];
    size_t len;
    size_t i;

    if (path == NULL || path[0] == '\0') {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset output dir is empty");
    }
    len = strlen(path);
    if (len >= sizeof(tmp)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dataset output dir too long");
    }
    memcpy(tmp, path, len + 1U);
    for (i = 1U; i < len; ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            if (tmp[0] != '\0' && mkdir(tmp, 0755) != 0 && errno != EEXIST) {
                return _gd_error(GD_ERR_IO, "failed to create dataset output directory");
            }
            tmp[i] = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return _gd_error(GD_ERR_IO, "failed to create dataset output directory");
    }
    return GD_OK;
}

gd_status gd_ds_join_path(const char *dir, const char *name, char **out)
{
    size_t dlen;
    size_t nlen;
    size_t need;
    char *path;
    int needs_slash;

    if (dir == NULL || name == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid dataset path join");
    }
    *out = NULL;
    dlen = strlen(dir);
    nlen = strlen(name);
    needs_slash = dlen > 0U && dir[dlen - 1U] != '/' ? 1 : 0;
    if (dlen > SIZE_MAX - nlen - 2U) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset path too large");
    }
    need = dlen + (needs_slash != 0 ? 1U : 0U) + nlen + 1U;
    path = (char *)malloc(need);
    if (path == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "dataset path allocation failed");
    }
    (void)snprintf(path, need, "%s%s%s", dir, needs_slash != 0 ? "/" : "", name);
    *out = path;
    return GD_OK;
}

static void gd_ds_put_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xffU);
    p[1] = (uint8_t)((v >> 8U) & 0xffU);
    p[2] = (uint8_t)((v >> 16U) & 0xffU);
    p[3] = (uint8_t)((v >> 24U) & 0xffU);
}

static void gd_ds_put_le64(uint8_t *p, uint64_t v)
{
    int i;
    for (i = 0; i < 8; ++i) {
        p[i] = (uint8_t)((v >> (8U * (uint32_t)i)) & 0xffU);
    }
}

static uint32_t gd_ds_get_le32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
           ((uint32_t)p[3] << 24U);
}

static uint64_t gd_ds_get_le64(const uint8_t *p)
{
    uint64_t v = 0U;
    int i;
    for (i = 7; i >= 0; --i) {
        v <<= 8U;
        v |= (uint64_t)p[i];
    }
    return v;
}

gd_status gd_ds_write_header(FILE *f,
                                    gd_gdtok_dtype dtype,
                                    uint32_t vocab_size,
                                    uint32_t block_len,
                                    uint64_t n_tokens,
                                    uint64_t n_samples,
                                    uint64_t tokenizer_hash)
{
    uint8_t header[GD_GDTOK_HEADER_SIZE];
    size_t nwrite;
    memset(header, 0, sizeof(header));
    memcpy(header, GD_GDTOK_MAGIC, strlen(GD_GDTOK_MAGIC));
    gd_ds_put_le32(&header[8], GD_GDTOK_VERSION);
    gd_ds_put_le32(&header[12], GD_GDTOK_HEADER_SIZE);
    gd_ds_put_le32(&header[16], (uint32_t)dtype);
    gd_ds_put_le32(&header[20], vocab_size);
    gd_ds_put_le32(&header[24], block_len);
    gd_ds_put_le32(&header[28], 0U);
    gd_ds_put_le64(&header[32], n_tokens);
    gd_ds_put_le64(&header[40], n_samples);
    gd_ds_put_le64(&header[48], tokenizer_hash);
    gd_ds_put_le64(&header[56], GD_GDTOK_HEADER_SIZE);
    nwrite = fwrite(header, 1U, sizeof(header), f);
    if (nwrite != sizeof(header)) {
        return _gd_error(GD_ERR_IO, "failed to write gdtok header");
    }
    return GD_OK;
}

gd_status gd_ds_write_payload(FILE *f,
                                     const gd_ds_i32_vec *tokens,
                                     uint64_t n_tokens,
                                     gd_gdtok_dtype dtype)
{
    uint64_t i;
    if (tokens == NULL || n_tokens > tokens->len) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok payload");
    }
    for (i = 0U; i < n_tokens; ++i) {
        int32_t id = tokens->data[i];
        if (id < 0) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "negative token id in dataset");
        }
        if (dtype == GD_GDTOK_DTYPE_U16) {
            uint8_t b[2];
            if (id > UINT16_MAX) {
                return _gd_error(GD_ERR_INVALID_ARGUMENT, "token id exceeds u16 shard dtype");
            }
            b[0] = (uint8_t)((uint32_t)id & 0xffU);
            b[1] = (uint8_t)(((uint32_t)id >> 8U) & 0xffU);
            if (fwrite(b, 1U, sizeof(b), f) != sizeof(b)) {
                return _gd_error(GD_ERR_IO, "failed to write u16 gdtok payload");
            }
        } else {
            uint8_t b[4];
            gd_ds_put_le32(b, (uint32_t)id);
            if (fwrite(b, 1U, sizeof(b), f) != sizeof(b)) {
                return _gd_error(GD_ERR_IO, "failed to write u32 gdtok payload");
            }
        }
    }
    return GD_OK;
}

gd_status gd_gdtok_read_header(const char *path, gd_gdtok_header *out)
{
    FILE *f;
    uint8_t header[GD_GDTOK_HEADER_SIZE];
    size_t nread;
    uint32_t dtype;

    if (path == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok header read arguments");
    }
    memset(out, 0, sizeof(*out));
    f = fopen(path, "rb");
    if (f == NULL) {
        return _gd_error(GD_ERR_IO, "failed to open gdtok shard");
    }
    nread = fread(header, 1U, sizeof(header), f);
    if (nread != sizeof(header)) {
        (void)fclose(f);
        return _gd_error(GD_ERR_IO, "failed to read gdtok header");
    }
    if (fclose(f) != 0) {
        return _gd_error(GD_ERR_IO, "failed to close gdtok shard");
    }
    if (memcmp(header, GD_GDTOK_MAGIC, strlen(GD_GDTOK_MAGIC)) != 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok magic");
    }
    dtype = gd_ds_get_le32(&header[16]);
    if (dtype != (uint32_t)GD_GDTOK_DTYPE_U16 && dtype != (uint32_t)GD_GDTOK_DTYPE_U32) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gdtok dtype");
    }
    out->version = gd_ds_get_le32(&header[8]);
    out->header_size = gd_ds_get_le32(&header[12]);
    out->dtype = (gd_gdtok_dtype)dtype;
    out->vocab_size = gd_ds_get_le32(&header[20]);
    out->block_len = gd_ds_get_le32(&header[24]);
    out->n_tokens = gd_ds_get_le64(&header[32]);
    out->n_samples = gd_ds_get_le64(&header[40]);
    out->tokenizer_hash = gd_ds_get_le64(&header[48]);
    out->payload_offset = gd_ds_get_le64(&header[56]);
    if (out->version != GD_GDTOK_VERSION || out->header_size != GD_GDTOK_HEADER_SIZE ||
        out->payload_offset != GD_GDTOK_HEADER_SIZE) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unsupported gdtok header version");
    }
    return GD_OK;
}
