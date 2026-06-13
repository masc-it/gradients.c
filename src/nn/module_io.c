#define _POSIX_C_SOURCE 200809L

#include <gradients/module.h>
#include <gradients/transfer.h>

#include "../core/memory_internal.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define GD_CKPT_MAGIC_LEN 8U
#define GD_CKPT_VERSION 1U
#define GD_CKPT_HEADER_SIZE 64U
#define GD_CKPT_DIR_ENTRY_FIXED_SIZE (16U + 8U * GD_MAX_DIMS + 16U)
#define GD_CKPT_IO_CHUNK (4U * 1024U * 1024U)

static const unsigned char gd_ckpt_magic[GD_CKPT_MAGIC_LEN] = {'G', 'D', 'C', 'K', 'P', 'T', '1', '\0'};

typedef enum gd_ckpt_entry_flags {
    GD_CKPT_ENTRY_PARAM = 1U,
    GD_CKPT_ENTRY_BUFFER = 2U,
} gd_ckpt_entry_flags;

typedef struct gd_ckpt_header {
    uint32_t version;
    uint32_t header_size;
    uint64_t tensor_count;
    uint64_t dir_offset;
    uint64_t dir_size;
    uint64_t metadata_offset;
    uint64_t metadata_len;
    uint64_t data_offset;
} gd_ckpt_header;

typedef struct gd_ckpt_state_entry {
    char path[GD_MODULE_PATH_MAX];
    gd_tensor *tensor;
    uint32_t flags;
    size_t nbytes;
    uint64_t data_offset;
    bool seen;
} gd_ckpt_state_entry;

typedef struct gd_ckpt_state_list {
    gd_ckpt_state_entry *items;
    uint32_t count;
    uint32_t capacity;
} gd_ckpt_state_list;

typedef struct gd_ckpt_file_entry {
    char path[GD_MODULE_PATH_MAX];
    uint32_t flags;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    uint64_t data_offset;
    uint64_t nbytes;
} gd_ckpt_file_entry;

typedef struct gd_ckpt_file_list {
    gd_ckpt_file_entry *items;
    uint32_t count;
} gd_ckpt_file_list;

static gd_status gd_ckpt_ctx_error(gd_context *ctx, gd_status status, const char *message)
{
    return ctx != NULL ? gd_context_set_error(ctx, status, message) : status;
}

static bool gd_ckpt_add_overflow_u64(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL || b > UINT64_MAX - a) {
        return true;
    }
    *out = a + b;
    return false;
}

static gd_status gd_ckpt_join_path(const char *prefix,
                                   const char *name,
                                   char out[GD_MODULE_PATH_MAX])
{
    int n;
    if (name == NULL || name[0] == '\0' || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (prefix == NULL || prefix[0] == '\0') {
        n = snprintf(out, GD_MODULE_PATH_MAX, "%s", name);
    } else {
        n = snprintf(out, GD_MODULE_PATH_MAX, "%s.%s", prefix, name);
    }
    if (n < 0 || (size_t)n >= GD_MODULE_PATH_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static gd_status gd_ckpt_state_list_reserve(gd_ckpt_state_list *list)
{
    gd_ckpt_state_entry *grown;
    size_t max_items;
    uint32_t new_capacity;
    if (list == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (list->count < list->capacity) {
        return GD_OK;
    }
    new_capacity = list->capacity == 0U ? 16U : list->capacity * 2U;
    max_items = SIZE_MAX / sizeof(list->items[0]);
    if (new_capacity <= list->capacity || (size_t)new_capacity > max_items) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    grown = (gd_ckpt_state_entry *)realloc(list->items,
                                           (size_t)new_capacity * sizeof(list->items[0]));
    if (grown == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    memset(grown + list->capacity,
           0,
           (size_t)(new_capacity - list->capacity) * sizeof(list->items[0]));
    list->items = grown;
    list->capacity = new_capacity;
    return GD_OK;
}

static void gd_ckpt_state_list_free(gd_ckpt_state_list *list)
{
    if (list == NULL) {
        return;
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static void gd_ckpt_file_list_free(gd_ckpt_file_list *list)
{
    if (list == NULL) {
        return;
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static gd_status gd_ckpt_tensor_save_nbytes(gd_context *ctx,
                                            const gd_tensor *tensor,
                                            size_t *out_nbytes)
{
    gd_status st;
    if (ctx == NULL || tensor == NULL || out_nbytes == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_ckpt_ctx_error(ctx,
                                 GD_ERR_UNSUPPORTED,
                                 "module checkpoint only supports contiguous tensors");
    }
    return gd_tensor_logical_nbytes(tensor, out_nbytes);
}

static gd_status gd_ckpt_state_list_append(gd_context *ctx,
                                           gd_ckpt_state_list *list,
                                           const char *path,
                                           gd_tensor *tensor,
                                           uint32_t flags)
{
    gd_status st;
    gd_ckpt_state_entry *entry;
    size_t nbytes = 0U;
    if (ctx == NULL || list == NULL || path == NULL || tensor == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (strlen(path) >= GD_MODULE_PATH_MAX) {
        return gd_ckpt_ctx_error(ctx, GD_ERR_INVALID_ARGUMENT, "module checkpoint path too long");
    }
    st = gd_ckpt_tensor_save_nbytes(ctx, tensor, &nbytes);
    if (st != GD_OK) {
        return st;
    }
    st = gd_ckpt_state_list_reserve(list);
    if (st != GD_OK) {
        return gd_ckpt_ctx_error(ctx, st, "module checkpoint state list allocation failed");
    }
    entry = &list->items[list->count];
    memset(entry, 0, sizeof(*entry));
    (void)snprintf(entry->path, sizeof(entry->path), "%s", path);
    entry->tensor = tensor;
    entry->flags = flags;
    entry->nbytes = nbytes;
    list->count += 1U;
    return GD_OK;
}

static gd_status gd_ckpt_collect_module_state_impl(gd_context *ctx,
                                                   const gd_module *module,
                                                   const char *prefix,
                                                   bool include_buffers,
                                                   gd_ckpt_state_list *out)
{
    uint32_t i;
    gd_status st;
    char path[GD_MODULE_PATH_MAX];
    if (ctx == NULL || module == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < module->param_count; ++i) {
        const gd_module_param_entry *entry = &module->params[i];
        if (entry->tensor == NULL) {
            continue;
        }
        st = gd_ckpt_join_path(prefix, entry->name, path);
        if (st != GD_OK) {
            return gd_ckpt_ctx_error(ctx, st, "module checkpoint parameter path too long");
        }
        st = gd_ckpt_state_list_append(ctx, out, path, entry->tensor, GD_CKPT_ENTRY_PARAM);
        if (st != GD_OK) {
            return st;
        }
    }
    if (include_buffers) {
        for (i = 0U; i < module->buffer_count; ++i) {
            const gd_module_buffer_entry *entry = &module->buffers[i];
            if (entry->tensor == NULL) {
                continue;
            }
            st = gd_ckpt_join_path(prefix, entry->name, path);
            if (st != GD_OK) {
                return gd_ckpt_ctx_error(ctx, st, "module checkpoint buffer path too long");
            }
            st = gd_ckpt_state_list_append(ctx, out, path, entry->tensor, GD_CKPT_ENTRY_BUFFER);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    for (i = 0U; i < module->child_count; ++i) {
        const gd_module_child_entry *child = &module->children[i];
        if (child->module == NULL) {
            continue;
        }
        st = gd_ckpt_join_path(prefix, child->name, path);
        if (st != GD_OK) {
            return gd_ckpt_ctx_error(ctx, st, "module checkpoint child path too long");
        }
        st = gd_ckpt_collect_module_state_impl(ctx, child->module, path, include_buffers, out);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

static gd_status gd_ckpt_collect_module_state(gd_context *ctx,
                                              const gd_module *module,
                                              bool include_buffers,
                                              gd_ckpt_state_list *out)
{
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    if (ctx == NULL || module == NULL || module->name[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_ckpt_collect_module_state_impl(ctx, module, module->name, include_buffers, out);
}

static gd_ckpt_state_entry *gd_ckpt_find_state_entry(gd_ckpt_state_list *list, const char *path)
{
    uint32_t i;
    if (list == NULL || path == NULL) {
        return NULL;
    }
    for (i = 0U; i < list->count; ++i) {
        if (strncmp(list->items[i].path, path, GD_MODULE_PATH_MAX) == 0) {
            return &list->items[i];
        }
    }
    return NULL;
}

static void gd_ckpt_put_u32(unsigned char dst[4], uint32_t value)
{
    dst[0] = (unsigned char)(value & 0xffU);
    dst[1] = (unsigned char)((value >> 8U) & 0xffU);
    dst[2] = (unsigned char)((value >> 16U) & 0xffU);
    dst[3] = (unsigned char)((value >> 24U) & 0xffU);
}

static void gd_ckpt_put_u64(unsigned char dst[8], uint64_t value)
{
    uint32_t i;
    for (i = 0U; i < 8U; ++i) {
        dst[i] = (unsigned char)((value >> (i * 8U)) & UINT64_C(0xff));
    }
}

static uint32_t gd_ckpt_get_u32(const unsigned char src[4])
{
    return ((uint32_t)src[0]) |
           ((uint32_t)src[1] << 8U) |
           ((uint32_t)src[2] << 16U) |
           ((uint32_t)src[3] << 24U);
}

static uint64_t gd_ckpt_get_u64(const unsigned char src[8])
{
    uint64_t value = 0U;
    uint32_t i;
    for (i = 0U; i < 8U; ++i) {
        value |= ((uint64_t)src[i]) << (i * 8U);
    }
    return value;
}

static gd_status gd_ckpt_write_exact(FILE *file, const void *data, size_t nbytes)
{
    if (file == NULL || (data == NULL && nbytes != 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (nbytes == 0U) {
        return GD_OK;
    }
    return fwrite(data, 1U, nbytes, file) == nbytes ? GD_OK : GD_ERR_IO;
}

static gd_status gd_ckpt_read_exact(FILE *file, void *data, size_t nbytes)
{
    if (file == NULL || (data == NULL && nbytes != 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (nbytes == 0U) {
        return GD_OK;
    }
    return fread(data, 1U, nbytes, file) == nbytes ? GD_OK : GD_ERR_IO;
}

static gd_status gd_ckpt_write_u32(FILE *file, uint32_t value)
{
    unsigned char bytes[4];
    gd_ckpt_put_u32(bytes, value);
    return gd_ckpt_write_exact(file, bytes, sizeof(bytes));
}

static gd_status gd_ckpt_write_u64(FILE *file, uint64_t value)
{
    unsigned char bytes[8];
    gd_ckpt_put_u64(bytes, value);
    return gd_ckpt_write_exact(file, bytes, sizeof(bytes));
}

static gd_status gd_ckpt_read_u32(FILE *file, uint32_t *out)
{
    gd_status st;
    unsigned char bytes[4];
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_ckpt_read_exact(file, bytes, sizeof(bytes));
    if (st != GD_OK) {
        return st;
    }
    *out = gd_ckpt_get_u32(bytes);
    return GD_OK;
}

static gd_status gd_ckpt_read_u64(FILE *file, uint64_t *out)
{
    gd_status st;
    unsigned char bytes[8];
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_ckpt_read_exact(file, bytes, sizeof(bytes));
    if (st != GD_OK) {
        return st;
    }
    *out = gd_ckpt_get_u64(bytes);
    return GD_OK;
}

static gd_status gd_ckpt_seek(FILE *file, uint64_t offset)
{
    if (file == NULL || offset > (uint64_t)INT64_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return fseeko(file, (off_t)offset, SEEK_SET) == 0 ? GD_OK : GD_ERR_IO;
}

static gd_status gd_ckpt_write_header(FILE *file, const gd_ckpt_header *header)
{
    gd_status st;
    if (file == NULL || header == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_ckpt_write_exact(file, gd_ckpt_magic, GD_CKPT_MAGIC_LEN);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u32(file, header->version);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u32(file, header->header_size);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, header->tensor_count);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, header->dir_offset);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, header->dir_size);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, header->metadata_offset);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, header->metadata_len);
    if (st != GD_OK) { return st; }
    return gd_ckpt_write_u64(file, header->data_offset);
}

static gd_status gd_ckpt_read_header(FILE *file, gd_ckpt_header *header)
{
    gd_status st;
    unsigned char magic[GD_CKPT_MAGIC_LEN];
    if (file == NULL || header == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(header, 0, sizeof(*header));
    st = gd_ckpt_read_exact(file, magic, sizeof(magic));
    if (st != GD_OK) { return st; }
    if (memcmp(magic, gd_ckpt_magic, sizeof(magic)) != 0) {
        return GD_ERR_BAD_STATE;
    }
    st = gd_ckpt_read_u32(file, &header->version);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u32(file, &header->header_size);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->tensor_count);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->dir_offset);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->dir_size);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->metadata_offset);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->metadata_len);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_read_u64(file, &header->data_offset);
    if (st != GD_OK) { return st; }
    if (header->version != GD_CKPT_VERSION || header->header_size != GD_CKPT_HEADER_SIZE ||
        header->dir_offset < header->header_size || header->metadata_offset < header->dir_offset ||
        header->data_offset < header->metadata_offset) {
        return GD_ERR_BAD_STATE;
    }
    return GD_OK;
}

static gd_status gd_ckpt_write_dir_entry(FILE *file, const gd_ckpt_state_entry *entry)
{
    gd_status st;
    uint32_t i;
    const size_t path_len_size = strlen(entry->path);
    uint32_t path_len;
    if (file == NULL || entry == NULL || entry->tensor == NULL || path_len_size >= GD_MODULE_PATH_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    path_len = (uint32_t)path_len_size;
    st = gd_ckpt_write_u32(file, path_len);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u32(file, entry->flags);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u32(file, (uint32_t)entry->tensor->dtype);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u32(file, entry->tensor->rank);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < GD_MAX_DIMS; ++i) {
        const uint64_t dim = i < entry->tensor->rank ? (uint64_t)entry->tensor->shape[i] : 0U;
        st = gd_ckpt_write_u64(file, dim);
        if (st != GD_OK) { return st; }
    }
    st = gd_ckpt_write_u64(file, entry->data_offset);
    if (st != GD_OK) { return st; }
    st = gd_ckpt_write_u64(file, (uint64_t)entry->nbytes);
    if (st != GD_OK) { return st; }
    return gd_ckpt_write_exact(file, entry->path, path_len_size);
}

static gd_status gd_ckpt_read_dir(FILE *file,
                                  const gd_ckpt_header *header,
                                  gd_ckpt_file_list *out)
{
    gd_status st;
    uint32_t i;
    uint64_t consumed = 0U;
    if (file == NULL || header == NULL || out == NULL || header->tensor_count > (uint64_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    st = gd_ckpt_seek(file, header->dir_offset);
    if (st != GD_OK) {
        return st;
    }
    out->count = (uint32_t)header->tensor_count;
    if (out->count == 0U) {
        return GD_OK;
    }
    {
        const size_t max_items = SIZE_MAX / sizeof(out->items[0]);
        if ((size_t)out->count > max_items) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    out->items = (gd_ckpt_file_entry *)calloc((size_t)out->count, sizeof(out->items[0]));
    if (out->items == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    for (i = 0U; i < out->count; ++i) {
        gd_ckpt_file_entry *entry = &out->items[i];
        uint32_t path_len;
        uint32_t dtype;
        uint32_t rank;
        uint32_t j;
        uint64_t fixed_and_path;
        st = gd_ckpt_read_u32(file, &path_len);
        if (st != GD_OK) { return st; }
        st = gd_ckpt_read_u32(file, &entry->flags);
        if (st != GD_OK) { return st; }
        st = gd_ckpt_read_u32(file, &dtype);
        if (st != GD_OK) { return st; }
        st = gd_ckpt_read_u32(file, &rank);
        if (st != GD_OK) { return st; }
        if (path_len == 0U || path_len >= GD_MODULE_PATH_MAX || rank > GD_MAX_DIMS ||
            (entry->flags != GD_CKPT_ENTRY_PARAM && entry->flags != GD_CKPT_ENTRY_BUFFER)) {
            return GD_ERR_BAD_STATE;
        }
        entry->dtype = (gd_dtype)dtype;
        entry->rank = rank;
        for (j = 0U; j < GD_MAX_DIMS; ++j) {
            uint64_t dim;
            st = gd_ckpt_read_u64(file, &dim);
            if (st != GD_OK) { return st; }
            if (j < rank && dim > (uint64_t)INT64_MAX) {
                return GD_ERR_BAD_STATE;
            }
            entry->shape[j] = j < rank ? (int64_t)dim : 0;
        }
        st = gd_ckpt_read_u64(file, &entry->data_offset);
        if (st != GD_OK) { return st; }
        st = gd_ckpt_read_u64(file, &entry->nbytes);
        if (st != GD_OK) { return st; }
        st = gd_ckpt_read_exact(file, entry->path, path_len);
        if (st != GD_OK) { return st; }
        entry->path[path_len] = '\0';
        fixed_and_path = (uint64_t)GD_CKPT_DIR_ENTRY_FIXED_SIZE + (uint64_t)path_len;
        if (gd_ckpt_add_overflow_u64(consumed, fixed_and_path, &consumed) ||
            consumed > header->dir_size || entry->data_offset < header->data_offset) {
            return GD_ERR_BAD_STATE;
        }
    }
    return consumed == header->dir_size ? GD_OK : GD_ERR_BAD_STATE;
}

static gd_status gd_ckpt_compute_layout(gd_ckpt_state_list *entries,
                                        size_t metadata_len,
                                        gd_ckpt_header *header)
{
    uint32_t i;
    uint64_t dir_size = 0U;
    uint64_t cursor;
    if (entries == NULL || header == NULL || entries->count > (uint32_t)UINT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(header, 0, sizeof(*header));
    for (i = 0U; i < entries->count; ++i) {
        const size_t path_len = strlen(entries->items[i].path);
        uint64_t entry_size;
        if (path_len == 0U || path_len >= GD_MODULE_PATH_MAX ||
            entries->items[i].nbytes > (size_t)UINT64_MAX) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        entry_size = (uint64_t)GD_CKPT_DIR_ENTRY_FIXED_SIZE + (uint64_t)path_len;
        if (gd_ckpt_add_overflow_u64(dir_size, entry_size, &dir_size)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    header->version = GD_CKPT_VERSION;
    header->header_size = GD_CKPT_HEADER_SIZE;
    header->tensor_count = entries->count;
    header->dir_offset = GD_CKPT_HEADER_SIZE;
    header->dir_size = dir_size;
    if (gd_ckpt_add_overflow_u64(header->dir_offset, header->dir_size, &header->metadata_offset) ||
        gd_ckpt_add_overflow_u64(header->metadata_offset, (uint64_t)metadata_len, &header->data_offset)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    cursor = header->data_offset;
    for (i = 0U; i < entries->count; ++i) {
        entries->items[i].data_offset = cursor;
        if (gd_ckpt_add_overflow_u64(cursor, (uint64_t)entries->items[i].nbytes, &cursor)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    header->metadata_len = (uint64_t)metadata_len;
    return GD_OK;
}

static char *gd_ckpt_tmp_path(const char *path)
{
    size_t len;
    char *tmp;
    const char suffix[] = ".tmp";
    if (path == NULL || path[0] == '\0') {
        return NULL;
    }
    len = strlen(path);
    if (len > SIZE_MAX - sizeof(suffix)) {
        return NULL;
    }
    tmp = (char *)malloc(len + sizeof(suffix));
    if (tmp == NULL) {
        return NULL;
    }
    (void)snprintf(tmp, len + sizeof(suffix), "%s%s", path, suffix);
    return tmp;
}

static gd_status gd_ckpt_flush_file(FILE *file)
{
    int fd;
    if (file == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (fflush(file) != 0) {
        return GD_ERR_IO;
    }
    fd = fileno(file);
    if (fd < 0) {
        return GD_ERR_IO;
    }
    if (fsync(fd) != 0) {
        return GD_ERR_IO;
    }
    return GD_OK;
}

static gd_status gd_ckpt_write_tensor_data(gd_context *ctx,
                                           FILE *file,
                                           const gd_tensor *tensor,
                                           size_t nbytes,
                                           unsigned char *buffer,
                                           size_t buffer_size)
{
    size_t offset = 0U;
    if (ctx == NULL || file == NULL || tensor == NULL || (buffer == NULL && buffer_size != 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    while (offset < nbytes) {
        gd_status st;
        size_t chunk = nbytes - offset;
        if (chunk > buffer_size) {
            chunk = buffer_size;
        }
        st = gd_span_download(ctx, &tensor->storage, tensor->view_offset + offset, buffer, chunk);
        if (st != GD_OK) {
            return st;
        }
        st = gd_ckpt_write_exact(file, buffer, chunk);
        if (st != GD_OK) {
            return st;
        }
        offset += chunk;
    }
    return GD_OK;
}

static gd_status gd_ckpt_read_tensor_data(gd_context *ctx,
                                          FILE *file,
                                          gd_tensor *tensor,
                                          uint64_t data_offset,
                                          size_t nbytes,
                                          unsigned char *buffer,
                                          size_t buffer_size)
{
    size_t offset = 0U;
    if (ctx == NULL || file == NULL || tensor == NULL || (buffer == NULL && buffer_size != 0U)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_status st = gd_ckpt_seek(file, data_offset);
    if (st != GD_OK) {
        return st;
    }
    while (offset < nbytes) {
        size_t chunk = nbytes - offset;
        if (chunk > buffer_size) {
            chunk = buffer_size;
        }
        st = gd_ckpt_read_exact(file, buffer, chunk);
        if (st != GD_OK) {
            return st;
        }
        st = gd_span_upload(ctx, &tensor->storage, tensor->view_offset + offset, buffer, chunk);
        if (st != GD_OK) {
            return st;
        }
        offset += chunk;
    }
    tensor->version += 1U;
    if (tensor->version == 0U) {
        tensor->version = 1U;
    }
    return GD_OK;
}

gd_status gd_module_save_state(gd_context *ctx,
                               const gd_module *module,
                               const char *path,
                               const gd_module_save_options *options)
{
    gd_status st;
    gd_ckpt_state_list entries;
    gd_ckpt_header header;
    FILE *file = NULL;
    char *tmp_path = NULL;
    unsigned char *buffer = NULL;
    bool include_buffers = true;
    const char *metadata = NULL;
    size_t metadata_len = 0U;
    uint32_t i;
    if (ctx == NULL || module == NULL || path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&entries, 0, sizeof(entries));
    if (options != NULL) {
        include_buffers = options->include_buffers;
        metadata = options->metadata;
        metadata_len = options->metadata_len;
    }
    if (metadata_len != 0U && metadata == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_ckpt_collect_module_state(ctx, module, include_buffers, &entries);
    if (st != GD_OK) {
        gd_ckpt_state_list_free(&entries);
        return st;
    }
    st = gd_ckpt_compute_layout(&entries, metadata_len, &header);
    if (st != GD_OK) {
        gd_ckpt_state_list_free(&entries);
        return gd_ckpt_ctx_error(ctx, st, "module checkpoint layout overflow");
    }
    st = gd_synchronize(ctx);
    if (st != GD_OK) {
        gd_ckpt_state_list_free(&entries);
        return st;
    }
    tmp_path = gd_ckpt_tmp_path(path);
    if (tmp_path == NULL) {
        gd_ckpt_state_list_free(&entries);
        return gd_ckpt_ctx_error(ctx, GD_ERR_OUT_OF_MEMORY, "module checkpoint temp path allocation failed");
    }
    file = fopen(tmp_path, "wb");
    if (file == NULL) {
        free(tmp_path);
        gd_ckpt_state_list_free(&entries);
        return gd_ckpt_ctx_error(ctx, GD_ERR_IO, "module checkpoint open failed");
    }
    buffer = (unsigned char *)malloc(GD_CKPT_IO_CHUNK);
    if (buffer == NULL) {
        (void)fclose(file);
        (void)remove(tmp_path);
        free(tmp_path);
        gd_ckpt_state_list_free(&entries);
        return gd_ckpt_ctx_error(ctx, GD_ERR_OUT_OF_MEMORY, "module checkpoint IO buffer allocation failed");
    }

    st = gd_ckpt_write_header(file, &header);
    for (i = 0U; st == GD_OK && i < entries.count; ++i) {
        st = gd_ckpt_write_dir_entry(file, &entries.items[i]);
    }
    if (st == GD_OK && metadata_len != 0U) {
        st = gd_ckpt_write_exact(file, metadata, metadata_len);
    }
    for (i = 0U; st == GD_OK && i < entries.count; ++i) {
        st = gd_ckpt_write_tensor_data(ctx,
                                       file,
                                       entries.items[i].tensor,
                                       entries.items[i].nbytes,
                                       buffer,
                                       GD_CKPT_IO_CHUNK);
    }
    if (st == GD_OK) {
        st = gd_ckpt_flush_file(file);
    }
    if (fclose(file) != 0 && st == GD_OK) {
        st = GD_ERR_IO;
    }
    file = NULL;
    if (st == GD_OK && rename(tmp_path, path) != 0) {
        st = GD_ERR_IO;
    }
    if (st != GD_OK) {
        (void)remove(tmp_path);
    }
    free(buffer);
    free(tmp_path);
    gd_ckpt_state_list_free(&entries);
    return st == GD_OK ? GD_OK : gd_ckpt_ctx_error(ctx, st, "module checkpoint save failed");
}

static bool gd_ckpt_file_entry_matches_tensor(const gd_ckpt_file_entry *file_entry,
                                              const gd_tensor *tensor,
                                              size_t nbytes)
{
    uint32_t i;
    if (file_entry == NULL || tensor == NULL || file_entry->dtype != tensor->dtype ||
        file_entry->rank != tensor->rank || file_entry->nbytes != (uint64_t)nbytes) {
        return false;
    }
    for (i = 0U; i < tensor->rank; ++i) {
        if (file_entry->shape[i] != tensor->shape[i]) {
            return false;
        }
    }
    return true;
}

gd_status gd_module_load_state(gd_context *ctx,
                               gd_module *module,
                               const char *path,
                               const gd_module_load_options *options)
{
    gd_status st;
    gd_ckpt_header header;
    gd_ckpt_file_list file_entries;
    gd_ckpt_state_list state_entries;
    FILE *file = NULL;
    unsigned char *buffer = NULL;
    bool strict = true;
    bool load_buffers = true;
    uint32_t i;
    if (ctx == NULL || module == NULL || path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&header, 0, sizeof(header));
    memset(&file_entries, 0, sizeof(file_entries));
    memset(&state_entries, 0, sizeof(state_entries));
    if (options != NULL) {
        strict = options->strict;
        load_buffers = options->load_buffers;
    }
    st = gd_ckpt_collect_module_state(ctx, module, load_buffers, &state_entries);
    if (st != GD_OK) {
        return st;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        gd_ckpt_state_list_free(&state_entries);
        return gd_ckpt_ctx_error(ctx, GD_ERR_IO, "module checkpoint open failed");
    }
    st = gd_ckpt_read_header(file, &header);
    if (st == GD_OK) {
        st = gd_ckpt_read_dir(file, &header, &file_entries);
    }
    if (st != GD_OK) {
        (void)fclose(file);
        gd_ckpt_file_list_free(&file_entries);
        gd_ckpt_state_list_free(&state_entries);
        return gd_ckpt_ctx_error(ctx, st, "module checkpoint header/directory invalid");
    }

    for (i = 0U; i < file_entries.count; ++i) {
        gd_ckpt_file_entry *file_entry = &file_entries.items[i];
        gd_ckpt_state_entry *state_entry;
        if (!load_buffers && file_entry->flags == GD_CKPT_ENTRY_BUFFER) {
            continue;
        }
        state_entry = gd_ckpt_find_state_entry(&state_entries, file_entry->path);
        if (state_entry == NULL) {
            if (strict) {
                st = gd_ckpt_ctx_error(ctx, GD_ERR_BAD_STATE, "module checkpoint has unexpected tensor");
                break;
            }
            continue;
        }
        state_entry->seen = true;
        if (!gd_ckpt_file_entry_matches_tensor(file_entry, state_entry->tensor, state_entry->nbytes)) {
            st = gd_ckpt_ctx_error(ctx, GD_ERR_BAD_STATE, "module checkpoint tensor metadata mismatch");
            break;
        }
    }
    if (st == GD_OK && strict) {
        for (i = 0U; i < state_entries.count; ++i) {
            if (!state_entries.items[i].seen) {
                st = gd_ckpt_ctx_error(ctx, GD_ERR_BAD_STATE, "module checkpoint missing tensor");
                break;
            }
        }
    }
    if (st != GD_OK) {
        (void)fclose(file);
        gd_ckpt_file_list_free(&file_entries);
        gd_ckpt_state_list_free(&state_entries);
        return st;
    }

    buffer = (unsigned char *)malloc(GD_CKPT_IO_CHUNK);
    if (buffer == NULL) {
        (void)fclose(file);
        gd_ckpt_file_list_free(&file_entries);
        gd_ckpt_state_list_free(&state_entries);
        return gd_ckpt_ctx_error(ctx, GD_ERR_OUT_OF_MEMORY, "module checkpoint IO buffer allocation failed");
    }
    for (i = 0U; i < file_entries.count; ++i) {
        const gd_ckpt_file_entry *file_entry = &file_entries.items[i];
        gd_ckpt_state_entry *state_entry;
        if (!load_buffers && file_entry->flags == GD_CKPT_ENTRY_BUFFER) {
            continue;
        }
        state_entry = gd_ckpt_find_state_entry(&state_entries, file_entry->path);
        if (state_entry == NULL) {
            continue;
        }
        st = gd_ckpt_read_tensor_data(ctx,
                                      file,
                                      state_entry->tensor,
                                      file_entry->data_offset,
                                      state_entry->nbytes,
                                      buffer,
                                      GD_CKPT_IO_CHUNK);
        if (st != GD_OK) {
            break;
        }
    }
    free(buffer);
    if (fclose(file) != 0 && st == GD_OK) {
        st = GD_ERR_IO;
    }
    gd_ckpt_file_list_free(&file_entries);
    gd_ckpt_state_list_free(&state_entries);
    return st == GD_OK ? GD_OK : gd_ckpt_ctx_error(ctx, st, "module checkpoint load failed");
}

gd_status gd_checkpoint_read_metadata(const char *path,
                                      char **metadata_out,
                                      size_t *metadata_len_out)
{
    gd_status st;
    gd_ckpt_header header;
    FILE *file;
    char *metadata;
    if (path == NULL || path[0] == '\0' || metadata_out == NULL || metadata_len_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *metadata_out = NULL;
    *metadata_len_out = 0U;
    file = fopen(path, "rb");
    if (file == NULL) {
        return GD_ERR_IO;
    }
    st = gd_ckpt_read_header(file, &header);
    if (st != GD_OK) {
        (void)fclose(file);
        return st;
    }
    if (header.metadata_len > (uint64_t)SIZE_MAX - 1U) {
        (void)fclose(file);
        return GD_ERR_OUT_OF_MEMORY;
    }
    metadata = (char *)malloc((size_t)header.metadata_len + 1U);
    if (metadata == NULL) {
        (void)fclose(file);
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gd_ckpt_seek(file, header.metadata_offset);
    if (st != GD_OK) {
        free(metadata);
        (void)fclose(file);
        return st;
    }
    st = gd_ckpt_read_exact(file, metadata, (size_t)header.metadata_len);
    if (fclose(file) != 0 && st == GD_OK) {
        st = GD_ERR_IO;
    }
    if (st != GD_OK) {
        free(metadata);
        return st;
    }
    metadata[(size_t)header.metadata_len] = '\0';
    *metadata_out = metadata;
    *metadata_len_out = (size_t)header.metadata_len;
    return GD_OK;
}
