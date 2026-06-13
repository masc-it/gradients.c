/* Shared C helpers for operation Python oracle runners.
 *
 * This file is intentionally included from generated temporary runner sources
 * instead of compiled as a standalone translation unit.  Functions are marked
 * maybe-unused because each op runner only needs a subset of the helpers, while
 * all runners are compiled with -Wall -Wextra -Werror.
 */
#ifndef GD_ORACLE_RUNNER_COMMON_C
#define GD_ORACLE_RUNNER_COMMON_C

#include <gradients/gradients.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define GD_ORACLE_MAYBE_UNUSED __attribute__((unused))
#else
#define GD_ORACLE_MAYBE_UNUSED
#endif

#ifndef AXIS_ALL
#define AXIS_ALL (-999)
#endif

static int GD_ORACLE_MAYBE_UNUSED read_file(const char *path, void *dst, size_t nbytes)
{
    FILE *f;
    if (path == NULL || dst == NULL) {
        return 1;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fread(dst, 1U, nbytes, f) != nbytes) {
        fprintf(stderr, "read %s: expected %zu bytes\n", path, nbytes);
        (void)fclose(f);
        return 1;
    }
    return fclose(f) != 0 ? 1 : 0;
}

static int GD_ORACLE_MAYBE_UNUSED write_file(const char *path, const void *src, size_t nbytes)
{
    FILE *f;
    if (path == NULL || src == NULL) {
        return 1;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "open %s: %s\n", path, strerror(errno));
        return 1;
    }
    if (fwrite(src, 1U, nbytes, f) != nbytes) {
        fprintf(stderr, "write %s: expected %zu bytes\n", path, nbytes);
        (void)fclose(f);
        return 1;
    }
    return fclose(f) != 0 ? 1 : 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_u32(const char *s, uint32_t *out)
{
    char *end = NULL;
    unsigned long v;
    if (s == NULL || out == NULL || *s == '\0') {
        return 1;
    }
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0' || v > UINT32_MAX) {
        return 1;
    }
    *out = (uint32_t)v;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_u64(const char *s, uint64_t *out)
{
    char *end = NULL;
    unsigned long long v;
    if (s == NULL || out == NULL || *s == '\0') {
        return 1;
    }
    errno = 0;
    v = strtoull(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') {
        return 1;
    }
    *out = (uint64_t)v;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_i32(const char *s, int32_t *out)
{
    char *end = NULL;
    long v;
    if (s == NULL || out == NULL || *s == '\0') {
        return 1;
    }
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0' || v < INT32_MIN || v > INT32_MAX) {
        return 1;
    }
    *out = (int32_t)v;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_i64(const char *s, int64_t *out)
{
    char *end = NULL;
    long long v;
    if (s == NULL || out == NULL || *s == '\0') {
        return 1;
    }
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno == ERANGE || end == s || *end != '\0') {
        return 1;
    }
    *out = (int64_t)v;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_i64_dim(const char *s, int64_t *out)
{
    if (parse_i64(s, out) != 0 || *out <= 0) {
        return 1;
    }
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_f32(const char *s, float *out)
{
    char *end = NULL;
    float v;
    if (s == NULL || out == NULL || *s == '\0') {
        return 1;
    }
    errno = 0;
    v = strtof(s, &end);
    if (errno == ERANGE || end == s || *end != '\0') {
        return 1;
    }
    *out = v;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED parse_float(const char *s, float *out)
{
    return parse_f32(s, out);
}

static int GD_ORACLE_MAYBE_UNUSED dtype_element_size(gd_dtype dtype, size_t *elem_size)
{
    if (elem_size == NULL) {
        return 1;
    }
    switch (dtype) {
    case GD_DTYPE_F16:
        *elem_size = 2U;
        return 0;
    case GD_DTYPE_F32:
        *elem_size = 4U;
        return 0;
    case GD_DTYPE_U8:
        *elem_size = 1U;
        return 0;
    case GD_DTYPE_I32:
        *elem_size = 4U;
        return 0;
    default:
        return 1;
    }
}

static int GD_ORACLE_MAYBE_UNUSED parse_dtype(const char *s, gd_dtype *dtype)
{
    if (s == NULL || dtype == NULL) {
        return 1;
    }
    if (strcmp(s, "f16") == 0) {
        *dtype = GD_DTYPE_F16;
        return 0;
    }
    if (strcmp(s, "f32") == 0) {
        *dtype = GD_DTYPE_F32;
        return 0;
    }
    if (strcmp(s, "u8") == 0) {
        *dtype = GD_DTYPE_U8;
        return 0;
    }
    if (strcmp(s, "i32") == 0) {
        *dtype = GD_DTYPE_I32;
        return 0;
    }
    return 1;
}

static int GD_ORACLE_MAYBE_UNUSED parse_dtype_with_size(const char *s,
                                                        gd_dtype *dtype,
                                                        size_t *elem_size)
{
    return parse_dtype(s, dtype) != 0 ? 1 : dtype_element_size(*dtype, elem_size);
}

static int GD_ORACLE_MAYBE_UNUSED check_status(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return 0;
    }
    fprintf(stderr,
            "%s failed: %s (%d), ctx=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return 1;
}

static size_t GD_ORACLE_MAYBE_UNUSED align_up(size_t value, size_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static void GD_ORACLE_MAYBE_UNUSED oracle_memory_config_slots(gd_memory_config *cfg,
                                                              size_t params_bytes,
                                                              size_t scratch_slot_bytes,
                                                              uint32_t scratch_slots,
                                                              uint32_t data_slots)
{
    memset(cfg, 0, sizeof(*cfg));
    cfg->params_bytes = params_bytes;
    cfg->state_bytes = 1024U * 1024U;
    cfg->scratch_slot_bytes = scratch_slot_bytes;
    cfg->data_slot_bytes = 1024U * 1024U;
    cfg->scratch_slots = scratch_slots;
    cfg->data_slots = data_slots;
    cfg->default_alignment = 256U;
}

static void GD_ORACLE_MAYBE_UNUSED oracle_memory_config(gd_memory_config *cfg,
                                                        size_t params_bytes,
                                                        size_t scratch_slot_bytes)
{
    oracle_memory_config_slots(cfg, params_bytes, scratch_slot_bytes, 2U, 2U);
}

static int GD_ORACLE_MAYBE_UNUSED mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (b != 0U && a > SIZE_MAX / b)) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED tensor_count(uint32_t rank, const int64_t *shape, size_t *out)
{
    uint32_t i;
    size_t count = 1U;
    if (out == NULL || (rank > 0U && shape == NULL)) {
        return 1;
    }
    for (i = 0U; i < rank; ++i) {
        if (shape[i] <= 0 || (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / count)) {
            return 1;
        }
        count *= (size_t)shape[i];
    }
    *out = count;
    return 0;
}

static int GD_ORACLE_MAYBE_UNUSED axis_shape(uint32_t rank,
                                             const int64_t *shape,
                                             int32_t axis,
                                             uint32_t *out_rank,
                                             int64_t *out_shape)
{
    uint32_t i;
    uint32_t j = 0U;
    int32_t normalized;
    if (shape == NULL || out_rank == NULL || out_shape == NULL) {
        return 1;
    }
    if (axis == AXIS_ALL) {
        *out_rank = 0U;
        return 0;
    }
    normalized = axis < 0 ? axis + (int32_t)rank : axis;
    if (normalized < 0 || normalized >= (int32_t)rank) {
        return 1;
    }
    *out_rank = rank - 1U;
    for (i = 0U; i < rank; ++i) {
        if (i == (uint32_t)normalized) {
            continue;
        }
        out_shape[j] = shape[i];
        j += 1U;
    }
    return 0;
}

#define CHECK(ctx, expr) \
    do { \
        if (check_status((ctx), (expr), #expr) != 0) { \
            goto fail; \
        } \
    } while (0)

#endif /* GD_ORACLE_RUNNER_COMMON_C */
