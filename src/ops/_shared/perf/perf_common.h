#ifndef GD_OPS_SHARED_PERF_COMMON_H
#define GD_OPS_SHARED_PERF_COMMON_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/gradients.h>

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

#define GD_PERF_GB (1000000000.0)
#define GD_PERF_GIB (1024.0 * 1024.0 * 1024.0)
#define GD_PERF_MIB (1024.0 * 1024.0)

#define GD_PERF_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

typedef bool (*gd_perf_run_fn)(void *user);

static inline double gd_perf_now_seconds(void)
{
#if defined(__APPLE__)
    static mach_timebase_info_data_t info;
    static double scale = 0.0;
    if (scale == 0.0) {
        if (mach_timebase_info(&info) != 0 || info.denom == 0U) {
            return 0.0;
        }
        scale = ((double)info.numer / (double)info.denom) * 1.0e-9;
    }
    return (double)mach_absolute_time() * scale;
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
#endif
}

static inline int gd_perf_env_int(const char *name, int fallback, int min_value, int max_value)
{
    const char *value = getenv(name);
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    parsed = strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        return fallback;
    }
    if (parsed < (long)min_value) {
        return min_value;
    }
    if (parsed > (long)max_value) {
        return max_value;
    }
    return (int)parsed;
}

static inline size_t gd_perf_align_up(size_t value, size_t alignment)
{
    if (alignment == 0U) {
        return value;
    }
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static inline bool gd_perf_checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

static inline bool gd_perf_checked_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || b > SIZE_MAX - a) {
        return false;
    }
    *out = a + b;
    return true;
}

static inline bool gd_perf_checked_mul_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL || a < 0 || b < 0 || (a != 0 && b > INT64_MAX / a)) {
        return false;
    }
    *out = a * b;
    return true;
}

static inline bool gd_perf_is_floating_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static inline bool gd_perf_shape_count(uint32_t rank,
                                       const int64_t shape[GD_MAX_DIMS],
                                       bool allow_scalar,
                                       size_t *out_count)
{
    uint32_t dim;
    size_t count = 1U;
    if (out_count == NULL || rank > GD_MAX_DIMS || (!allow_scalar && rank == 0U) ||
        (rank != 0U && shape == NULL)) {
        return false;
    }
    for (dim = 0U; dim < rank; ++dim) {
        if (shape[dim] <= 0 || !gd_perf_checked_mul_size(count, (size_t)shape[dim], &count)) {
            return false;
        }
    }
    *out_count = count;
    return true;
}

static inline bool gd_perf_axis_shape(uint32_t rank,
                                      const int64_t shape[GD_MAX_DIMS],
                                      int32_t axis,
                                      bool keepdims,
                                      uint32_t *out_rank,
                                      int64_t out_shape[GD_MAX_DIMS])
{
    int32_t normalized;
    uint32_t dim;
    uint32_t j = 0U;
    if (shape == NULL || out_rank == NULL || out_shape == NULL || rank == 0U || rank > GD_MAX_DIMS) {
        return false;
    }
    normalized = axis < 0 ? axis + (int32_t)rank : axis;
    if (normalized < 0 || normalized >= (int32_t)rank) {
        return false;
    }
    memset(out_shape, 0, GD_MAX_DIMS * sizeof(out_shape[0]));
    if (keepdims) {
        *out_rank = rank;
        for (dim = 0U; dim < rank; ++dim) {
            out_shape[dim] = dim == (uint32_t)normalized ? 1 : shape[dim];
        }
        return true;
    }
    *out_rank = rank - 1U;
    for (dim = 0U; dim < rank; ++dim) {
        if (dim == (uint32_t)normalized) {
            continue;
        }
        out_shape[j] = shape[dim];
        j += 1U;
    }
    return true;
}

static inline void gd_perf_print_shape(uint32_t rank,
                                       const int64_t shape[GD_MAX_DIMS],
                                       const char *separator)
{
    uint32_t dim;
    const char *sep = separator != NULL ? separator : "x";
    printf("[");
    for (dim = 0U; dim < rank; ++dim) {
        printf("%s%lld", dim == 0U ? "" : sep, (long long)shape[dim]);
    }
    printf("]");
}

static inline void gd_perf_format_shape(uint32_t rank,
                                        const int64_t shape[GD_MAX_DIMS],
                                        const char *separator,
                                        char *out,
                                        size_t out_size)
{
    size_t used = 0U;
    uint32_t dim;
    const char *sep = separator != NULL ? separator : "x";
    if (out == NULL || out_size == 0U) {
        return;
    }
    out[0] = '\0';
    for (dim = 0U; dim < rank && used + 1U < out_size; ++dim) {
        int written = snprintf(out + used,
                               out_size - used,
                               "%s%lld",
                               dim == 0U ? "" : sep,
                               (long long)shape[dim]);
        if (written < 0 || (size_t)written >= out_size - used) {
            out[out_size - 1U] = '\0';
            return;
        }
        used += (size_t)written;
    }
}

static inline bool gd_perf_case_selected(const char *case_name,
                                         const char *profile,
                                         bool smoke_case,
                                         bool default_all)
{
    if (case_name == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0') {
        return default_all || smoke_case;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    if (strcmp(profile, "smoke") == 0) {
        return smoke_case;
    }
    return strcmp(profile, case_name) == 0;
}

static inline bool gd_perf_status_ok(const char *tag, gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[%s][FAIL] %s -> %s (%d), context_error=%s\n",
            tag != NULL ? tag : "PERF",
            expr != NULL ? expr : "(unknown)",
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define GD_PERF_REQUIRE_OK(tag, ctx, expr)                             \
    do {                                                               \
        gd_status gd_perf_status__ = (expr);                           \
        if (!gd_perf_status_ok((tag), (ctx), gd_perf_status__, #expr)) { \
            return false;                                              \
        }                                                              \
    } while (0)

static inline gd_memory_config gd_perf_memory_config(size_t params_bytes,
                                                     size_t state_bytes,
                                                     size_t scratch_slot_bytes,
                                                     size_t data_slot_bytes,
                                                     uint32_t scratch_slots,
                                                     uint32_t data_slots,
                                                     size_t default_alignment)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = state_bytes;
    cfg.scratch_slot_bytes = scratch_slot_bytes;
    cfg.data_slot_bytes = data_slot_bytes;
    cfg.scratch_slots = scratch_slots;
    cfg.data_slots = data_slots;
    cfg.default_alignment = default_alignment;
    return cfg;
}

static inline bool gd_perf_measure_seconds(void *user,
                                           gd_perf_run_fn run,
                                           int warmup,
                                           int iters,
                                           double *out_seconds)
{
    double start;
    double elapsed;
    int i;
    if (run == NULL || out_seconds == NULL || warmup < 0 || iters <= 0) {
        return false;
    }
    for (i = 0; i < warmup; ++i) {
        if (!run(user)) {
            return false;
        }
    }
    start = gd_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!run(user)) {
            return false;
        }
    }
    elapsed = gd_perf_now_seconds() - start;
    if (elapsed <= 0.0) {
        return false;
    }
    *out_seconds = elapsed / (double)iters;
    return true;
}

static inline bool gd_perf_measure_labeled(void *user,
                                           gd_perf_run_fn run,
                                           int warmup,
                                           int iters,
                                           const char *tag,
                                           const char *label,
                                           double logical_bytes)
{
    double seconds;
    double avg_ms;
    double gib_s;
    if (!gd_perf_measure_seconds(user, run, warmup, iters, &seconds)) {
        return false;
    }
    avg_ms = seconds * 1.0e3;
    gib_s = (logical_bytes / GD_PERF_GIB) / seconds;
    printf("[%s][%s] avg_ms=%.4f logical_GiB/s=%.2f\n", tag, label, avg_ms, gib_s);
    return true;
}

static inline bool gd_perf_write_scalar_one(gd_context *ctx,
                                            gd_tensor *tensor,
                                            gd_dtype dtype,
                                            const char *tag)
{
    if (dtype == GD_DTYPE_F16) {
        const uint16_t one = 0x3C00U;
        GD_PERF_REQUIRE_OK(tag, ctx, gd_tensor_write(ctx, tensor, &one, sizeof(one)));
        return true;
    }
    if (dtype == GD_DTYPE_F32) {
        const float one = 1.0f;
        GD_PERF_REQUIRE_OK(tag, ctx, gd_tensor_write(ctx, tensor, &one, sizeof(one)));
        return true;
    }
    return false;
}

#endif /* GD_OPS_SHARED_PERF_COMMON_H */
