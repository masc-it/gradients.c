/*
 * gd_huber Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=huber
 *
 * Optional environment:
 *   GD_HUBER_PERF_PROFILE=smoke|all|<case-name>
 *   GD_HUBER_PERF_WARMUP=10
 *   GD_HUBER_PERF_ITERS=100
 */

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

#define GD_HUBER_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct huber_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
} huber_perf_case;

typedef struct huber_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad;
    size_t count;
    size_t elem_size;
    size_t data_bytes;
} huber_perf_model;

typedef bool (*huber_perf_run_fn)(huber_perf_model *model);

static double huber_perf_now_seconds(void)
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

static int huber_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t huber_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static const char *huber_perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static size_t huber_perf_dtype_size(gd_dtype dtype)
{
    if (dtype == GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool huber_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[HUBER][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define HUBER_PERF_REQUIRE_OK(ctx, expr)                                  \
    do {                                                                 \
        gd_status huber_perf_st__ = (expr);                              \
        if (!huber_perf_status_ok((ctx), huber_perf_st__, #expr)) {      \
            return false;                                                \
        }                                                                \
    } while (0)

static bool huber_perf_count(uint32_t rank,
                             const int64_t shape[GD_MAX_DIMS],
                             size_t *out_count)
{
    uint32_t dim;
    size_t count = 1U;
    if (rank == 0U || rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (dim = 0U; dim < rank; ++dim) {
        if (shape[dim] <= 0 || (uint64_t)shape[dim] > (uint64_t)(SIZE_MAX / count)) {
            return false;
        }
        count *= (size_t)shape[dim];
    }
    *out_count = count;
    return true;
}

static bool huber_perf_case_selected(const huber_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tail_1x513_f16") == 0 ||
               strcmp(pcase->name, "activation_256x1024_f16") == 0 ||
               strcmp(pcase->name, "regression_64x1024_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static void huber_perf_print_shape(const huber_perf_case *pcase)
{
    uint32_t dim;
    printf("[");
    for (dim = 0U; dim < pcase->rank; ++dim) {
        printf("%s%lld", dim == 0U ? "" : "x", (long long)pcase->shape[dim]);
    }
    printf("]");
}

static gd_memory_config huber_perf_config(size_t data_bytes)
{
    gd_memory_config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = huber_perf_align_up(data_bytes * 2U + 1024U * 1024U, 4096U);
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = huber_perf_align_up(data_bytes * 8U + 32U * 1024U * 1024U, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool huber_perf_setup(const huber_perf_case *pcase, huber_perf_model *model)
{
    gd_memory_config cfg;
    float grad_one = 1.0f;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->elem_size = huber_perf_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || !huber_perf_count(pcase->rank, pcase->shape, &model->count) ||
        model->count > SIZE_MAX / model->elem_size) {
        fprintf(stderr, "[HUBER][FAIL] invalid case shape/dtype: %s\n", pcase->name);
        return false;
    }
    model->data_bytes = model->count * model->elem_size;
    cfg = huber_perf_config(model->data_bytes);
    if (!huber_perf_status_ok(NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
        return false;
    }
    HUBER_PERF_REQUIRE_OK(model->ctx,
                          gd_tensor_rand_uniform(model->ctx,
                                                 GD_ARENA_PARAMS,
                                                 pcase->dtype,
                                                 gd_shape_make(pcase->rank, pcase->shape),
                                                 256U,
                                                 UINT64_C(0x48554245),
                                                 -2.0f,
                                                 2.0f,
                                                 &model->x));
    HUBER_PERF_REQUIRE_OK(model->ctx,
                          gd_tensor_rand_uniform(model->ctx,
                                                 GD_ARENA_PARAMS,
                                                 pcase->dtype,
                                                 gd_shape_make(pcase->rank, pcase->shape),
                                                 256U,
                                                 UINT64_C(0x48554246),
                                                 -2.0f,
                                                 2.0f,
                                                 &model->y));
    HUBER_PERF_REQUIRE_OK(model->ctx,
                          gd_tensor_empty(model->ctx,
                                          GD_ARENA_PARAMS,
                                          GD_DTYPE_F32,
                                          gd_shape_make(0U, NULL),
                                          256U,
                                          &model->grad));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_tensor_write(model->ctx, &model->grad, &grad_one, sizeof(grad_one)));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void huber_perf_destroy(huber_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool huber_perf_run_forward(huber_perf_model *model)
{
    gd_tensor loss;
    model->x.requires_grad = false;
    model->y.requires_grad = false;
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_huber(model->ctx, &model->x, &model->y, &loss));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool huber_perf_run_backward(huber_perf_model *model)
{
    gd_tensor dx;
    gd_tensor dy;
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_huber_backward(model->ctx, &model->x, &model->y, &model->grad, &dx, &dy));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool huber_perf_run_autograd(huber_perf_model *model)
{
    gd_tensor loss;
    model->x.requires_grad = true;
    model->y.requires_grad = true;
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_huber(model->ctx, &model->x, &model->y, &loss));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &loss, NULL));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    HUBER_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool huber_perf_measure(huber_perf_model *model,
                               const char *label,
                               huber_perf_run_fn fn,
                               int warmup,
                               int iters,
                               double logical_bytes)
{
    int i;
    double t0;
    double elapsed;
    double avg_ms;
    double gib_s;
    for (i = 0; i < warmup; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    t0 = huber_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = huber_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (logical_bytes / GD_HUBER_PERF_GIB) / (elapsed / (double)iters);
    printf("[HUBER][%s] avg_ms=%.4f logical_GiB/s=%.2f\n", label, avg_ms, gib_s);
    return true;
}

int main(void)
{
    static const huber_perf_case cases[] = {
        {.name = "tail_1x513_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {1, 513}},
        {.name = "activation_256x1024_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {256, 1024}},
        {.name = "image_32x3x224x224_f16", .dtype = GD_DTYPE_F16, .rank = 4U, .shape = {32, 3, 224, 224}},
        {.name = "activation_4096x4096_f16", .dtype = GD_DTYPE_F16, .rank = 2U, .shape = {4096, 4096}},
        {.name = "regression_64x1024_f32", .dtype = GD_DTYPE_F32, .rank = 2U, .shape = {64, 1024}},
        {.name = "regression_1024x4096_f32", .dtype = GD_DTYPE_F32, .rank = 2U, .shape = {1024, 4096}},
    };
    const char *profile = getenv("GD_HUBER_PERF_PROFILE");
    int warmup = huber_perf_env_int("GD_HUBER_PERF_WARMUP", 10, 0, 10000);
    int iters = huber_perf_env_int("GD_HUBER_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[HUBER] warmup=%d iters=%d profile=%s delta=1 reduction=mean\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        huber_perf_model model;
        double data_bytes;
        if (!huber_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!huber_perf_setup(&cases[i], &model)) {
            return 1;
        }
        data_bytes = (double)model.data_bytes;
        printf("[HUBER] case=%s dtype=%s shape=", cases[i].name, huber_perf_dtype_name(cases[i].dtype));
        huber_perf_print_shape(&cases[i]);
        printf(" elems=%zu bytes=%zu\n", model.count, model.data_bytes);
        if (!huber_perf_measure(&model, "fwd", huber_perf_run_forward, warmup, iters, 2.0 * data_bytes) ||
            !huber_perf_measure(&model, "bwd", huber_perf_run_backward, warmup, iters, 4.0 * data_bytes) ||
            !huber_perf_measure(&model, "autograd", huber_perf_run_autograd, warmup, iters, 6.0 * data_bytes)) {
            huber_perf_destroy(&model);
            return 1;
        }
        huber_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[HUBER][FAIL] no cases selected for GD_HUBER_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
