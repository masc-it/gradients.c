/*
 * gd_reshape performance probe.
 *
 * Reshape is intentionally metadata-only: forward and direct backward create
 * aliasing views and submit no Metal kernels. Autograd numbers include the
 * current gradient-slot zero/fill + accumulation traffic.
 *
 * Run from the repository root with:
 *   make op-perf OP=reshape
 *
 * Optional environment:
 *   GD_RESHAPE_PERF_PROFILE=smoke|all|<case-name>
 *   GD_RESHAPE_PERF_WARMUP=20
 *   GD_RESHAPE_PERF_ITERS=200
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

#define GD_RESHAPE_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct reshape_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t input_rank;
    uint32_t target_rank;
    int64_t input_shape[GD_MAX_DIMS];
    int64_t target_shape[GD_MAX_DIMS];
} reshape_perf_case;

typedef struct reshape_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad_out;
    gd_shape target;
    size_t logical_bytes;
    bool floating;
    uint64_t sink;
} reshape_perf_model;

typedef bool (*reshape_perf_run_fn)(reshape_perf_model *model);

static double reshape_perf_now_seconds(void)
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

static int reshape_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t reshape_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool reshape_perf_checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return false;
    }
    if (a != 0U && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool reshape_perf_checked_mul_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL || a < 0 || b < 0) {
        return false;
    }
    if (a != 0 && b > INT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool reshape_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[RESHAPE][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define RESHAPE_PERF_REQUIRE_OK(ctx, expr)                                  \
    do {                                                                    \
        gd_status reshape_perf_st__ = (expr);                               \
        if (!reshape_perf_status_ok((ctx), reshape_perf_st__, #expr)) {     \
            return false;                                                   \
        }                                                                   \
    } while (0)

static bool reshape_perf_is_floating(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool reshape_perf_case_selected(const reshape_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tokens_to_matrix_f16") == 0 ||
               strcmp(pcase->name, "heads_to_sequence_f16") == 0 ||
               strcmp(pcase->name, "mlp_features_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool reshape_perf_shape_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t d;
    size_t count = 1U;
    if (rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (d = 0U; d < rank; ++d) {
        if (shape[d] <= 0 || !reshape_perf_checked_mul_size(count, (size_t)shape[d], &count)) {
            return false;
        }
    }
    *out_count = count;
    return true;
}

static bool reshape_perf_resolve_target(const reshape_perf_case *pcase, int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t d;
    int32_t infer_index = -1;
    int64_t input_count = 1;
    int64_t known_target = 1;
    if (pcase == NULL || out_shape == NULL || pcase->input_rank > GD_MAX_DIMS ||
        pcase->target_rank > GD_MAX_DIMS) {
        return false;
    }
    for (d = 0U; d < pcase->input_rank; ++d) {
        if (pcase->input_shape[d] <= 0 ||
            !reshape_perf_checked_mul_i64(input_count, pcase->input_shape[d], &input_count)) {
            return false;
        }
    }
    for (d = 0U; d < GD_MAX_DIMS; ++d) {
        out_shape[d] = 0;
    }
    for (d = 0U; d < pcase->target_rank; ++d) {
        int64_t dim = pcase->target_shape[d];
        if (dim == -1) {
            if (infer_index >= 0) {
                return false;
            }
            infer_index = (int32_t)d;
            out_shape[d] = -1;
            continue;
        }
        if (dim <= 0 || !reshape_perf_checked_mul_i64(known_target, dim, &known_target)) {
            return false;
        }
        out_shape[d] = dim;
    }
    if (infer_index >= 0) {
        if (known_target <= 0 || input_count % known_target != 0) {
            return false;
        }
        out_shape[infer_index] = input_count / known_target;
    } else if (known_target != input_count) {
        return false;
    }
    return true;
}

static gd_memory_config reshape_perf_config(size_t logical_bytes, bool floating)
{
    gd_memory_config cfg;
    size_t params_bytes = logical_bytes * (floating ? 2U : 1U);
    size_t scratch_bytes = logical_bytes * (floating ? 10U : 2U) + 64U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = reshape_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = reshape_perf_align_up(scratch_bytes, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool reshape_perf_setup(const reshape_perf_case *pcase, reshape_perf_model *model)
{
    gd_memory_config cfg;
    size_t count;
    size_t elem_size;
    int64_t resolved_target[GD_MAX_DIMS];
    uint32_t d;
    if (pcase == NULL || model == NULL || pcase->input_rank > GD_MAX_DIMS ||
        pcase->target_rank > GD_MAX_DIMS) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    elem_size = gd_dtype_size(pcase->dtype);
    if (elem_size == 0U || !reshape_perf_shape_count(pcase->input_rank, pcase->input_shape, &count) ||
        !reshape_perf_checked_mul_size(count, elem_size, &model->logical_bytes) ||
        !reshape_perf_resolve_target(pcase, resolved_target)) {
        return false;
    }
    model->floating = reshape_perf_is_floating(pcase->dtype);
    model->target = gd_shape_make(pcase->target_rank, pcase->target_shape);
    cfg = reshape_perf_config(model->logical_bytes, model->floating);
    if (!reshape_perf_status_ok(NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
        return false;
    }
    if (model->floating) {
        RESHAPE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(pcase->input_rank, pcase->input_shape),
                                                       256U,
                                                       UINT64_C(0x5e5a9e00),
                                                       -0.25f,
                                                       0.25f,
                                                       &model->x));
        RESHAPE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(pcase->target_rank, resolved_target),
                                                       256U,
                                                       UINT64_C(0x5e5a9eff),
                                                       -0.25f,
                                                       0.25f,
                                                       &model->grad_out));
    } else {
        RESHAPE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_zeros(model->ctx,
                                                GD_ARENA_PARAMS,
                                                pcase->dtype,
                                                gd_shape_make(pcase->input_rank, pcase->input_shape),
                                                256U,
                                                &model->x));
    }
    for (d = 0U; d < pcase->target_rank; ++d) {
        if (model->target.dims[d] != pcase->target_shape[d]) {
            return false;
        }
    }
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void reshape_perf_destroy(reshape_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool reshape_perf_run_forward(reshape_perf_model *model)
{
    gd_tensor out;
    model->x.requires_grad = false;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_reshape(model->ctx, &model->x, model->target, &out));
    model->sink += out.id + (uint64_t)out.rank;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reshape_perf_run_backward(reshape_perf_model *model)
{
    gd_tensor grad_x;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = false;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_reshape_backward(model->ctx, &model->x, &model->grad_out, &grad_x));
    model->sink += grad_x.id + (uint64_t)grad_x.rank;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reshape_perf_run_autograd(reshape_perf_model *model)
{
    gd_tensor out;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = true;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_reshape(model->ctx, &model->x, model->target, &out));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad_out));
    model->sink += out.id + (uint64_t)out.rank;
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RESHAPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reshape_perf_measure(reshape_perf_model *model,
                                 const char *label,
                                 reshape_perf_run_fn fn,
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
    t0 = reshape_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = reshape_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (logical_bytes / GD_RESHAPE_PERF_GIB) / (elapsed / (double)iters);
    printf("[RESHAPE][%s] avg_ms=%.6f avg_us=%.3f logical_GiB/s=%.2f\n",
           label,
           avg_ms,
           avg_ms * 1000.0,
           gib_s);
    return true;
}

static void reshape_perf_print_shape(uint32_t rank, const int64_t shape[GD_MAX_DIMS])
{
    uint32_t d;
    printf("[");
    for (d = 0U; d < rank; ++d) {
        printf("%s%lld", d == 0U ? "" : ",", (long long)shape[d]);
    }
    printf("]");
}

static void reshape_perf_print_case(const reshape_perf_case *pcase, const reshape_perf_model *model)
{
    printf("[RESHAPE] case=%s dtype=%s input=",
           pcase->name,
           gd_dtype_name(pcase->dtype));
    reshape_perf_print_shape(pcase->input_rank, pcase->input_shape);
    printf(" target=");
    reshape_perf_print_shape(pcase->target_rank, pcase->target_shape);
    printf(" logical_bytes=%zu materialized_bytes=0 floating=%d\n",
           model->logical_bytes,
           model->floating ? 1 : 0);
}

int main(void)
{
    static const reshape_perf_case cases[] = {
        {.name = "tokens_to_matrix_f16",
         .dtype = GD_DTYPE_F16,
         .input_rank = 3U,
         .target_rank = 2U,
         .input_shape = {4, 1024, 1024},
         .target_shape = {-1, 1024}},
        {.name = "heads_to_sequence_f16",
         .dtype = GD_DTYPE_F16,
         .input_rank = 4U,
         .target_rank = 3U,
         .input_shape = {2, 16, 2048, 64},
         .target_shape = {2, 2048, -1}},
        {.name = "mlp_features_f32",
         .dtype = GD_DTYPE_F32,
         .input_rank = 2U,
         .target_rank = 3U,
         .input_shape = {2048, 2048},
         .target_shape = {2048, 8, 256}},
        {.name = "tokens_to_matrix_large_f16",
         .dtype = GD_DTYPE_F16,
         .input_rank = 3U,
         .target_rank = 2U,
         .input_shape = {8, 2048, 1024},
         .target_shape = {-1, 1024}},
        {.name = "patches_to_sequence_f16",
         .dtype = GD_DTYPE_F16,
         .input_rank = 4U,
         .target_rank = 3U,
         .input_shape = {32, 14, 14, 768},
         .target_shape = {32, -1, 768}},
        {.name = "token_ids_flat_i32",
         .dtype = GD_DTYPE_I32,
         .input_rank = 2U,
         .target_rank = 1U,
         .input_shape = {64, 2048},
         .target_shape = {-1}},
    };
    const char *profile = getenv("GD_RESHAPE_PERF_PROFILE");
    int warmup = reshape_perf_env_int("GD_RESHAPE_PERF_WARMUP", 20, 0, 10000);
    int iters = reshape_perf_env_int("GD_RESHAPE_PERF_ITERS", 200, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[RESHAPE] warmup=%d iters=%d profile=%s note=metadata-only-view\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        reshape_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!reshape_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!reshape_perf_setup(&cases[i], &model)) {
            return 1;
        }
        reshape_perf_print_case(&cases[i], &model);
        fwd_bytes = (double)model.logical_bytes;
        bwd_bytes = (double)model.logical_bytes;
        autograd_bytes = 4.0 * (double)model.logical_bytes;
        if (!reshape_perf_measure(&model, "fwd", reshape_perf_run_forward, warmup, iters, fwd_bytes)) {
            reshape_perf_destroy(&model);
            return 1;
        }
        if (model.floating) {
            if (!reshape_perf_measure(&model, "bwd", reshape_perf_run_backward, warmup, iters, bwd_bytes) ||
                !reshape_perf_measure(&model,
                                      "autograd",
                                      reshape_perf_run_autograd,
                                      warmup,
                                      iters,
                                      autograd_bytes)) {
                reshape_perf_destroy(&model);
                return 1;
            }
        }
        if (model.sink == 0U) {
            fprintf(stderr, "[RESHAPE][FAIL] timing sink was not updated\n");
            reshape_perf_destroy(&model);
            return 1;
        }
        reshape_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[RESHAPE][FAIL] no cases selected for GD_RESHAPE_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
