/*
 * gd_permute Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=permute
 *
 * Optional environment:
 *   GD_PERMUTE_PERF_PROFILE=smoke|all|<case-name>
 *   GD_PERMUTE_PERF_WARMUP=10
 *   GD_PERMUTE_PERF_ITERS=100
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

#define GD_PERMUTE_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct permute_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    int32_t axes[GD_MAX_DIMS];
} permute_perf_case;

typedef struct permute_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad_out;
    int32_t axes[GD_MAX_DIMS];
    uint32_t rank;
    size_t logical_bytes;
    bool floating;
    uint64_t sink;
} permute_perf_model;

typedef bool (*permute_perf_run_fn)(permute_perf_model *model);

static double permute_perf_now_seconds(void)
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

static int permute_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t permute_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool permute_perf_checked_mul_size(size_t a, size_t b, size_t *out)
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

static bool permute_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[PERMUTE][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define PERMUTE_PERF_REQUIRE_OK(ctx, expr)                                  \
    do {                                                                    \
        gd_status permute_perf_st__ = (expr);                               \
        if (!permute_perf_status_ok((ctx), permute_perf_st__, #expr)) {     \
            return false;                                                   \
        }                                                                   \
    } while (0)

static bool permute_perf_is_floating(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool permute_perf_case_selected(const permute_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "qkv_bthd_to_bhtd_f16") == 0 ||
               strcmp(pcase->name, "vision_nhwc_to_nchw_f16") == 0 ||
               strcmp(pcase->name, "mlp_tiles_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool permute_perf_shape_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t d;
    size_t count = 1U;
    if (rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (d = 0U; d < rank; ++d) {
        if (shape[d] <= 0 || !permute_perf_checked_mul_size(count, (size_t)shape[d], &count)) {
            return false;
        }
    }
    *out_count = count;
    return true;
}

static bool permute_perf_resolve_output_shape(const permute_perf_case *pcase,
                                              int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t d;
    uint32_t seen = 0U;
    if (pcase == NULL || out_shape == NULL || pcase->rank > GD_MAX_DIMS) {
        return false;
    }
    for (d = 0U; d < GD_MAX_DIMS; ++d) {
        out_shape[d] = 0;
    }
    for (d = 0U; d < pcase->rank; ++d) {
        int32_t axis = pcase->axes[d];
        uint32_t bit;
        if (axis < 0) {
            axis += (int32_t)pcase->rank;
        }
        if (axis < 0 || axis >= (int32_t)pcase->rank) {
            return false;
        }
        bit = 1U << (uint32_t)axis;
        if ((seen & bit) != 0U) {
            return false;
        }
        seen |= bit;
        out_shape[d] = pcase->shape[(uint32_t)axis];
    }
    return true;
}

static gd_memory_config permute_perf_config(size_t logical_bytes, bool floating)
{
    gd_memory_config cfg;
    size_t params_bytes = logical_bytes * (floating ? 2U : 1U);
    size_t scratch_bytes = logical_bytes * (floating ? 14U : 4U) + 64U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = permute_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = permute_perf_align_up(scratch_bytes, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 4U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool permute_perf_setup(const permute_perf_case *pcase, permute_perf_model *model)
{
    gd_memory_config cfg;
    int64_t out_shape[GD_MAX_DIMS];
    size_t count;
    size_t elem_size;
    uint32_t d;
    if (pcase == NULL || model == NULL || pcase->rank > GD_MAX_DIMS) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    elem_size = gd_dtype_size(pcase->dtype);
    if (elem_size == 0U || !permute_perf_shape_count(pcase->rank, pcase->shape, &count) ||
        !permute_perf_checked_mul_size(count, elem_size, &model->logical_bytes) ||
        !permute_perf_resolve_output_shape(pcase, out_shape)) {
        return false;
    }
    if (count > UINT32_MAX) {
        return false;
    }
    model->rank = pcase->rank;
    model->floating = permute_perf_is_floating(pcase->dtype);
    for (d = 0U; d < pcase->rank; ++d) {
        model->axes[d] = pcase->axes[d];
    }
    cfg = permute_perf_config(model->logical_bytes, model->floating);
    if (!permute_perf_status_ok(NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
        return false;
    }
    if (model->floating) {
        PERMUTE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(pcase->rank, pcase->shape),
                                                       256U,
                                                       UINT64_C(0x9e377900),
                                                       -0.25f,
                                                       0.25f,
                                                       &model->x));
        PERMUTE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(pcase->rank, out_shape),
                                                       256U,
                                                       UINT64_C(0x9e3779ff),
                                                       -0.25f,
                                                       0.25f,
                                                       &model->grad_out));
    } else {
        PERMUTE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_zeros(model->ctx,
                                                GD_ARENA_PARAMS,
                                                pcase->dtype,
                                                gd_shape_make(pcase->rank, pcase->shape),
                                                256U,
                                                &model->x));
        PERMUTE_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_zeros(model->ctx,
                                                GD_ARENA_PARAMS,
                                                pcase->dtype,
                                                gd_shape_make(pcase->rank, out_shape),
                                                256U,
                                                &model->grad_out));
    }
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void permute_perf_destroy(permute_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool permute_perf_run_forward(permute_perf_model *model)
{
    gd_tensor out;
    model->x.requires_grad = false;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_permute(model->ctx, &model->x, model->axes, model->rank, &out));
    model->sink += out.id + (uint64_t)out.rank;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool permute_perf_run_backward(permute_perf_model *model)
{
    gd_tensor grad_x;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = false;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    PERMUTE_PERF_REQUIRE_OK(model->ctx,
                            gd_permute_backward(model->ctx,
                                                &model->x,
                                                &model->grad_out,
                                                model->axes,
                                                model->rank,
                                                &grad_x));
    model->sink += grad_x.id + (uint64_t)grad_x.rank;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool permute_perf_run_autograd(permute_perf_model *model)
{
    gd_tensor out;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = true;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_permute(model->ctx, &model->x, model->axes, model->rank, &out));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad_out));
    model->sink += out.id + (uint64_t)out.rank;
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    PERMUTE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool permute_perf_measure(permute_perf_model *model,
                                 const char *label,
                                 permute_perf_run_fn fn,
                                 int warmup,
                                 int iters,
                                 double traffic_bytes)
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
    t0 = permute_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = permute_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (traffic_bytes / GD_PERMUTE_PERF_GIB) / (elapsed / (double)iters);
    printf("[PERMUTE][%s] avg_ms=%.6f avg_us=%.3f effective_GiB/s=%.2f\n",
           label,
           avg_ms,
           avg_ms * 1000.0,
           gib_s);
    return true;
}

static void permute_perf_print_shape(uint32_t rank, const int64_t shape[GD_MAX_DIMS])
{
    uint32_t d;
    printf("[");
    for (d = 0U; d < rank; ++d) {
        printf("%s%lld", d == 0U ? "" : ",", (long long)shape[d]);
    }
    printf("]");
}

static void permute_perf_print_axes(uint32_t rank, const int32_t axes[GD_MAX_DIMS])
{
    uint32_t d;
    printf("[");
    for (d = 0U; d < rank; ++d) {
        printf("%s%d", d == 0U ? "" : ",", (int)axes[d]);
    }
    printf("]");
}

static void permute_perf_print_case(const permute_perf_case *pcase, const permute_perf_model *model)
{
    int64_t out_shape[GD_MAX_DIMS];
    (void)permute_perf_resolve_output_shape(pcase, out_shape);
    printf("[PERMUTE] case=%s dtype=%s input=", pcase->name, gd_dtype_name(pcase->dtype));
    permute_perf_print_shape(pcase->rank, pcase->shape);
    printf(" axes=");
    permute_perf_print_axes(pcase->rank, pcase->axes);
    printf(" output=");
    permute_perf_print_shape(pcase->rank, out_shape);
    printf(" logical_bytes=%zu fwd_materialized_bytes=%zu floating=%d\n",
           model->logical_bytes,
           model->logical_bytes * 2U,
           model->floating ? 1 : 0);
}

int main(void)
{
    static const permute_perf_case cases[] = {
        {.name = "qkv_bthd_to_bhtd_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 4U,
         .shape = {4, 2048, 16, 64},
         .axes = {0, 2, 1, 3}},
        {.name = "vision_nhwc_to_nchw_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 4U,
         .shape = {32, 224, 224, 3},
         .axes = {0, 3, 1, 2}},
        {.name = "mlp_tiles_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 3U,
         .shape = {2048, 8, 256},
         .axes = {1, 0, 2}},
        {.name = "qkv_packed_to_heads_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 5U,
         .shape = {3, 2, 2048, 16, 64},
         .axes = {1, 3, 2, 0, 4}},
        {.name = "logits_btv_to_bvt_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .shape = {2, 512, 8192},
         .axes = {0, 2, 1}},
        {.name = "token_ids_bt_to_tb_i32",
         .dtype = GD_DTYPE_I32,
         .rank = 2U,
         .shape = {64, 2048},
         .axes = {1, 0}},
        {.name = "image_bytes_hwc_to_chw_u8",
         .dtype = GD_DTYPE_U8,
         .rank = 3U,
         .shape = {1024, 1024, 3},
         .axes = {2, 0, 1}},
    };
    const char *profile = getenv("GD_PERMUTE_PERF_PROFILE");
    int warmup = permute_perf_env_int("GD_PERMUTE_PERF_WARMUP", 10, 0, 10000);
    int iters = permute_perf_env_int("GD_PERMUTE_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[PERMUTE] warmup=%d iters=%d profile=%s note=materialized-contiguous-output\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        permute_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!permute_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!permute_perf_setup(&cases[i], &model)) {
            return 1;
        }
        permute_perf_print_case(&cases[i], &model);
        fwd_bytes = 2.0 * (double)model.logical_bytes;
        bwd_bytes = 2.0 * (double)model.logical_bytes;
        autograd_bytes = 8.0 * (double)model.logical_bytes;
        if (!permute_perf_measure(&model, "fwd", permute_perf_run_forward, warmup, iters, fwd_bytes)) {
            permute_perf_destroy(&model);
            return 1;
        }
        if (model.floating) {
            if (!permute_perf_measure(&model, "bwd", permute_perf_run_backward, warmup, iters, bwd_bytes) ||
                !permute_perf_measure(&model,
                                      "autograd",
                                      permute_perf_run_autograd,
                                      warmup,
                                      iters,
                                      autograd_bytes)) {
                permute_perf_destroy(&model);
                return 1;
            }
        }
        if (model.sink == 0U) {
            fprintf(stderr, "[PERMUTE][FAIL] timing sink was not updated\n");
            permute_perf_destroy(&model);
            return 1;
        }
        permute_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[PERMUTE][FAIL] no cases selected for GD_PERMUTE_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
