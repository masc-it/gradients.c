/*
 * gd_split Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=split
 *
 * Optional environment:
 *   GD_SPLIT_PERF_PROFILE=smoke|all|<case-name>
 *   GD_SPLIT_PERF_WARMUP=10
 *   GD_SPLIT_PERF_ITERS=100
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

#define GD_SPLIT_PERF_MAX_OUTPUTS 4U
#define GD_SPLIT_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct split_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int32_t axis;
    uint32_t n_outputs;
    int64_t shape[GD_MAX_DIMS];
    int64_t sizes[GD_SPLIT_PERF_MAX_OUTPUTS];
} split_perf_case;

typedef struct split_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad_outputs[GD_SPLIT_PERF_MAX_OUTPUTS];
    const gd_tensor *grad_output_ptrs[GD_SPLIT_PERF_MAX_OUTPUTS];
    int64_t sizes[GD_SPLIT_PERF_MAX_OUTPUTS];
    uint32_t n_outputs;
    int32_t axis;
    size_t input_bytes;
    bool floating;
} split_perf_model;

typedef bool (*split_perf_run_fn)(split_perf_model *model);

static double split_perf_now_seconds(void)
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

static int split_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t split_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool split_perf_checked_mul(size_t a, size_t b, size_t *out)
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

static bool split_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[SPLIT][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define SPLIT_PERF_REQUIRE_OK(ctx, expr)                                  \
    do {                                                                  \
        gd_status split_perf_st__ = (expr);                               \
        if (!split_perf_status_ok((ctx), split_perf_st__, #expr)) {       \
            return false;                                                 \
        }                                                                 \
    } while (0)

static bool split_perf_is_floating(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool split_perf_case_selected(const split_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "qkv_bth3d_axis2_f16") == 0 ||
               strcmp(pcase->name, "qkv_flat_last_f16") == 0 ||
               strcmp(pcase->name, "mlp_branches_last_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool split_perf_shape_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t d;
    size_t count = 1U;
    if (rank == 0U || rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (d = 0U; d < rank; ++d) {
        if (shape[d] <= 0 || !split_perf_checked_mul(count, (size_t)shape[d], &count)) {
            return false;
        }
    }
    *out_count = count;
    return true;
}

static gd_memory_config split_perf_config(size_t input_bytes, bool floating)
{
    gd_memory_config cfg;
    size_t params_bytes = input_bytes * (floating ? 2U : 1U);
    size_t scratch_bytes = input_bytes * (floating ? 10U : 3U) + 64U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = split_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = split_perf_align_up(scratch_bytes, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool split_perf_setup(const split_perf_case *pcase, split_perf_model *model)
{
    gd_memory_config cfg;
    size_t elem_size;
    size_t input_elems = 0U;
    int32_t normalized_axis;
    int64_t axis_sum = 0;
    uint32_t i;
    if (pcase == NULL || model == NULL || pcase->rank == 0U || pcase->rank > GD_MAX_DIMS ||
        pcase->n_outputs == 0U || pcase->n_outputs > GD_SPLIT_PERF_MAX_OUTPUTS) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    elem_size = gd_dtype_size(pcase->dtype);
    if (elem_size == 0U) {
        return false;
    }
    normalized_axis = pcase->axis < 0 ? (int32_t)pcase->rank + pcase->axis : pcase->axis;
    if (normalized_axis < 0 || normalized_axis >= (int32_t)pcase->rank) {
        return false;
    }
    for (i = 0U; i < pcase->n_outputs; ++i) {
        if (pcase->sizes[i] <= 0) {
            return false;
        }
        axis_sum += pcase->sizes[i];
        model->sizes[i] = pcase->sizes[i];
    }
    if (axis_sum != pcase->shape[(uint32_t)normalized_axis]) {
        return false;
    }
    if (!split_perf_shape_count(pcase->rank, pcase->shape, &input_elems) ||
        !split_perf_checked_mul(input_elems, elem_size, &model->input_bytes)) {
        return false;
    }
    model->n_outputs = pcase->n_outputs;
    model->axis = pcase->axis;
    model->floating = split_perf_is_floating(pcase->dtype);
    cfg = split_perf_config(model->input_bytes, model->floating);
    if (!split_perf_status_ok(NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
        return false;
    }
    if (model->floating) {
        SPLIT_PERF_REQUIRE_OK(model->ctx,
                              gd_tensor_rand_uniform(model->ctx,
                                                     GD_ARENA_PARAMS,
                                                     pcase->dtype,
                                                     gd_shape_make(pcase->rank, pcase->shape),
                                                     256U,
                                                     UINT64_C(0x51717000),
                                                     -0.25f,
                                                     0.25f,
                                                     &model->x));
    } else {
        SPLIT_PERF_REQUIRE_OK(model->ctx,
                              gd_tensor_zeros(model->ctx,
                                              GD_ARENA_PARAMS,
                                              pcase->dtype,
                                              gd_shape_make(pcase->rank, pcase->shape),
                                              256U,
                                              &model->x));
    }
    if (model->floating) {
        for (i = 0U; i < pcase->n_outputs; ++i) {
            int64_t grad_shape[GD_MAX_DIMS];
            uint32_t d;
            for (d = 0U; d < pcase->rank; ++d) {
                grad_shape[d] = pcase->shape[d];
            }
            grad_shape[(uint32_t)normalized_axis] = pcase->sizes[i];
            SPLIT_PERF_REQUIRE_OK(model->ctx,
                                  gd_tensor_rand_uniform(model->ctx,
                                                         GD_ARENA_PARAMS,
                                                         pcase->dtype,
                                                         gd_shape_make(pcase->rank, grad_shape),
                                                         256U,
                                                         UINT64_C(0x51717080) + (uint64_t)i,
                                                         -0.25f,
                                                         0.25f,
                                                         &model->grad_outputs[i]));
            model->grad_output_ptrs[i] = &model->grad_outputs[i];
        }
    }
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void split_perf_destroy(split_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool split_perf_run_forward(split_perf_model *model)
{
    gd_tensor outputs[GD_SPLIT_PERF_MAX_OUTPUTS];
    model->x.requires_grad = false;
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    SPLIT_PERF_REQUIRE_OK(model->ctx,
                          gd_split(model->ctx,
                                   &model->x,
                                   model->sizes,
                                   model->n_outputs,
                                   model->axis,
                                   outputs));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool split_perf_run_backward(split_perf_model *model)
{
    gd_tensor dx;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = false;
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    SPLIT_PERF_REQUIRE_OK(model->ctx,
                          gd_split_backward(model->ctx,
                                            &model->x,
                                            model->grad_output_ptrs,
                                            model->sizes,
                                            model->n_outputs,
                                            model->axis,
                                            &dx));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool split_perf_run_autograd(split_perf_model *model)
{
    gd_tensor outputs[GD_SPLIT_PERF_MAX_OUTPUTS];
    const gd_tensor *output_ptrs[GD_SPLIT_PERF_MAX_OUTPUTS];
    uint32_t i;
    if (!model->floating) {
        return true;
    }
    model->x.requires_grad = true;
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    SPLIT_PERF_REQUIRE_OK(model->ctx,
                          gd_split(model->ctx,
                                   &model->x,
                                   model->sizes,
                                   model->n_outputs,
                                   model->axis,
                                   outputs));
    for (i = 0U; i < model->n_outputs; ++i) {
        output_ptrs[i] = &outputs[i];
    }
    SPLIT_PERF_REQUIRE_OK(model->ctx,
                          gd_backward_many(model->ctx,
                                           model->n_outputs,
                                           output_ptrs,
                                           model->grad_output_ptrs));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SPLIT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool split_perf_measure(split_perf_model *model,
                               const char *label,
                               split_perf_run_fn fn,
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
    t0 = split_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = split_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (logical_bytes / GD_SPLIT_PERF_GIB) / (elapsed / (double)iters);
    printf("[SPLIT][%s] avg_ms=%.4f effective_GiB/s=%.2f\n", label, avg_ms, gib_s);
    return true;
}

static void split_perf_print_case(const split_perf_case *pcase, const split_perf_model *model)
{
    uint32_t i;
    printf("[SPLIT] case=%s dtype=%s input=[",
           pcase->name,
           gd_dtype_name(pcase->dtype));
    for (i = 0U; i < pcase->rank; ++i) {
        printf("%s%lld", i == 0U ? "" : ",", (long long)pcase->shape[i]);
    }
    printf("] axis=%d sizes=[", pcase->axis);
    for (i = 0U; i < pcase->n_outputs; ++i) {
        printf("%s%lld", i == 0U ? "" : ",", (long long)pcase->sizes[i]);
    }
    printf("] logical_bytes=%zu fwd_materialized_bytes=%zu floating=%d\n",
           model->input_bytes,
           model->input_bytes * 2U,
           model->floating ? 1 : 0);
}

int main(void)
{
    static const split_perf_case cases[] = {
        {.name = "qkv_bth3d_axis2_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 5U,
         .axis = 2,
         .n_outputs = 3U,
         .shape = {4, 2048, 3, 16, 64},
         .sizes = {1, 1, 1}},
        {.name = "qkv_flat_last_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .axis = -1,
         .n_outputs = 3U,
         .shape = {4, 1024, 3072},
         .sizes = {1024, 1024, 1024}},
        {.name = "mlp_branches_last_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = -1,
         .n_outputs = 2U,
         .shape = {4096, 4096},
         .sizes = {1024, 3072}},
        {.name = "vision_qkv_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 5U,
         .axis = 1,
         .n_outputs = 3U,
         .shape = {1, 3, 196, 12, 64},
         .sizes = {1, 1, 1}},
        {.name = "token_ids_last_i32",
         .dtype = GD_DTYPE_I32,
         .rank = 3U,
         .axis = -1,
         .n_outputs = 3U,
         .shape = {64, 2048, 3},
         .sizes = {1, 1, 1}},
        {.name = "image_rgb_alpha_u8",
         .dtype = GD_DTYPE_U8,
         .rank = 3U,
         .axis = -1,
         .n_outputs = 2U,
         .shape = {1024, 1024, 4},
         .sizes = {3, 1}},
    };
    const char *profile = getenv("GD_SPLIT_PERF_PROFILE");
    int warmup = split_perf_env_int("GD_SPLIT_PERF_WARMUP", 10, 0, 10000);
    int iters = split_perf_env_int("GD_SPLIT_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[SPLIT] warmup=%d iters=%d profile=%s note=materialized-contiguous-outputs\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        split_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!split_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!split_perf_setup(&cases[i], &model)) {
            return 1;
        }
        split_perf_print_case(&cases[i], &model);
        fwd_bytes = 2.0 * (double)model.input_bytes;
        bwd_bytes = 2.0 * (double)model.input_bytes;
        autograd_bytes = 4.0 * (double)model.input_bytes;
        if (!split_perf_measure(&model, "fwd", split_perf_run_forward, warmup, iters, fwd_bytes)) {
            split_perf_destroy(&model);
            return 1;
        }
        if (model.floating) {
            if (!split_perf_measure(&model, "bwd", split_perf_run_backward, warmup, iters, bwd_bytes) ||
                !split_perf_measure(&model,
                                    "autograd",
                                    split_perf_run_autograd,
                                    warmup,
                                    iters,
                                    autograd_bytes)) {
                split_perf_destroy(&model);
                return 1;
            }
        }
        split_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[SPLIT][FAIL] no cases selected for GD_SPLIT_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
