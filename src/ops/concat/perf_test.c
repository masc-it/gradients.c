/*
 * gd_concat Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=concat
 *
 * Optional environment:
 *   GD_CONCAT_PERF_PROFILE=smoke|all|<case-name>
 *   GD_CONCAT_PERF_WARMUP=10
 *   GD_CONCAT_PERF_ITERS=100
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

#define GD_CONCAT_PERF_MAX_INPUTS 4U
#define GD_CONCAT_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct concat_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int32_t axis;
    uint32_t n_inputs;
    int64_t base_shape[GD_MAX_DIMS];
    int64_t axis_lens[GD_CONCAT_PERF_MAX_INPUTS];
} concat_perf_case;

typedef struct concat_perf_model {
    gd_context *ctx;
    gd_tensor inputs[GD_CONCAT_PERF_MAX_INPUTS];
    const gd_tensor *input_ptrs[GD_CONCAT_PERF_MAX_INPUTS];
    gd_tensor grad_out;
    uint32_t n_inputs;
    int32_t axis;
    size_t output_bytes;
    size_t input_bytes[GD_CONCAT_PERF_MAX_INPUTS];
    bool floating;
} concat_perf_model;

typedef bool (*concat_perf_run_fn)(concat_perf_model *model);

static double concat_perf_now_seconds(void)
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

static int concat_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t concat_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool concat_perf_checked_mul(size_t a, size_t b, size_t *out)
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

static bool concat_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[CONCAT][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define CONCAT_PERF_REQUIRE_OK(ctx, expr)                                  \
    do {                                                                   \
        gd_status concat_perf_st__ = (expr);                               \
        if (!concat_perf_status_ok((ctx), concat_perf_st__, #expr)) {      \
            return false;                                                  \
        }                                                                  \
    } while (0)

static bool concat_perf_is_floating(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool concat_perf_case_selected(const concat_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tokens_axis0_f16") == 0 ||
               strcmp(pcase->name, "features_axis1_f16") == 0 ||
               strcmp(pcase->name, "features_axis1_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool concat_perf_shape_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t d;
    size_t count = 1U;
    if (rank == 0U || rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (d = 0U; d < rank; ++d) {
        if (shape[d] <= 0 || !concat_perf_checked_mul(count, (size_t)shape[d], &count)) {
            return false;
        }
    }
    *out_count = count;
    return true;
}

static gd_memory_config concat_perf_config(size_t output_bytes, bool floating)
{
    gd_memory_config cfg;
    size_t params_bytes = output_bytes * (floating ? 2U : 1U);
    size_t scratch_bytes = output_bytes * (floating ? 8U : 2U) + 64U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = concat_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = concat_perf_align_up(scratch_bytes, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool concat_perf_setup(const concat_perf_case *pcase, concat_perf_model *model)
{
    gd_memory_config cfg;
    size_t elem_size;
    size_t output_elems = 0U;
    uint32_t normalized_axis;
    uint32_t i;
    int64_t output_shape[GD_MAX_DIMS];
    if (pcase == NULL || model == NULL || pcase->rank == 0U || pcase->rank > GD_MAX_DIMS ||
        pcase->n_inputs == 0U || pcase->n_inputs > GD_CONCAT_PERF_MAX_INPUTS) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    elem_size = gd_dtype_size(pcase->dtype);
    if (elem_size == 0U) {
        return false;
    }
    normalized_axis = pcase->axis < 0 ? (uint32_t)((int32_t)pcase->rank + pcase->axis) :
                                        (uint32_t)pcase->axis;
    if (normalized_axis >= pcase->rank) {
        return false;
    }
    for (i = 0U; i < pcase->rank; ++i) {
        output_shape[i] = pcase->base_shape[i];
    }
    output_shape[normalized_axis] = 0;
    for (i = 0U; i < pcase->n_inputs; ++i) {
        if (pcase->axis_lens[i] <= 0 || output_shape[normalized_axis] > INT64_MAX - pcase->axis_lens[i]) {
            return false;
        }
        output_shape[normalized_axis] += pcase->axis_lens[i];
    }
    if (!concat_perf_shape_count(pcase->rank, output_shape, &output_elems) ||
        !concat_perf_checked_mul(output_elems, elem_size, &model->output_bytes)) {
        return false;
    }
    model->n_inputs = pcase->n_inputs;
    model->axis = pcase->axis;
    model->floating = concat_perf_is_floating(pcase->dtype);
    cfg = concat_perf_config(model->output_bytes, model->floating);
    if (!concat_perf_status_ok(NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
        return false;
    }
    for (i = 0U; i < pcase->n_inputs; ++i) {
        int64_t shape[GD_MAX_DIMS];
        size_t count;
        uint32_t d;
        for (d = 0U; d < pcase->rank; ++d) {
            shape[d] = pcase->base_shape[d];
        }
        shape[normalized_axis] = pcase->axis_lens[i];
        if (!concat_perf_shape_count(pcase->rank, shape, &count) ||
            !concat_perf_checked_mul(count, elem_size, &model->input_bytes[i])) {
            return false;
        }
        if (model->floating) {
            CONCAT_PERF_REQUIRE_OK(model->ctx,
                                   gd_tensor_rand_uniform(model->ctx,
                                                          GD_ARENA_PARAMS,
                                                          pcase->dtype,
                                                          gd_shape_make(pcase->rank, shape),
                                                          256U,
                                                          UINT64_C(0xc0aca700) + (uint64_t)i,
                                                          -0.25f,
                                                          0.25f,
                                                          &model->inputs[i]));
        } else {
            CONCAT_PERF_REQUIRE_OK(model->ctx,
                                   gd_tensor_zeros(model->ctx,
                                                   GD_ARENA_PARAMS,
                                                   pcase->dtype,
                                                   gd_shape_make(pcase->rank, shape),
                                                   256U,
                                                   &model->inputs[i]));
        }
        model->input_ptrs[i] = &model->inputs[i];
    }
    if (model->floating) {
        CONCAT_PERF_REQUIRE_OK(model->ctx,
                               gd_tensor_rand_uniform(model->ctx,
                                                      GD_ARENA_PARAMS,
                                                      pcase->dtype,
                                                      gd_shape_make(pcase->rank, output_shape),
                                                      256U,
                                                      UINT64_C(0xc0aca7ff),
                                                      -0.25f,
                                                      0.25f,
                                                      &model->grad_out));
    }
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void concat_perf_destroy(concat_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool concat_perf_run_forward(concat_perf_model *model)
{
    gd_tensor out;
    uint32_t i;
    for (i = 0U; i < model->n_inputs; ++i) {
        model->inputs[i].requires_grad = false;
    }
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    CONCAT_PERF_REQUIRE_OK(model->ctx,
                           gd_concat(model->ctx,
                                     model->input_ptrs,
                                     model->n_inputs,
                                     model->axis,
                                     &out));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool concat_perf_run_backward(concat_perf_model *model)
{
    gd_tensor grad_inputs[GD_CONCAT_PERF_MAX_INPUTS];
    uint32_t i;
    if (!model->floating) {
        return true;
    }
    for (i = 0U; i < model->n_inputs; ++i) {
        model->inputs[i].requires_grad = false;
    }
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CONCAT_PERF_REQUIRE_OK(model->ctx,
                           gd_concat_backward(model->ctx,
                                              &model->grad_out,
                                              model->input_ptrs,
                                              model->n_inputs,
                                              model->axis,
                                              grad_inputs));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool concat_perf_run_autograd(concat_perf_model *model)
{
    gd_tensor out;
    uint32_t i;
    if (!model->floating) {
        return true;
    }
    for (i = 0U; i < model->n_inputs; ++i) {
        model->inputs[i].requires_grad = true;
    }
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    CONCAT_PERF_REQUIRE_OK(model->ctx,
                           gd_concat(model->ctx,
                                     model->input_ptrs,
                                     model->n_inputs,
                                     model->axis,
                                     &out));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad_out));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    CONCAT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool concat_perf_measure(concat_perf_model *model,
                                const char *label,
                                concat_perf_run_fn fn,
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
    t0 = concat_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = concat_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (logical_bytes / GD_CONCAT_PERF_GIB) / (elapsed / (double)iters);
    printf("[CONCAT][%s] avg_ms=%.4f logical_GiB/s=%.2f\n", label, avg_ms, gib_s);
    return true;
}

static void concat_perf_print_case(const concat_perf_case *pcase, const concat_perf_model *model)
{
    uint32_t i;
    printf("[CONCAT] case=%s dtype=%s rank=%u axis=%d inputs=",
           pcase->name,
           gd_dtype_name(pcase->dtype),
           pcase->rank,
           pcase->axis);
    for (i = 0U; i < pcase->n_inputs; ++i) {
        printf("%s%lld", i == 0U ? "" : ",", (long long)pcase->axis_lens[i]);
    }
    printf(" output_bytes=%zu floating=%d\n", model->output_bytes, model->floating ? 1 : 0);
}

int main(void)
{
    static const concat_perf_case cases[] = {
        {.name = "tokens_axis0_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 2U,
         .axis = 0,
         .n_inputs = 2U,
         .base_shape = {0, 1024},
         .axis_lens = {4096, 4096}},
        {.name = "features_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 2U,
         .axis = 1,
         .n_inputs = 3U,
         .base_shape = {8192, 0},
         .axis_lens = {256, 512, 256}},
        {.name = "features_axis1_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = 1,
         .n_inputs = 2U,
         .base_shape = {2048, 0},
         .axis_lens = {1024, 1024}},
        {.name = "heads_axis1_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .axis = 1,
         .n_inputs = 2U,
         .base_shape = {4096, 0, 64},
         .axis_lens = {8, 8}},
        {.name = "three_way_axis0_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 2U,
         .axis = 0,
         .n_inputs = 3U,
         .base_shape = {0, 2048},
         .axis_lens = {1024, 2048, 1024}},
        {.name = "token_ids_axis0_i32",
         .dtype = GD_DTYPE_I32,
         .rank = 1U,
         .axis = 0,
         .n_inputs = 3U,
         .base_shape = {0},
         .axis_lens = {65536, 65536, 32768}},
    };
    const char *profile = getenv("GD_CONCAT_PERF_PROFILE");
    int warmup = concat_perf_env_int("GD_CONCAT_PERF_WARMUP", 10, 0, 10000);
    int iters = concat_perf_env_int("GD_CONCAT_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[CONCAT] warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        concat_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!concat_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!concat_perf_setup(&cases[i], &model)) {
            return 1;
        }
        concat_perf_print_case(&cases[i], &model);
        fwd_bytes = 2.0 * (double)model.output_bytes;
        bwd_bytes = 2.0 * (double)model.output_bytes;
        autograd_bytes = 4.0 * (double)model.output_bytes;
        if (!concat_perf_measure(&model, "fwd", concat_perf_run_forward, warmup, iters, fwd_bytes)) {
            concat_perf_destroy(&model);
            return 1;
        }
        if (model.floating) {
            if (!concat_perf_measure(&model, "bwd", concat_perf_run_backward, warmup, iters, bwd_bytes) ||
                !concat_perf_measure(&model,
                                     "autograd",
                                     concat_perf_run_autograd,
                                     warmup,
                                     iters,
                                     autograd_bytes)) {
                concat_perf_destroy(&model);
                return 1;
            }
        }
        concat_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[CONCAT][FAIL] no cases selected for GD_CONCAT_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
