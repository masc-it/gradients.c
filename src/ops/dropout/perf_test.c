/*
 * Dropout public API performance probe.
 *
 * Run with:
 *   make op-perf OP=dropout
 *
 * Optional environment:
 *   GD_DROPOUT_PERF_PROFILE=all|smoke|<case-name>
 *   GD_DROPOUT_PERF_WARMUP=10
 *   GD_DROPOUT_PERF_ITERS=100
 *
 * Timings include public validation, scratch allocation, Metal encoding, command
 * buffer submission, and synchronization. Logical bandwidth counts activation
 * reads/writes plus compact u8 mask traffic for training dropout.
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

#define DROPOUT_PERF_GIB (1024.0 * 1024.0 * 1024.0)

#if defined(__APPLE__)
static double dropout_perf_now_seconds(void)
{
    static mach_timebase_info_data_t info;
    static double scale = 0.0;
    if (scale == 0.0) {
        if (mach_timebase_info(&info) != 0 || info.denom == 0U) {
            return 0.0;
        }
        scale = ((double)info.numer / (double)info.denom) * 1.0e-9;
    }
    return (double)mach_absolute_time() * scale;
}
#else
static double dropout_perf_now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#endif

static int dropout_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t dropout_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool dropout_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[DROPOUT][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define DROPOUT_PERF_REQUIRE_OK(ctx, expr)                                      \
    do {                                                                        \
        gd_status dropout_perf_st__ = (expr);                                   \
        if (!dropout_perf_status_ok((ctx), dropout_perf_st__, #expr)) {         \
            return false;                                                       \
        }                                                                       \
    } while (0)

typedef struct dropout_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    float p;
} dropout_perf_case;

typedef struct dropout_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad;
    size_t count;
    size_t elem_size;
    size_t tensor_bytes;
    uint64_t seed;
} dropout_perf_model;

typedef bool (*dropout_perf_run_fn)(dropout_perf_model *model, const dropout_perf_case *pcase);

static bool dropout_perf_case_selected(const dropout_perf_case *pcase, const char *profile)
{
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0) {
        return true;
    }
    if (strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tail_1x513_f16") == 0 ||
               strcmp(pcase->name, "act_4096x1024_f16") == 0 ||
               strcmp(pcase->name, "act_4096x1024_f32") == 0;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool dropout_perf_count_shape(const dropout_perf_case *pcase, size_t *out_count)
{
    size_t count = 1U;
    uint32_t i;
    if (pcase == NULL || out_count == NULL || pcase->rank == 0U || pcase->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = 0U; i < pcase->rank; ++i) {
        if (pcase->shape[i] <= 0 || (uint64_t)pcase->shape[i] > (uint64_t)(SIZE_MAX / count)) {
            return false;
        }
        count *= (size_t)pcase->shape[i];
    }
    *out_count = count;
    return count != 0U;
}

static void dropout_perf_shape_string(const dropout_perf_case *pcase, char *out, size_t out_size)
{
    size_t used = 0U;
    uint32_t i;
    if (out == NULL || out_size == 0U) {
        return;
    }
    out[0] = '\0';
    for (i = 0U; pcase != NULL && i < pcase->rank && used + 1U < out_size; ++i) {
        int n = snprintf(out + used,
                         out_size - used,
                         "%s%lld",
                         i == 0U ? "" : "x",
                         (long long)pcase->shape[i]);
        if (n < 0 || (size_t)n >= out_size - used) {
            out[out_size - 1U] = '\0';
            return;
        }
        used += (size_t)n;
    }
}

static bool dropout_perf_create_model(const dropout_perf_case *pcase, dropout_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->elem_size = gd_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || !dropout_perf_count_shape(pcase, &model->count) ||
        model->count > SIZE_MAX / model->elem_size) {
        return false;
    }
    model->tensor_bytes = model->count * model->elem_size;
    if (model->tensor_bytes > (SIZE_MAX - 256U * 1024U * 1024U) / 10U ||
        model->count > (SIZE_MAX - 256U * 1024U * 1024U) / 4U) {
        return false;
    }
    params_bytes = dropout_perf_align_up(model->tensor_bytes * 2U + 64U * 1024U * 1024U, 4096U);
    scratch_bytes = dropout_perf_align_up(model->tensor_bytes * 10U + model->count * 4U +
                                              256U * 1024U * 1024U,
                                          4096U);
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = 8U * 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 8U * 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &model->ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[DROPOUT] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!dropout_perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    DROPOUT_PERF_REQUIRE_OK(model->ctx,
                            gd_tensor_rand_uniform(model->ctx,
                                                   GD_ARENA_PARAMS,
                                                   pcase->dtype,
                                                   gd_shape_make(pcase->rank, pcase->shape),
                                                   256U,
                                                   1234U,
                                                   -2.0f,
                                                   2.0f,
                                                   &model->x));
    DROPOUT_PERF_REQUIRE_OK(model->ctx,
                            gd_tensor_rand_uniform(model->ctx,
                                                   GD_ARENA_PARAMS,
                                                   pcase->dtype,
                                                   gd_shape_make(pcase->rank, pcase->shape),
                                                   256U,
                                                   5678U,
                                                   -0.5f,
                                                   0.5f,
                                                   &model->grad));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    model->seed = UINT64_C(0xD00D000000000000) ^ (uint64_t)model->count ^ ((uint64_t)pcase->rank << 48);
    return true;
}

static void dropout_perf_destroy_model(dropout_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool dropout_perf_run_forward(dropout_perf_model *model, const dropout_perf_case *pcase)
{
    gd_tensor y;
    model->x.requires_grad = false;
    model->seed += UINT64_C(0x9E3779B97F4A7C15);
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_dropout(model->ctx, &model->x, pcase->p, true, model->seed, &y));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool dropout_perf_run_backward_direct(dropout_perf_model *model, const dropout_perf_case *pcase)
{
    gd_tensor dx;
    model->x.requires_grad = false;
    model->seed += UINT64_C(0x9E3779B97F4A7C15);
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    DROPOUT_PERF_REQUIRE_OK(model->ctx,
                            gd_dropout_backward(model->ctx,
                                                &model->x,
                                                &model->grad,
                                                pcase->p,
                                                model->seed,
                                                &dx));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool dropout_perf_run_forward_backward_autograd(dropout_perf_model *model,
                                                       const dropout_perf_case *pcase)
{
    gd_tensor y;
    model->x.requires_grad = true;
    model->seed += UINT64_C(0x9E3779B97F4A7C15);
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_dropout(model->ctx, &model->x, pcase->p, true, model->seed, &y));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &y, &model->grad));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool dropout_perf_run_eval_identity(dropout_perf_model *model, const dropout_perf_case *pcase)
{
    gd_tensor y;
    model->x.requires_grad = false;
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_dropout(model->ctx, &model->x, pcase->p, false, 0U, &y));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    DROPOUT_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static double dropout_perf_measure(dropout_perf_model *model,
                                   const dropout_perf_case *pcase,
                                   dropout_perf_run_fn fn,
                                   int warmup,
                                   int iters)
{
    double start;
    double elapsed;
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!fn(model, pcase)) {
            return -1.0;
        }
    }
    start = dropout_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model, pcase)) {
            return -1.0;
        }
    }
    elapsed = dropout_perf_now_seconds() - start;
    if (elapsed <= 0.0 || iters <= 0) {
        return -1.0;
    }
    return elapsed / (double)iters;
}

static void dropout_perf_print_result(const char *tag,
                                      const dropout_perf_case *pcase,
                                      const dropout_perf_model *model,
                                      double seconds,
                                      int warmup,
                                      int iters,
                                      double logical_bytes)
{
    char shape[128];
    double ms = seconds * 1.0e3;
    double logical_gib_s = logical_bytes > 0.0 ? (logical_bytes / DROPOUT_PERF_GIB) / seconds : 0.0;
    double gelem_s = ((double)model->count * 1.0e-9) / seconds;
    dropout_perf_shape_string(pcase, shape, sizeof(shape));
    printf("[DROPOUT][%s] case=%s dtype=%s shape=%s p=%.3f elems=%zu warmup=%d iters=%d avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f\n",
           tag,
           pcase->name,
           gd_dtype_name(pcase->dtype),
           shape,
           (double)pcase->p,
           model->count,
           warmup,
           iters,
           ms,
           logical_gib_s,
           gelem_s);
}

int main(void)
{
    static const dropout_perf_case cases[] = {
        {"tail_1x513_f16", GD_DTYPE_F16, 2U, {1, 513}, 0.20f},
        {"mlp_hidden_128x128_f16", GD_DTYPE_F16, 2U, {128, 128}, 0.10f},
        {"act_4096x1024_f16", GD_DTYPE_F16, 2U, {4096, 1024}, 0.10f},
        {"act_8192x2048_f16", GD_DTYPE_F16, 2U, {8192, 2048}, 0.10f},
        {"rank3_8x512x1024_f16", GD_DTYPE_F16, 3U, {8, 512, 1024}, 0.20f},
        {"vision_4x64x128x128_f16", GD_DTYPE_F16, 4U, {4, 64, 128, 128}, 0.20f},
        {"act_4096x1024_f32", GD_DTYPE_F32, 2U, {4096, 1024}, 0.10f},
        {"rank3_4x256x1024_f32", GD_DTYPE_F32, 3U, {4, 256, 1024}, 0.20f},
    };
    const char *profile = getenv("GD_DROPOUT_PERF_PROFILE");
    int warmup = dropout_perf_env_int("GD_DROPOUT_PERF_WARMUP", 10, 0, 10000);
    int iters = dropout_perf_env_int("GD_DROPOUT_PERF_ITERS", 100, 1, 1000000);
    size_t i;
    bool ran = false;
    printf("[DROPOUT] public API perf: warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "all");
    printf("[DROPOUT] logical bytes: fwd=read_x+write_y+write_mask, direct_bwd=recompute mask, autograd=saved-mask backward.\n");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        dropout_perf_model model;
        double fwd_s;
        double bwd_s;
        double pair_s;
        double eval_s;
        double elem_size;
        if (!dropout_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!dropout_perf_create_model(&cases[i], &model)) {
            continue;
        }
        elem_size = (double)model.elem_size;
        printf("[DROPOUT] case=%s dtype=%s elems=%zu tensor=%.1fMiB p=%.3f\n",
               cases[i].name,
               gd_dtype_name(cases[i].dtype),
               model.count,
               (double)model.tensor_bytes / (1024.0 * 1024.0),
               (double)cases[i].p);
        fwd_s = dropout_perf_measure(&model, &cases[i], dropout_perf_run_forward, warmup, iters);
        bwd_s = dropout_perf_measure(&model, &cases[i], dropout_perf_run_backward_direct, warmup, iters);
        pair_s = dropout_perf_measure(&model,
                                      &cases[i],
                                      dropout_perf_run_forward_backward_autograd,
                                      warmup,
                                      iters);
        eval_s = dropout_perf_measure(&model, &cases[i], dropout_perf_run_eval_identity, warmup, iters);
        if (fwd_s > 0.0) {
            dropout_perf_print_result("train_fwd",
                                      &cases[i],
                                      &model,
                                      fwd_s,
                                      warmup,
                                      iters,
                                      (double)model.count * (elem_size * 2.0 + 1.0));
        }
        if (bwd_s > 0.0) {
            dropout_perf_print_result("bwd_direct_recompute",
                                      &cases[i],
                                      &model,
                                      bwd_s,
                                      warmup,
                                      iters,
                                      (double)model.count * elem_size * 2.0);
        }
        if (pair_s > 0.0) {
            dropout_perf_print_result("fwd_bwd_autograd_saved_mask",
                                      &cases[i],
                                      &model,
                                      pair_s,
                                      warmup,
                                      iters,
                                      (double)model.count * (elem_size * 4.0 + 2.0));
        }
        if (eval_s > 0.0) {
            dropout_perf_print_result("eval_identity",
                                      &cases[i],
                                      &model,
                                      eval_s,
                                      warmup,
                                      iters,
                                      0.0);
        }
        dropout_perf_destroy_model(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[DROPOUT][FAIL] no cases selected for GD_DROPOUT_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
