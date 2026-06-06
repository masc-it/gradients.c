/*
 * gd_rms_norm Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=rms_norm
 *
 * Optional environment:
 *   GD_RMS_NORM_PERF_PROFILE=smoke|all|<case-name>
 *   GD_RMS_NORM_PERF_WARMUP=10
 *   GD_RMS_NORM_PERF_ITERS=100
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

#define RMS_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct rms_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    float eps;
} rms_perf_case;

typedef struct rms_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor weight;
    gd_tensor grad;
    size_t count;
    size_t rows;
    size_t cols;
    size_t elem_size;
    size_t x_bytes;
    size_t partial_bytes;
} rms_perf_model;

typedef bool (*rms_perf_run_fn)(rms_perf_model *model);

static double rms_perf_now_seconds(void)
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

static int rms_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t rms_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static const char *rms_perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static size_t rms_perf_dtype_size(gd_dtype dtype)
{
    if (dtype == GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool rms_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[RMS_NORM][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define RMS_PERF_REQUIRE_OK(ctx, expr)                             \
    do {                                                           \
        gd_status rms_perf_st__ = (expr);                          \
        if (!rms_perf_status_ok((ctx), rms_perf_st__, #expr)) {    \
            return false;                                          \
        }                                                          \
    } while (0)

static bool rms_perf_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
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

static bool rms_perf_case_selected(const rms_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "transformer_512x4096_f16") == 0 ||
               strcmp(pcase->name, "small_hidden_8192x64_f16") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool rms_perf_init(rms_perf_model *model, const rms_perf_case *pcase)
{
    gd_memory_config cfg;
    size_t weight_bytes;
    size_t params_bytes;
    size_t scratch_bytes;
    size_t row_blocks;
    if (model == NULL || pcase == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    if (!rms_perf_count(pcase->rank, pcase->shape, &model->count) ||
        pcase->shape[pcase->rank - 1U] <= 0) {
        return false;
    }
    model->cols = (size_t)pcase->shape[pcase->rank - 1U];
    model->rows = model->count / model->cols;
    model->elem_size = rms_perf_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || model->count > SIZE_MAX / model->elem_size) {
        return false;
    }
    model->x_bytes = model->count * model->elem_size;
    weight_bytes = model->cols * model->elem_size;
    row_blocks = (model->rows + 63U) / 64U;
    if (row_blocks > SIZE_MAX / model->cols || row_blocks * model->cols > SIZE_MAX / sizeof(float)) {
        return false;
    }
    model->partial_bytes = row_blocks * model->cols * sizeof(float);
    params_bytes = rms_perf_align_up(model->x_bytes * 2U + weight_bytes + 1024U * 1024U, 4096U);
    scratch_bytes = rms_perf_align_up(model->x_bytes * 8U + model->partial_bytes * 2U +
                                      weight_bytes * 8U + 1024U * 1024U, 4096U);
    cfg = gd_memory_config_default();
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = rms_perf_align_up(weight_bytes * 4U + 1024U * 1024U, 4096U);
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    RMS_PERF_REQUIRE_OK(NULL, gd_context_create(&cfg, &model->ctx));
    RMS_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx,
                                               GD_ARENA_PARAMS,
                                               pcase->dtype,
                                               gd_shape_make(pcase->rank, pcase->shape),
                                               256U,
                                               UINT64_C(0x524D5301),
                                               -0.5f,
                                               0.5f,
                                               &model->x));
    {
        int64_t w_shape[1];
        w_shape[0] = (int64_t)model->cols;
        RMS_PERF_REQUIRE_OK(model->ctx,
                            gd_tensor_rand_uniform(model->ctx,
                                                   GD_ARENA_PARAMS,
                                                   pcase->dtype,
                                                   gd_shape_make(1U, w_shape),
                                                   256U,
                                                   UINT64_C(0x524D5302),
                                                   0.75f,
                                                   1.25f,
                                                   &model->weight));
    }
    RMS_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx,
                                               GD_ARENA_PARAMS,
                                               pcase->dtype,
                                               gd_shape_make(pcase->rank, pcase->shape),
                                               256U,
                                               UINT64_C(0x524D5303),
                                               -0.25f,
                                               0.25f,
                                               &model->grad));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void rms_perf_destroy(rms_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool rms_perf_run_forward(rms_perf_model *model)
{
    gd_tensor y;
    model->x.requires_grad = false;
    model->weight.requires_grad = false;
    RMS_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_rms_norm(model->ctx, &model->x, &model->weight, 1.0e-5f, &y));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rms_perf_run_backward(rms_perf_model *model)
{
    gd_tensor dx;
    gd_tensor dw;
    model->x.requires_grad = false;
    model->weight.requires_grad = false;
    RMS_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    RMS_PERF_REQUIRE_OK(model->ctx,
                        gd_rms_norm_backward(model->ctx,
                                             &model->x,
                                             &model->weight,
                                             &model->grad,
                                             1.0e-5f,
                                             &dx,
                                             &dw));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rms_perf_run_autograd(rms_perf_model *model)
{
    gd_tensor y;
    model->x.requires_grad = true;
    model->weight.requires_grad = true;
    RMS_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_rms_norm(model->ctx, &model->x, &model->weight, 1.0e-5f, &y));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &y, &model->grad));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    RMS_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rms_perf_measure(rms_perf_model *model,
                             const char *label,
                             rms_perf_run_fn fn,
                             int warmup,
                             int iters,
                             double effective_bytes)
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
    t0 = rms_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = rms_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (effective_bytes / RMS_PERF_GIB) / (elapsed / (double)iters);
    printf("[RMS_NORM][%s] avg_ms=%.4f effective_GiB/s=%.2f\n", label, avg_ms, gib_s);
    return true;
}

static void rms_perf_print_case(const rms_perf_case *pcase, const rms_perf_model *model)
{
    uint32_t i;
    printf("[RMS_NORM] case=%s dtype=%s input=[",
           pcase->name,
           rms_perf_dtype_name(pcase->dtype));
    for (i = 0U; i < pcase->rank; ++i) {
        printf("%s%lld", i == 0U ? "" : ",", (long long)pcase->shape[i]);
    }
    printf("] rows=%zu cols=%zu bytes=%zu partial_bytes=%zu eps=%.1e\n",
           model->rows,
           model->cols,
           model->x_bytes,
           model->partial_bytes,
           (double)pcase->eps);
}

int main(void)
{
    const rms_perf_case cases[] = {
        {"small_hidden_8192x64_f16", GD_DTYPE_F16, 2U, {8192, 64, 0, 0, 0, 0, 0, 0}, 1.0e-5f},
        {"transformer_512x4096_f16", GD_DTYPE_F16, 2U, {512, 4096, 0, 0, 0, 0, 0, 0}, 1.0e-5f},
        {"qkv_tokens_4096x1024_f16", GD_DTYPE_F16, 2U, {4096, 1024, 0, 0, 0, 0, 0, 0}, 1.0e-5f},
        {"vision_tokens_1024x768_f16", GD_DTYPE_F16, 2U, {1024, 768, 0, 0, 0, 0, 0, 0}, 1.0e-5f},
        {"wide_mlp_4096x1024_f32", GD_DTYPE_F32, 2U, {4096, 1024, 0, 0, 0, 0, 0, 0}, 1.0e-5f},
        {"batched_rank3_8x512x4096_f16", GD_DTYPE_F16, 3U, {8, 512, 4096, 0, 0, 0, 0, 0}, 1.0e-5f},
    };
    const char *profile = getenv("GD_RMS_NORM_PERF_PROFILE");
    int warmup = rms_perf_env_int("GD_RMS_NORM_PERF_WARMUP", 10, 0, 100000);
    int iters = rms_perf_env_int("GD_RMS_NORM_PERF_ITERS", 100, 1, 1000000);
    size_t i;
    bool any = false;
    bool ok = true;
    printf("[RMS_NORM] warmup=%d iters=%d profile=%s note=last-dim-fp32-accum-saved-inv-rms\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        rms_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!rms_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        any = true;
        if (!rms_perf_init(&model, &cases[i])) {
            ok = false;
            break;
        }
        rms_perf_print_case(&cases[i], &model);
        fwd_bytes = (double)model.x_bytes * 4.0;
        bwd_bytes = (double)model.x_bytes * 10.0 + (double)model.partial_bytes * 2.0;
        autograd_bytes = (double)model.x_bytes * 15.0 + (double)model.partial_bytes * 2.0;
        ok = rms_perf_measure(&model, "fwd", rms_perf_run_forward, warmup, iters, fwd_bytes) &&
             rms_perf_measure(&model, "bwd", rms_perf_run_backward, warmup, iters, bwd_bytes) &&
             rms_perf_measure(&model, "autograd", rms_perf_run_autograd, warmup, iters, autograd_bytes);
        rms_perf_destroy(&model);
        if (!ok) {
            break;
        }
    }
    if (!any) {
        fprintf(stderr, "[RMS_NORM][FAIL] no cases selected for profile=%s\n", profile != NULL ? profile : "smoke");
        return 2;
    }
    return ok ? 0 : 1;
}
