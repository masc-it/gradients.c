/*
 * FP16 ReLU public API performance probe.
 *
 * Run with:
 *   make op-perf OP=relu
 *
 * Optional environment:
 *   GD_RELU_PERF_PROFILE=all|smoke|<case-name>
 *   GD_RELU_PERF_WARMUP=10
 *   GD_RELU_PERF_ITERS=100
 *
 * Times include public validation, scratch allocation, Metal encoding, command
 * buffer submission, and synchronization. Bandwidth is reported as a logical
 * lower bound for the kernel work; the autograd pair also reports an estimated
 * public-training bandwidth that includes grad-slot zero/accumulate traffic.
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

#define RELU_PERF_GIB (1024.0 * 1024.0 * 1024.0)

#if defined(__APPLE__)
static double relu_perf_now_seconds(void)
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
static double relu_perf_now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#endif

static int relu_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t relu_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool relu_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[RELU][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define RELU_PERF_REQUIRE_OK(ctx, expr)                         \
    do {                                                        \
        gd_status relu_perf_st__ = (expr);                      \
        if (!relu_perf_status_ok((ctx), relu_perf_st__, #expr)) { \
            return false;                                       \
        }                                                       \
    } while (0)

typedef struct relu_perf_case {
    const char *name;
    int64_t rows;
    int64_t cols;
} relu_perf_case;

typedef struct relu_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad;
    size_t count;
    size_t tensor_bytes;
    int64_t rows;
    int64_t cols;
} relu_perf_model;

typedef bool (*relu_perf_run_fn)(relu_perf_model *model);

static bool relu_perf_case_selected(const relu_perf_case *pcase, const char *profile)
{
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0) {
        return true;
    }
    if (strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tail_1x513") == 0 ||
               strcmp(pcase->name, "bert_4096x768") == 0;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool relu_perf_create_model(const relu_perf_case *pcase, relu_perf_model *model)
{
    gd_memory_config cfg;
    int64_t shape[2];
    size_t params_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (pcase == NULL || model == NULL || pcase->rows <= 0 || pcase->cols <= 0 ||
        (uint64_t)pcase->rows > (uint64_t)(SIZE_MAX / (size_t)pcase->cols)) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->rows = pcase->rows;
    model->cols = pcase->cols;
    model->count = (size_t)pcase->rows * (size_t)pcase->cols;
    if (model->count == 0U || model->count > SIZE_MAX / sizeof(uint16_t)) {
        return false;
    }
    model->tensor_bytes = model->count * sizeof(uint16_t);
    params_bytes = relu_perf_align_up(model->tensor_bytes * 2U + 64U * 1024U * 1024U, 4096U);
    scratch_bytes = relu_perf_align_up(model->tensor_bytes * 8U + 128U * 1024U * 1024U, 4096U);
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &model->ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[RELU] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!relu_perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    shape[0] = pcase->rows;
    shape[1] = pcase->cols;
    RELU_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_rand_uniform(model->ctx,
                                                GD_ARENA_PARAMS,
                                                GD_DTYPE_F16,
                                                2U,
                                                shape,
                                                256U,
                                                1234U,
                                                -1.0f,
                                                1.0f,
                                                &model->x));
    RELU_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_rand_uniform(model->ctx,
                                                GD_ARENA_PARAMS,
                                                GD_DTYPE_F16,
                                                2U,
                                                shape,
                                                256U,
                                                5678U,
                                                -0.5f,
                                                0.5f,
                                                &model->grad));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void relu_perf_destroy_model(relu_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool relu_perf_run_forward(relu_perf_model *model)
{
    gd_tensor y;
    model->x.requires_grad = false;
    RELU_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_relu(model->ctx, &model->x, &y));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool relu_perf_run_backward_direct(relu_perf_model *model)
{
    gd_tensor dx;
    model->x.requires_grad = false;
    RELU_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_relu_backward(model->ctx, &model->x, &model->grad, &dx));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool relu_perf_run_forward_backward_autograd(relu_perf_model *model)
{
    gd_tensor y;
    model->x.requires_grad = true;
    RELU_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_relu(model->ctx, &model->x, &y));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &y, &model->grad));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    RELU_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static double relu_perf_measure(relu_perf_model *model,
                                relu_perf_run_fn fn,
                                int warmup,
                                int iters)
{
    double start;
    double elapsed;
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!fn(model)) {
            return -1.0;
        }
    }
    start = relu_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return -1.0;
        }
    }
    elapsed = relu_perf_now_seconds() - start;
    if (elapsed <= 0.0 || iters <= 0) {
        return -1.0;
    }
    return elapsed / (double)iters;
}

static void relu_perf_print_result(const char *tag,
                                   const relu_perf_case *pcase,
                                   const relu_perf_model *model,
                                   double seconds,
                                   double logical_bytes,
                                   double estimated_public_bytes)
{
    double ms = seconds * 1.0e3;
    double logical_gib_s = (logical_bytes / RELU_PERF_GIB) / seconds;
    double gelem_s = ((double)model->count * 1.0e-9) / seconds;
    printf("[RELU][%s] case=%s shape=%lldx%lld elems=%zu avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f",
           tag,
           pcase->name,
           (long long)model->rows,
           (long long)model->cols,
           model->count,
           ms,
           logical_gib_s,
           gelem_s);
    if (estimated_public_bytes > 0.0) {
        double estimated_gib_s = (estimated_public_bytes / RELU_PERF_GIB) / seconds;
        printf(" estimated_public_GiB/s=%.2f", estimated_gib_s);
    }
    printf("\n");
}

int main(void)
{
    static const relu_perf_case cases[] = {
        {"tail_1x513", 1, 513},
        {"bert_4096x768", 4096, 768},
        {"gpt_4096x4096", 4096, 4096},
        {"mlp_2048x11008", 2048, 11008},
    };
    const char *profile = getenv("GD_RELU_PERF_PROFILE");
    int warmup = relu_perf_env_int("GD_RELU_PERF_WARMUP", 10, 0, 10000);
    int iters = relu_perf_env_int("GD_RELU_PERF_ITERS", 100, 1, 1000000);
    size_t i;
    bool ran = false;
    printf("[RELU] FP16 public API perf: warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "all");
    printf("[RELU] logical bytes: fwd=read_x+write_y, bwd=read_x+read_grad+write_dx; "
           "autograd estimate includes grad zero/accumulate traffic.\n");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        relu_perf_model model;
        double fwd_s;
        double bwd_s;
        double pair_s;
        double count = (double)((size_t)cases[i].rows * (size_t)cases[i].cols);
        if (!relu_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!relu_perf_create_model(&cases[i], &model)) {
            continue;
        }
        fwd_s = relu_perf_measure(&model, relu_perf_run_forward, warmup, iters);
        bwd_s = relu_perf_measure(&model, relu_perf_run_backward_direct, warmup, iters);
        pair_s = relu_perf_measure(&model, relu_perf_run_forward_backward_autograd, warmup, iters);
        if (fwd_s > 0.0) {
            relu_perf_print_result("fwd", &cases[i], &model, fwd_s, count * 4.0, 0.0);
        }
        if (bwd_s > 0.0) {
            relu_perf_print_result("bwd_direct", &cases[i], &model, bwd_s, count * 6.0, 0.0);
        }
        if (pair_s > 0.0) {
            relu_perf_print_result("fwd_bwd_autograd", &cases[i], &model, pair_s, count * 10.0, count * 26.0);
        }
        relu_perf_destroy_model(&model);
    }
    if (!ran) {
        fprintf(stderr, "[RELU][FAIL] no cases selected for GD_RELU_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
