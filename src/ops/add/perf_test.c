/*
 * FP16 gd_add Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=add
 *
 * Optional environment:
 *   GD_ADD_PERF_PROFILE=all|smoke|<case-name>
 *   GD_ADD_PERF_WARMUP=10
 *   GD_ADD_PERF_ITERS=100
 *
 * Wall time includes public validation, scratch allocation, Metal command
 * encoding/submission, and synchronization. Bandwidth is the logical FP16
 * tensor traffic for the public op contract, not a hardware-counter value.
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

#define ADD_PERF_GIB (1024.0 * 1024.0 * 1024.0)
#define ADD_PERF_F16_BYTES 2U

typedef struct add_perf_case {
    const char *name;
    uint32_t x_rank;
    int64_t x_shape[GD_MAX_DIMS];
    uint32_t y_rank;
    int64_t y_shape[GD_MAX_DIMS];
} add_perf_case;

typedef struct add_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad;
    uint32_t out_rank;
    int64_t out_shape[GD_MAX_DIMS];
    size_t x_count;
    size_t y_count;
    size_t out_count;
    size_t x_bytes;
    size_t y_bytes;
    size_t out_bytes;
} add_perf_model;

typedef bool (*add_perf_run_fn)(add_perf_model *model);

static double add_perf_now_seconds(void)
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

static int add_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t add_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool add_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[ADD][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define ADD_PERF_REQUIRE_OK(ctx, expr)                         \
    do {                                                       \
        gd_status add_perf_st__ = (expr);                      \
        if (!add_perf_status_ok((ctx), add_perf_st__, #expr)) {\
            return false;                                      \
        }                                                      \
    } while (0)

static bool add_perf_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
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

static bool add_perf_broadcast_shape(const add_perf_case *pcase,
                                     uint32_t *out_rank,
                                     int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t rank;
    uint32_t dim;
    if (pcase == NULL || out_rank == NULL || out_shape == NULL || pcase->x_rank > GD_MAX_DIMS ||
        pcase->y_rank > GD_MAX_DIMS) {
        return false;
    }
    rank = pcase->x_rank > pcase->y_rank ? pcase->x_rank : pcase->y_rank;
    if (rank == 0U) {
        return false;
    }
    memset(out_shape, 0, GD_MAX_DIMS * sizeof(out_shape[0]));
    for (dim = 0U; dim < rank; ++dim) {
        int64_t xd = 1;
        int64_t yd = 1;
        if (dim + pcase->x_rank >= rank) {
            xd = pcase->x_shape[dim + pcase->x_rank - rank];
        }
        if (dim + pcase->y_rank >= rank) {
            yd = pcase->y_shape[dim + pcase->y_rank - rank];
        }
        if (xd <= 0 || yd <= 0 || (xd != yd && xd != 1 && yd != 1)) {
            return false;
        }
        out_shape[dim] = xd > yd ? xd : yd;
    }
    *out_rank = rank;
    return true;
}

static bool add_perf_case_selected(const add_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0) {
        return true;
    }
    if (strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tail_1x513_direct") == 0 ||
               strcmp(pcase->name, "direct_4096x1024") == 0 ||
               strcmp(pcase->name, "row_4096x1024") == 0;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool add_perf_create_model(const add_perf_case *pcase, add_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    if (!add_perf_broadcast_shape(pcase, &model->out_rank, model->out_shape) ||
        !add_perf_count(pcase->x_rank, pcase->x_shape, &model->x_count) ||
        !add_perf_count(pcase->y_rank, pcase->y_shape, &model->y_count) ||
        !add_perf_count(model->out_rank, model->out_shape, &model->out_count)) {
        fprintf(stderr, "[ADD][FAIL] invalid perf shape for case=%s\n", pcase->name);
        return false;
    }
    if (model->x_count > SIZE_MAX / ADD_PERF_F16_BYTES ||
        model->y_count > SIZE_MAX / ADD_PERF_F16_BYTES ||
        model->out_count > SIZE_MAX / ADD_PERF_F16_BYTES) {
        return false;
    }
    model->x_bytes = model->x_count * ADD_PERF_F16_BYTES;
    model->y_bytes = model->y_count * ADD_PERF_F16_BYTES;
    model->out_bytes = model->out_count * ADD_PERF_F16_BYTES;
    if (model->out_bytes > SIZE_MAX / 16U) {
        return false;
    }
    params_bytes = add_perf_align_up(model->x_bytes + model->y_bytes + model->out_bytes +
                                         64U * 1024U * 1024U,
                                     4096U);
    scratch_bytes = add_perf_align_up(model->out_bytes * 16U + 128U * 1024U * 1024U, 4096U);
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
        printf("[ADD] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!add_perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    ADD_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(pcase->x_rank, pcase->x_shape), 256U, 11U, -1.0f, 1.0f, &model->x));
    ADD_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(pcase->y_rank, pcase->y_shape), 256U, 22U, -1.0f, 1.0f, &model->y));
    ADD_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(model->out_rank, model->out_shape), 256U, 33U, -0.5f, 0.5f, &model->grad));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void add_perf_destroy_model(add_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool add_perf_run_forward(add_perf_model *model)
{
    gd_tensor out;
    model->x.requires_grad = false;
    model->y.requires_grad = false;
    ADD_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_add(model->ctx, &model->x, &model->y, &out));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool add_perf_run_backward_direct(add_perf_model *model)
{
    gd_tensor dx;
    gd_tensor dy;
    model->x.requires_grad = false;
    model->y.requires_grad = false;
    ADD_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_add_backward(model->ctx, &model->x, &model->y, &model->grad, &dx, &dy));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool add_perf_run_forward_backward_autograd(add_perf_model *model)
{
    gd_tensor out;
    gd_tensor dx;
    gd_tensor dy;
    model->x.requires_grad = true;
    model->y.requires_grad = true;
    ADD_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_add(model->ctx, &model->x, &model->y, &out));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->y, &dy));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    ADD_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static double add_perf_measure(add_perf_model *model, add_perf_run_fn run, int warmup, int iters)
{
    double start;
    double elapsed;
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!run(model)) {
            return -1.0;
        }
    }
    start = add_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!run(model)) {
            return -1.0;
        }
    }
    elapsed = add_perf_now_seconds() - start;
    return elapsed / (double)iters;
}

static void add_perf_print_shape(uint32_t rank, const int64_t shape[GD_MAX_DIMS])
{
    uint32_t dim;
    printf("[");
    for (dim = 0U; dim < rank; ++dim) {
        printf("%s%lld", dim == 0U ? "" : "x", (long long)shape[dim]);
    }
    printf("]");
}

static void add_perf_print_result(const char *op,
                                  const add_perf_case *pcase,
                                  const add_perf_model *model,
                                  double seconds,
                                  double logical_bytes,
                                  int warmup,
                                  int iters)
{
    double gib_s = (logical_bytes / ADD_PERF_GIB) / seconds;
    double gelem_s = ((double)model->out_count / seconds) / 1.0e9;
    printf("[ADD][%s] case=%s dtype=f16 x=", op, pcase->name);
    add_perf_print_shape(pcase->x_rank, pcase->x_shape);
    printf(" y=");
    add_perf_print_shape(pcase->y_rank, pcase->y_shape);
    printf(" out=");
    add_perf_print_shape(model->out_rank, model->out_shape);
    printf(" warmup=%d iters=%d avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f\n",
           warmup,
           iters,
           seconds * 1000.0,
           gib_s,
           gelem_s);
}

static bool add_perf_run_case(const add_perf_case *pcase, int warmup, int iters)
{
    add_perf_model model;
    double fwd_s;
    double bwd_s;
    double pair_s;
    double fwd_bytes;
    double bwd_bytes;
    if (!add_perf_create_model(pcase, &model)) {
        return false;
    }
    fwd_bytes = (double)model.x_bytes + (double)model.y_bytes + (double)model.out_bytes;
    bwd_bytes = (double)model.out_bytes + (double)model.x_bytes + (double)model.y_bytes;
    printf("[ADD] case=%s out_elems=%zu out_tensor=%.1fMiB x_elems=%zu y_elems=%zu\n",
           pcase->name,
           model.out_count,
           (double)model.out_bytes / (1024.0 * 1024.0),
           model.x_count,
           model.y_count);
    fwd_s = add_perf_measure(&model, add_perf_run_forward, warmup, iters);
    bwd_s = add_perf_measure(&model, add_perf_run_backward_direct, warmup, iters);
    pair_s = add_perf_measure(&model, add_perf_run_forward_backward_autograd, warmup, iters);
    if (fwd_s > 0.0) {
        add_perf_print_result("fwd", pcase, &model, fwd_s, fwd_bytes, warmup, iters);
    }
    if (bwd_s > 0.0) {
        add_perf_print_result("bwd_direct", pcase, &model, bwd_s, bwd_bytes, warmup, iters);
    }
    if (pair_s > 0.0) {
        add_perf_print_result("fwd_bwd_autograd",
                              pcase,
                              &model,
                              pair_s,
                              fwd_bytes + bwd_bytes,
                              warmup,
                              iters);
    }
    add_perf_destroy_model(&model);
    return fwd_s > 0.0 && bwd_s > 0.0 && pair_s > 0.0;
}

int main(void)
{
    static const add_perf_case cases[] = {
        {"tail_1x513_direct", 2U, {1, 513}, 2U, {1, 513}},
        {"direct_4096x1024", 2U, {4096, 1024}, 2U, {4096, 1024}},
        {"direct_8192x2048", 2U, {8192, 2048}, 2U, {8192, 2048}},
        {"row_4096x1024", 2U, {4096, 1024}, 1U, {1024}},
        {"row_8192x2048", 2U, {8192, 2048}, 1U, {2048}},
        {"rank3_8x512x1024_bias", 3U, {8, 512, 1024}, 1U, {1024}},
        {"rank3_4x2048x4096_bias", 3U, {4, 2048, 4096}, 1U, {4096}},
        {"mlp_2048x11008_bias", 2U, {2048, 11008}, 1U, {11008}},
    };
    const char *profile = getenv("GD_ADD_PERF_PROFILE");
    int warmup = add_perf_env_int("GD_ADD_PERF_WARMUP", 10, 0, 10000);
    int iters = add_perf_env_int("GD_ADD_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[ADD] FP16 public API perf: warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "all");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!add_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!add_perf_run_case(&cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr, "[ADD][FAIL] no case matched GD_ADD_PERF_PROFILE=%s\n", profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}
