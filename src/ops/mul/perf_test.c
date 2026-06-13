/*
 * Focused F16 gd_mul Metal performance probe.
 *
 * Build example:
 *   make BUILD_DIR=build-mul-perf \
 *     CFLAGS='-std=c11 -O3 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes -Wno-unused-parameter' \
 *     OBJCFLAGS='-Iinclude -DGD_ENABLE_METAL=1 -O3 -DNDEBUG -Wall -Wextra -Werror -fobjc-arc' build
 *   cc -DGD_MUL_PERF_TEST_MAIN -Iinclude -std=c11 -O3 -DNDEBUG -Wall -Wextra -Werror -Wpedantic \
 *     src/ops/mul/perf_test.c build-mul-perf/libgradients.a \
 *     -framework Foundation -framework Metal -o build-mul-perf/mul_perf_test
 *   GRADIENTS_METALLIB=build-mul-perf/gradients.metallib build-mul-perf/mul_perf_test
 *
 * Wall time includes public validation, scratch allocation, Metal command buffer
 * submission, and synchronization. Reported bandwidth is logical minimum global
 * traffic for the public op sequence; real hardware traffic can differ because
 * broadcast operands may be cached or reread.
 */

#ifndef GD_MUL_PERF_TEST_MAIN
typedef int gd_mul_perf_test_build_guard;
#else

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

#define MUL_PERF_GB 1000000000.0
#define MUL_PERF_GIB (1024.0 * 1024.0 * 1024.0)
#define MUL_PERF_MIB (1024.0 * 1024.0)
#define MUL_PERF_F16_BYTES 2U

typedef enum mul_perf_op {
    MUL_PERF_FWD = 1,
    MUL_PERF_BWD_X = 2,
    MUL_PERF_BWD_Y = 3,
    MUL_PERF_BWD_XY = 4,
} mul_perf_op;

typedef struct mul_perf_case {
    const char *name;
    uint32_t rank;
    int64_t x_shape[GD_MAX_DIMS];
    uint32_t y_rank;
    int64_t y_shape[GD_MAX_DIMS];
    mul_perf_op op;
} mul_perf_case;

typedef struct mul_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad;
    uint32_t rank;
    int64_t out_shape[GD_MAX_DIMS];
    size_t out_count;
    size_t x_count;
    size_t y_count;
    size_t out_bytes;
    size_t x_bytes;
    size_t y_bytes;
} mul_perf_model;

static double mul_perf_now_seconds(void)
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

static int mul_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t mul_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool mul_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[MUL][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define MUL_PERF_REQUIRE_OK(ctx, expr)                         \
    do {                                                       \
        gd_status mul_perf_st__ = (expr);                      \
        if (!mul_perf_status_ok((ctx), mul_perf_st__, #expr)) {\
            return false;                                      \
        }                                                      \
    } while (0)

static bool mul_perf_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t dim;
    size_t count = 1U;
    if (rank == 0U || rank > GD_MAX_DIMS || out_count == NULL) {
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

static bool mul_perf_broadcast_shape(const mul_perf_case *pcase,
                                     uint32_t *out_rank,
                                     int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t rank;
    uint32_t dim;
    if (pcase == NULL || out_rank == NULL || out_shape == NULL || pcase->rank > GD_MAX_DIMS ||
        pcase->y_rank > GD_MAX_DIMS) {
        return false;
    }
    rank = pcase->rank > pcase->y_rank ? pcase->rank : pcase->y_rank;
    if (rank == 0U) {
        return false;
    }
    memset(out_shape, 0, GD_MAX_DIMS * sizeof(out_shape[0]));
    for (dim = 0U; dim < rank; ++dim) {
        int64_t xd = 1;
        int64_t yd = 1;
        if (dim + pcase->rank >= rank) {
            xd = pcase->x_shape[dim + pcase->rank - rank];
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

static const char *mul_perf_op_name(mul_perf_op op)
{
    switch (op) {
    case MUL_PERF_FWD:
        return "fwd";
    case MUL_PERF_BWD_X:
        return "bwd_x";
    case MUL_PERF_BWD_Y:
        return "bwd_y";
    case MUL_PERF_BWD_XY:
        return "bwd_xy";
    default:
        return "unknown";
    }
}

static double mul_perf_logical_bytes(const mul_perf_model *model, mul_perf_op op)
{
    if (model == NULL) {
        return 0.0;
    }
    switch (op) {
    case MUL_PERF_FWD:
        return (double)model->x_bytes + (double)model->y_bytes + (double)model->out_bytes;
    case MUL_PERF_BWD_X:
        return (double)model->out_bytes + (double)model->y_bytes + (double)model->x_bytes;
    case MUL_PERF_BWD_Y:
        if (model->y_count == model->out_count) {
            return (double)model->out_bytes + (double)model->x_bytes + (double)model->y_bytes;
        }
        return (double)model->out_bytes * 2.0 + (double)model->x_bytes + (double)model->y_bytes;
    case MUL_PERF_BWD_XY:
        if (model->y_count == model->out_count && model->x_count == model->out_count) {
            return (double)model->out_bytes * 6.0;
        }
        return (double)model->out_bytes * 4.0 + (double)model->x_bytes + (double)model->y_bytes * 2.0;
    default:
        return 0.0;
    }
}

static double mul_perf_flops(const mul_perf_model *model, mul_perf_op op)
{
    if (model == NULL) {
        return 0.0;
    }
    switch (op) {
    case MUL_PERF_FWD:
    case MUL_PERF_BWD_X:
        return (double)model->out_count;
    case MUL_PERF_BWD_Y:
        return model->y_count == model->out_count ? (double)model->out_count : (double)model->out_count * 2.0;
    case MUL_PERF_BWD_XY:
        return model->y_count == model->out_count ? (double)model->out_count * 2.0 : (double)model->out_count * 3.0;
    default:
        return 0.0;
    }
}

static bool mul_perf_create_model(const mul_perf_case *pcase, mul_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    if (!mul_perf_broadcast_shape(pcase, &model->rank, model->out_shape) ||
        !mul_perf_count(pcase->rank, pcase->x_shape, &model->x_count) ||
        !mul_perf_count(pcase->y_rank, pcase->y_shape, &model->y_count) ||
        !mul_perf_count(model->rank, model->out_shape, &model->out_count)) {
        return false;
    }
    if (model->out_count > SIZE_MAX / MUL_PERF_F16_BYTES ||
        model->x_count > SIZE_MAX / MUL_PERF_F16_BYTES ||
        model->y_count > SIZE_MAX / MUL_PERF_F16_BYTES) {
        return false;
    }
    model->out_bytes = model->out_count * MUL_PERF_F16_BYTES;
    model->x_bytes = model->x_count * MUL_PERF_F16_BYTES;
    model->y_bytes = model->y_count * MUL_PERF_F16_BYTES;
    params_bytes = mul_perf_align_up(model->x_bytes + model->y_bytes + model->out_bytes + 64U * 1024U * 1024U,
                                     4096U);
    scratch_bytes = mul_perf_align_up(model->out_bytes * 10U + 128U * 1024U * 1024U, 4096U);
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    {
        gd_status st = gd_context_create(&cfg, &model->ctx);
        if (st == GD_ERR_UNSUPPORTED) {
            printf("[MUL] skipped case=%s: no supported GPU backend\n", pcase->name);
            return false;
        }
        if (!mul_perf_status_ok(model->ctx, st, "gd_context_create")) {
            return false;
        }
    }
    MUL_PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(pcase->rank, pcase->x_shape), 256U, 101U, -1.0f, 1.0f, &model->x));
    MUL_PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(pcase->y_rank, pcase->y_shape), 256U, 202U, -1.0f, 1.0f, &model->y));
    MUL_PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(model->rank, model->out_shape), 256U, 303U, -1.0f, 1.0f, &model->grad));
    MUL_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    MUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    printf("[MUL] case=%s op=%s out_rank=%u out_elems=%zu out_tensor=%.1fMiB x_elems=%zu y_elems=%zu scratch_slot=%.1fMiB\n",
           pcase->name,
           mul_perf_op_name(pcase->op),
           model->rank,
           model->out_count,
           (double)model->out_bytes / MUL_PERF_MIB,
           model->x_count,
           model->y_count,
           (double)scratch_bytes / MUL_PERF_MIB);
    return true;
}

static void mul_perf_destroy_model(mul_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool mul_perf_run_once(mul_perf_model *model, mul_perf_op op)
{
    gd_tensor out;
    gd_tensor dx;
    gd_tensor dy;
    gd_scope_mode mode = op == MUL_PERF_FWD ? GD_SCOPE_INFER : GD_SCOPE_TRAIN;
    MUL_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, mode, gd_batch_empty()));
    if (op == MUL_PERF_FWD) {
        MUL_PERF_REQUIRE_OK(model->ctx, gd_mul(model->ctx, &model->x, &model->y, &out));
    } else if (op == MUL_PERF_BWD_X) {
        MUL_PERF_REQUIRE_OK(model->ctx,
                            gd_mul_backward(model->ctx, &model->x, &model->y, &model->grad, &dx, NULL));
    } else if (op == MUL_PERF_BWD_Y) {
        MUL_PERF_REQUIRE_OK(model->ctx,
                            gd_mul_backward(model->ctx, &model->x, &model->y, &model->grad, NULL, &dy));
    } else if (op == MUL_PERF_BWD_XY) {
        MUL_PERF_REQUIRE_OK(model->ctx,
                            gd_mul_backward(model->ctx, &model->x, &model->y, &model->grad, &dx, &dy));
    } else {
        return false;
    }
    MUL_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    MUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool mul_perf_bench_case(const mul_perf_case *pcase, int warmup, int iters)
{
    mul_perf_model model;
    double start;
    double elapsed;
    double seconds_per_iter;
    double bytes;
    double flops;
    int i;
    if (!mul_perf_create_model(pcase, &model)) {
        return false;
    }
    for (i = 0; i < warmup; ++i) {
        if (!mul_perf_run_once(&model, pcase->op)) {
            mul_perf_destroy_model(&model);
            return false;
        }
    }
    start = mul_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!mul_perf_run_once(&model, pcase->op)) {
            mul_perf_destroy_model(&model);
            return false;
        }
    }
    elapsed = mul_perf_now_seconds() - start;
    seconds_per_iter = elapsed / (double)iters;
    bytes = mul_perf_logical_bytes(&model, pcase->op);
    flops = mul_perf_flops(&model, pcase->op);
    printf("[MUL][PERF] case=%-24s op=%-6s dtype=f16 iters=%d ms=%.3f elems/s=%.2fG gflop/s=%.2f logical_bw=%.2fGB/s logical_bwi=%.2fGiB/s\n",
           pcase->name,
           mul_perf_op_name(pcase->op),
           iters,
           seconds_per_iter * 1.0e3,
           ((double)model.out_count / seconds_per_iter) / MUL_PERF_GB,
           (flops / seconds_per_iter) / MUL_PERF_GB,
           (bytes / seconds_per_iter) / MUL_PERF_GB,
           (bytes / seconds_per_iter) / MUL_PERF_GIB);
    mul_perf_destroy_model(&model);
    return true;
}

static bool mul_perf_case_selected(const char *profile, const char *name)
{
    return profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0 || strcmp(profile, name) == 0;
}

int main(void)
{
    const mul_perf_case cases[] = {
        {"direct_4096x1024_fwd", 2U, {4096, 1024}, 2U, {4096, 1024}, MUL_PERF_FWD},
        {"direct_8192x2048_fwd", 2U, {8192, 2048}, 2U, {8192, 2048}, MUL_PERF_FWD},
        {"row_4096x1024_fwd", 2U, {4096, 1024}, 1U, {1024}, MUL_PERF_FWD},
        {"row_8192x2048_fwd", 2U, {8192, 2048}, 1U, {2048}, MUL_PERF_FWD},
        {"generic_8x512x1024_fwd", 3U, {8, 512, 1024}, 3U, {8, 1, 1024}, MUL_PERF_FWD},
        {"suffix8_8x512x1024_bwd_y", 3U, {8, 512, 1024}, 2U, {512, 1024}, MUL_PERF_BWD_Y},
        {"direct_4096x1024_bwd_xy", 2U, {4096, 1024}, 2U, {4096, 1024}, MUL_PERF_BWD_XY},
        {"direct_8192x2048_bwd_xy", 2U, {8192, 2048}, 2U, {8192, 2048}, MUL_PERF_BWD_XY},
        {"row_4096x1024_bwd_x", 2U, {4096, 1024}, 1U, {1024}, MUL_PERF_BWD_X},
        {"row_4096x1024_bwd_y", 2U, {4096, 1024}, 1U, {1024}, MUL_PERF_BWD_Y},
        {"row_4096x1024_bwd_xy", 2U, {4096, 1024}, 1U, {1024}, MUL_PERF_BWD_XY},
        {"row_8192x2048_bwd_xy", 2U, {8192, 2048}, 1U, {2048}, MUL_PERF_BWD_XY},
    };
    const char *profile = getenv("GD_MUL_PERF_PROFILE");
    int warmup = mul_perf_env_int("GD_MUL_PERF_WARMUP", 3, 0, 100);
    int iters = mul_perf_env_int("GD_MUL_PERF_ITERS", 10, 1, 1000);
    bool ran = false;
    size_t i;
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!mul_perf_case_selected(profile, cases[i].name)) {
            continue;
        }
        ran = true;
        if (!mul_perf_bench_case(&cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr, "[MUL][FAIL] no case matched GD_MUL_PERF_PROFILE=%s\n", profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}

#endif /* GD_MUL_PERF_TEST_MAIN */
