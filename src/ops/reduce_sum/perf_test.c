/*
 * gd_reduce_sum Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=reduce_sum
 *
 * Optional environment:
 *   GD_REDUCE_SUM_PERF_PROFILE=smoke|all|<case-name>
 *   GD_REDUCE_SUM_PERF_WARMUP=10
 *   GD_REDUCE_SUM_PERF_ITERS=100
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

#define REDUCE_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct reduce_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    int32_t axis;
} reduce_perf_case;

typedef struct reduce_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor scalar_grad;
    gd_tensor axis_grad;
    uint32_t axis_rank;
    int64_t axis_shape[GD_MAX_DIMS];
    size_t x_count;
    size_t axis_count;
    size_t elem_size;
    size_t x_bytes;
    size_t axis_bytes;
} reduce_perf_model;

typedef bool (*reduce_perf_run_fn)(reduce_perf_model *model, int32_t axis);

static double reduce_perf_now_seconds(void)
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

static int reduce_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t reduce_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static const char *reduce_perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static size_t reduce_perf_dtype_size(gd_dtype dtype)
{
    if (dtype == GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool reduce_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[REDUCE_SUM][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define REDUCE_PERF_REQUIRE_OK(ctx, expr)                             \
    do {                                                              \
        gd_status reduce_perf_st__ = (expr);                          \
        if (!reduce_perf_status_ok((ctx), reduce_perf_st__, #expr)) { \
            return false;                                             \
        }                                                             \
    } while (0)

static bool reduce_perf_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t dim;
    size_t count = 1U;
    if (rank > GD_MAX_DIMS || out_count == NULL || (rank != 0U && shape == NULL)) {
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

static bool reduce_perf_axis_shape(const reduce_perf_case *pcase,
                                   uint32_t *out_rank,
                                   int64_t out_shape[GD_MAX_DIMS])
{
    int32_t axis;
    uint32_t dim;
    uint32_t j = 0U;
    if (pcase == NULL || out_rank == NULL || out_shape == NULL || pcase->rank == 0U ||
        pcase->rank > GD_MAX_DIMS) {
        return false;
    }
    axis = pcase->axis < 0 ? pcase->axis + (int32_t)pcase->rank : pcase->axis;
    if (axis < 0 || axis >= (int32_t)pcase->rank) {
        return false;
    }
    memset(out_shape, 0, sizeof(int64_t) * GD_MAX_DIMS);
    *out_rank = pcase->rank - 1U;
    for (dim = 0U; dim < pcase->rank; ++dim) {
        if (dim == (uint32_t)axis) {
            continue;
        }
        out_shape[j] = pcase->shape[dim];
        j += 1U;
    }
    return true;
}

static bool reduce_perf_case_selected(const reduce_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tail_1x513_f16_last") == 0 ||
               strcmp(pcase->name, "act_4096x1024_f16_last") == 0 ||
               strcmp(pcase->name, "act_4096x1024_f16_axis0") == 0 ||
               strcmp(pcase->name, "act_4096x1024_f32_last") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool reduce_perf_write_scalar_one(gd_context *ctx, gd_tensor *tensor, gd_dtype dtype)
{
    if (dtype == GD_DTYPE_F16) {
        const uint16_t one = 0x3C00U;
        REDUCE_PERF_REQUIRE_OK(ctx, gd_tensor_write(ctx, tensor, &one, sizeof(one)));
        return true;
    }
    if (dtype == GD_DTYPE_F32) {
        const float one = 1.0f;
        REDUCE_PERF_REQUIRE_OK(ctx, gd_tensor_write(ctx, tensor, &one, sizeof(one)));
        return true;
    }
    return false;
}

static bool reduce_perf_create_model(const reduce_perf_case *pcase, reduce_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->elem_size = reduce_perf_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || !reduce_perf_count(pcase->rank, pcase->shape, &model->x_count) ||
        !reduce_perf_axis_shape(pcase, &model->axis_rank, model->axis_shape) ||
        !reduce_perf_count(model->axis_rank, model->axis_shape, &model->axis_count)) {
        fprintf(stderr, "[REDUCE_SUM][FAIL] invalid case=%s\n", pcase->name);
        return false;
    }
    if (model->x_count > SIZE_MAX / model->elem_size || model->axis_count > SIZE_MAX / model->elem_size) {
        return false;
    }
    model->x_bytes = model->x_count * model->elem_size;
    model->axis_bytes = model->axis_count * model->elem_size;
    params_bytes = reduce_perf_align_up(model->x_bytes + model->axis_bytes + 64U * 1024U * 1024U, 4096U);
    scratch_bytes = reduce_perf_align_up(model->x_bytes * 8U + model->axis_bytes * 4U +
                                             256U * 1024U * 1024U,
                                         4096U);
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
        printf("[REDUCE_SUM] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!reduce_perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    REDUCE_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  pcase->dtype,
                                                  pcase->rank,
                                                  pcase->shape,
                                                  256U,
                                                  1001U,
                                                  -1.0f,
                                                  1.0f,
                                                  &model->x));
    REDUCE_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_empty(model->ctx,
                                           GD_ARENA_PARAMS,
                                           pcase->dtype,
                                           0U,
                                           NULL,
                                           256U,
                                           &model->scalar_grad));
    if (!reduce_perf_write_scalar_one(model->ctx, &model->scalar_grad, pcase->dtype)) {
        return false;
    }
    REDUCE_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  pcase->dtype,
                                                  model->axis_rank,
                                                  model->axis_rank == 0U ? NULL : model->axis_shape,
                                                  256U,
                                                  2002U,
                                                  -0.5f,
                                                  0.5f,
                                                  &model->axis_grad));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void reduce_perf_destroy_model(reduce_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool reduce_perf_run_all_forward(reduce_perf_model *model, int32_t axis)
{
    gd_tensor out;
    (void)axis;
    model->x.requires_grad = false;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_reduce_sum(model->ctx, &model->x, &out));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reduce_perf_run_all_backward(reduce_perf_model *model, int32_t axis)
{
    gd_tensor dx;
    (void)axis;
    model->x.requires_grad = false;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    REDUCE_PERF_REQUIRE_OK(model->ctx,
                           gd_reduce_sum_backward(model->ctx, &model->x, &model->scalar_grad, &dx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reduce_perf_run_all_autograd(reduce_perf_model *model, int32_t axis)
{
    gd_tensor out;
    gd_tensor dx;
    (void)axis;
    model->x.requires_grad = true;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_reduce_sum(model->ctx, &model->x, &out));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->scalar_grad));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reduce_perf_run_axis_forward(reduce_perf_model *model, int32_t axis)
{
    gd_tensor out;
    model->x.requires_grad = false;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_reduce_sum_axis(model->ctx, &model->x, axis, false, &out));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reduce_perf_run_axis_backward(reduce_perf_model *model, int32_t axis)
{
    gd_tensor dx;
    model->x.requires_grad = false;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_EVAL));
    REDUCE_PERF_REQUIRE_OK(model->ctx,
                           gd_reduce_sum_axis_backward(model->ctx,
                                                       &model->x,
                                                       &model->axis_grad,
                                                       axis,
                                                       false,
                                                       &dx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool reduce_perf_run_axis_autograd(reduce_perf_model *model, int32_t axis)
{
    gd_tensor out;
    gd_tensor dx;
    model->x.requires_grad = true;
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, GD_SCOPE_TRAIN));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_reduce_sum_axis(model->ctx, &model->x, axis, false, &out));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->axis_grad));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    REDUCE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static double reduce_perf_measure(reduce_perf_model *model,
                                  reduce_perf_run_fn run,
                                  int32_t axis,
                                  int warmup,
                                  int iters)
{
    double start;
    double elapsed;
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!run(model, axis)) {
            return -1.0;
        }
    }
    start = reduce_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!run(model, axis)) {
            return -1.0;
        }
    }
    elapsed = reduce_perf_now_seconds() - start;
    return elapsed / (double)iters;
}

static void reduce_perf_print_shape(uint32_t rank, const int64_t shape[GD_MAX_DIMS])
{
    uint32_t dim;
    printf("[");
    for (dim = 0U; dim < rank; ++dim) {
        printf("%s%lld", dim == 0U ? "" : "x", (long long)shape[dim]);
    }
    printf("]");
}

static void reduce_perf_print_result(const char *op,
                                     const reduce_perf_case *pcase,
                                     const reduce_perf_model *model,
                                     double seconds,
                                     double logical_bytes,
                                     int warmup,
                                     int iters)
{
    double gib_s = (logical_bytes / REDUCE_PERF_GIB) / seconds;
    double gelem_s = ((double)model->x_count / seconds) / 1.0e9;
    printf("[REDUCE_SUM][%s] case=%s dtype=%s shape=", op, pcase->name, reduce_perf_dtype_name(pcase->dtype));
    reduce_perf_print_shape(pcase->rank, pcase->shape);
    printf(" axis=%d axis_out=", pcase->axis);
    reduce_perf_print_shape(model->axis_rank, model->axis_shape);
    printf(" warmup=%d iters=%d avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f\n",
           warmup,
           iters,
           seconds * 1000.0,
           gib_s,
           gelem_s);
}

static bool reduce_perf_run_case(const reduce_perf_case *pcase, int warmup, int iters)
{
    reduce_perf_model model;
    double all_fwd_s;
    double all_bwd_s;
    double all_auto_s;
    double axis_fwd_s;
    double axis_bwd_s;
    double axis_auto_s;
    double scalar_bytes;
    if (!reduce_perf_create_model(pcase, &model)) {
        return false;
    }
    scalar_bytes = (double)model.elem_size;
    printf("[REDUCE_SUM] case=%s dtype=%s elems=%zu tensor=%.1fMiB axis_count=%zu\n",
           pcase->name,
           reduce_perf_dtype_name(pcase->dtype),
           model.x_count,
           (double)model.x_bytes / (1024.0 * 1024.0),
           model.axis_count);
    all_fwd_s = reduce_perf_measure(&model, reduce_perf_run_all_forward, pcase->axis, warmup, iters);
    all_bwd_s = reduce_perf_measure(&model, reduce_perf_run_all_backward, pcase->axis, warmup, iters);
    all_auto_s = reduce_perf_measure(&model, reduce_perf_run_all_autograd, pcase->axis, warmup, iters);
    axis_fwd_s = reduce_perf_measure(&model, reduce_perf_run_axis_forward, pcase->axis, warmup, iters);
    axis_bwd_s = reduce_perf_measure(&model, reduce_perf_run_axis_backward, pcase->axis, warmup, iters);
    axis_auto_s = reduce_perf_measure(&model, reduce_perf_run_axis_autograd, pcase->axis, warmup, iters);
    if (all_fwd_s > 0.0) {
        reduce_perf_print_result("all_fwd", pcase, &model, all_fwd_s, (double)model.x_bytes + scalar_bytes, warmup, iters);
    }
    if (all_bwd_s > 0.0) {
        reduce_perf_print_result("all_bwd_direct", pcase, &model, all_bwd_s, (double)model.x_bytes + scalar_bytes, warmup, iters);
    }
    if (all_auto_s > 0.0) {
        reduce_perf_print_result("all_fwd_bwd_autograd",
                                 pcase,
                                 &model,
                                 all_auto_s,
                                 (double)model.x_bytes * 2.0 + scalar_bytes * 2.0,
                                 warmup,
                                 iters);
    }
    if (axis_fwd_s > 0.0) {
        reduce_perf_print_result("axis_fwd", pcase, &model, axis_fwd_s, (double)model.x_bytes + (double)model.axis_bytes, warmup, iters);
    }
    if (axis_bwd_s > 0.0) {
        reduce_perf_print_result("axis_bwd_direct",
                                 pcase,
                                 &model,
                                 axis_bwd_s,
                                 (double)model.x_bytes + (double)model.axis_bytes,
                                 warmup,
                                 iters);
    }
    if (axis_auto_s > 0.0) {
        reduce_perf_print_result("axis_fwd_bwd_autograd",
                                 pcase,
                                 &model,
                                 axis_auto_s,
                                 (double)model.x_bytes * 2.0 + (double)model.axis_bytes * 2.0,
                                 warmup,
                                 iters);
    }
    reduce_perf_destroy_model(&model);
    return all_fwd_s > 0.0 && all_bwd_s > 0.0 && all_auto_s > 0.0 && axis_fwd_s > 0.0 &&
           axis_bwd_s > 0.0 && axis_auto_s > 0.0;
}

int main(void)
{
    static const reduce_perf_case cases[] = {
        {"tail_1x513_f16_last", GD_DTYPE_F16, 2U, {1, 513}, -1},
        {"act_4096x1024_f16_last", GD_DTYPE_F16, 2U, {4096, 1024}, -1},
        {"act_4096x1024_f16_axis0", GD_DTYPE_F16, 2U, {4096, 1024}, 0},
        {"act_8192x2048_f16_last", GD_DTYPE_F16, 2U, {8192, 2048}, -1},
        {"rank3_8x512x1024_f16_last", GD_DTYPE_F16, 3U, {8, 512, 1024}, -1},
        {"rank3_8x512x1024_f16_middle", GD_DTYPE_F16, 3U, {8, 512, 1024}, 1},
        {"mlp_2048x11008_f16_last", GD_DTYPE_F16, 2U, {2048, 11008}, -1},
        {"act_4096x1024_f32_last", GD_DTYPE_F32, 2U, {4096, 1024}, -1},
    };
    const char *profile = getenv("GD_REDUCE_SUM_PERF_PROFILE");
    int warmup = reduce_perf_env_int("GD_REDUCE_SUM_PERF_WARMUP", 10, 0, 10000);
    int iters = reduce_perf_env_int("GD_REDUCE_SUM_PERF_ITERS", 100, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[REDUCE_SUM] public API perf: warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!reduce_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!reduce_perf_run_case(&cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr,
                "[REDUCE_SUM][FAIL] no case matched GD_REDUCE_SUM_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}
