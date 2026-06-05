/*
 * v2 public elementwise/reduction performance probe.
 *
 * Measures the public gd_add/gd_sub/gd_mul broadcast paths plus all-elements and
 * axis gd_reduce_sum/gd_reduce_mean, sparse cross entropy, and scalar-loss MSE
 * graph on real activation-sized tensors.
 * Wall time includes public validation, scoped arena allocation, Metal command
 * buffer submit, and synchronization.
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

#define PERF_GB (1024.0 * 1024.0 * 1024.0)
#define PERF_MB (1024.0 * 1024.0)

#if defined(__APPLE__)
static double perf_now_seconds(void)
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
static double perf_now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#endif

static int perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static bool perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[ELEM][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define PERF_REQUIRE_OK(ctx, expr)                         \
    do {                                                   \
        gd_status perf_st__ = (expr);                      \
        if (!perf_status_ok((ctx), perf_st__, #expr)) {    \
            return false;                                  \
        }                                                  \
    } while (0)

typedef enum perf_op_kind {
    PERF_OP_ADD = 1,
    PERF_OP_SUB = 2,
    PERF_OP_MUL = 3,
    PERF_OP_ADD_BIAS = 4,
    PERF_OP_MUL_BIAS = 5,
    PERF_OP_REDUCE_SUM = 6,
    PERF_OP_REDUCE_MEAN = 7,
    PERF_OP_REDUCE_SUM_AXIS1 = 8,
    PERF_OP_REDUCE_MEAN_AXIS1 = 9,
    PERF_OP_REDUCE_MEAN_AXIS1_BWD = 10,
    PERF_OP_CROSS_ENTROPY = 11,
    PERF_OP_CROSS_ENTROPY_BWD = 12,
    PERF_OP_MSE_GRAPH = 13,
} perf_op_kind;

typedef struct perf_case {
    const char *name;
    int64_t tokens;
    int64_t hidden;
    gd_dtype dtype;
} perf_case;

typedef struct perf_model {
    const char *name;
    gd_context *ctx;
    gd_tensor x;
    gd_tensor y;
    gd_tensor bias;
    gd_tensor target;
    gd_tensor labels;
    int64_t tokens;
    int64_t hidden;
    size_t count;
    size_t elem_size;
    gd_dtype dtype;
} perf_model;

static size_t perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static const char *perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static bool perf_create_model(const perf_case *pcase, perf_model *model)
{
    gd_memory_config cfg;
    int64_t x_shape[2];
    int64_t bias_shape[1];
    int64_t label_shape[1];
    int32_t *labels;
    size_t tensor_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (pcase == NULL || model == NULL || pcase->tokens <= 0 || pcase->hidden <= 0) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->name = pcase->name;
    model->tokens = pcase->tokens;
    model->hidden = pcase->hidden;
    model->dtype = pcase->dtype;
    model->elem_size = gd_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || (uint64_t)pcase->tokens > (uint64_t)(SIZE_MAX / pcase->hidden)) {
        return false;
    }
    model->count = (size_t)pcase->tokens * (size_t)pcase->hidden;
    if (model->count > SIZE_MAX / model->elem_size) {
        return false;
    }
    tensor_bytes = model->count * model->elem_size;
    scratch_bytes = perf_align_up(tensor_bytes * 12U + 128U * 1024U * 1024U, 4096U);
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = perf_align_up(tensor_bytes * 3U + (size_t)pcase->hidden * model->elem_size +
                                         (size_t)pcase->tokens * sizeof(int32_t) +
                                         64U * 1024U * 1024U,
                                     4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    st = gd_context_create(&cfg, &model->ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[ELEM] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    x_shape[0] = pcase->tokens;
    x_shape[1] = pcase->hidden;
    bias_shape[0] = pcase->hidden;
    label_shape[0] = pcase->tokens;
    PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       2U,
                                                       x_shape,
                                                       256U,
                                                       11U,
                                                       -1.0f,
                                                       1.0f,
                                                       &model->x));
    PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       2U,
                                                       x_shape,
                                                       256U,
                                                       22U,
                                                       -1.0f,
                                                       1.0f,
                                                       &model->y));
    PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       2U,
                                                       x_shape,
                                                       256U,
                                                       33U,
                                                       -1.0f,
                                                       1.0f,
                                                       &model->target));
    PERF_REQUIRE_OK(model->ctx, gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       1U,
                                                       bias_shape,
                                                       256U,
                                                       44U,
                                                       -0.25f,
                                                       0.25f,
                                                       &model->bias));
    labels = (int32_t *)malloc((size_t)pcase->tokens * sizeof(*labels));
    if (labels == NULL) {
        return false;
    }
    for (int64_t i = 0; i < pcase->tokens; ++i) {
        labels[i] = (int32_t)(i % pcase->hidden);
    }
    st = gd_tensor_empty(model->ctx,
                         GD_ARENA_PARAMS,
                         GD_DTYPE_I32,
                         1U,
                         label_shape,
                         256U,
                         &model->labels);
    if (!perf_status_ok(model->ctx, st, "gd_tensor_empty labels")) {
        free(labels);
        return false;
    }
    st = gd_tensor_write(model->ctx,
                         &model->labels,
                         labels,
                         (size_t)pcase->tokens * sizeof(*labels));
    free(labels);
    if (!perf_status_ok(model->ctx, st, "gd_tensor_write labels")) {
        return false;
    }
    model->x.requires_grad = true;
    model->y.requires_grad = false;
    model->target.requires_grad = false;
    model->bias.requires_grad = false;
    model->labels.requires_grad = false;
    PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    printf("[ELEM] case=%s dtype=%s shape=%lldx%lld elems=%zu tensor=%.1f MiB scratch_slot=%.1f MiB\n",
           pcase->name,
           perf_dtype_name(pcase->dtype),
           (long long)pcase->tokens,
           (long long)pcase->hidden,
           model->count,
           (double)tensor_bytes / PERF_MB,
           (double)scratch_bytes / PERF_MB);
    return true;
}

static void perf_destroy_model(perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool perf_run_once(perf_model *model, perf_op_kind kind)
{
    gd_tensor out;
    gd_tensor tmp;
    gd_tensor loss;
    gd_tensor sq;
    gd_tensor diff;
    gd_scope_mode mode = (kind == PERF_OP_MSE_GRAPH || kind == PERF_OP_REDUCE_MEAN_AXIS1_BWD ||
                          kind == PERF_OP_CROSS_ENTROPY_BWD) ?
                             GD_SCOPE_TRAIN : GD_SCOPE_INFER;
    PERF_REQUIRE_OK(model->ctx, gd_begin(model->ctx, mode));
    switch (kind) {
    case PERF_OP_ADD:
        PERF_REQUIRE_OK(model->ctx, gd_add(model->ctx, &model->x, &model->y, &out));
        break;
    case PERF_OP_SUB:
        PERF_REQUIRE_OK(model->ctx, gd_sub(model->ctx, &model->x, &model->y, &out));
        break;
    case PERF_OP_MUL:
        PERF_REQUIRE_OK(model->ctx, gd_mul(model->ctx, &model->x, &model->y, &out));
        break;
    case PERF_OP_ADD_BIAS:
        PERF_REQUIRE_OK(model->ctx, gd_add(model->ctx, &model->x, &model->bias, &out));
        break;
    case PERF_OP_MUL_BIAS:
        PERF_REQUIRE_OK(model->ctx, gd_mul(model->ctx, &model->x, &model->bias, &out));
        break;
    case PERF_OP_REDUCE_SUM:
        PERF_REQUIRE_OK(model->ctx, gd_reduce_sum(model->ctx, &model->x, &out));
        break;
    case PERF_OP_REDUCE_MEAN:
        PERF_REQUIRE_OK(model->ctx, gd_reduce_mean(model->ctx, &model->x, &out));
        break;
    case PERF_OP_REDUCE_SUM_AXIS1:
        PERF_REQUIRE_OK(model->ctx, gd_reduce_sum_axis(model->ctx, &model->x, 1, false, &out));
        break;
    case PERF_OP_REDUCE_MEAN_AXIS1:
        PERF_REQUIRE_OK(model->ctx, gd_reduce_mean_axis(model->ctx, &model->x, 1, false, &out));
        break;
    case PERF_OP_REDUCE_MEAN_AXIS1_BWD:
        PERF_REQUIRE_OK(model->ctx, gd_reduce_mean_axis(model->ctx, &model->x, 1, false, &out));
        PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, NULL));
        PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &tmp));
        break;
    case PERF_OP_CROSS_ENTROPY:
        PERF_REQUIRE_OK(model->ctx, gd_cross_entropy(model->ctx, &model->x, &model->labels, &loss));
        break;
    case PERF_OP_CROSS_ENTROPY_BWD:
        PERF_REQUIRE_OK(model->ctx, gd_cross_entropy(model->ctx, &model->x, &model->labels, &loss));
        PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &loss, NULL));
        PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &tmp));
        break;
    case PERF_OP_MSE_GRAPH:
        PERF_REQUIRE_OK(model->ctx, gd_sub(model->ctx, &model->x, &model->target, &diff));
        PERF_REQUIRE_OK(model->ctx, gd_mul(model->ctx, &diff, &diff, &sq));
        PERF_REQUIRE_OK(model->ctx, gd_reduce_mean(model->ctx, &sq, &loss));
        PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &loss, NULL));
        PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &tmp));
        break;
    default:
        return false;
    }
    PERF_REQUIRE_OK(model->ctx, gd_end(model->ctx));
    PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static const char *perf_op_name(perf_op_kind kind)
{
    switch (kind) {
    case PERF_OP_ADD:
        return "add";
    case PERF_OP_SUB:
        return "sub";
    case PERF_OP_MUL:
        return "mul";
    case PERF_OP_ADD_BIAS:
        return "add_bias_bcast";
    case PERF_OP_MUL_BIAS:
        return "mul_bias_bcast";
    case PERF_OP_REDUCE_SUM:
        return "reduce_sum";
    case PERF_OP_REDUCE_MEAN:
        return "reduce_mean";
    case PERF_OP_REDUCE_SUM_AXIS1:
        return "reduce_sum_axis1";
    case PERF_OP_REDUCE_MEAN_AXIS1:
        return "reduce_mean_axis1";
    case PERF_OP_REDUCE_MEAN_AXIS1_BWD:
        return "reduce_mean_axis1_bwd";
    case PERF_OP_CROSS_ENTROPY:
        return "cross_entropy";
    case PERF_OP_CROSS_ENTROPY_BWD:
        return "cross_entropy_bwd";
    case PERF_OP_MSE_GRAPH:
        return "mse_fwd_bwd";
    default:
        return "unknown";
    }
}

static double perf_effective_bytes(const perf_model *model, perf_op_kind kind)
{
    const double tensor = (double)model->count * (double)model->elem_size;
    const double bias = (double)model->hidden * (double)model->elem_size;
    switch (kind) {
    case PERF_OP_ADD:
    case PERF_OP_SUB:
    case PERF_OP_MUL:
        return tensor * 3.0;
    case PERF_OP_ADD_BIAS:
    case PERF_OP_MUL_BIAS:
        return tensor * 2.0 + bias;
    case PERF_OP_REDUCE_SUM:
    case PERF_OP_REDUCE_MEAN:
        return tensor;
    case PERF_OP_REDUCE_SUM_AXIS1:
    case PERF_OP_REDUCE_MEAN_AXIS1:
        return tensor + (double)model->tokens * (double)model->elem_size;
    case PERF_OP_REDUCE_MEAN_AXIS1_BWD:
        return tensor * 4.0 + (double)model->tokens * (double)model->elem_size * 2.0;
    case PERF_OP_CROSS_ENTROPY:
        return tensor * 2.0 + (double)model->tokens * (double)(sizeof(int32_t) + sizeof(float));
    case PERF_OP_CROSS_ENTROPY_BWD:
        return tensor * 6.0 + (double)model->tokens * (double)(sizeof(int32_t) + sizeof(float));
    case PERF_OP_MSE_GRAPH:
        /* sub fwd + mul fwd + reduce read + reduce bwd broadcast + two mul bwd + two accumulates + sub bwd */
        return tensor * 16.0;
    default:
        return tensor;
    }
}

static bool perf_bench_op(perf_model *model, perf_op_kind kind, int warmup, int iters)
{
    double start;
    double elapsed;
    double seconds_per_iter;
    double bytes;
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!perf_run_once(model, kind)) {
            return false;
        }
    }
    start = perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!perf_run_once(model, kind)) {
            return false;
        }
    }
    elapsed = perf_now_seconds() - start;
    seconds_per_iter = elapsed / (double)iters;
    bytes = perf_effective_bytes(model, kind);
    printf("[ELEM][PERF] case=%s dtype=%s op=%-22s iters=%d ms=%.3f elems/s=%.2fG effective_bw=%.2fGB/s\n",
           model->name,
           perf_dtype_name(model->dtype),
           perf_op_name(kind),
           iters,
           seconds_per_iter * 1.0e3,
           ((double)model->count / seconds_per_iter) / 1.0e9,
           (bytes / seconds_per_iter) / PERF_GB);
    return true;
}

static bool perf_run_case(const perf_case *pcase, int warmup, int iters)
{
    static const perf_op_kind ops[] = {
        PERF_OP_ADD,
        PERF_OP_SUB,
        PERF_OP_MUL,
        PERF_OP_ADD_BIAS,
        PERF_OP_MUL_BIAS,
        PERF_OP_REDUCE_SUM,
        PERF_OP_REDUCE_MEAN,
        PERF_OP_REDUCE_SUM_AXIS1,
        PERF_OP_REDUCE_MEAN_AXIS1,
        PERF_OP_REDUCE_MEAN_AXIS1_BWD,
        PERF_OP_CROSS_ENTROPY,
        PERF_OP_CROSS_ENTROPY_BWD,
        PERF_OP_MSE_GRAPH,
    };
    perf_model model;
    size_t i;
    if (!perf_create_model(pcase, &model)) {
        return false;
    }
    for (i = 0U; i < sizeof(ops) / sizeof(ops[0]); ++i) {
        if (pcase->dtype != GD_DTYPE_F16 &&
            (ops[i] == PERF_OP_MUL || ops[i] == PERF_OP_MUL_BIAS ||
             ops[i] == PERF_OP_CROSS_ENTROPY || ops[i] == PERF_OP_CROSS_ENTROPY_BWD ||
             ops[i] == PERF_OP_MSE_GRAPH)) {
            printf("[ELEM][PERF] case=%s dtype=%s op=%-22s skipped=f16_only\n",
                   pcase->name,
                   perf_dtype_name(pcase->dtype),
                   perf_op_name(ops[i]));
            continue;
        }
        if (!perf_bench_op(&model, ops[i], warmup, iters)) {
            perf_destroy_model(&model);
            return false;
        }
    }
    perf_destroy_model(&model);
    return true;
}

static bool perf_case_selected(const char *profile, const char *name)
{
    return profile == NULL || profile[0] == '\0' || strcmp(profile, "all") == 0 || strcmp(profile, name) == 0;
}

int main(void)
{
    const perf_case cases[] = {
        {"h1024_f16", 4096, 1024, GD_DTYPE_F16},
        {"h2048_f16", 8192, 2048, GD_DTYPE_F16},
        {"h1024_f32", 4096, 1024, GD_DTYPE_F32},
    };
    const char *profile = getenv("GD_QA_ELEM_PROFILE");
    int warmup = perf_env_int("GD_QA_ELEM_WARMUP", 3, 0, 100);
    int iters = perf_env_int("GD_QA_ELEM_ITERS", 10, 1, 1000);
    bool ran = false;
    size_t i;
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (!perf_case_selected(profile, cases[i].name)) {
            continue;
        }
        ran = true;
        if (!perf_run_case(&cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr, "[ELEM][FAIL] no case matched GD_QA_ELEM_PROFILE=%s\n", profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}
