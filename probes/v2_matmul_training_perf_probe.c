/*
 * v2 public matmul training performance probe.
 *
 * Links against libgradients and measures the public gd_matmul / gd_matmul_backward
 * path on transformer-like workloads. This exercises the production Metal
 * metallib F16 GEMM kernels through the same arena/scoped execution API that a
 * training loop uses, so wall time includes public API validation, arena
 * allocation, command-buffer submit, and synchronization.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/gradients.h>

#include <inttypes.h>
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

#define PERF_MB (1024.0 * 1024.0)
#define PERF_GB (1024.0 * 1024.0 * 1024.0)

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

static bool perf_env_bool(const char *name, bool fallback)
{
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') {
        return fallback;
    }
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "FALSE") != 0;
}

static float perf_abs_f32(float x)
{
    return x < 0.0f ? -x : x;
}

static uint16_t perf_f32_to_f16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign;
    int32_t exp;
    uint32_t mant;
    uint32_t out_exp;
    uint32_t out_mant;
    v.f = value;
    sign = (v.u >> 16) & 0x8000U;
    exp = (int32_t)((v.u >> 23) & 0xffU) - 127;
    mant = v.u & 0x7fffffU;
    if (((v.u >> 23) & 0xffU) == 0xffU) {
        return (uint16_t)(sign | (mant == 0U ? 0x7c00U : 0x7e00U));
    }
    if (exp > 15) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (exp < -14) {
        uint32_t shifted;
        uint32_t remainder;
        uint32_t halfway;
        int32_t shift = -14 - exp;
        if (shift > 24) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        shifted = mant >> (uint32_t)(shift + 13);
        remainder = mant & ((1U << (uint32_t)(shift + 13)) - 1U);
        halfway = 1U << (uint32_t)(shift + 12);
        if (remainder > halfway || (remainder == halfway && (shifted & 1U) != 0U)) {
            shifted += 1U;
        }
        return (uint16_t)(sign | shifted);
    }
    out_exp = (uint32_t)(exp + 15);
    out_mant = mant >> 13;
    {
        uint32_t remainder = mant & 0x1fffU;
        if (remainder > 0x1000U || (remainder == 0x1000U && (out_mant & 1U) != 0U)) {
            out_mant += 1U;
            if (out_mant == 0x400U) {
                out_mant = 0U;
                out_exp += 1U;
                if (out_exp >= 31U) {
                    return (uint16_t)(sign | 0x7c00U);
                }
            }
        }
    }
    return (uint16_t)(sign | (out_exp << 10) | out_mant);
}

static float perf_f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    uint32_t exp = ((uint32_t)bits >> 10) & 0x1fU;
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0U) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1;
            exp -= 1U;
        }
        mant &= 0x3ffU;
        exp += 1U;
    } else if (exp == 31U) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((exp + (127U - 15U)) << 23) | (mant << 13);
    return v.f;
}

static void perf_fill_x(uint16_t *x, uint32_t m, uint32_t k)
{
    uint32_t i;
    uint32_t j;
    for (i = 0U; i < m; ++i) {
        for (j = 0U; j < k; ++j) {
            x[i * k + j] = perf_f32_to_f16_bits(0.25f + 0.125f * (float)i - 0.0625f * (float)j);
        }
    }
}

static void perf_fill_w(uint16_t *w, uint32_t k, uint32_t n)
{
    uint32_t i;
    uint32_t j;
    for (i = 0U; i < k; ++i) {
        for (j = 0U; j < n; ++j) {
            w[i * n + j] = perf_f32_to_f16_bits(-0.5f + 0.125f * (float)j + 0.03125f * (float)i);
        }
    }
}

static float perf_ref_matmul_value(const uint16_t *x,
                                   const uint16_t *w,
                                   uint32_t row,
                                   uint32_t col,
                                   uint32_t k,
                                   uint32_t n)
{
    uint32_t p;
    float sum = 0.0f;
    for (p = 0U; p < k; ++p) {
        sum += perf_f16_bits_to_f32(x[row * k + p]) * perf_f16_bits_to_f32(w[p * n + col]);
    }
    return perf_f16_bits_to_f32(perf_f32_to_f16_bits(sum));
}

static bool perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[PERF][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define PERF_REQUIRE_OK(ctx, expr)                   \
    do {                                             \
        gd_status perf_st__ = (expr);                \
        if (!perf_status_ok((ctx), perf_st__, #expr)) { \
            return false;                            \
        }                                            \
    } while (0)

typedef struct perf_case {
    const char *name;
    int64_t tokens;
    int64_t hidden;
    int64_t intermediate;
    int64_t heads;
} perf_case;

typedef struct perf_model {
    gd_tensor w_qkv;
    gd_tensor w_proj;
    gd_tensor w_gate;
    gd_tensor w_up;
    gd_tensor w_down;
    gd_tensor b_qkv;
    gd_tensor b_proj;
    gd_tensor b_gate;
    gd_tensor b_up;
    gd_tensor b_down;
    int64_t tokens;
    int64_t hidden;
    int64_t intermediate;
    int64_t heads;
    int64_t head_dim;
    uint32_t matmuls;
    uint32_t bias_reductions;
    double forward_flops;
    double backward_flops;
    size_t params_bytes;
    size_t scratch_bytes;
    size_t backward_scratch_bytes;
    size_t data_bytes;
} perf_model;

typedef struct perf_totals {
    uint32_t profiles_run;
    uint32_t failures;
    uint32_t warnings;
    bool skipped;
    double best_sync_tflops;
    double best_pipeline_tflops;
    const char *best_sync_profile;
    const char *best_pipeline_profile;
    bool matmul_backward_ready;
} perf_totals;

static bool perf_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool perf_align_up_size(size_t value, size_t alignment, size_t *out)
{
    if (out == NULL || alignment == 0U || (alignment & (alignment - 1U)) != 0U ||
        value > SIZE_MAX - (alignment - 1U)) {
        return false;
    }
    *out = (value + alignment - 1U) & ~(alignment - 1U);
    return true;
}

static bool perf_tensor_bytes_2d(int64_t rows, int64_t cols, size_t *out)
{
    size_t r;
    size_t c;
    if (out == NULL || rows <= 0 || cols <= 0) {
        return false;
    }
    r = (size_t)rows;
    c = (size_t)cols;
    if (r > SIZE_MAX / c || r * c > SIZE_MAX / 2U) {
        return false;
    }
    *out = r * c * 2U;
    return true;
}

static bool perf_tensor_bytes_1d(int64_t count, size_t *out)
{
    if (out == NULL || count <= 0 || (uint64_t)count > (uint64_t)(SIZE_MAX / 2U)) {
        return false;
    }
    *out = (size_t)count * 2U;
    return true;
}

static double perf_matmul_flops(int64_t m, int64_t k, int64_t n)
{
    return 2.0 * (double)m * (double)k * (double)n;
}

static bool perf_add_matmul_backward_scratch(size_t current,
                                             int64_t m,
                                             int64_t k,
                                             int64_t n,
                                             size_t *out)
{
    size_t dx;
    size_t dw;
    size_t tmp;
    return perf_tensor_bytes_2d(m, k, &dx) &&
           perf_tensor_bytes_2d(k, n, &dw) &&
           perf_add_size(current, dx, &tmp) &&
           perf_add_size(tmp, dw, out);
}

static bool perf_model_plan(const perf_case *shape, perf_model *model)
{
    size_t w_qkv;
    size_t w_proj;
    size_t w_gate;
    size_t w_up;
    size_t w_down;
    size_t b_qkv;
    size_t b_proj;
    size_t b_gate;
    size_t b_up;
    size_t b_down;
    size_t y_qkv;
    size_t y_proj;
    size_t y_gate;
    size_t y_up;
    size_t y_down;
    size_t head_q;
    size_t head_kt;
    size_t head_scores;
    size_t head_v;
    size_t head_ctx;
    size_t one_head_scratch;
    size_t heads_scratch;
    size_t bwd_scratch;
    size_t tmp;
    int64_t head_dim;
    if (shape == NULL || model == NULL || shape->heads <= 0 ||
        shape->hidden <= 0 || (shape->hidden % shape->heads) != 0 ||
        shape->hidden > INT64_MAX / 3 || shape->hidden > INT64_MAX / 4) {
        return false;
    }
    head_dim = shape->hidden / shape->heads;
    memset(model, 0, sizeof(*model));
    model->tokens = shape->tokens;
    model->hidden = shape->hidden;
    model->intermediate = shape->intermediate;
    model->heads = shape->heads;
    model->head_dim = head_dim;
    model->matmuls = (uint32_t)(5 + 2 * shape->heads);
    if (!perf_tensor_bytes_2d(shape->hidden, 3 * shape->hidden, &w_qkv) ||
        !perf_tensor_bytes_2d(shape->hidden, shape->hidden, &w_proj) ||
        !perf_tensor_bytes_2d(shape->hidden, shape->intermediate, &w_gate) ||
        !perf_tensor_bytes_2d(shape->hidden, shape->intermediate, &w_up) ||
        !perf_tensor_bytes_2d(shape->intermediate, shape->hidden, &w_down) ||
        !perf_tensor_bytes_1d(3 * shape->hidden, &b_qkv) ||
        !perf_tensor_bytes_1d(shape->hidden, &b_proj) ||
        !perf_tensor_bytes_1d(shape->intermediate, &b_gate) ||
        !perf_tensor_bytes_1d(shape->intermediate, &b_up) ||
        !perf_tensor_bytes_1d(shape->hidden, &b_down) ||
        !perf_tensor_bytes_2d(shape->tokens, 3 * shape->hidden, &y_qkv) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->hidden, &y_proj) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->intermediate, &y_gate) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->intermediate, &y_up) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->hidden, &y_down) ||
        !perf_tensor_bytes_2d(shape->tokens, head_dim, &head_q) ||
        !perf_tensor_bytes_2d(head_dim, shape->tokens, &head_kt) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->tokens, &head_scores) ||
        !perf_tensor_bytes_2d(shape->tokens, head_dim, &head_v) ||
        !perf_tensor_bytes_2d(shape->tokens, head_dim, &head_ctx) ||
        !perf_tensor_bytes_2d(shape->tokens, shape->hidden, &model->data_bytes)) {
        return false;
    }
    if (!perf_add_size(w_qkv, w_proj, &tmp) ||
        !perf_add_size(tmp, w_gate, &tmp) ||
        !perf_add_size(tmp, w_up, &tmp) ||
        !perf_add_size(tmp, w_down, &tmp) ||
        !perf_add_size(tmp, b_qkv, &tmp) ||
        !perf_add_size(tmp, b_proj, &tmp) ||
        !perf_add_size(tmp, b_gate, &tmp) ||
        !perf_add_size(tmp, b_up, &tmp) ||
        !perf_add_size(tmp, b_down, &model->params_bytes) ||
        !perf_add_size(head_q, head_kt, &one_head_scratch) ||
        !perf_add_size(one_head_scratch, head_scores, &one_head_scratch) ||
        !perf_add_size(one_head_scratch, head_v, &one_head_scratch) ||
        !perf_add_size(one_head_scratch, head_ctx, &one_head_scratch) ||
        (size_t)shape->heads > SIZE_MAX / one_head_scratch) {
        return false;
    }
    heads_scratch = (size_t)shape->heads * one_head_scratch;
    if (!perf_add_size(y_qkv, heads_scratch, &tmp) ||
        !perf_add_size(tmp, y_proj, &tmp) ||
        !perf_add_size(tmp, y_gate, &tmp) ||
        !perf_add_size(tmp, y_up, &tmp) ||
        !perf_add_size(tmp, y_down, &model->scratch_bytes)) {
        return false;
    }
    bwd_scratch = model->scratch_bytes;
    model->bias_reductions = 5U;
    if (!perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->hidden,
                                          3 * shape->hidden, &bwd_scratch) ||
        !perf_add_size(bwd_scratch, b_qkv, &bwd_scratch) ||
        !perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->hidden,
                                          shape->hidden, &bwd_scratch) ||
        !perf_add_size(bwd_scratch, b_proj, &bwd_scratch) ||
        !perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->hidden,
                                          shape->intermediate, &bwd_scratch) ||
        !perf_add_size(bwd_scratch, b_gate, &bwd_scratch) ||
        !perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->hidden,
                                          shape->intermediate, &bwd_scratch) ||
        !perf_add_size(bwd_scratch, b_up, &bwd_scratch) ||
        !perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->intermediate,
                                          shape->hidden, &bwd_scratch) ||
        !perf_add_size(bwd_scratch, b_down, &bwd_scratch)) {
        return false;
    }
    for (int64_t h = 0; h < shape->heads; ++h) {
        if (!perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, head_dim,
                                              shape->tokens, &bwd_scratch) ||
            !perf_add_matmul_backward_scratch(bwd_scratch, shape->tokens, shape->tokens,
                                              head_dim, &bwd_scratch)) {
            return false;
        }
    }
    model->backward_scratch_bytes = bwd_scratch;
    model->forward_flops =
        perf_matmul_flops(shape->tokens, shape->hidden, 3 * shape->hidden) +
        (double)shape->heads * perf_matmul_flops(shape->tokens, head_dim, shape->tokens) +
        (double)shape->heads * perf_matmul_flops(shape->tokens, shape->tokens, head_dim) +
        perf_matmul_flops(shape->tokens, shape->hidden, shape->hidden) +
        perf_matmul_flops(shape->tokens, shape->hidden, shape->intermediate) +
        perf_matmul_flops(shape->tokens, shape->hidden, shape->intermediate) +
        perf_matmul_flops(shape->tokens, shape->intermediate, shape->hidden);
    model->backward_flops = 2.0 * model->forward_flops;
    return true;
}

static gd_memory_config perf_config_for_model(const perf_model *model,
                                              uint32_t ring_slots,
                                              bool include_backward)
{
    gd_memory_config cfg;
    size_t params;
    size_t scratch;
    size_t data;
    const size_t alignment = 4096U;
    const size_t params_slack = 32U * 1024U * 1024U;
    const size_t ring_slack = 32U * 1024U * 1024U;
    const size_t data_slack = 8U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    (void)perf_add_size(model->params_bytes, params_slack, &params);
    (void)perf_add_size(include_backward ? model->backward_scratch_bytes : model->scratch_bytes,
                        ring_slack,
                        &scratch);
    (void)perf_add_size(model->data_bytes, data_slack, &data);
    (void)perf_align_up_size(params, alignment, &params);
    (void)perf_align_up_size(scratch, alignment, &scratch);
    (void)perf_align_up_size(data, alignment, &data);
    cfg.params_bytes = params;
    cfg.state_bytes = 16U * 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch;
    cfg.data_slot_bytes = data;
    cfg.scratch_slots = ring_slots;
    cfg.data_slots = ring_slots;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool perf_model_init(gd_context *ctx, const perf_case *shape, perf_model *model)
{
    int64_t qkv_shape[2];
    int64_t proj_shape[2];
    int64_t gate_shape[2];
    int64_t down_shape[2];
    int64_t qkv_bias_shape[1];
    int64_t proj_bias_shape[1];
    int64_t gate_bias_shape[1];
    if (!perf_model_plan(shape, model)) {
        fprintf(stderr, "[PERF][FAIL] invalid/overflowing profile shape %s\n", shape->name);
        return false;
    }
    qkv_shape[0] = shape->hidden;
    qkv_shape[1] = 3 * shape->hidden;
    proj_shape[0] = shape->hidden;
    proj_shape[1] = shape->hidden;
    gate_shape[0] = shape->hidden;
    gate_shape[1] = shape->intermediate;
    down_shape[0] = shape->intermediate;
    down_shape[1] = shape->hidden;
    qkv_bias_shape[0] = 3 * shape->hidden;
    proj_bias_shape[0] = shape->hidden;
    gate_bias_shape[0] = shape->intermediate;
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, qkv_shape), 256U, 101U, -0.02f, 0.02f, &model->w_qkv));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, proj_shape), 256U, 102U, -0.02f, 0.02f, &model->w_proj));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, gate_shape), 256U, 103U, -0.02f, 0.02f, &model->w_gate));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, gate_shape), 256U, 104U, -0.02f, 0.02f, &model->w_up));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, down_shape), 256U, 105U, -0.02f, 0.02f, &model->w_down));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, qkv_bias_shape), 256U, 106U, -0.02f, 0.02f, &model->b_qkv));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, proj_bias_shape), 256U, 107U, -0.02f, 0.02f, &model->b_proj));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, gate_bias_shape), 256U, 108U, -0.02f, 0.02f, &model->b_gate));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, gate_bias_shape), 256U, 109U, -0.02f, 0.02f, &model->b_up));
    PERF_REQUIRE_OK(ctx, gd_tensor_rand_uniform(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(1U, proj_bias_shape), 256U, 110U, -0.02f, 0.02f, &model->b_down));
    PERF_REQUIRE_OK(ctx, gd_context_seal_params(ctx));
    return true;
}

static bool perf_matmul_step(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             bool include_backward,
                             gd_tensor *out)
{
    gd_tensor dx;
    gd_tensor dw;
    PERF_REQUIRE_OK(ctx, gd_matmul(ctx, x, w, out));
    if (include_backward) {
        PERF_REQUIRE_OK(ctx, gd_matmul_backward(ctx, x, w, out, &dx, &dw));
        (void)dx;
        (void)dw;
    }
    return true;
}

static bool perf_linear_step(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *bias,
                             bool include_backward,
                             gd_tensor *out)
{
    gd_tensor dx;
    gd_tensor dw;
    gd_tensor db;
    PERF_REQUIRE_OK(ctx, gd_linear(ctx, x, w, bias, out));
    if (include_backward) {
        PERF_REQUIRE_OK(ctx, gd_linear_backward(ctx, x, w, bias, out, &dx, &dw, &db));
        (void)dx;
        (void)dw;
        (void)db;
    }
    return true;
}

static bool perf_model_step(gd_context *ctx, const perf_model *model, bool include_backward)
{
    gd_tensor x;
    gd_tensor qkv;
    gd_tensor proj;
    gd_tensor gate;
    gd_tensor up;
    gd_tensor down;
    int64_t x_shape[2];
    int64_t head_q_shape[2];
    int64_t head_kt_shape[2];
    int64_t head_v_shape[2];
    int64_t h;
    x_shape[0] = model->tokens;
    x_shape[1] = model->hidden;
    head_q_shape[0] = model->tokens;
    head_q_shape[1] = model->head_dim;
    head_kt_shape[0] = model->head_dim;
    head_kt_shape[1] = model->tokens;
    head_v_shape[0] = model->tokens;
    head_v_shape[1] = model->head_dim;
    PERF_REQUIRE_OK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    if (!perf_linear_step(ctx, &x, &model->w_qkv, &model->b_qkv, include_backward, &qkv)) {
        return false;
    }
    for (h = 0; h < model->heads; ++h) {
        gd_tensor q;
        gd_tensor kt;
        gd_tensor scores;
        gd_tensor v;
        gd_tensor head_ctx;
        PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, head_q_shape), 256U, &q));
        PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, head_kt_shape), 256U, &kt));
        PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(2U, head_v_shape), 256U, &v));
        if (!perf_matmul_step(ctx, &q, &kt, include_backward, &scores)) {
            return false;
        }
        if (!perf_matmul_step(ctx, &scores, &v, include_backward, &head_ctx)) {
            return false;
        }
        (void)head_ctx;
    }
    if (!perf_linear_step(ctx, &x, &model->w_proj, &model->b_proj, include_backward, &proj)) {
        return false;
    }
    if (!perf_linear_step(ctx, &x, &model->w_gate, &model->b_gate, include_backward, &gate)) {
        return false;
    }
    if (!perf_linear_step(ctx, &x, &model->w_up, &model->b_up, include_backward, &up)) {
        return false;
    }
    if (!perf_linear_step(ctx, &up, &model->w_down, &model->b_down, include_backward, &down)) {
        return false;
    }
    PERF_REQUIRE_OK(ctx, gd_end_step(ctx));
    (void)qkv;
    (void)proj;
    (void)gate;
    (void)down;
    return true;
}

static bool perf_run_warmup(gd_context *ctx,
                            const perf_model *model,
                            int warmup,
                            bool include_backward)
{
    int i;
    for (i = 0; i < warmup; ++i) {
        if (!perf_model_step(ctx, model, include_backward)) {
            return false;
        }
        PERF_REQUIRE_OK(ctx, gd_synchronize(ctx));
    }
    return true;
}

static bool perf_run_sync(gd_context *ctx,
                          const perf_model *model,
                          int iters,
                          double *out_submit_s,
                          double *out_step_s,
                          bool include_backward)
{
    int i;
    double submit_total = 0.0;
    double step_total = 0.0;
    for (i = 0; i < iters; ++i) {
        double t0 = perf_now_seconds();
        double t1;
        double t2;
        if (!perf_model_step(ctx, model, include_backward)) {
            return false;
        }
        t1 = perf_now_seconds();
        PERF_REQUIRE_OK(ctx, gd_synchronize(ctx));
        t2 = perf_now_seconds();
        submit_total += t1 - t0;
        step_total += t2 - t0;
    }
    *out_submit_s = submit_total / (double)iters;
    *out_step_s = step_total / (double)iters;
    return true;
}

static bool perf_run_pipeline(gd_context *ctx,
                              const perf_model *model,
                              int iters,
                              double *out_step_s,
                              gd_memory_stats *out_before,
                              gd_memory_stats *out_after,
                              bool include_backward)
{
    int i;
    double t0;
    double t1;
    PERF_REQUIRE_OK(ctx, gd_memory_stats_query(ctx, out_before));
    t0 = perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!perf_model_step(ctx, model, include_backward)) {
            return false;
        }
    }
    PERF_REQUIRE_OK(ctx, gd_synchronize(ctx));
    t1 = perf_now_seconds();
    PERF_REQUIRE_OK(ctx, gd_memory_stats_query(ctx, out_after));
    *out_step_s = (t1 - t0) / (double)iters;
    return true;
}

static bool perf_smoke_correctness(perf_totals *totals)
{
    enum { M = 4, K = 7, N = 6 };
    gd_memory_config cfg;
    gd_context *ctx = NULL;
    gd_tensor x;
    gd_tensor w;
    gd_tensor y;
    gd_tensor dx;
    gd_tensor dw;
    const int64_t x_shape[2] = {M, K};
    const int64_t w_shape[2] = {K, N};
    uint16_t x_data[M * K];
    uint16_t w_data[K * N];
    uint16_t got[M * N];
    uint32_t i;
    uint32_t j;
    bool ok = true;
    gd_status st;

    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = 64U * 1024U;
    cfg.state_bytes = 16U * 1024U;
    cfg.scratch_slot_bytes = 64U * 1024U;
    cfg.data_slot_bytes = 16U * 1024U;
    cfg.scratch_slots = 2U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("v2_matmul_training_perf_probe: skipped (no supported GPU backend)\n");
        totals->skipped = true;
        return true;
    }
    if (st != GD_OK || ctx == NULL) {
        fprintf(stderr, "[PERF][FAIL] gd_context_create(smoke) -> %s (%d)\n",
                gd_status_string(st), (int)st);
        totals->failures += 1U;
        return false;
    }

    perf_fill_x(x_data, M, K);
    perf_fill_w(w_data, K, N);
    PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, x_shape), 256U, &x));
    PERF_REQUIRE_OK(ctx, gd_tensor_empty(ctx, GD_ARENA_PARAMS, GD_DTYPE_F16, gd_shape_make(2U, w_shape), 256U, &w));
    PERF_REQUIRE_OK(ctx, gd_tensor_write(ctx, &x, x_data, sizeof(x_data)));
    PERF_REQUIRE_OK(ctx, gd_tensor_write(ctx, &w, w_data, sizeof(w_data)));
    PERF_REQUIRE_OK(ctx, gd_context_seal_params(ctx));
    PERF_REQUIRE_OK(ctx, gd_begin_step(ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    PERF_REQUIRE_OK(ctx, gd_matmul(ctx, &x, &w, &y));
    memset(&dx, 0, sizeof(dx));
    memset(&dw, 0, sizeof(dw));
    st = gd_matmul_backward(ctx, &x, &w, &y, &dx, &dw);
    totals->matmul_backward_ready = st == GD_OK;
    if (st == GD_ERR_NOT_IMPLEMENTED) {
        totals->warnings += 1U;
        printf("[PERF][WARN] matmul_backward_status=%s training_ready=no reason=backward_not_implemented\n",
               gd_status_string(st));
        gd_context_clear_error(ctx);
    } else if (st != GD_OK) {
        totals->warnings += 1U;
        printf("[PERF][WARN] matmul_backward_status=%s (%d) training_ready=no\n",
               gd_status_string(st), (int)st);
        gd_context_clear_error(ctx);
    } else {
        printf("[PERF] matmul_backward_status=%s training_ready=partial\n", gd_status_string(st));
    }
    PERF_REQUIRE_OK(ctx, gd_end_step(ctx));
    PERF_REQUIRE_OK(ctx, gd_synchronize(ctx));
    memset(got, 0, sizeof(got));
    PERF_REQUIRE_OK(ctx, gd_tensor_read(ctx, &y, got, sizeof(got)));

    for (i = 0U; i < (uint32_t)M; ++i) {
        for (j = 0U; j < (uint32_t)N; ++j) {
            float want = perf_ref_matmul_value(x_data, w_data, i, j, K, N);
            float have = perf_f16_bits_to_f32(got[i * (uint32_t)N + j]);
            if (perf_abs_f32(want - have) > 0.02f) {
                fprintf(stderr,
                        "[PERF][FAIL] smoke matmul mismatch row=%" PRIu32 " col=%" PRIu32
                        " got=%g want=%g\n",
                        i,
                        j,
                        (double)have,
                        (double)want);
                ok = false;
            }
        }
    }
    if (ok) {
        printf("[PERF] public_matmul_smoke correctness=pass M=%d K=%d N=%d\n", M, K, N);
    } else {
        totals->failures += 1U;
    }

    gd_context_destroy(ctx);
    return ok;
}

static bool perf_profile_enabled(const char *filter, const char *profile)
{
    return filter == NULL || filter[0] == '\0' || strcmp(filter, "all") == 0 ||
           strcmp(filter, profile) == 0;
}

static bool perf_run_profile(const perf_case *shape,
                             int warmup,
                             int iters,
                             int pipeline_iters,
                             bool run_pipeline,
                             bool include_backward,
                             uint32_t ring_slots,
                             perf_totals *totals)
{
    gd_context *ctx = NULL;
    gd_status st;
    gd_memory_config cfg;
    gd_memory_stats before;
    gd_memory_stats after;
    perf_model plan;
    perf_model model;
    double submit_s = 0.0;
    double step_s = 0.0;
    double pipe_step_s = 0.0;
    double sync_tflops;
    double pipe_tflops;
    double flops;
    uint32_t matmul_calls;
    const char *workload;
    size_t active_scratch_bytes;
    uint64_t scratch_waits;
    uint64_t data_waits;
    uint64_t backend_waits;

    if (!perf_model_plan(shape, &plan)) {
        fprintf(stderr, "[PERF][FAIL] could not plan profile %s\n", shape->name);
        totals->failures += 1U;
        return false;
    }
    cfg = perf_config_for_model(&plan, ring_slots, include_backward);
    workload = include_backward ? "fwd_bwd" : "forward";
    flops = plan.forward_flops + (include_backward ? plan.backward_flops : 0.0);
    matmul_calls = plan.matmuls * (include_backward ? 3U : 1U);
    active_scratch_bytes = include_backward ? plan.backward_scratch_bytes : plan.scratch_bytes;
    printf("[PERF] profile=%s workload=%s tokens=%lld hidden=%lld heads=%lld head_dim=%lld intermediate=%lld forward_matmuls=%" PRIu32 " matmul_calls=%" PRIu32 " bias_reductions=%" PRIu32 " flops=%.3fG forward_flops=%.3fG params=%.1fMB scratch_slot=%.1fMB data_slot=%.1fMB ring_slots=%" PRIu32 "\n",
           shape->name,
           workload,
           (long long)shape->tokens,
           (long long)shape->hidden,
           (long long)shape->heads,
           (long long)(shape->hidden / shape->heads),
           (long long)shape->intermediate,
           plan.matmuls,
           matmul_calls,
           include_backward ? plan.bias_reductions : 0U,
           flops / 1.0e9,
           plan.forward_flops / 1.0e9,
           (double)cfg.params_bytes / PERF_MB,
           (double)cfg.scratch_slot_bytes / PERF_MB,
           (double)cfg.data_slot_bytes / PERF_MB,
           ring_slots);

    st = gd_context_create(&cfg, &ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[PERF] profile=%s skipped no_supported_gpu_backend\n", shape->name);
        totals->skipped = true;
        return true;
    }
    if (st != GD_OK || ctx == NULL) {
        fprintf(stderr, "[PERF][FAIL] gd_context_create(profile=%s) -> %s (%d)\n",
                shape->name, gd_status_string(st), (int)st);
        totals->failures += 1U;
        return false;
    }
    if (!perf_model_init(ctx, shape, &model)) {
        gd_context_destroy(ctx);
        totals->failures += 1U;
        return false;
    }
    if (!perf_run_warmup(ctx, &model, warmup, include_backward)) {
        gd_context_destroy(ctx);
        totals->failures += 1U;
        return false;
    }
    if (!perf_run_sync(ctx, &model, iters, &submit_s, &step_s, include_backward)) {
        gd_context_destroy(ctx);
        totals->failures += 1U;
        return false;
    }

    flops = model.forward_flops + (include_backward ? model.backward_flops : 0.0);
    matmul_calls = model.matmuls * (include_backward ? 3U : 1U);
    active_scratch_bytes = include_backward ? model.backward_scratch_bytes : model.scratch_bytes;
    sync_tflops = step_s > 0.0 ? flops / step_s / 1.0e12 : 0.0;
    printf("[PERF] sync profile=%s workload=%s avg_submit_ms=%8.3f avg_step_ms=%8.3f work_tflops=%7.2f matmuls_per_s=%8.1f activation_gb_s=%7.2f\n",
           shape->name,
           workload,
           submit_s * 1000.0,
           step_s * 1000.0,
           sync_tflops,
           (double)matmul_calls / step_s,
           ((double)active_scratch_bytes / PERF_GB) / step_s);
    if (sync_tflops > totals->best_sync_tflops) {
        totals->best_sync_tflops = sync_tflops;
        totals->best_sync_profile = shape->name;
    }
    if (sync_tflops > 0.0 && sync_tflops < 1.0 && shape->hidden >= 512) {
        totals->warnings += 1U;
        printf("[PERF][WARN] sync profile=%s work_tflops_below_1.0 wall_time_includes_public_api_overhead\n",
               shape->name);
    }

    if (run_pipeline) {
        if (!perf_run_pipeline(ctx, &model, pipeline_iters, &pipe_step_s, &before, &after,
                               include_backward)) {
            gd_context_destroy(ctx);
            totals->failures += 1U;
            return false;
        }
        pipe_tflops = pipe_step_s > 0.0 ? flops / pipe_step_s / 1.0e12 : 0.0;
        scratch_waits = after.scratch.waits - before.scratch.waits;
        data_waits = after.data.waits - before.data.waits;
        backend_waits = after.backend_waits - before.backend_waits;
        printf("[PERF] pipeline profile=%s workload=%s queued_steps=%d avg_step_ms=%8.3f work_tflops=%7.2f matmuls_per_s=%8.1f scratch_waits=%" PRIu64 " data_waits=%" PRIu64 " backend_waits=%" PRIu64 "\n",
               shape->name,
               workload,
               pipeline_iters,
               pipe_step_s * 1000.0,
               pipe_tflops,
               (double)matmul_calls / pipe_step_s,
               scratch_waits,
               data_waits,
               backend_waits);
        if (pipe_tflops > totals->best_pipeline_tflops) {
            totals->best_pipeline_tflops = pipe_tflops;
            totals->best_pipeline_profile = shape->name;
        }
    }

    totals->profiles_run += 1U;
    gd_context_destroy(ctx);
    return true;
}

int main(void)
{
    static const perf_case cases[] = {
        {"h128x4h4", 512, 128, 512, 4},
        {"h256x4h4", 512, 256, 1024, 4},
        {"h512x4h4", 512, 512, 2048, 4},
    };
    const char *profile_filter = getenv("GD_QA_PERF_PROFILE");
    int warmup = perf_env_int("GD_QA_PERF_WARMUP", 2, 0, 100);
    int iters = perf_env_int("GD_QA_PERF_ITERS", 5, 1, 1000);
    int pipeline_iters = perf_env_int("GD_QA_PERF_PIPELINE_ITERS", iters, 1, 1000);
    bool run_pipeline = perf_env_bool("GD_QA_PERF_PIPELINE", true);
    bool include_backward = perf_env_bool("GD_QA_PERF_BACKWARD", true);
    uint32_t ring_slots = (uint32_t)perf_env_int("GD_QA_PERF_RING_SLOTS", 3, 1, 64);
    perf_totals totals;
    size_t i;

    memset(&totals, 0, sizeof(totals));
    totals.best_sync_profile = "none";
    totals.best_pipeline_profile = "none";

    printf("v2_matmul_training_perf_probe: start warmup=%d iters=%d pipeline_iters=%d pipeline=%s backward=%s profile=%s\n",
           warmup,
           iters,
           pipeline_iters,
           run_pipeline ? "on" : "off",
           include_backward ? "on" : "off",
           profile_filter != NULL && profile_filter[0] != '\0' ? profile_filter : "all");

    if (!perf_smoke_correctness(&totals)) {
        printf("v2_matmul_training_perf_probe: summary profiles=0 fail=%" PRIu32 " warn=%" PRIu32 "\n",
               totals.failures,
               totals.warnings);
        return totals.failures == 0U ? 0 : 1;
    }
    if (totals.skipped) {
        printf("v2_matmul_training_perf_probe: summary skipped=1 fail=0 warn=%" PRIu32 "\n",
               totals.warnings);
        return 0;
    }

    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        if (perf_profile_enabled(profile_filter, cases[i].name)) {
            (void)perf_run_profile(&cases[i], warmup, iters, pipeline_iters,
                                   run_pipeline, include_backward, ring_slots, &totals);
        }
    }

    if (totals.profiles_run == 0U) {
        totals.warnings += 1U;
        printf("[PERF][WARN] no profiles matched GD_QA_PERF_PROFILE=%s\n",
               profile_filter != NULL ? profile_filter : "(null)");
    }

    printf("\n[PERF][REPORT] public_api_%s_best_sync profile=%s tflops=%.2f\n",
           include_backward ? "fwd_bwd" : "forward",
           totals.best_sync_profile,
           totals.best_sync_tflops);
    if (run_pipeline) {
        printf("[PERF][REPORT] public_api_%s_best_pipeline profile=%s tflops=%.2f\n",
               include_backward ? "fwd_bwd" : "forward",
               totals.best_pipeline_profile,
               totals.best_pipeline_tflops);
    }
    printf("[PERF][REPORT] training_ready=%s blockers=%s\n",
           totals.matmul_backward_ready ? "partial" : "no",
           totals.matmul_backward_ready ? "autograd_optimizer_amp_not_covered_by_probe" :
                                          "matmul_backward_not_implemented");
    printf("[PERF][REPORT] kernel_path=metal_metallib_f16_reg_tiled_simdgroup_gemm_linear_bias_reduce\n");
    printf("[PERF][REPORT] interpretation=%s_wall_time_includes_current_public_api_overheads\n",
           include_backward ? "fwd_bwd" : "forward_only");
    printf("v2_matmul_training_perf_probe: summary profiles=%" PRIu32 " fail=%" PRIu32 " warn=%" PRIu32 "\n",
           totals.profiles_run,
           totals.failures,
           totals.warnings);
    return totals.failures == 0U ? 0 : 1;
}
