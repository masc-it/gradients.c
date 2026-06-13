/*
 * RoPE public API performance probe.
 *
 * Run:
 *   make op-perf OP=rope
 *
 * Optional environment:
 *   GD_ROPE_PERF_PROFILE=smoke|all|<case-name>
 *   GD_ROPE_PERF_WARMUP=5
 *   GD_ROPE_PERF_ITERS=20
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

#define ROPE_PERF_GIB (1024.0 * 1024.0 * 1024.0)

#if defined(__APPLE__)
static double rope_perf_now_seconds(void)
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
static double rope_perf_now_seconds(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}
#endif

static int rope_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t rope_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool rope_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[ROPE][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define ROPE_PERF_REQUIRE_OK(ctx, expr)                                      \
    do {                                                                     \
        gd_status rope_perf_st__ = (expr);                                   \
        if (!rope_perf_status_ok((ctx), rope_perf_st__, #expr)) {            \
            return false;                                                    \
        }                                                                    \
    } while (0)

typedef struct rope_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    int32_t n_dims;
    bool interleaved;
} rope_perf_case;

typedef struct rope_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad;
    gd_tensor pos;
    size_t count;
    size_t pos_count;
    size_t elem_size;
    size_t tensor_bytes;
} rope_perf_model;

typedef bool (*rope_perf_run_fn)(rope_perf_model *model, const rope_perf_case *pcase);

static const char *rope_perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static bool rope_perf_case_selected(const rope_perf_case *pcase, const char *profile)
{
    const char *selected = profile != NULL && profile[0] != '\0' ? profile : "smoke";
    if (strcmp(selected, "all") == 0) {
        return true;
    }
    if (strcmp(selected, "smoke") == 0) {
        return strcmp(pcase->name, "decode_1x1x32x64_f16") == 0 ||
               strcmp(pcase->name, "train_4x1024x32x64_f16") == 0;
    }
    return strcmp(selected, pcase->name) == 0;
}

static bool rope_perf_count_shape(const rope_perf_case *pcase, size_t *out_count, size_t *out_pos_count)
{
    uint64_t count = 1U;
    uint64_t pos_count = 1U;
    uint32_t i;
    if (pcase == NULL || out_count == NULL || out_pos_count == NULL || pcase->rank < 2U ||
        pcase->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = 0U; i < pcase->rank; ++i) {
        if (pcase->shape[i] <= 0 || (uint64_t)pcase->shape[i] > UINT64_MAX / count) {
            return false;
        }
        count *= (uint64_t)pcase->shape[i];
        if (i + 2U < pcase->rank) {
            if ((uint64_t)pcase->shape[i] > UINT64_MAX / pos_count) {
                return false;
            }
            pos_count *= (uint64_t)pcase->shape[i];
        }
    }
    if (count > (uint64_t)SIZE_MAX || pos_count > (uint64_t)SIZE_MAX) {
        return false;
    }
    *out_count = (size_t)count;
    *out_pos_count = (size_t)pos_count;
    return count != 0U && pos_count != 0U;
}

static void rope_perf_shape_string(const rope_perf_case *pcase, char *out, size_t out_size)
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

static bool rope_perf_create_model(const rope_perf_case *pcase, rope_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    int64_t pos_shape[1];
    int32_t *pos_host;
    size_t i;
    gd_status st;
    if (pcase == NULL || model == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->elem_size = gd_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || !rope_perf_count_shape(pcase, &model->count, &model->pos_count) ||
        model->count > SIZE_MAX / model->elem_size) {
        return false;
    }
    model->tensor_bytes = model->count * model->elem_size;
    if (model->tensor_bytes > (SIZE_MAX - 256U * 1024U * 1024U) / 12U) {
        return false;
    }
    params_bytes = rope_perf_align_up(model->tensor_bytes * 2U + model->pos_count * sizeof(int32_t) +
                                          64U * 1024U * 1024U,
                                      4096U);
    scratch_bytes = rope_perf_align_up(model->tensor_bytes * 12U + 256U * 1024U * 1024U, 4096U);
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
        printf("[ROPE] skipped case=%s: no supported GPU backend\n", pcase->name);
        return false;
    }
    if (!rope_perf_status_ok(model->ctx, st, "gd_context_create")) {
        return false;
    }
    ROPE_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_rand_uniform(model->ctx,
                                                GD_ARENA_PARAMS,
                                                pcase->dtype,
                                                gd_shape_make(pcase->rank, pcase->shape),
                                                256U,
                                                0x1234U,
                                                -1.0f,
                                                1.0f,
                                                &model->x));
    ROPE_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_rand_uniform(model->ctx,
                                                GD_ARENA_PARAMS,
                                                pcase->dtype,
                                                gd_shape_make(pcase->rank, pcase->shape),
                                                256U,
                                                0x5678U,
                                                -1.0f,
                                                1.0f,
                                                &model->grad));
    pos_shape[0] = (int64_t)model->pos_count;
    ROPE_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_empty(model->ctx,
                                         GD_ARENA_PARAMS,
                                         GD_DTYPE_I32,
                                         gd_shape_make(1U, pos_shape),
                                         256U,
                                         &model->pos));
    pos_host = (int32_t *)malloc(model->pos_count * sizeof(pos_host[0]));
    if (pos_host == NULL) {
        return false;
    }
    for (i = 0U; i < model->pos_count; ++i) {
        pos_host[i] = (int32_t)(i & 0x7fffU);
    }
    ROPE_PERF_REQUIRE_OK(model->ctx,
                         gd_tensor_write(model->ctx,
                                         &model->pos,
                                         pos_host,
                                         model->pos_count * sizeof(pos_host[0])));
    free(pos_host);
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void rope_perf_destroy_model(rope_perf_model *model)
{
    if (model != NULL && model->ctx != NULL) {
        gd_context_destroy(model->ctx);
        model->ctx = NULL;
    }
}

static gd_rope_config rope_perf_config(const rope_perf_case *pcase)
{
    gd_rope_config config;
    config.theta = 10000.0f;
    config.n_dims = pcase->n_dims;
    config.interleaved = pcase->interleaved;
    return config;
}

static bool rope_perf_run_forward(rope_perf_model *model, const rope_perf_case *pcase)
{
    gd_tensor y;
    gd_rope_config config = rope_perf_config(pcase);
    model->x.requires_grad = false;
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_rope(model->ctx, &model->x, &model->pos, &config, &y));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rope_perf_run_backward(rope_perf_model *model, const rope_perf_case *pcase)
{
    gd_tensor dx;
    gd_rope_config config = rope_perf_config(pcase);
    model->x.requires_grad = false;
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    ROPE_PERF_REQUIRE_OK(model->ctx,
                         gd_rope_backward(model->ctx, &model->x, &model->pos, &model->grad, &config, &dx));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rope_perf_run_autograd(rope_perf_model *model, const rope_perf_case *pcase)
{
    gd_tensor y;
    gd_tensor dx;
    gd_rope_config config = rope_perf_config(pcase);
    model->x.requires_grad = true;
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_rope(model->ctx, &model->x, &model->pos, &config, &y));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &y, &model->grad));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    ROPE_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool rope_perf_measure(const rope_perf_case *pcase,
                              rope_perf_model *model,
                              const char *op,
                              rope_perf_run_fn fn,
                              int warmup,
                              int iters,
                              double logical_bytes)
{
    double start;
    double elapsed;
    double avg_ms;
    double gib_s;
    int i;
    char shape[128];
    for (i = 0; i < warmup; ++i) {
        if (!fn(model, pcase)) {
            return false;
        }
    }
    start = rope_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model, pcase)) {
            return false;
        }
    }
    elapsed = rope_perf_now_seconds() - start;
    avg_ms = elapsed * 1000.0 / (double)iters;
    gib_s = (logical_bytes / ROPE_PERF_GIB) / (avg_ms / 1000.0);
    rope_perf_shape_string(pcase, shape, sizeof(shape));
    printf("[ROPE][PERF] case=%-28s op=%-8s dtype=%s shape=%s n_dims=%d interleaved=%u warmup=%d iters=%d avg_ms=%.4f logical_GiB/s=%.2f\n",
           pcase->name,
           op,
           rope_perf_dtype_name(pcase->dtype),
           shape,
           (int)pcase->n_dims,
           pcase->interleaved ? 1U : 0U,
           warmup,
           iters,
           avg_ms,
           gib_s);
    return true;
}

static bool rope_perf_run_case(const rope_perf_case *pcase, int warmup, int iters)
{
    rope_perf_model model;
    double one_pass_bytes;
    if (!rope_perf_create_model(pcase, &model)) {
        return false;
    }
    one_pass_bytes = (double)model.tensor_bytes * 2.0 + (double)model.pos_count * (double)sizeof(int32_t);
    if (!rope_perf_measure(pcase, &model, "fwd", rope_perf_run_forward, warmup, iters, one_pass_bytes) ||
        !rope_perf_measure(pcase, &model, "bwd", rope_perf_run_backward, warmup, iters, one_pass_bytes) ||
        !rope_perf_measure(pcase,
                           &model,
                           "autograd",
                           rope_perf_run_autograd,
                           warmup,
                           iters,
                           one_pass_bytes * 2.0)) {
        rope_perf_destroy_model(&model);
        return false;
    }
    rope_perf_destroy_model(&model);
    return true;
}

int main(void)
{
    const rope_perf_case cases[] = {
        {.name = "decode_1x1x32x64_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 4U,
         .shape = {1, 1, 32, 64},
         .n_dims = 64,
         .interleaved = false},
        {.name = "train_4x1024x32x64_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 4U,
         .shape = {4, 1024, 32, 64},
         .n_dims = 64,
         .interleaved = false},
        {.name = "gpt_lm_b1_512x4x64_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .shape = {512, 4, 64},
         .n_dims = 64,
         .interleaved = false},
        {.name = "gpt_lm_b32_16384x4x64_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 3U,
         .shape = {16384, 4, 64},
         .n_dims = 64,
         .interleaved = false},
        {.name = "partial_4x1024x32x128_f16",
         .dtype = GD_DTYPE_F16,
         .rank = 4U,
         .shape = {4, 1024, 32, 128},
         .n_dims = 64,
         .interleaved = true},
        {.name = "train_2x512x16x64_f32",
         .dtype = GD_DTYPE_F32,
         .rank = 4U,
         .shape = {2, 512, 16, 64},
         .n_dims = 64,
         .interleaved = false},
    };
    const char *profile = getenv("GD_ROPE_PERF_PROFILE");
    const int warmup = rope_perf_env_int("GD_ROPE_PERF_WARMUP", 5, 0, 1000);
    const int iters = rope_perf_env_int("GD_ROPE_PERF_ITERS", 20, 1, 100000);
    bool ran = false;
    uint32_t i;
    for (i = 0U; i < (uint32_t)(sizeof(cases) / sizeof(cases[0])); ++i) {
        if (!rope_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        ran = true;
        if (!rope_perf_run_case(&cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr, "[ROPE][FAIL] no cases selected (profile=%s)\n", profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}
