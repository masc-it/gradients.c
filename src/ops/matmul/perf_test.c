/*
 * Batched/broadcasted gd_matmul Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=matmul
 *
 * Optional environment:
 *   GD_MATMUL_PERF_PROFILE=smoke|all|<case-name>
 *   GD_MATMUL_PERF_WARMUP=5
 *   GD_MATMUL_PERF_ITERS=30
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

#define MATMUL_PERF_TERA 1000000000000.0

typedef struct matmul_perf_case {
    const char *name;
    uint32_t x_rank;
    int64_t x_shape[GD_MAX_DIMS];
    uint32_t w_rank;
    int64_t w_shape[GD_MAX_DIMS];
} matmul_perf_case;

typedef struct matmul_perf_model {
    gd_context *ctx;
    gd_tensor x;
    gd_tensor w;
    gd_tensor grad;
    uint32_t out_rank;
    int64_t out_shape[GD_MAX_DIMS];
    size_t x_count;
    size_t w_count;
    size_t y_count;
    size_t x_bytes;
    size_t w_bytes;
    size_t y_bytes;
    double fwd_flops;
    double bwd_flops;
} matmul_perf_model;

typedef bool (*matmul_perf_run_fn)(matmul_perf_model *model);

static double matmul_perf_now_seconds(void)
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

static int matmul_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t matmul_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool matmul_perf_count(uint32_t rank, const int64_t shape[GD_MAX_DIMS], size_t *out_count)
{
    uint32_t i;
    size_t count = 1U;
    if (rank > GD_MAX_DIMS || shape == NULL || out_count == NULL) {
        return false;
    }
    for (i = 0U; i < rank; ++i) {
        if (shape[i] <= 0 || (uint64_t)shape[i] > (uint64_t)(SIZE_MAX / count)) {
            return false;
        }
        count *= (size_t)shape[i];
    }
    *out_count = count;
    return true;
}

static bool matmul_perf_shape(const matmul_perf_case *pcase,
                              uint32_t *out_rank,
                              int64_t out_shape[GD_MAX_DIMS],
                              uint32_t *out_batch_count)
{
    uint32_t xb;
    uint32_t wb;
    uint32_t br;
    uint32_t axis;
    uint32_t batch_count = 1U;
    if (pcase == NULL || out_rank == NULL || out_shape == NULL || out_batch_count == NULL ||
        pcase->x_rank < 2U || pcase->w_rank < 2U) {
        return false;
    }
    xb = pcase->x_rank - 2U;
    wb = pcase->w_rank - 2U;
    br = xb > wb ? xb : wb;
    if (br > GD_MAX_DIMS - 2U || pcase->x_shape[pcase->x_rank - 1U] != pcase->w_shape[pcase->w_rank - 2U]) {
        return false;
    }
    memset(out_shape, 0, sizeof(int64_t) * GD_MAX_DIMS);
    for (axis = 0U; axis < br; ++axis) {
        uint32_t x_pad = br - xb;
        uint32_t w_pad = br - wb;
        int64_t xd = axis < x_pad ? 1 : pcase->x_shape[axis - x_pad];
        int64_t wd = axis < w_pad ? 1 : pcase->w_shape[axis - w_pad];
        int64_t od;
        if (xd != wd && xd != 1 && wd != 1) {
            return false;
        }
        od = xd > wd ? xd : wd;
        if (od <= 0 || od > (int64_t)(UINT32_MAX / batch_count)) {
            return false;
        }
        out_shape[axis] = od;
        batch_count *= (uint32_t)od;
    }
    out_shape[br] = pcase->x_shape[pcase->x_rank - 2U];
    out_shape[br + 1U] = pcase->w_shape[pcase->w_rank - 1U];
    *out_rank = br + 2U;
    *out_batch_count = batch_count;
    return true;
}

static bool matmul_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[MATMUL][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define MATMUL_PERF_REQUIRE_OK(ctx, expr)                             \
    do {                                                              \
        gd_status matmul_perf_st__ = (expr);                          \
        if (!matmul_perf_status_ok((ctx), matmul_perf_st__, #expr)) { \
            return false;                                             \
        }                                                             \
    } while (0)

static bool matmul_perf_selected(const matmul_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "proj_b1_t128_c768_n2304") == 0 ||
               strcmp(pcase->name, "attn_b1_h12_t128_d64") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool matmul_perf_init(matmul_perf_model *model, const matmul_perf_case *pcase)
{
    gd_memory_config cfg;
    uint32_t batch_count;
    size_t params_bytes;
    size_t scratch_bytes;
    size_t state_bytes;
    double m;
    double k;
    double n;
    if (model == NULL || pcase == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    if (!matmul_perf_shape(pcase, &model->out_rank, model->out_shape, &batch_count) ||
        !matmul_perf_count(pcase->x_rank, pcase->x_shape, &model->x_count) ||
        !matmul_perf_count(pcase->w_rank, pcase->w_shape, &model->w_count) ||
        !matmul_perf_count(model->out_rank, model->out_shape, &model->y_count)) {
        return false;
    }
    model->x_bytes = model->x_count * sizeof(uint16_t);
    model->w_bytes = model->w_count * sizeof(uint16_t);
    model->y_bytes = model->y_count * sizeof(uint16_t);
    m = (double)pcase->x_shape[pcase->x_rank - 2U];
    k = (double)pcase->x_shape[pcase->x_rank - 1U];
    n = (double)pcase->w_shape[pcase->w_rank - 1U];
    model->fwd_flops = 2.0 * (double)batch_count * m * n * k;
    model->bwd_flops = 2.0 * model->fwd_flops;
    params_bytes = matmul_perf_align_up(model->x_bytes + model->w_bytes + model->y_bytes + 16U * 1024U * 1024U, 4096U);
    scratch_bytes = matmul_perf_align_up(model->y_bytes * 4U + (model->x_bytes + model->w_bytes) * 8U +
                                         128U * 1024U * 1024U, 4096U);
    state_bytes = matmul_perf_align_up((model->x_bytes + model->w_bytes) * 2U + 16U * 1024U * 1024U, 4096U);
    cfg = gd_memory_config_default();
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = state_bytes;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    MATMUL_PERF_REQUIRE_OK(NULL, gd_context_create(&cfg, &model->ctx));
    MATMUL_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  GD_DTYPE_F16,
                                                  gd_shape_make(pcase->x_rank, pcase->x_shape),
                                                  256U,
                                                  UINT64_C(0x4D4D0101),
                                                  -0.25f,
                                                  0.25f,
                                                  &model->x));
    MATMUL_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  GD_DTYPE_F16,
                                                  gd_shape_make(pcase->w_rank, pcase->w_shape),
                                                  256U,
                                                  UINT64_C(0x4D4D0202),
                                                  -0.25f,
                                                  0.25f,
                                                  &model->w));
    MATMUL_PERF_REQUIRE_OK(model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  GD_DTYPE_F16,
                                                  gd_shape_make(model->out_rank, model->out_shape),
                                                  256U,
                                                  UINT64_C(0x4D4D0303),
                                                  -0.125f,
                                                  0.125f,
                                                  &model->grad));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static void matmul_perf_destroy(matmul_perf_model *model)
{
    if (model != NULL && model->ctx != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool matmul_perf_run_forward(matmul_perf_model *model)
{
    gd_tensor out;
    model->x.requires_grad = false;
    model->w.requires_grad = false;
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_matmul(model->ctx, &model->x, &model->w, &out));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool matmul_perf_run_backward(matmul_perf_model *model)
{
    gd_tensor dx;
    gd_tensor dw;
    model->x.requires_grad = false;
    model->w.requires_grad = false;
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_matmul_backward(model->ctx, &model->x, &model->w, &model->grad, &dx, &dw));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool matmul_perf_run_autograd(matmul_perf_model *model)
{
    gd_tensor out;
    model->x.requires_grad = true;
    model->w.requires_grad = true;
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_matmul(model->ctx, &model->x, &model->w, &out));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    MATMUL_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool matmul_perf_measure(matmul_perf_model *model,
                                const char *label,
                                matmul_perf_run_fn fn,
                                int warmup,
                                int iters,
                                double flops)
{
    int i;
    double t0;
    double elapsed;
    double avg_ms;
    double tflops;
    for (i = 0; i < warmup; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    t0 = matmul_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = matmul_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    tflops = flops / (elapsed / (double)iters) / MATMUL_PERF_TERA;
    printf("[MATMUL][%s] avg_ms=%.4f TFLOP/s=%.2f\n", label, avg_ms, tflops);
    return true;
}

static void matmul_perf_print_shape(uint32_t rank, const int64_t shape[GD_MAX_DIMS])
{
    uint32_t i;
    printf("[");
    for (i = 0U; i < rank; ++i) {
        printf("%s%lld", i == 0U ? "" : ",", (long long)shape[i]);
    }
    printf("]");
}

static void matmul_perf_print_case(const matmul_perf_case *pcase, const matmul_perf_model *model)
{
    printf("[MATMUL] case=%s x=", pcase->name);
    matmul_perf_print_shape(pcase->x_rank, pcase->x_shape);
    printf(" w=");
    matmul_perf_print_shape(pcase->w_rank, pcase->w_shape);
    printf(" out=");
    matmul_perf_print_shape(model->out_rank, model->out_shape);
    printf(" bytes(x=%zu,w=%zu,out=%zu) fwd_GFLOP=%.2f\n",
           model->x_bytes,
           model->w_bytes,
           model->y_bytes,
           model->fwd_flops / 1.0e9);
}

int main(void)
{
    const matmul_perf_case cases[] = {
        {"proj_b1_t128_c768_n2304", 3U, {1, 128, 768, 0, 0, 0, 0, 0}, 2U, {768, 2304, 0, 0, 0, 0, 0, 0}},
        {"proj_b4_t128_c768_n2304", 3U, {4, 128, 768, 0, 0, 0, 0, 0}, 2U, {768, 2304, 0, 0, 0, 0, 0, 0}},
        {"proj_b2_t512_c768_n2304", 3U, {2, 512, 768, 0, 0, 0, 0, 0}, 2U, {768, 2304, 0, 0, 0, 0, 0, 0}},
        {"proj_b1_t2048_c768_n2304", 3U, {1, 2048, 768, 0, 0, 0, 0, 0}, 2U, {768, 2304, 0, 0, 0, 0, 0, 0}},
        {"attn_b1_h12_t128_d64", 4U, {1, 12, 128, 64, 0, 0, 0, 0}, 4U, {1, 12, 64, 128, 0, 0, 0, 0}},
        {"attn_b1_h8_t512_d64", 4U, {1, 8, 512, 64, 0, 0, 0, 0}, 4U, {1, 8, 64, 512, 0, 0, 0, 0}},
        {"attn_b1_h8_t2048_d64", 4U, {1, 8, 2048, 64, 0, 0, 0, 0}, 4U, {1, 8, 64, 2048, 0, 0, 0, 0}},
        {"gpt10_proj_b4_t256_c384_n1152", 3U, {4, 256, 384, 0, 0, 0, 0, 0}, 2U, {384, 1152, 0, 0, 0, 0, 0, 0}},
        {"gpt10_proj_b8_t256_c384_n1152", 3U, {8, 256, 384, 0, 0, 0, 0, 0}, 2U, {384, 1152, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_qkv_b1_t512_c256_n768", 3U, {1, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 768, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_proj_b1_t512_c256_n256", 3U, {1, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 256, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_mlp_up_b1_t512_c256_n2048", 3U, {1, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 2048, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_mlp_down_b1_t512_c1024_n256", 3U, {1, 512, 1024, 0, 0, 0, 0, 0}, 2U, {1024, 256, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_qkv_b32_t512_c256_n768", 3U, {32, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 768, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_proj_b32_t512_c256_n256", 3U, {32, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 256, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_mlp_up_b32_t512_c256_n2048", 3U, {32, 512, 256, 0, 0, 0, 0, 0}, 2U, {256, 2048, 0, 0, 0, 0, 0, 0}},
        {"gpt_lm_mlp_down_b32_t512_c1024_n256", 3U, {32, 512, 1024, 0, 0, 0, 0, 0}, 2U, {1024, 256, 0, 0, 0, 0, 0, 0}},
        {"gpt10_mlp_up_b4_t256_c384_n1536", 3U, {4, 256, 384, 0, 0, 0, 0, 0}, 2U, {384, 1536, 0, 0, 0, 0, 0, 0}},
        {"gpt10_mlp_down_b4_t256_c1536_n384", 3U, {4, 256, 1536, 0, 0, 0, 0, 0}, 2U, {1536, 384, 0, 0, 0, 0, 0, 0}},
        {"gpt10_attn_b4_h6_t256_d64", 4U, {4, 6, 256, 64, 0, 0, 0, 0}, 4U, {4, 6, 64, 256, 0, 0, 0, 0}},
        {"gpt10_attn_b2_h6_t512_d64", 4U, {2, 6, 512, 64, 0, 0, 0, 0}, 4U, {2, 6, 64, 512, 0, 0, 0, 0}},
    };
    const char *profile = getenv("GD_MATMUL_PERF_PROFILE");
    int warmup = matmul_perf_env_int("GD_MATMUL_PERF_WARMUP", 5, 0, 1000);
    int iters = matmul_perf_env_int("GD_MATMUL_PERF_ITERS", 30, 1, 10000);
    size_t i;
    bool any = false;
    printf("[MATMUL] warmup=%d iters=%d profile=%s note=batched-broadcast-f16\n",
           warmup,
           iters,
           profile != NULL ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        matmul_perf_model model;
        if (!matmul_perf_selected(&cases[i], profile)) {
            continue;
        }
        any = true;
        if (!matmul_perf_init(&model, &cases[i])) {
            fprintf(stderr, "[MATMUL][FAIL] init case=%s\n", cases[i].name);
            return 1;
        }
        matmul_perf_print_case(&cases[i], &model);
        if (!matmul_perf_measure(&model, "fwd", matmul_perf_run_forward, warmup, iters, model.fwd_flops) ||
            !matmul_perf_measure(&model, "bwd", matmul_perf_run_backward, warmup, iters, model.bwd_flops) ||
            !matmul_perf_measure(&model, "autograd", matmul_perf_run_autograd, warmup, iters,
                                 model.fwd_flops + model.bwd_flops)) {
            matmul_perf_destroy(&model);
            return 1;
        }
        matmul_perf_destroy(&model);
    }
    if (!any) {
        fprintf(stderr, "[MATMUL][FAIL] no cases selected for profile=%s\n", profile != NULL ? profile : "smoke");
        return 2;
    }
    return 0;
}
