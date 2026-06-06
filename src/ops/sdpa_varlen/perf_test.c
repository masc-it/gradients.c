/*
 * gd_sdpa_varlen Metal performance probe.
 *
 * Run from the repository root with:
 *   make op-perf OP=sdpa_varlen
 *
 * Optional environment:
 *   GD_SDPA_VARLEN_PERF_PROFILE=smoke|all|<case-name>
 *   GD_SDPA_VARLEN_PERF_WARMUP=5
 *   GD_SDPA_VARLEN_PERF_ITERS=30
 *
 * Fast-path split tuning can also be exercised with:
 *   GD_METAL_SDPA_SPLIT_MIN=128 GD_METAL_SDPA_SPLIT_MAX=8 make op-perf OP=sdpa_varlen
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <gradients/gradients.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <time.h>
#endif

#define GD_SDPA_VARLEN_PERF_TFLOP 1.0e12

#define SDPA_VARLEN_PERF_REQUIRE_OK(ctx, expr)                                      \
    do {                                                                           \
        gd_status sdpa_varlen_perf_st__ = (expr);                                  \
        if (!sdpa_varlen_perf_status_ok((ctx), sdpa_varlen_perf_st__, #expr)) {    \
            return false;                                                          \
        }                                                                          \
    } while (0)

typedef struct sdpa_varlen_perf_case {
    const char *name;
    gd_dtype dtype;
    int32_t batch;
    int32_t seqlen;
    int32_t hq;
    int32_t hkv;
    int32_t dh;
    bool causal;
    int32_t sliding_window;
    int32_t prefix_len;
} sdpa_varlen_perf_case;

typedef struct sdpa_varlen_perf_model {
    gd_context *ctx;
    gd_tensor q;
    gd_tensor k;
    gd_tensor v;
    gd_tensor cu_seqlens;
    gd_tensor grad_out;
    gd_sdpa_varlen_config config;
    size_t total_tokens;
    size_t q_bytes;
    size_t kv_bytes;
    double pair_count;
} sdpa_varlen_perf_model;

typedef bool (*sdpa_varlen_perf_run_fn)(sdpa_varlen_perf_model *model);

static double sdpa_varlen_perf_now_seconds(void)
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

static int sdpa_varlen_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t sdpa_varlen_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static bool sdpa_varlen_perf_checked_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return false;
    }
    if (a != 0U && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool sdpa_varlen_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[SDPA_VARLEN][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

static int sdpa_varlen_perf_allowed(int i,
                                    int j,
                                    bool causal,
                                    int32_t window,
                                    int32_t prefix_len)
{
    if (causal) {
        if (prefix_len > 0) {
            if (i < prefix_len) {
                if (j >= prefix_len) {
                    return 0;
                }
            } else if (j > i) {
                return 0;
            }
        } else if (j > i) {
            return 0;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (i >= prefix_len && j >= prefix_len && (i - j) >= window) {
                return 0;
            }
        } else if ((i - j) >= window) {
            return 0;
        }
    }
    return 1;
}

static double sdpa_varlen_perf_pair_count(const sdpa_varlen_perf_case *pcase)
{
    int32_t i;
    int32_t j;
    double per_sequence = 0.0;
    if (pcase == NULL) {
        return 0.0;
    }
    for (i = 0; i < pcase->seqlen; ++i) {
        for (j = 0; j < pcase->seqlen; ++j) {
            if (sdpa_varlen_perf_allowed(i,
                                          j,
                                          pcase->causal,
                                          pcase->sliding_window,
                                          pcase->prefix_len)) {
                per_sequence += 1.0;
            }
        }
    }
    return per_sequence * (double)pcase->batch * (double)pcase->hq;
}

static bool sdpa_varlen_perf_case_selected(const sdpa_varlen_perf_case *pcase,
                                           const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "generic_tail_f16") == 0 ||
               strcmp(pcase->name, "fast_dh64_window_f16") == 0 ||
               strcmp(pcase->name, "generic_causal_f32") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static gd_memory_config sdpa_varlen_perf_config(size_t q_bytes,
                                                size_t kv_bytes,
                                                size_t cu_bytes,
                                                size_t stats_bytes)
{
    gd_memory_config cfg;
    size_t params_bytes = q_bytes + kv_bytes * 2U + q_bytes + cu_bytes;
    size_t scratch_bytes = q_bytes * 6U + kv_bytes * 8U + stats_bytes * 4U + 64U * 1024U * 1024U;
    memset(&cfg, 0, sizeof(cfg));
    cfg.params_bytes = sdpa_varlen_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U);
    cfg.state_bytes = 4U * 1024U * 1024U;
    cfg.scratch_slot_bytes = sdpa_varlen_perf_align_up(scratch_bytes, 4096U);
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    return cfg;
}

static bool sdpa_varlen_perf_setup(const sdpa_varlen_perf_case *pcase,
                                   sdpa_varlen_perf_model *model)
{
    size_t elem_size;
    size_t total_tokens;
    size_t q_elems;
    size_t kv_elems;
    size_t cu_count;
    size_t cu_bytes;
    size_t stats_elems;
    size_t stats_bytes;
    gd_memory_config mem_cfg;
    int64_t q_shape[3];
    int64_t k_shape[3];
    int64_t cu_shape[1];
    int32_t *cu_host;
    int32_t b;
    if (pcase == NULL || model == NULL || pcase->batch <= 0 || pcase->seqlen <= 0 ||
        pcase->hq <= 0 || pcase->hkv <= 0 || pcase->dh <= 0 ||
        (pcase->hq % pcase->hkv) != 0) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    elem_size = gd_dtype_size(pcase->dtype);
    if (elem_size == 0U ||
        !sdpa_varlen_perf_checked_mul((size_t)pcase->batch, (size_t)pcase->seqlen, &total_tokens) ||
        !sdpa_varlen_perf_checked_mul(total_tokens, (size_t)pcase->hq, &q_elems) ||
        !sdpa_varlen_perf_checked_mul(q_elems, (size_t)pcase->dh, &q_elems) ||
        !sdpa_varlen_perf_checked_mul(total_tokens, (size_t)pcase->hkv, &kv_elems) ||
        !sdpa_varlen_perf_checked_mul(kv_elems, (size_t)pcase->dh, &kv_elems) ||
        !sdpa_varlen_perf_checked_mul(q_elems, elem_size, &model->q_bytes) ||
        !sdpa_varlen_perf_checked_mul(kv_elems, elem_size, &model->kv_bytes) ||
        !sdpa_varlen_perf_checked_mul(total_tokens, (size_t)pcase->hq, &stats_elems) ||
        !sdpa_varlen_perf_checked_mul(stats_elems, 3U * sizeof(float), &stats_bytes)) {
        fprintf(stderr, "[SDPA_VARLEN][FAIL] invalid case shape/dtype: %s\n", pcase->name);
        return false;
    }
    cu_count = (size_t)pcase->batch + 1U;
    if (!sdpa_varlen_perf_checked_mul(cu_count, sizeof(int32_t), &cu_bytes)) {
        return false;
    }
    cu_host = (int32_t *)calloc(cu_count, sizeof(*cu_host));
    if (cu_host == NULL) {
        fprintf(stderr, "[SDPA_VARLEN][FAIL] cu_seqlens allocation failed\n");
        return false;
    }
    for (b = 0; b <= pcase->batch; ++b) {
        cu_host[b] = b * pcase->seqlen;
    }

    model->total_tokens = total_tokens;
    model->pair_count = sdpa_varlen_perf_pair_count(pcase);
    model->config.scale = 0.0f;
    model->config.causal = pcase->causal;
    model->config.sliding_window = pcase->sliding_window;
    model->config.prefix_len = pcase->prefix_len;
    model->config.max_seqlen = pcase->seqlen;

    mem_cfg = sdpa_varlen_perf_config(model->q_bytes, model->kv_bytes, cu_bytes, stats_bytes);
    if (!sdpa_varlen_perf_status_ok(NULL, gd_context_create(&mem_cfg, &model->ctx), "gd_context_create")) {
        free(cu_host);
        return false;
    }

    q_shape[0] = (int64_t)total_tokens;
    q_shape[1] = pcase->hq;
    q_shape[2] = pcase->dh;
    k_shape[0] = (int64_t)total_tokens;
    k_shape[1] = pcase->hkv;
    k_shape[2] = pcase->dh;
    cu_shape[0] = (int64_t)cu_count;

    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(3U, q_shape),
                                                       256U,
                                                       UINT64_C(0x53565041514),
                                                       -0.125f,
                                                       0.125f,
                                                       &model->q));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(3U, k_shape),
                                                       256U,
                                                       UINT64_C(0x535650414b),
                                                       -0.125f,
                                                       0.125f,
                                                       &model->k));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(3U, k_shape),
                                                       256U,
                                                       UINT64_C(0x5356504156),
                                                       -0.125f,
                                                       0.125f,
                                                       &model->v));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_rand_uniform(model->ctx,
                                                       GD_ARENA_PARAMS,
                                                       pcase->dtype,
                                                       gd_shape_make(3U, q_shape),
                                                       256U,
                                                       UINT64_C(0x5356504147),
                                                       -0.125f,
                                                       0.125f,
                                                       &model->grad_out));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_empty(model->ctx,
                                                GD_ARENA_PARAMS,
                                                GD_DTYPE_I32,
                                                gd_shape_make(1U, cu_shape),
                                                256U,
                                                &model->cu_seqlens));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_tensor_write(model->ctx, &model->cu_seqlens, cu_host, cu_bytes));
    free(cu_host);
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void sdpa_varlen_perf_destroy(sdpa_varlen_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool sdpa_varlen_perf_run_forward(sdpa_varlen_perf_model *model)
{
    gd_tensor out;
    model->q.requires_grad = false;
    model->k.requires_grad = false;
    model->v.requires_grad = false;
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_sdpa_varlen(model->ctx,
                                                &model->q,
                                                &model->k,
                                                &model->v,
                                                &model->cu_seqlens,
                                                &model->config,
                                                &out));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool sdpa_varlen_perf_run_backward(sdpa_varlen_perf_model *model)
{
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    model->q.requires_grad = false;
    model->k.requires_grad = false;
    model->v.requires_grad = false;
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_sdpa_varlen_backward(model->ctx,
                                                         &model->q,
                                                         &model->k,
                                                         &model->v,
                                                         &model->cu_seqlens,
                                                         &model->grad_out,
                                                         &model->config,
                                                         &dq,
                                                         &dk,
                                                         &dv));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool sdpa_varlen_perf_run_autograd(sdpa_varlen_perf_model *model)
{
    gd_tensor out;
    model->q.requires_grad = true;
    model->k.requires_grad = true;
    model->v.requires_grad = true;
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx,
                                gd_sdpa_varlen(model->ctx,
                                                &model->q,
                                                &model->k,
                                                &model->v,
                                                &model->cu_seqlens,
                                                &model->config,
                                                &out));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad_out));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    SDPA_VARLEN_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool sdpa_varlen_perf_measure(sdpa_varlen_perf_model *model,
                                     const char *label,
                                     sdpa_varlen_perf_run_fn fn,
                                     int warmup,
                                     int iters,
                                     double logical_flops)
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
    t0 = sdpa_varlen_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = sdpa_varlen_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    tflops = (logical_flops / GD_SDPA_VARLEN_PERF_TFLOP) / (elapsed / (double)iters);
    printf("[SDPA_VARLEN][%s] avg_ms=%.4f logical_TFLOP/s=%.3f\n", label, avg_ms, tflops);
    return true;
}

int main(void)
{
    static const sdpa_varlen_perf_case cases[] = {
        {.name = "generic_tail_f16",
         .dtype = GD_DTYPE_F16,
         .batch = 3,
         .seqlen = 73,
         .hq = 4,
         .hkv = 2,
         .dh = 40,
         .causal = true,
         .sliding_window = 17,
         .prefix_len = 5},
        {.name = "fast_dh64_window_f16",
         .dtype = GD_DTYPE_F16,
         .batch = 1,
         .seqlen = 256,
         .hq = 8,
         .hkv = 2,
         .dh = 64,
         .causal = true,
         .sliding_window = 128,
         .prefix_len = 16},
        {.name = "generic_causal_f32",
         .dtype = GD_DTYPE_F32,
         .batch = 1,
         .seqlen = 96,
         .hq = 4,
         .hkv = 2,
         .dh = 32,
         .causal = true,
         .sliding_window = 0,
         .prefix_len = 0},
        {.name = "fast_full_causal_f16",
         .dtype = GD_DTYPE_F16,
         .batch = 1,
         .seqlen = 512,
         .hq = 8,
         .hkv = 2,
         .dh = 64,
         .causal = true,
         .sliding_window = 0,
         .prefix_len = 0},
        {.name = "fast_long_window_f16",
         .dtype = GD_DTYPE_F16,
         .batch = 2,
         .seqlen = 2048,
         .hq = 16,
         .hkv = 4,
         .dh = 64,
         .causal = true,
         .sliding_window = 256,
         .prefix_len = 32},
        {.name = "generic_noncausal_f16",
         .dtype = GD_DTYPE_F16,
         .batch = 4,
         .seqlen = 128,
         .hq = 8,
         .hkv = 2,
         .dh = 48,
         .causal = false,
         .sliding_window = 0,
         .prefix_len = 0},
    };
    const char *profile = getenv("GD_SDPA_VARLEN_PERF_PROFILE");
    int warmup = sdpa_varlen_perf_env_int("GD_SDPA_VARLEN_PERF_WARMUP", 5, 0, 10000);
    int iters = sdpa_varlen_perf_env_int("GD_SDPA_VARLEN_PERF_ITERS", 30, 1, 1000000);
    bool ran = false;
    size_t i;
    printf("[SDPA_VARLEN] warmup=%d iters=%d profile=%s\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        sdpa_varlen_perf_model model;
        const sdpa_varlen_perf_case *pcase = &cases[i];
        double fwd_flops;
        double bwd_flops;
        double autograd_flops;
        if (!sdpa_varlen_perf_case_selected(pcase, profile)) {
            continue;
        }
        ran = true;
        if (!sdpa_varlen_perf_setup(pcase, &model)) {
            return 1;
        }
        fwd_flops = model.pair_count * 4.0 * (double)pcase->dh;
        bwd_flops = model.pair_count * 18.0 * (double)pcase->dh;
        autograd_flops = fwd_flops + bwd_flops;
        printf("[SDPA_VARLEN] case=%s dtype=%s B=%d T=%d N=%zu Hq=%d Hkv=%d Dh=%d causal=%d window=%d prefix=%d pairs=%.0f q_bytes=%zu kv_bytes=%zu\n",
               pcase->name,
               gd_dtype_name(pcase->dtype),
               pcase->batch,
               pcase->seqlen,
               model.total_tokens,
               pcase->hq,
               pcase->hkv,
               pcase->dh,
               pcase->causal ? 1 : 0,
               pcase->sliding_window,
               pcase->prefix_len,
               model.pair_count,
               model.q_bytes,
               model.kv_bytes);
        if (!sdpa_varlen_perf_measure(&model,
                                      "fwd",
                                      sdpa_varlen_perf_run_forward,
                                      warmup,
                                      iters,
                                      fwd_flops) ||
            !sdpa_varlen_perf_measure(&model,
                                      "bwd",
                                      sdpa_varlen_perf_run_backward,
                                      warmup,
                                      iters,
                                      bwd_flops) ||
            !sdpa_varlen_perf_measure(&model,
                                      "autograd",
                                      sdpa_varlen_perf_run_autograd,
                                      warmup,
                                      iters,
                                      autograd_flops)) {
            sdpa_varlen_perf_destroy(&model);
            return 1;
        }
        sdpa_varlen_perf_destroy(&model);
    }
    if (!ran) {
        fprintf(stderr,
                "[SDPA_VARLEN][FAIL] no cases selected for GD_SDPA_VARLEN_PERF_PROFILE=%s\n",
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}
