/*
 * gd_embedding Metal performance probe.
 *
 * Run with:
 *   make op-perf OP=embedding
 *
 * Optional environment:
 *   GD_EMBEDDING_PERF_PROFILE=smoke|all|<case-name>
 *   GD_EMBEDDING_PERF_WARMUP=10
 *   GD_EMBEDDING_PERF_ITERS=100
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

#define EMB_PERF_GIB (1024.0 * 1024.0 * 1024.0)

typedef struct emb_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t vocab;
    uint32_t dim;
    uint32_t ids_rank;
    int64_t ids_shape[GD_MAX_DIMS];
} emb_perf_case;

typedef struct emb_perf_model {
    gd_context *ctx;
    gd_tensor table;
    gd_tensor ids;
    gd_tensor grad;
    size_t ids_count;
    size_t table_count;
    size_t out_count;
    size_t elem_size;
    size_t table_bytes;
    size_t ids_bytes;
    size_t out_bytes;
    size_t scratch_f32_bytes;
} emb_perf_model;

typedef bool (*emb_perf_run_fn)(emb_perf_model *model);

static double emb_perf_now_seconds(void)
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

static int emb_perf_env_int(const char *name, int fallback, int min_value, int max_value)
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

static size_t emb_perf_align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1U) & ~(alignment - 1U);
}

static const char *emb_perf_dtype_name(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 ? "f32" : "f16";
}

static size_t emb_perf_dtype_size(gd_dtype dtype)
{
    if (dtype == GD_DTYPE_F16) {
        return 2U;
    }
    if (dtype == GD_DTYPE_F32) {
        return 4U;
    }
    return 0U;
}

static bool emb_perf_status_ok(gd_context *ctx, gd_status st, const char *expr)
{
    if (st == GD_OK) {
        return true;
    }
    fprintf(stderr,
            "[EMBEDDING][FAIL] %s -> %s (%d), context_error=%s\n",
            expr,
            gd_status_string(st),
            (int)st,
            ctx != NULL ? gd_context_error(ctx) : "no context");
    return false;
}

#define EMB_PERF_REQUIRE_OK(ctx, expr)                            \
    do {                                                          \
        gd_status emb_perf_st__ = (expr);                         \
        if (!emb_perf_status_ok((ctx), emb_perf_st__, #expr)) {   \
            return false;                                         \
        }                                                         \
    } while (0)

static bool emb_perf_case_selected(const emb_perf_case *pcase, const char *profile)
{
    if (pcase == NULL) {
        return false;
    }
    if (profile == NULL || profile[0] == '\0' || strcmp(profile, "smoke") == 0) {
        return strcmp(pcase->name, "tokens_8k_x64_f16") == 0 ||
               strcmp(pcase->name, "bert_32x128_d768_f16") == 0;
    }
    if (strcmp(profile, "all") == 0) {
        return true;
    }
    return strcmp(profile, pcase->name) == 0;
}

static bool emb_perf_count_ids(const emb_perf_case *pcase, size_t *out_count)
{
    uint32_t d;
    size_t count = 1U;
    if (pcase == NULL || out_count == NULL || pcase->ids_rank == 0U ||
        pcase->ids_rank > GD_MAX_DIMS) {
        return false;
    }
    for (d = 0U; d < pcase->ids_rank; ++d) {
        if (pcase->ids_shape[d] <= 0 ||
            (uint64_t)pcase->ids_shape[d] > (uint64_t)(SIZE_MAX / count)) {
            return false;
        }
        count *= (size_t)pcase->ids_shape[d];
    }
    *out_count = count;
    return true;
}

static void emb_perf_make_ids(int32_t *ids, size_t count, uint32_t vocab, uint32_t seed)
{
    size_t i;
    uint32_t state = seed != 0U ? seed : 1U;
    if (ids == NULL || vocab == 0U) {
        return;
    }
    for (i = 0U; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        ids[i] = (int32_t)(state % vocab);
        if ((i & 15U) == 0U && i > 0U) {
            ids[i] = ids[i - 1U];
        }
    }
}

static bool emb_perf_init(emb_perf_model *model, const emb_perf_case *pcase)
{
    gd_memory_config cfg;
    int64_t table_shape[2];
    int64_t grad_shape[GD_MAX_DIMS];
    int32_t *ids_host = NULL;
    uint32_t d;
    size_t params_bytes;
    size_t scratch_bytes;
    if (model == NULL || pcase == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    if (!emb_perf_count_ids(pcase, &model->ids_count) || pcase->vocab == 0U || pcase->dim == 0U) {
        return false;
    }
    model->elem_size = emb_perf_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || (size_t)pcase->vocab > SIZE_MAX / (size_t)pcase->dim ||
        model->ids_count > SIZE_MAX / (size_t)pcase->dim) {
        return false;
    }
    model->table_count = (size_t)pcase->vocab * (size_t)pcase->dim;
    model->out_count = model->ids_count * (size_t)pcase->dim;
    if (model->table_count > SIZE_MAX / model->elem_size ||
        model->out_count > SIZE_MAX / model->elem_size ||
        model->ids_count > SIZE_MAX / sizeof(int32_t) ||
        model->table_count > SIZE_MAX / sizeof(float)) {
        return false;
    }
    model->table_bytes = model->table_count * model->elem_size;
    model->out_bytes = model->out_count * model->elem_size;
    model->ids_bytes = model->ids_count * sizeof(int32_t);
    model->scratch_f32_bytes = pcase->dtype == GD_DTYPE_F16 ? model->table_count * sizeof(float) : 0U;
    params_bytes = emb_perf_align_up(model->table_bytes + model->out_bytes + model->ids_bytes +
                                     32U * 1024U * 1024U, 4096U);
    scratch_bytes = emb_perf_align_up(model->out_bytes * 4U + model->table_bytes * 4U +
                                      model->scratch_f32_bytes * 2U + 64U * 1024U * 1024U, 4096U);
    cfg = gd_memory_config_default();
    cfg.params_bytes = params_bytes;
    cfg.state_bytes = 1024U * 1024U;
    cfg.scratch_slot_bytes = scratch_bytes;
    cfg.data_slot_bytes = 1024U * 1024U;
    cfg.scratch_slots = 3U;
    cfg.data_slots = 2U;
    cfg.default_alignment = 256U;
    EMB_PERF_REQUIRE_OK(NULL, gd_context_create(&cfg, &model->ctx));
    table_shape[0] = (int64_t)pcase->vocab;
    table_shape[1] = (int64_t)pcase->dim;
    EMB_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx,
                                               GD_ARENA_PARAMS,
                                               pcase->dtype,
                                               gd_shape_make(2U, table_shape),
                                               256U,
                                               UINT64_C(0x454D4201),
                                               -0.5f,
                                               0.5f,
                                               &model->table));
    EMB_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_empty(model->ctx,
                                        GD_ARENA_PARAMS,
                                        GD_DTYPE_I32,
                                        gd_shape_make(pcase->ids_rank, pcase->ids_shape),
                                        256U,
                                        &model->ids));
    ids_host = (int32_t *)malloc(model->ids_bytes);
    if (ids_host == NULL) {
        return false;
    }
    emb_perf_make_ids(ids_host, model->ids_count, pcase->vocab, 0x454D4202U);
    EMB_PERF_REQUIRE_OK(model->ctx, gd_tensor_write(model->ctx, &model->ids, ids_host, model->ids_bytes));
    free(ids_host);
    ids_host = NULL;
    for (d = 0U; d < GD_MAX_DIMS; ++d) {
        grad_shape[d] = 0;
    }
    for (d = 0U; d < pcase->ids_rank; ++d) {
        grad_shape[d] = pcase->ids_shape[d];
    }
    grad_shape[pcase->ids_rank] = (int64_t)pcase->dim;
    EMB_PERF_REQUIRE_OK(model->ctx,
                        gd_tensor_rand_uniform(model->ctx,
                                               GD_ARENA_PARAMS,
                                               pcase->dtype,
                                               gd_shape_make(pcase->ids_rank + 1U, grad_shape),
                                               256U,
                                               UINT64_C(0x454D4203),
                                               -0.25f,
                                               0.25f,
                                               &model->grad));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void emb_perf_destroy(emb_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool emb_perf_run_forward(emb_perf_model *model)
{
    gd_tensor out;
    model->table.requires_grad = false;
    EMB_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_embedding(model->ctx, &model->table, &model->ids, &out));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool emb_perf_run_backward(emb_perf_model *model)
{
    gd_tensor grad_table;
    model->table.requires_grad = false;
    EMB_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    EMB_PERF_REQUIRE_OK(model->ctx,
                        gd_embedding_backward(model->ctx,
                                              &model->table,
                                              &model->ids,
                                              &model->grad,
                                              &grad_table));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool emb_perf_run_autograd(emb_perf_model *model)
{
    gd_tensor out;
    model->table.requires_grad = true;
    EMB_PERF_REQUIRE_OK(model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_embedding(model->ctx, &model->table, &model->ids, &out));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_backward(model->ctx, &out, &model->grad));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_end_step(model->ctx));
    EMB_PERF_REQUIRE_OK(model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool emb_perf_measure(emb_perf_model *model,
                             const char *label,
                             emb_perf_run_fn fn,
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
    t0 = emb_perf_now_seconds();
    for (i = 0; i < iters; ++i) {
        if (!fn(model)) {
            return false;
        }
    }
    elapsed = emb_perf_now_seconds() - t0;
    avg_ms = elapsed * 1.0e3 / (double)iters;
    gib_s = (effective_bytes / EMB_PERF_GIB) / (elapsed / (double)iters);
    printf("[EMBEDDING][%s] avg_ms=%.4f effective_GiB/s=%.2f\n", label, avg_ms, gib_s);
    return true;
}

static void emb_perf_print_case(const emb_perf_case *pcase, const emb_perf_model *model)
{
    uint32_t d;
    printf("[EMBEDDING] case=%s dtype=%s vocab=%u dim=%u ids=[",
           pcase->name,
           emb_perf_dtype_name(pcase->dtype),
           pcase->vocab,
           pcase->dim);
    for (d = 0U; d < pcase->ids_rank; ++d) {
        printf("%s%lld", d == 0U ? "" : ",", (long long)pcase->ids_shape[d]);
    }
    printf("] ids_count=%zu table_bytes=%zu out_bytes=%zu scratch_f32_bytes=%zu\n",
           model->ids_count,
           model->table_bytes,
           model->out_bytes,
           model->scratch_f32_bytes);
}

int main(void)
{
    const emb_perf_case cases[] = {
        {"tokens_8k_x64_f16", GD_DTYPE_F16, 8192U, 64U, 2U, {64, 128, 0, 0, 0, 0, 0, 0}},
        {"bert_32x128_d768_f16", GD_DTYPE_F16, 30522U, 768U, 2U, {32, 128, 0, 0, 0, 0, 0, 0}},
        {"gpt_8x1024_d768_f16", GD_DTYPE_F16, 50257U, 768U, 2U, {8, 1024, 0, 0, 0, 0, 0, 0}},
        {"wide_hidden_1x1024_d4096_f16", GD_DTYPE_F16, 8192U, 4096U, 2U, {1, 1024, 0, 0, 0, 0, 0, 0}},
        {"f32_table_16x256_d256", GD_DTYPE_F32, 8192U, 256U, 2U, {16, 256, 0, 0, 0, 0, 0, 0}},
    };
    const char *profile = getenv("GD_EMBEDDING_PERF_PROFILE");
    int warmup = emb_perf_env_int("GD_EMBEDDING_PERF_WARMUP", 10, 0, 100000);
    int iters = emb_perf_env_int("GD_EMBEDDING_PERF_ITERS", 100, 1, 1000000);
    size_t i;
    bool any = false;
    bool ok = true;
    printf("[EMBEDDING] warmup=%d iters=%d profile=%s note=forward-lookup/backward-dense-atomic-scatter\n",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        emb_perf_model model;
        double fwd_bytes;
        double bwd_bytes;
        double autograd_bytes;
        if (!emb_perf_case_selected(&cases[i], profile)) {
            continue;
        }
        any = true;
        if (!emb_perf_init(&model, &cases[i])) {
            ok = false;
            break;
        }
        emb_perf_print_case(&cases[i], &model);
        fwd_bytes = (double)model.ids_bytes + (double)model.out_bytes * 2.0;
        bwd_bytes = (double)model.ids_bytes + (double)model.out_bytes +
                    (double)model.out_count * sizeof(float) * 2.0 +
                    (double)model.table_count * sizeof(float) +
                    (double)model.table_bytes;
        if (model.scratch_f32_bytes != 0U) {
            bwd_bytes += (double)model.scratch_f32_bytes;
        }
        autograd_bytes = fwd_bytes + bwd_bytes + (double)model.out_bytes;
        ok = emb_perf_measure(&model, "fwd", emb_perf_run_forward, warmup, iters, fwd_bytes) &&
             emb_perf_measure(&model, "bwd", emb_perf_run_backward, warmup, iters, bwd_bytes) &&
             emb_perf_measure(&model, "autograd", emb_perf_run_autograd, warmup, iters, autograd_bytes);
        emb_perf_destroy(&model);
        if (!ok) {
            break;
        }
    }
    if (!any) {
        fprintf(stderr, "[EMBEDDING][FAIL] no cases selected for profile=%s\n", profile != NULL ? profile : "smoke");
        return 2;
    }
    return ok ? 0 : 1;
}
