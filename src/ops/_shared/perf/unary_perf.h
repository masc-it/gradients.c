#ifndef GD_OPS_SHARED_PERF_UNARY_PERF_H
#define GD_OPS_SHARED_PERF_UNARY_PERF_H

#include "perf_common.h"

typedef struct gd_unary_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    bool smoke;
} gd_unary_perf_case;

typedef gd_status (*gd_unary_perf_forward_fn)(gd_context *ctx,
                                             const gd_tensor *x,
                                             gd_tensor *out);
typedef gd_status (*gd_unary_perf_backward_fn)(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *grad_out,
                                              gd_tensor *grad_x);

typedef struct gd_unary_perf_spec {
    const char *tag;
    const char *profile_env;
    const char *warmup_env;
    const char *iters_env;
    const char *title;
    const char *note;
    uint64_t x_seed;
    uint64_t grad_seed;
    float x_min;
    float x_max;
    float grad_min;
    float grad_max;
    size_t state_bytes;
    size_t data_slot_bytes;
    double autograd_estimated_public_multiplier;
    gd_unary_perf_forward_fn forward;
    gd_unary_perf_backward_fn backward;
} gd_unary_perf_spec;

typedef struct gd_unary_perf_model {
    const gd_unary_perf_spec *spec;
    gd_context *ctx;
    gd_tensor x;
    gd_tensor grad;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    size_t count;
    size_t elem_size;
    size_t tensor_bytes;
} gd_unary_perf_model;

static bool gd_unary_perf_setup(const gd_unary_perf_spec *spec,
                                const gd_unary_perf_case *pcase,
                                gd_unary_perf_model *model)
{
    gd_memory_config cfg;
    size_t params_bytes;
    size_t scratch_bytes;
    uint32_t i;
    gd_status st;
    if (spec == NULL || pcase == NULL || model == NULL || spec->forward == NULL || spec->backward == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->spec = spec;
    model->elem_size = gd_dtype_size(pcase->dtype);
    if (model->elem_size == 0U || !gd_perf_shape_count(pcase->rank, pcase->shape, false, &model->count) ||
        model->count > SIZE_MAX / model->elem_size) {
        fprintf(stderr, "[%s][FAIL] invalid case shape/dtype: %s\n", spec->tag, pcase->name);
        return false;
    }
    model->rank = pcase->rank;
    for (i = 0U; i < pcase->rank; ++i) {
        model->shape[i] = pcase->shape[i];
    }
    model->tensor_bytes = model->count * model->elem_size;
    if (model->tensor_bytes > (SIZE_MAX - 128U * 1024U * 1024U) / 8U) {
        return false;
    }
    params_bytes = gd_perf_align_up(model->tensor_bytes * 2U + 64U * 1024U * 1024U, 4096U);
    scratch_bytes = gd_perf_align_up(model->tensor_bytes * 8U + 128U * 1024U * 1024U, 4096U);
    cfg = gd_perf_memory_config(params_bytes,
                                spec->state_bytes,
                                scratch_bytes,
                                spec->data_slot_bytes,
                                2U,
                                2U,
                                256U);
    st = gd_context_create(&cfg, &model->ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[%s] skipped case=%s: no supported GPU backend\n", spec->tag, pcase->name);
        return false;
    }
    if (!gd_perf_status_ok(spec->tag, model->ctx, st, "gd_context_create")) {
        return false;
    }
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_rand_uniform(model->ctx,
                                              GD_ARENA_PARAMS,
                                              pcase->dtype,
                                              gd_shape_make(pcase->rank, pcase->shape),
                                              256U,
                                              spec->x_seed,
                                              spec->x_min,
                                              spec->x_max,
                                              &model->x));
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_rand_uniform(model->ctx,
                                              GD_ARENA_PARAMS,
                                              pcase->dtype,
                                              gd_shape_make(pcase->rank, pcase->shape),
                                              256U,
                                              spec->grad_seed,
                                              spec->grad_min,
                                              spec->grad_max,
                                              &model->grad));
    GD_PERF_REQUIRE_OK(spec->tag, model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void gd_unary_perf_destroy(gd_unary_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool gd_unary_perf_run_forward(void *user)
{
    gd_unary_perf_model *model = (gd_unary_perf_model *)user;
    gd_tensor y;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->forward(model->ctx, &model->x, &y));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_unary_perf_run_backward_direct(void *user)
{
    gd_unary_perf_model *model = (gd_unary_perf_model *)user;
    gd_tensor dx;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->backward(model->ctx, &model->x, &model->grad, &dx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_unary_perf_run_forward_backward_autograd(void *user)
{
    gd_unary_perf_model *model = (gd_unary_perf_model *)user;
    gd_tensor y;
    model->x.requires_grad = true;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->forward(model->ctx, &model->x, &y));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_backward(model->ctx, &y, &model->grad));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static void gd_unary_perf_print_result(const char *label,
                                       const gd_unary_perf_case *pcase,
                                       const gd_unary_perf_model *model,
                                       double seconds,
                                       int warmup,
                                       int iters,
                                       double logical_bytes,
                                       double estimated_public_bytes)
{
    char shape[128];
    double logical_gib_s = (logical_bytes / GD_PERF_GIB) / seconds;
    double gelem_s = ((double)model->count * 1.0e-9) / seconds;
    gd_perf_format_shape(pcase->rank, pcase->shape, "x", shape, sizeof(shape));
    printf("[%s][%s] case=%s dtype=%s shape=%s elems=%zu warmup=%d iters=%d "
           "avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f",
           model->spec->tag,
           label,
           pcase->name,
           gd_dtype_name(pcase->dtype),
           shape,
           model->count,
           warmup,
           iters,
           seconds * 1.0e3,
           logical_gib_s,
           gelem_s);
    if (estimated_public_bytes > 0.0) {
        double estimated_gib_s = (estimated_public_bytes / GD_PERF_GIB) / seconds;
        printf(" estimated_public_GiB/s=%.2f", estimated_gib_s);
    }
    printf("\n");
}

static bool gd_unary_perf_run_case(const gd_unary_perf_spec *spec,
                                   const gd_unary_perf_case *pcase,
                                   int warmup,
                                   int iters)
{
    gd_unary_perf_model model;
    double fwd_s;
    double bwd_s;
    double pair_s;
    double bytes_per_elem;
    memset(&model, 0, sizeof(model));
    if (!gd_unary_perf_setup(spec, pcase, &model)) {
        gd_unary_perf_destroy(&model);
        return false;
    }
    bytes_per_elem = (double)model.elem_size;
    printf("[%s] case=%s dtype=%s elems=%zu tensor=%.1fMiB\n",
           spec->tag,
           pcase->name,
           gd_dtype_name(pcase->dtype),
           model.count,
           (double)model.tensor_bytes / GD_PERF_MIB);
    if (!gd_perf_measure_seconds(&model, gd_unary_perf_run_forward, warmup, iters, &fwd_s) ||
        !gd_perf_measure_seconds(&model, gd_unary_perf_run_backward_direct, warmup, iters, &bwd_s) ||
        !gd_perf_measure_seconds(&model, gd_unary_perf_run_forward_backward_autograd, warmup, iters, &pair_s)) {
        gd_unary_perf_destroy(&model);
        return false;
    }
    gd_unary_perf_print_result("fwd", pcase, &model, fwd_s, warmup, iters, model.count * bytes_per_elem * 2.0, 0.0);
    gd_unary_perf_print_result("bwd_direct",
                               pcase,
                               &model,
                               bwd_s,
                               warmup,
                               iters,
                               model.count * bytes_per_elem * 3.0,
                               0.0);
    gd_unary_perf_print_result("fwd_bwd_autograd",
                               pcase,
                               &model,
                               pair_s,
                               warmup,
                               iters,
                               model.count * bytes_per_elem * 5.0,
                               spec->autograd_estimated_public_multiplier > 0.0
                                   ? model.count * bytes_per_elem * spec->autograd_estimated_public_multiplier
                                   : 0.0);
    gd_unary_perf_destroy(&model);
    return true;
}

static int gd_unary_perf_main(const gd_unary_perf_spec *spec,
                              const gd_unary_perf_case *cases,
                              size_t case_count)
{
    const char *profile;
    int warmup;
    int iters;
    bool ran = false;
    size_t i;
    if (spec == NULL || cases == NULL || spec->tag == NULL || spec->profile_env == NULL ||
        spec->warmup_env == NULL || spec->iters_env == NULL) {
        return 1;
    }
    profile = getenv(spec->profile_env);
    warmup = gd_perf_env_int(spec->warmup_env, 10, 0, 10000);
    iters = gd_perf_env_int(spec->iters_env, 100, 1, 1000000);
    printf("[%s] %s: warmup=%d iters=%d profile=%s\n",
           spec->tag,
           spec->title != NULL ? spec->title : "public API perf",
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "all");
    if (spec->note != NULL && spec->note[0] != '\0') {
        printf("[%s] %s\n", spec->tag, spec->note);
    }
    for (i = 0U; i < case_count; ++i) {
        if (!gd_perf_case_selected(cases[i].name, profile, cases[i].smoke, true)) {
            continue;
        }
        ran = true;
        if (!gd_unary_perf_run_case(spec, &cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr,
                "[%s][FAIL] no cases selected for %s=%s\n",
                spec->tag,
                spec->profile_env,
                profile != NULL ? profile : "(null)");
        return 2;
    }
    return 0;
}

#endif /* GD_OPS_SHARED_PERF_UNARY_PERF_H */
