#ifndef GD_OPS_SHARED_PERF_PAIRWISE_LOSS_PERF_H
#define GD_OPS_SHARED_PERF_PAIRWISE_LOSS_PERF_H

#include "perf_common.h"

typedef struct gd_pairwise_loss_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    bool smoke;
} gd_pairwise_loss_perf_case;

typedef gd_status (*gd_pairwise_loss_perf_forward_fn)(gd_context *ctx,
                                                     const gd_tensor *x,
                                                     const gd_tensor *y,
                                                     gd_tensor *out);
typedef gd_status (*gd_pairwise_loss_perf_backward_fn)(gd_context *ctx,
                                                      const gd_tensor *x,
                                                      const gd_tensor *y,
                                                      const gd_tensor *grad_out,
                                                      gd_tensor *grad_x,
                                                      gd_tensor *grad_y);

typedef struct gd_pairwise_loss_perf_spec {
    const char *tag;
    const char *profile_env;
    const char *warmup_env;
    const char *iters_env;
    const char *banner_suffix;
    uint64_t x_seed;
    uint64_t y_seed;
    float random_min;
    float random_max;
    gd_pairwise_loss_perf_forward_fn forward;
    gd_pairwise_loss_perf_backward_fn backward;
} gd_pairwise_loss_perf_spec;

typedef struct gd_pairwise_loss_perf_model {
    const gd_pairwise_loss_perf_spec *spec;
    gd_context *ctx;
    gd_tensor x;
    gd_tensor y;
    gd_tensor grad;
    size_t count;
    size_t elem_size;
    size_t data_bytes;
} gd_pairwise_loss_perf_model;

static bool gd_pairwise_loss_perf_setup(const gd_pairwise_loss_perf_spec *spec,
                                        const gd_pairwise_loss_perf_case *pcase,
                                        gd_pairwise_loss_perf_model *model)
{
    gd_memory_config cfg;
    float grad_one = 1.0f;
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
    model->data_bytes = model->count * model->elem_size;
    cfg = gd_perf_memory_config(gd_perf_align_up(model->data_bytes * 2U + 1024U * 1024U, 4096U),
                                1024U * 1024U,
                                gd_perf_align_up(model->data_bytes * 8U + 32U * 1024U * 1024U, 4096U),
                                1024U * 1024U,
                                3U,
                                2U,
                                256U);
    if (!gd_perf_status_ok(spec->tag, NULL, gd_context_create(&cfg, &model->ctx), "gd_context_create")) {
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
                                              spec->random_min,
                                              spec->random_max,
                                              &model->x));
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_rand_uniform(model->ctx,
                                              GD_ARENA_PARAMS,
                                              pcase->dtype,
                                              gd_shape_make(pcase->rank, pcase->shape),
                                              256U,
                                              spec->y_seed,
                                              spec->random_min,
                                              spec->random_max,
                                              &model->y));
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_empty(model->ctx,
                                       GD_ARENA_PARAMS,
                                       GD_DTYPE_F32,
                                       gd_shape_make(0U, NULL),
                                       256U,
                                       &model->grad));
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_write(model->ctx, &model->grad, &grad_one, sizeof(grad_one)));
    GD_PERF_REQUIRE_OK(spec->tag, model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void gd_pairwise_loss_perf_destroy(gd_pairwise_loss_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool gd_pairwise_loss_perf_run_forward(void *user)
{
    gd_pairwise_loss_perf_model *model = (gd_pairwise_loss_perf_model *)user;
    gd_tensor loss;
    model->x.requires_grad = false;
    model->y.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->forward(model->ctx, &model->x, &model->y, &loss));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_pairwise_loss_perf_run_backward(void *user)
{
    gd_pairwise_loss_perf_model *model = (gd_pairwise_loss_perf_model *)user;
    gd_tensor dx;
    gd_tensor dy;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->backward(model->ctx, &model->x, &model->y, &model->grad, &dx, &dy));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_pairwise_loss_perf_run_autograd(void *user)
{
    gd_pairwise_loss_perf_model *model = (gd_pairwise_loss_perf_model *)user;
    gd_tensor loss;
    model->x.requires_grad = true;
    model->y.requires_grad = true;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->forward(model->ctx, &model->x, &model->y, &loss));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_backward(model->ctx, &loss, NULL));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static int gd_pairwise_loss_perf_main(const gd_pairwise_loss_perf_spec *spec,
                                      const gd_pairwise_loss_perf_case *cases,
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
    printf("[%s] warmup=%d iters=%d profile=%s%s%s\n",
           spec->tag,
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke",
           spec->banner_suffix != NULL && spec->banner_suffix[0] != '\0' ? " " : "",
           spec->banner_suffix != NULL ? spec->banner_suffix : "");
    for (i = 0U; i < case_count; ++i) {
        gd_pairwise_loss_perf_model model;
        double data_bytes;
        memset(&model, 0, sizeof(model));
        if (!gd_perf_case_selected(cases[i].name, profile, cases[i].smoke, false)) {
            continue;
        }
        ran = true;
        if (!gd_pairwise_loss_perf_setup(spec, &cases[i], &model)) {
            gd_pairwise_loss_perf_destroy(&model);
            return 1;
        }
        data_bytes = (double)model.data_bytes;
        printf("[%s] case=%s dtype=%s shape=", spec->tag, cases[i].name, gd_dtype_name(cases[i].dtype));
        gd_perf_print_shape(cases[i].rank, cases[i].shape, "x");
        printf(" elems=%zu bytes=%zu\n", model.count, model.data_bytes);
        if (!gd_perf_measure_labeled(&model,
                                     gd_pairwise_loss_perf_run_forward,
                                     warmup,
                                     iters,
                                     spec->tag,
                                     "fwd",
                                     2.0 * data_bytes) ||
            !gd_perf_measure_labeled(&model,
                                     gd_pairwise_loss_perf_run_backward,
                                     warmup,
                                     iters,
                                     spec->tag,
                                     "bwd",
                                     4.0 * data_bytes) ||
            !gd_perf_measure_labeled(&model,
                                     gd_pairwise_loss_perf_run_autograd,
                                     warmup,
                                     iters,
                                     spec->tag,
                                     "autograd",
                                     6.0 * data_bytes)) {
            gd_pairwise_loss_perf_destroy(&model);
            return 1;
        }
        gd_pairwise_loss_perf_destroy(&model);
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

#endif /* GD_OPS_SHARED_PERF_PAIRWISE_LOSS_PERF_H */
