#ifndef GD_OPS_SHARED_PERF_REDUCE_PERF_H
#define GD_OPS_SHARED_PERF_REDUCE_PERF_H

#include "perf_common.h"

typedef struct gd_reduce_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    int32_t axis;
    bool smoke;
} gd_reduce_perf_case;

typedef gd_status (*gd_reduce_perf_all_forward_fn)(gd_context *ctx,
                                                  const gd_tensor *x,
                                                  gd_tensor *out);
typedef gd_status (*gd_reduce_perf_all_backward_fn)(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   const gd_tensor *grad_out,
                                                   gd_tensor *grad_x);
typedef gd_status (*gd_reduce_perf_axis_forward_fn)(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   int32_t axis,
                                                   bool keepdims,
                                                   gd_tensor *out);
typedef gd_status (*gd_reduce_perf_axis_backward_fn)(gd_context *ctx,
                                                    const gd_tensor *x,
                                                    const gd_tensor *grad_out,
                                                    int32_t axis,
                                                    bool keepdims,
                                                    gd_tensor *grad_x);

typedef struct gd_reduce_perf_spec {
    const char *tag;
    const char *profile_env;
    const char *warmup_env;
    const char *iters_env;
    bool include_keepdims;
    bool scalar_f16_uses_f32;
    gd_reduce_perf_all_forward_fn all_forward;
    gd_reduce_perf_all_backward_fn all_backward;
    gd_reduce_perf_axis_forward_fn axis_forward;
    gd_reduce_perf_axis_backward_fn axis_backward;
} gd_reduce_perf_spec;

typedef struct gd_reduce_perf_model {
    const gd_reduce_perf_spec *spec;
    gd_context *ctx;
    gd_tensor x;
    gd_tensor scalar_grad;
    gd_tensor axis_grad;
    gd_tensor axis_keep_grad;
    int32_t axis;
    uint32_t axis_rank;
    int64_t axis_shape[GD_MAX_DIMS];
    uint32_t axis_keep_rank;
    int64_t axis_keep_shape[GD_MAX_DIMS];
    size_t x_count;
    size_t axis_count;
    size_t axis_keep_count;
    size_t elem_size;
    size_t scalar_elem_size;
    size_t x_bytes;
    size_t axis_bytes;
    size_t axis_keep_bytes;
} gd_reduce_perf_model;

static bool gd_reduce_perf_setup(const gd_reduce_perf_spec *spec,
                                 const gd_reduce_perf_case *pcase,
                                 gd_reduce_perf_model *model)
{
    gd_memory_config cfg;
    gd_dtype scalar_dtype;
    size_t params_bytes;
    size_t scratch_bytes;
    gd_status st;
    if (spec == NULL || pcase == NULL || model == NULL || spec->all_forward == NULL ||
        spec->all_backward == NULL || spec->axis_forward == NULL || spec->axis_backward == NULL) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->spec = spec;
    model->axis = pcase->axis;
    model->elem_size = gd_dtype_size(pcase->dtype);
    scalar_dtype = spec->scalar_f16_uses_f32 && pcase->dtype == GD_DTYPE_F16 ? GD_DTYPE_F32 : pcase->dtype;
    model->scalar_elem_size = gd_dtype_size(scalar_dtype);
    if (model->elem_size == 0U || model->scalar_elem_size == 0U ||
        !gd_perf_shape_count(pcase->rank, pcase->shape, false, &model->x_count) ||
        !gd_perf_axis_shape(pcase->rank, pcase->shape, pcase->axis, false, &model->axis_rank, model->axis_shape) ||
        !gd_perf_shape_count(model->axis_rank, model->axis_shape, true, &model->axis_count)) {
        fprintf(stderr, "[%s][FAIL] invalid case=%s\n", spec->tag, pcase->name);
        return false;
    }
    if (spec->include_keepdims &&
        (!gd_perf_axis_shape(pcase->rank,
                             pcase->shape,
                             pcase->axis,
                             true,
                             &model->axis_keep_rank,
                             model->axis_keep_shape) ||
         !gd_perf_shape_count(model->axis_keep_rank, model->axis_keep_shape, true, &model->axis_keep_count))) {
        fprintf(stderr, "[%s][FAIL] invalid keepdims case=%s\n", spec->tag, pcase->name);
        return false;
    }
    if (model->x_count > SIZE_MAX / model->elem_size || model->axis_count > SIZE_MAX / model->elem_size ||
        (spec->include_keepdims && model->axis_keep_count > SIZE_MAX / model->elem_size)) {
        return false;
    }
    model->x_bytes = model->x_count * model->elem_size;
    model->axis_bytes = model->axis_count * model->elem_size;
    model->axis_keep_bytes = spec->include_keepdims ? model->axis_keep_count * model->elem_size : 0U;
    params_bytes = gd_perf_align_up(model->x_bytes + model->axis_bytes + model->axis_keep_bytes +
                                        64U * 1024U * 1024U,
                                    4096U);
    scratch_bytes = gd_perf_align_up(model->x_bytes * 8U + model->axis_bytes * 4U +
                                         model->axis_keep_bytes * 4U + 256U * 1024U * 1024U,
                                     4096U);
    cfg = gd_perf_memory_config(params_bytes,
                                4U * 1024U * 1024U,
                                scratch_bytes,
                                4U * 1024U * 1024U,
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
                                              1001U,
                                              -1.0f,
                                              1.0f,
                                              &model->x));
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_empty(model->ctx,
                                       GD_ARENA_PARAMS,
                                       scalar_dtype,
                                       gd_shape_make(0U, NULL),
                                       256U,
                                       &model->scalar_grad));
    if (!gd_perf_write_scalar_one(model->ctx, &model->scalar_grad, scalar_dtype, spec->tag)) {
        return false;
    }
    GD_PERF_REQUIRE_OK(spec->tag,
                       model->ctx,
                       gd_tensor_rand_uniform(model->ctx,
                                              GD_ARENA_PARAMS,
                                              pcase->dtype,
                                              gd_shape_make(model->axis_rank,
                                                            model->axis_rank == 0U ? NULL : model->axis_shape),
                                              256U,
                                              2002U,
                                              -0.5f,
                                              0.5f,
                                              &model->axis_grad));
    if (spec->include_keepdims) {
        GD_PERF_REQUIRE_OK(spec->tag,
                           model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  pcase->dtype,
                                                  gd_shape_make(model->axis_keep_rank,
                                                                model->axis_keep_rank == 0U ? NULL
                                                                                             : model->axis_keep_shape),
                                                  256U,
                                                  3003U,
                                                  -0.5f,
                                                  0.5f,
                                                  &model->axis_keep_grad));
    }
    GD_PERF_REQUIRE_OK(spec->tag, model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void gd_reduce_perf_destroy(gd_reduce_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool gd_reduce_perf_run_all_forward(void *user)
{
    gd_reduce_perf_model *model = (gd_reduce_perf_model *)user;
    gd_tensor out;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->all_forward(model->ctx, &model->x, &out));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_all_backward(void *user)
{
    gd_reduce_perf_model *model = (gd_reduce_perf_model *)user;
    gd_tensor dx;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->all_backward(model->ctx, &model->x, &model->scalar_grad, &dx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_all_autograd(void *user)
{
    gd_reduce_perf_model *model = (gd_reduce_perf_model *)user;
    gd_tensor out;
    gd_tensor dx;
    model->x.requires_grad = true;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, model->spec->all_forward(model->ctx, &model->x, &out));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_backward(model->ctx, &out, &model->scalar_grad));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_axis_forward_impl(gd_reduce_perf_model *model, bool keepdims)
{
    gd_tensor out;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->axis_forward(model->ctx, &model->x, model->axis, keepdims, &out));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_axis_backward_impl(gd_reduce_perf_model *model, bool keepdims)
{
    gd_tensor dx;
    gd_tensor *grad = keepdims ? &model->axis_keep_grad : &model->axis_grad;
    model->x.requires_grad = false;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->axis_backward(model->ctx, &model->x, grad, model->axis, keepdims, &dx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_axis_autograd_impl(gd_reduce_perf_model *model, bool keepdims)
{
    gd_tensor out;
    gd_tensor dx;
    gd_tensor *grad = keepdims ? &model->axis_keep_grad : &model->axis_grad;
    model->x.requires_grad = true;
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    GD_PERF_REQUIRE_OK(model->spec->tag,
                       model->ctx,
                       model->spec->axis_forward(model->ctx, &model->x, model->axis, keepdims, &out));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_backward(model->ctx, &out, grad));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_tensor_grad(model->ctx, &model->x, &dx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_reduce_perf_run_axis_forward(void *user)
{
    return gd_reduce_perf_run_axis_forward_impl((gd_reduce_perf_model *)user, false);
}

static bool gd_reduce_perf_run_axis_backward(void *user)
{
    return gd_reduce_perf_run_axis_backward_impl((gd_reduce_perf_model *)user, false);
}

static bool gd_reduce_perf_run_axis_autograd(void *user)
{
    return gd_reduce_perf_run_axis_autograd_impl((gd_reduce_perf_model *)user, false);
}

static bool gd_reduce_perf_run_axis_forward_keepdims(void *user)
{
    return gd_reduce_perf_run_axis_forward_impl((gd_reduce_perf_model *)user, true);
}

static bool gd_reduce_perf_run_axis_backward_keepdims(void *user)
{
    return gd_reduce_perf_run_axis_backward_impl((gd_reduce_perf_model *)user, true);
}

static bool gd_reduce_perf_run_axis_autograd_keepdims(void *user)
{
    return gd_reduce_perf_run_axis_autograd_impl((gd_reduce_perf_model *)user, true);
}

static void gd_reduce_perf_print_result(const char *op,
                                        const gd_reduce_perf_case *pcase,
                                        const gd_reduce_perf_model *model,
                                        bool keepdims,
                                        double seconds,
                                        double logical_bytes,
                                        int warmup,
                                        int iters)
{
    double gib_s = (logical_bytes / GD_PERF_GIB) / seconds;
    double gelem_s = ((double)model->x_count / seconds) / 1.0e9;
    printf("[%s][%s] case=%s dtype=%s shape=", model->spec->tag, op, pcase->name, gd_dtype_name(pcase->dtype));
    gd_perf_print_shape(pcase->rank, pcase->shape, "x");
    printf(" axis=%d", pcase->axis);
    if (model->spec->include_keepdims) {
        printf(" keepdims=%s", keepdims ? "true" : "false");
    }
    printf(" axis_out=");
    if (keepdims) {
        gd_perf_print_shape(model->axis_keep_rank, model->axis_keep_shape, "x");
    } else {
        gd_perf_print_shape(model->axis_rank, model->axis_shape, "x");
    }
    printf(" warmup=%d iters=%d avg_ms=%.4f logical_GiB/s=%.2f Gelem/s=%.2f\n",
           warmup,
           iters,
           seconds * 1000.0,
           gib_s,
           gelem_s);
}

static bool gd_reduce_perf_measure_and_print(gd_reduce_perf_model *model,
                                             const gd_reduce_perf_case *pcase,
                                             const char *label,
                                             gd_perf_run_fn run,
                                             bool keepdims,
                                             double logical_bytes,
                                             int warmup,
                                             int iters)
{
    double seconds;
    if (!gd_perf_measure_seconds(model, run, warmup, iters, &seconds)) {
        return false;
    }
    gd_reduce_perf_print_result(label, pcase, model, keepdims, seconds, logical_bytes, warmup, iters);
    return true;
}

static bool gd_reduce_perf_run_case(const gd_reduce_perf_spec *spec,
                                    const gd_reduce_perf_case *pcase,
                                    int warmup,
                                    int iters)
{
    gd_reduce_perf_model model;
    double scalar_bytes;
    memset(&model, 0, sizeof(model));
    if (!gd_reduce_perf_setup(spec, pcase, &model)) {
        gd_reduce_perf_destroy(&model);
        return false;
    }
    scalar_bytes = (double)model.scalar_elem_size;
    printf("[%s] case=%s dtype=%s", spec->tag, pcase->name, gd_dtype_name(pcase->dtype));
    if (spec->scalar_f16_uses_f32) {
        printf(" scalar_dtype=%s", gd_dtype_name(model.scalar_grad.dtype));
    }
    printf(" elems=%zu tensor=%.1fMiB axis_count=%zu\n",
           model.x_count,
           (double)model.x_bytes / GD_PERF_MIB,
           model.axis_count);
    if (!gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "all_fwd",
                                          gd_reduce_perf_run_all_forward,
                                          false,
                                          (double)model.x_bytes + scalar_bytes,
                                          warmup,
                                          iters) ||
        !gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "all_bwd_direct",
                                          gd_reduce_perf_run_all_backward,
                                          false,
                                          (double)model.x_bytes + scalar_bytes,
                                          warmup,
                                          iters) ||
        !gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "all_fwd_bwd_autograd",
                                          gd_reduce_perf_run_all_autograd,
                                          false,
                                          (double)model.x_bytes * 2.0 + scalar_bytes * 2.0,
                                          warmup,
                                          iters) ||
        !gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "axis_fwd",
                                          gd_reduce_perf_run_axis_forward,
                                          false,
                                          (double)model.x_bytes + (double)model.axis_bytes,
                                          warmup,
                                          iters) ||
        !gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "axis_bwd_direct",
                                          gd_reduce_perf_run_axis_backward,
                                          false,
                                          (double)model.x_bytes + (double)model.axis_bytes,
                                          warmup,
                                          iters) ||
        !gd_reduce_perf_measure_and_print(&model,
                                          pcase,
                                          "axis_fwd_bwd_autograd",
                                          gd_reduce_perf_run_axis_autograd,
                                          false,
                                          (double)model.x_bytes * 2.0 + (double)model.axis_bytes * 2.0,
                                          warmup,
                                          iters)) {
        gd_reduce_perf_destroy(&model);
        return false;
    }
    if (spec->include_keepdims &&
        (!gd_reduce_perf_measure_and_print(&model,
                                           pcase,
                                           "axis_fwd_keepdims",
                                           gd_reduce_perf_run_axis_forward_keepdims,
                                           true,
                                           (double)model.x_bytes + (double)model.axis_keep_bytes,
                                           warmup,
                                           iters) ||
         !gd_reduce_perf_measure_and_print(&model,
                                           pcase,
                                           "axis_bwd_direct_keepdims",
                                           gd_reduce_perf_run_axis_backward_keepdims,
                                           true,
                                           (double)model.x_bytes + (double)model.axis_keep_bytes,
                                           warmup,
                                           iters) ||
         !gd_reduce_perf_measure_and_print(&model,
                                           pcase,
                                           "axis_fwd_bwd_autograd_keepdims",
                                           gd_reduce_perf_run_axis_autograd_keepdims,
                                           true,
                                           (double)model.x_bytes * 2.0 + (double)model.axis_keep_bytes * 2.0,
                                           warmup,
                                           iters))) {
        gd_reduce_perf_destroy(&model);
        return false;
    }
    gd_reduce_perf_destroy(&model);
    return true;
}

static int gd_reduce_perf_main(const gd_reduce_perf_spec *spec,
                               const gd_reduce_perf_case *cases,
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
    printf("[%s] public API perf: warmup=%d iters=%d profile=%s\n",
           spec->tag,
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    for (i = 0U; i < case_count; ++i) {
        if (!gd_perf_case_selected(cases[i].name, profile, cases[i].smoke, false)) {
            continue;
        }
        ran = true;
        if (!gd_reduce_perf_run_case(spec, &cases[i], warmup, iters)) {
            return 1;
        }
    }
    if (!ran) {
        fprintf(stderr,
                "[%s][FAIL] no case matched %s=%s\n",
                spec->tag,
                spec->profile_env,
                profile != NULL ? profile : "");
        return 2;
    }
    return 0;
}

#endif /* GD_OPS_SHARED_PERF_REDUCE_PERF_H */
