#ifndef GD_OPS_SHARED_PERF_PARTITION_PERF_H
#define GD_OPS_SHARED_PERF_PARTITION_PERF_H

#include "perf_common.h"

#define GD_PARTITION_PERF_MAX_PARTS 4U

typedef enum gd_partition_perf_kind {
    GD_PARTITION_PERF_CONCAT = 1,
    GD_PARTITION_PERF_SPLIT = 2,
} gd_partition_perf_kind;

typedef struct gd_partition_perf_case {
    const char *name;
    gd_dtype dtype;
    uint32_t rank;
    int32_t axis;
    uint32_t n_parts;
    int64_t shape[GD_MAX_DIMS];
    int64_t part_sizes[GD_PARTITION_PERF_MAX_PARTS];
    bool smoke;
} gd_partition_perf_case;

typedef struct gd_partition_perf_spec {
    const char *tag;
    const char *profile_env;
    const char *warmup_env;
    const char *iters_env;
    const char *note;
    const char *bandwidth_label;
    gd_partition_perf_kind kind;
} gd_partition_perf_spec;

typedef struct gd_partition_perf_model {
    const gd_partition_perf_spec *spec;
    gd_context *ctx;
    gd_tensor x;
    gd_tensor inputs[GD_PARTITION_PERF_MAX_PARTS];
    const gd_tensor *input_ptrs[GD_PARTITION_PERF_MAX_PARTS];
    gd_tensor grad_out;
    gd_tensor grad_outputs[GD_PARTITION_PERF_MAX_PARTS];
    const gd_tensor *grad_output_ptrs[GD_PARTITION_PERF_MAX_PARTS];
    int64_t part_sizes[GD_PARTITION_PERF_MAX_PARTS];
    int64_t materialized_shape[GD_MAX_DIMS];
    uint32_t n_parts;
    uint32_t rank;
    int32_t axis;
    size_t logical_bytes;
    bool floating;
} gd_partition_perf_model;

static gd_memory_config gd_partition_perf_config(const gd_partition_perf_spec *spec,
                                                 size_t logical_bytes,
                                                 bool floating)
{
    size_t params_bytes = logical_bytes * (floating ? 2U : 1U);
    size_t scratch_scale;
    size_t scratch_bytes;
    if (spec->kind == GD_PARTITION_PERF_CONCAT) {
        scratch_scale = floating ? 8U : 2U;
    } else {
        scratch_scale = floating ? 10U : 3U;
    }
    scratch_bytes = logical_bytes * scratch_scale + 64U * 1024U * 1024U;
    return gd_perf_memory_config(gd_perf_align_up(params_bytes + 16U * 1024U * 1024U, 4096U),
                                 4U * 1024U * 1024U,
                                 gd_perf_align_up(scratch_bytes, 4096U),
                                 1024U * 1024U,
                                 3U,
                                 2U,
                                 256U);
}

static bool gd_partition_perf_prepare_concat(const gd_partition_perf_case *pcase,
                                             gd_partition_perf_model *model,
                                             size_t elem_size,
                                             uint32_t normalized_axis)
{
    size_t output_elems;
    uint32_t i;
    uint32_t d;
    for (d = 0U; d < pcase->rank; ++d) {
        model->materialized_shape[d] = pcase->shape[d];
    }
    model->materialized_shape[normalized_axis] = 0;
    for (i = 0U; i < pcase->n_parts; ++i) {
        if (pcase->part_sizes[i] <= 0 ||
            model->materialized_shape[normalized_axis] > INT64_MAX - pcase->part_sizes[i]) {
            return false;
        }
        model->materialized_shape[normalized_axis] += pcase->part_sizes[i];
        model->part_sizes[i] = pcase->part_sizes[i];
    }
    return gd_perf_shape_count(pcase->rank, model->materialized_shape, false, &output_elems) &&
           gd_perf_checked_mul_size(output_elems, elem_size, &model->logical_bytes);
}

static bool gd_partition_perf_prepare_split(const gd_partition_perf_case *pcase,
                                            gd_partition_perf_model *model,
                                            size_t elem_size,
                                            uint32_t normalized_axis)
{
    size_t input_elems;
    int64_t axis_sum = 0;
    uint32_t i;
    uint32_t d;
    for (d = 0U; d < pcase->rank; ++d) {
        model->materialized_shape[d] = pcase->shape[d];
    }
    for (i = 0U; i < pcase->n_parts; ++i) {
        if (pcase->part_sizes[i] <= 0 || axis_sum > INT64_MAX - pcase->part_sizes[i]) {
            return false;
        }
        axis_sum += pcase->part_sizes[i];
        model->part_sizes[i] = pcase->part_sizes[i];
    }
    return axis_sum == pcase->shape[normalized_axis] &&
           gd_perf_shape_count(pcase->rank, pcase->shape, false, &input_elems) &&
           gd_perf_checked_mul_size(input_elems, elem_size, &model->logical_bytes);
}

static bool gd_partition_perf_populate_concat(const gd_partition_perf_case *pcase,
                                              gd_partition_perf_model *model,
                                              uint32_t normalized_axis)
{
    uint32_t i;
    uint32_t d;
    for (i = 0U; i < pcase->n_parts; ++i) {
        int64_t input_shape[GD_MAX_DIMS];
        for (d = 0U; d < pcase->rank; ++d) {
            input_shape[d] = pcase->shape[d];
        }
        input_shape[normalized_axis] = pcase->part_sizes[i];
        if (model->floating) {
            GD_PERF_REQUIRE_OK(model->spec->tag,
                               model->ctx,
                               gd_tensor_rand_uniform(model->ctx,
                                                      GD_ARENA_PARAMS,
                                                      pcase->dtype,
                                                      gd_shape_make(pcase->rank, input_shape),
                                                      256U,
                                                      UINT64_C(0xc0aca700) + (uint64_t)i,
                                                      -0.25f,
                                                      0.25f,
                                                      &model->inputs[i]));
        } else {
            GD_PERF_REQUIRE_OK(model->spec->tag,
                               model->ctx,
                               gd_tensor_zeros(model->ctx,
                                               GD_ARENA_PARAMS,
                                               pcase->dtype,
                                               gd_shape_make(pcase->rank, input_shape),
                                               256U,
                                               &model->inputs[i]));
        }
        model->input_ptrs[i] = &model->inputs[i];
    }
    if (model->floating) {
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  pcase->dtype,
                                                  gd_shape_make(pcase->rank, model->materialized_shape),
                                                  256U,
                                                  UINT64_C(0xc0aca7ff),
                                                  -0.25f,
                                                  0.25f,
                                                  &model->grad_out));
    }
    return true;
}

static bool gd_partition_perf_populate_split(const gd_partition_perf_case *pcase,
                                             gd_partition_perf_model *model,
                                             uint32_t normalized_axis)
{
    uint32_t i;
    uint32_t d;
    if (model->floating) {
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_tensor_rand_uniform(model->ctx,
                                                  GD_ARENA_PARAMS,
                                                  pcase->dtype,
                                                  gd_shape_make(pcase->rank, pcase->shape),
                                                  256U,
                                                  UINT64_C(0x51717000),
                                                  -0.25f,
                                                  0.25f,
                                                  &model->x));
    } else {
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_tensor_zeros(model->ctx,
                                           GD_ARENA_PARAMS,
                                           pcase->dtype,
                                           gd_shape_make(pcase->rank, pcase->shape),
                                           256U,
                                           &model->x));
    }
    if (model->floating) {
        for (i = 0U; i < pcase->n_parts; ++i) {
            int64_t grad_shape[GD_MAX_DIMS];
            for (d = 0U; d < pcase->rank; ++d) {
                grad_shape[d] = pcase->shape[d];
            }
            grad_shape[normalized_axis] = pcase->part_sizes[i];
            GD_PERF_REQUIRE_OK(model->spec->tag,
                               model->ctx,
                               gd_tensor_rand_uniform(model->ctx,
                                                      GD_ARENA_PARAMS,
                                                      pcase->dtype,
                                                      gd_shape_make(pcase->rank, grad_shape),
                                                      256U,
                                                      UINT64_C(0x51717080) + (uint64_t)i,
                                                      -0.25f,
                                                      0.25f,
                                                      &model->grad_outputs[i]));
            model->grad_output_ptrs[i] = &model->grad_outputs[i];
        }
    }
    return true;
}

static bool gd_partition_perf_setup(const gd_partition_perf_spec *spec,
                                    const gd_partition_perf_case *pcase,
                                    gd_partition_perf_model *model)
{
    gd_memory_config cfg;
    size_t elem_size;
    int32_t normalized_axis;
    gd_status st;
    if (spec == NULL || pcase == NULL || model == NULL || pcase->rank == 0U || pcase->rank > GD_MAX_DIMS ||
        pcase->n_parts == 0U || pcase->n_parts > GD_PARTITION_PERF_MAX_PARTS) {
        return false;
    }
    memset(model, 0, sizeof(*model));
    model->spec = spec;
    model->n_parts = pcase->n_parts;
    model->rank = pcase->rank;
    model->axis = pcase->axis;
    model->floating = gd_perf_is_floating_dtype(pcase->dtype);
    elem_size = gd_dtype_size(pcase->dtype);
    normalized_axis = pcase->axis < 0 ? (int32_t)pcase->rank + pcase->axis : pcase->axis;
    if (elem_size == 0U || normalized_axis < 0 || normalized_axis >= (int32_t)pcase->rank) {
        return false;
    }
    if (spec->kind == GD_PARTITION_PERF_CONCAT) {
        if (!gd_partition_perf_prepare_concat(pcase, model, elem_size, (uint32_t)normalized_axis)) {
            return false;
        }
    } else if (spec->kind == GD_PARTITION_PERF_SPLIT) {
        if (!gd_partition_perf_prepare_split(pcase, model, elem_size, (uint32_t)normalized_axis)) {
            return false;
        }
    } else {
        return false;
    }
    cfg = gd_partition_perf_config(spec, model->logical_bytes, model->floating);
    st = gd_context_create(&cfg, &model->ctx);
    if (st == GD_ERR_UNSUPPORTED) {
        printf("[%s] skipped case=%s: no supported GPU backend\n", spec->tag, pcase->name);
        return false;
    }
    if (!gd_perf_status_ok(spec->tag, model->ctx, st, "gd_context_create")) {
        return false;
    }
    if (spec->kind == GD_PARTITION_PERF_CONCAT) {
        if (!gd_partition_perf_populate_concat(pcase, model, (uint32_t)normalized_axis)) {
            return false;
        }
    } else if (!gd_partition_perf_populate_split(pcase, model, (uint32_t)normalized_axis)) {
        return false;
    }
    GD_PERF_REQUIRE_OK(spec->tag, model->ctx, gd_context_seal_params(model->ctx));
    return true;
}

static void gd_partition_perf_destroy(gd_partition_perf_model *model)
{
    if (model != NULL) {
        gd_context_destroy(model->ctx);
        memset(model, 0, sizeof(*model));
    }
}

static bool gd_partition_perf_run_forward(void *user)
{
    gd_partition_perf_model *model = (gd_partition_perf_model *)user;
    if (model->spec->kind == GD_PARTITION_PERF_CONCAT) {
        gd_tensor out;
        uint32_t i;
        for (i = 0U; i < model->n_parts; ++i) {
            model->inputs[i].requires_grad = false;
        }
        GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_concat(model->ctx, model->input_ptrs, model->n_parts, model->axis, &out));
    } else {
        gd_tensor outputs[GD_PARTITION_PERF_MAX_PARTS];
        model->x.requires_grad = false;
        GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_EVAL, gd_batch_empty()));
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_split(model->ctx, &model->x, model->part_sizes, model->n_parts, model->axis, outputs));
    }
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_partition_perf_run_backward(void *user)
{
    gd_partition_perf_model *model = (gd_partition_perf_model *)user;
    if (!model->floating) {
        return true;
    }
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    if (model->spec->kind == GD_PARTITION_PERF_CONCAT) {
        gd_tensor grad_inputs[GD_PARTITION_PERF_MAX_PARTS];
        uint32_t i;
        for (i = 0U; i < model->n_parts; ++i) {
            model->inputs[i].requires_grad = false;
        }
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_concat_backward(model->ctx,
                                              &model->grad_out,
                                              model->input_ptrs,
                                              model->n_parts,
                                              model->axis,
                                              grad_inputs));
    } else {
        gd_tensor dx;
        model->x.requires_grad = false;
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_split_backward(model->ctx,
                                             &model->x,
                                             model->grad_output_ptrs,
                                             model->part_sizes,
                                             model->n_parts,
                                             model->axis,
                                             &dx));
    }
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_partition_perf_run_autograd(void *user)
{
    gd_partition_perf_model *model = (gd_partition_perf_model *)user;
    if (!model->floating) {
        return true;
    }
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_begin_step(model->ctx, GD_SCOPE_TRAIN, gd_batch_empty()));
    if (model->spec->kind == GD_PARTITION_PERF_CONCAT) {
        gd_tensor out;
        uint32_t i;
        for (i = 0U; i < model->n_parts; ++i) {
            model->inputs[i].requires_grad = true;
        }
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_concat(model->ctx, model->input_ptrs, model->n_parts, model->axis, &out));
        GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_backward(model->ctx, &out, &model->grad_out));
    } else {
        gd_tensor outputs[GD_PARTITION_PERF_MAX_PARTS];
        const gd_tensor *output_ptrs[GD_PARTITION_PERF_MAX_PARTS];
        uint32_t i;
        model->x.requires_grad = true;
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_split(model->ctx, &model->x, model->part_sizes, model->n_parts, model->axis, outputs));
        for (i = 0U; i < model->n_parts; ++i) {
            output_ptrs[i] = &outputs[i];
        }
        GD_PERF_REQUIRE_OK(model->spec->tag,
                           model->ctx,
                           gd_backward_many(model->ctx, model->n_parts, output_ptrs, model->grad_output_ptrs));
    }
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_end_step(model->ctx));
    GD_PERF_REQUIRE_OK(model->spec->tag, model->ctx, gd_synchronize(model->ctx));
    return true;
}

static bool gd_partition_perf_measure(gd_partition_perf_model *model,
                                      const char *label,
                                      gd_perf_run_fn run,
                                      int warmup,
                                      int iters,
                                      double logical_bytes)
{
    double seconds;
    double avg_ms;
    double gib_s;
    if (!gd_perf_measure_seconds(model, run, warmup, iters, &seconds)) {
        return false;
    }
    avg_ms = seconds * 1.0e3;
    gib_s = (logical_bytes / GD_PERF_GIB) / seconds;
    printf("[%s][%s] avg_ms=%.4f %s_GiB/s=%.2f\n",
           model->spec->tag,
           label,
           avg_ms,
           model->spec->bandwidth_label != NULL ? model->spec->bandwidth_label : "logical",
           gib_s);
    return true;
}

static void gd_partition_perf_print_case(const gd_partition_perf_case *pcase,
                                         const gd_partition_perf_model *model)
{
    uint32_t i;
    printf("[%s] case=%s dtype=%s ", model->spec->tag, pcase->name, gd_dtype_name(pcase->dtype));
    if (model->spec->kind == GD_PARTITION_PERF_CONCAT) {
        printf("rank=%u axis=%d inputs=", pcase->rank, pcase->axis);
        for (i = 0U; i < pcase->n_parts; ++i) {
            printf("%s%lld", i == 0U ? "" : ",", (long long)pcase->part_sizes[i]);
        }
        printf(" output_bytes=%zu floating=%d\n", model->logical_bytes, model->floating ? 1 : 0);
    } else {
        printf("input=");
        gd_perf_print_shape(pcase->rank, pcase->shape, ",");
        printf(" axis=%d sizes=", pcase->axis);
        gd_perf_print_shape(pcase->n_parts, pcase->part_sizes, ",");
        printf(" logical_bytes=%zu fwd_materialized_bytes=%zu floating=%d\n",
               model->logical_bytes,
               model->logical_bytes * 2U,
               model->floating ? 1 : 0);
    }
}

static bool gd_partition_perf_run_case(const gd_partition_perf_spec *spec,
                                       const gd_partition_perf_case *pcase,
                                       int warmup,
                                       int iters)
{
    gd_partition_perf_model model;
    double fwd_bytes;
    double bwd_bytes;
    double autograd_bytes;
    memset(&model, 0, sizeof(model));
    if (!gd_partition_perf_setup(spec, pcase, &model)) {
        gd_partition_perf_destroy(&model);
        return false;
    }
    gd_partition_perf_print_case(pcase, &model);
    fwd_bytes = 2.0 * (double)model.logical_bytes;
    bwd_bytes = 2.0 * (double)model.logical_bytes;
    autograd_bytes = 4.0 * (double)model.logical_bytes;
    if (!gd_partition_perf_measure(&model, "fwd", gd_partition_perf_run_forward, warmup, iters, fwd_bytes)) {
        gd_partition_perf_destroy(&model);
        return false;
    }
    if (model.floating &&
        (!gd_partition_perf_measure(&model, "bwd", gd_partition_perf_run_backward, warmup, iters, bwd_bytes) ||
         !gd_partition_perf_measure(&model,
                                    "autograd",
                                    gd_partition_perf_run_autograd,
                                    warmup,
                                    iters,
                                    autograd_bytes))) {
        gd_partition_perf_destroy(&model);
        return false;
    }
    gd_partition_perf_destroy(&model);
    return true;
}

static int gd_partition_perf_main(const gd_partition_perf_spec *spec,
                                  const gd_partition_perf_case *cases,
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
    printf("[%s] warmup=%d iters=%d profile=%s",
           spec->tag,
           warmup,
           iters,
           profile != NULL && profile[0] != '\0' ? profile : "smoke");
    if (spec->note != NULL && spec->note[0] != '\0') {
        printf(" note=%s", spec->note);
    }
    printf("\n");
    for (i = 0U; i < case_count; ++i) {
        if (!gd_perf_case_selected(cases[i].name, profile, cases[i].smoke, false)) {
            continue;
        }
        ran = true;
        if (!gd_partition_perf_run_case(spec, &cases[i], warmup, iters)) {
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

#endif /* GD_OPS_SHARED_PERF_PARTITION_PERF_H */
