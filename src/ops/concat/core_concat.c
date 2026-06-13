#include "concat_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static bool gd_concat_autograd_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_concat_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL) {
        return false;
    }
    if (a != 0U && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static gd_status gd_concat_normalize_axis(gd_context *ctx,
                                          uint32_t rank,
                                          int32_t axis,
                                          uint32_t *out_axis)
{
    int32_t normalized;
    if (ctx == NULL || out_axis == NULL || rank == 0U || rank > GD_MAX_DIMS ||
        rank > (uint32_t)INT32_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    normalized = axis < 0 ? axis + (int32_t)rank : axis;
    if (normalized < 0 || normalized >= (int32_t)rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat axis out of range");
    }
    *out_axis = (uint32_t)normalized;
    return GD_OK;
}

static gd_status gd_concat_tensor_factors(gd_context *ctx,
                                          const gd_tensor *tensor,
                                          uint32_t axis,
                                          uint64_t *out_outer,
                                          uint64_t *out_inner,
                                          uint64_t *out_axis_len,
                                          uint64_t *out_count)
{
    uint32_t i;
    uint64_t outer = 1U;
    uint64_t inner = 1U;
    uint64_t count;
    uint64_t block;
    if (ctx == NULL || tensor == NULL || out_outer == NULL || out_inner == NULL ||
        out_axis_len == NULL || out_count == NULL || axis >= tensor->rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < axis; ++i) {
        if (tensor->shape[i] <= 0 ||
            !gd_concat_u64_mul(outer, (uint64_t)tensor->shape[i], &outer)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "concat outer dimension overflow");
        }
    }
    for (i = axis + 1U; i < tensor->rank; ++i) {
        if (tensor->shape[i] <= 0 ||
            !gd_concat_u64_mul(inner, (uint64_t)tensor->shape[i], &inner)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "concat inner dimension overflow");
        }
    }
    if (tensor->shape[axis] <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat axis dimension must be positive");
    }
    if (!gd_concat_u64_mul((uint64_t)tensor->shape[axis], inner, &block) ||
        !gd_concat_u64_mul(outer, block, &count)) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "concat element count overflow");
    }
    *out_outer = outer;
    *out_inner = inner;
    *out_axis_len = (uint64_t)tensor->shape[axis];
    *out_count = count;
    return GD_OK;
}

static gd_status gd_concat_validate_inputs(gd_context *ctx,
                                           const gd_tensor *const *inputs,
                                           uint32_t n_inputs,
                                           int32_t axis,
                                           uint32_t *out_axis,
                                           int64_t out_shape[GD_MAX_DIMS],
                                           bool *out_needs_grad)
{
    gd_status st;
    const gd_tensor *first;
    uint32_t normalized_axis;
    uint32_t i;
    uint32_t d;
    int64_t axis_sum;
    bool needs_grad = false;
    if (ctx == NULL || inputs == NULL || out_axis == NULL || out_shape == NULL ||
        n_inputs == 0U || n_inputs > GD_CONCAT_MAX_INPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    first = inputs[0];
    if (first == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, first);
    if (st != GD_OK) {
        return st;
    }
    if (first->rank == 0U) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat does not support scalar tensors");
    }
    if (!gd_tensor_is_contiguous(first)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "concat requires contiguous inputs");
    }
    st = gd_concat_normalize_axis(ctx, first->rank, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    axis_sum = 0;
    for (d = 0U; d < first->rank; ++d) {
        out_shape[d] = first->shape[d];
    }
    for (i = 0U; i < n_inputs; ++i) {
        const gd_tensor *x = inputs[i];
        if (x == NULL) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat input is NULL");
        }
        st = gd_tensor_validate(ctx, x);
        if (st != GD_OK) {
            return st;
        }
        if (x->dtype != first->dtype || x->rank != first->rank) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat dtype/rank mismatch");
        }
        if (!gd_tensor_is_contiguous(x)) {
            return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "concat requires contiguous inputs");
        }
        for (d = 0U; d < first->rank; ++d) {
            if (d != normalized_axis && x->shape[d] != first->shape[d]) {
                return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat non-axis shape mismatch");
            }
        }
        if (x->shape[normalized_axis] <= 0 ||
            axis_sum > INT64_MAX - x->shape[normalized_axis]) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "concat axis dimension overflow");
        }
        axis_sum += x->shape[normalized_axis];
        needs_grad = needs_grad || x->requires_grad;
    }
    out_shape[normalized_axis] = axis_sum;
    *out_axis = normalized_axis;
    if (out_needs_grad != NULL) {
        *out_needs_grad = needs_grad;
    }
    return GD_OK;
}

static gd_status gd_concat_copy_to_full(gd_context *ctx,
                                        const gd_tensor *src,
                                        const gd_tensor *dst,
                                        uint32_t axis,
                                        int64_t axis_offset,
                                        int64_t full_axis)
{
    gd_status st;
    gd_backend_tensor_view sv;
    gd_backend_tensor_view dv;
    gd_backend_concat_args args;
    uint64_t outer;
    uint64_t inner;
    uint64_t slice_axis;
    uint64_t count;
    if (ctx == NULL || src == NULL || dst == NULL || axis_offset < 0 || full_axis <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_concat_tensor_factors(ctx, src, axis, &outer, &inner, &slice_axis, &count);
    if (st != GD_OK) {
        return st;
    }
    if ((uint64_t)full_axis < slice_axis || (uint64_t)axis_offset > (uint64_t)full_axis - slice_axis) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat invalid axis offset");
    }
    if (!gd_op_tensor_view_from_tensor(src, &sv) || !gd_op_tensor_view_from_tensor(dst, &dv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat invalid tensor view");
    }
    memset(&args, 0, sizeof(args));
    args.count = count;
    args.inner = inner;
    args.slice_axis = slice_axis;
    args.full_axis = (uint64_t)full_axis;
    args.axis_offset = (uint64_t)axis_offset;
    st = gd_backend_concat_to_full(gd_context_backend(ctx), &sv, &dv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend concat failed");
    }
    (void)outer;
    return GD_OK;
}

gd_status gd_concat_backward_input_impl(gd_context *ctx,
                                        const gd_tensor *grad_out,
                                        const gd_tensor *like_input,
                                        uint32_t axis,
                                        int64_t axis_offset,
                                        int64_t full_axis,
                                        gd_tensor *grad_input)
{
    gd_status st;
    gd_tensor dx;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_backend_concat_args args;
    uint32_t d;
    uint64_t outer;
    uint64_t inner;
    uint64_t slice_axis;
    uint64_t count;
    if (grad_input != NULL) {
        memset(grad_input, 0, sizeof(*grad_input));
    }
    if (ctx == NULL || grad_out == NULL || like_input == NULL || grad_input == NULL ||
        axis >= like_input->rank || axis_offset < 0 || full_axis <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, like_input);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != like_input->dtype || grad_out->rank != like_input->rank ||
        grad_out->shape[axis] != full_axis || !gd_tensor_is_contiguous(grad_out) ||
        !gd_tensor_is_contiguous(like_input)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward shape mismatch");
    }
    for (d = 0U; d < like_input->rank; ++d) {
        if (d != axis && grad_out->shape[d] != like_input->shape[d]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward non-axis shape mismatch");
        }
    }
    st = gd_concat_tensor_factors(ctx, like_input, axis, &outer, &inner, &slice_axis, &count);
    if (st != GD_OK) {
        return st;
    }
    if ((uint64_t)full_axis < slice_axis || (uint64_t)axis_offset > (uint64_t)full_axis - slice_axis) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward invalid axis offset");
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         like_input->dtype,
                         gd_shape_make(like_input->rank, like_input->shape),
                         256U,
                         &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.requires_grad = false;
    dx.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(grad_out, &gv) || !gd_op_tensor_view_from_tensor(&dx, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward invalid tensor view");
    }
    memset(&args, 0, sizeof(args));
    args.count = count;
    args.inner = inner;
    args.slice_axis = slice_axis;
    args.full_axis = (uint64_t)full_axis;
    args.axis_offset = (uint64_t)axis_offset;
    st = gd_backend_concat_from_full(gd_context_backend(ctx), &gv, &dxv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend concat backward failed");
    }
    *grad_input = dx;
    (void)outer;
    return GD_OK;
}

gd_status gd_concat(gd_context *ctx,
                    const gd_tensor *const *inputs,
                    uint32_t n_inputs,
                    int32_t axis,
                    gd_tensor *out)
{
    gd_status st;
    uint32_t normalized_axis;
    int64_t out_shape[GD_MAX_DIMS];
    gd_tensor y;
    bool needs_grad;
    uint32_t i;
    int64_t axis_offset = 0;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || inputs == NULL || out == NULL || n_inputs == 0U ||
        n_inputs > GD_CONCAT_MAX_INPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_concat_validate_inputs(ctx, inputs, n_inputs, axis, &normalized_axis, out_shape, &needs_grad);
    if (st != GD_OK) {
        return st;
    }
    if (needs_grad && !gd_concat_autograd_dtype(inputs[0]->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "concat autograd supports f16/f32 tensors only");
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         inputs[0]->dtype,
                         gd_shape_make(inputs[0]->rank, out_shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    for (i = 0U; i < n_inputs; ++i) {
        st = gd_concat_copy_to_full(ctx,
                                    inputs[i],
                                    &y,
                                    normalized_axis,
                                    axis_offset,
                                    out_shape[normalized_axis]);
        if (st != GD_OK) {
            return st;
        }
        axis_offset += inputs[i]->shape[normalized_axis];
    }
    if (needs_grad) {
        gd_tensor *outputs[1] = {&y};
        gd_concat_attrs attrs;
        memset(&attrs, 0, sizeof(attrs));
        attrs.axis = (int32_t)normalized_axis;
        attrs.n_inputs = n_inputs;
        st = gd_autograd_record(ctx,
                                GD_OP_CONCAT,
                                inputs,
                                (uint16_t)n_inputs,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_concat_backward(gd_context *ctx,
                             const gd_tensor *grad_out,
                             const gd_tensor *const *inputs,
                             uint32_t n_inputs,
                             int32_t axis,
                             gd_tensor *grad_inputs)
{
    gd_status st;
    uint32_t normalized_axis;
    int64_t out_shape[GD_MAX_DIMS];
    uint32_t i;
    int64_t axis_offset = 0;
    if (ctx == NULL || grad_out == NULL || inputs == NULL ||
        n_inputs == 0U || n_inputs > GD_CONCAT_MAX_INPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_inputs != NULL) {
        for (i = 0U; i < n_inputs; ++i) {
            memset(&grad_inputs[i], 0, sizeof(grad_inputs[i]));
        }
    }
    st = gd_concat_validate_inputs(ctx, inputs, n_inputs, axis, &normalized_axis, out_shape, NULL);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != inputs[0]->dtype || grad_out->rank != inputs[0]->rank ||
        !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward grad_out mismatch");
    }
    for (i = 0U; i < grad_out->rank; ++i) {
        if (grad_out->shape[i] != out_shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "concat backward grad_out shape mismatch");
        }
    }
    if (grad_inputs == NULL) {
        return GD_OK;
    }
    for (i = 0U; i < n_inputs; ++i) {
        st = gd_concat_backward_input_impl(ctx,
                                           grad_out,
                                           inputs[i],
                                           normalized_axis,
                                           axis_offset,
                                           out_shape[normalized_axis],
                                           &grad_inputs[i]);
        if (st != GD_OK) {
            return st;
        }
        axis_offset += inputs[i]->shape[normalized_axis];
    }
    return GD_OK;
}
