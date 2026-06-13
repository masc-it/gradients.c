#include "split_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static bool gd_split_autograd_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_split_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
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

static gd_status gd_split_normalize_axis(gd_context *ctx,
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
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split axis out of range");
    }
    *out_axis = (uint32_t)normalized;
    return GD_OK;
}

static gd_status gd_split_tensor_factors(gd_context *ctx,
                                         const gd_tensor *tensor,
                                         uint32_t axis,
                                         uint64_t *out_inner,
                                         uint64_t *out_axis_len,
                                         uint64_t *out_count)
{
    uint32_t i;
    uint64_t outer = 1U;
    uint64_t inner = 1U;
    uint64_t block;
    uint64_t count;
    if (ctx == NULL || tensor == NULL || out_inner == NULL || out_axis_len == NULL ||
        out_count == NULL || axis >= tensor->rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < axis; ++i) {
        if (tensor->shape[i] <= 0 ||
            !gd_split_u64_mul(outer, (uint64_t)tensor->shape[i], &outer)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "split outer dimension overflow");
        }
    }
    for (i = axis + 1U; i < tensor->rank; ++i) {
        if (tensor->shape[i] <= 0 ||
            !gd_split_u64_mul(inner, (uint64_t)tensor->shape[i], &inner)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "split inner dimension overflow");
        }
    }
    if (tensor->shape[axis] <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split axis dimension must be positive");
    }
    if (!gd_split_u64_mul((uint64_t)tensor->shape[axis], inner, &block) ||
        !gd_split_u64_mul(outer, block, &count)) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "split element count overflow");
    }
    *out_inner = inner;
    *out_axis_len = (uint64_t)tensor->shape[axis];
    *out_count = count;
    return GD_OK;
}

static gd_status gd_split_validate_sizes(gd_context *ctx,
                                         const gd_tensor *x,
                                         const int64_t *sizes,
                                         uint32_t n_outputs,
                                         int32_t axis,
                                         uint32_t *out_axis,
                                         bool *out_needs_grad)
{
    gd_status st;
    uint32_t normalized_axis;
    uint32_t i;
    int64_t axis_sum = 0;
    if (ctx == NULL || x == NULL || sizes == NULL || out_axis == NULL ||
        n_outputs == 0U || n_outputs > GD_SPLIT_MAX_OUTPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (x->rank == 0U) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split does not support scalar tensors");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "split requires contiguous input");
    }
    st = gd_split_normalize_axis(ctx, x->rank, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < n_outputs; ++i) {
        if (sizes[i] <= 0) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split sizes must be positive");
        }
        if (axis_sum > INT64_MAX - sizes[i]) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "split size sum overflow");
        }
        axis_sum += sizes[i];
    }
    if (axis_sum != x->shape[normalized_axis]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split sizes must sum to the input axis dimension");
    }
    *out_axis = normalized_axis;
    if (out_needs_grad != NULL) {
        *out_needs_grad = x->requires_grad;
    }
    return GD_OK;
}

static gd_status gd_split_validate_slice_shape(gd_context *ctx,
                                               const gd_tensor *full,
                                               const gd_tensor *slice,
                                               uint32_t axis,
                                               int64_t axis_offset,
                                               int64_t full_axis)
{
    gd_status st;
    uint32_t d;
    if (ctx == NULL || full == NULL || slice == NULL || axis >= full->rank ||
        axis_offset < 0 || full_axis <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, full);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, slice);
    if (st != GD_OK) {
        return st;
    }
    if (full->dtype != slice->dtype || full->rank != slice->rank || full->shape[axis] != full_axis ||
        !gd_tensor_is_contiguous(full) || !gd_tensor_is_contiguous(slice)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split slice shape mismatch");
    }
    for (d = 0U; d < full->rank; ++d) {
        if (d != axis && full->shape[d] != slice->shape[d]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split slice non-axis shape mismatch");
        }
    }
    if (slice->shape[axis] <= 0 || (uint64_t)full_axis < (uint64_t)slice->shape[axis] ||
        (uint64_t)axis_offset > (uint64_t)full_axis - (uint64_t)slice->shape[axis]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split slice axis range mismatch");
    }
    return GD_OK;
}

gd_status gd_split_forward_output_impl(gd_context *ctx,
                                       const gd_tensor *x,
                                       uint32_t axis,
                                       int64_t axis_offset,
                                       gd_tensor *out)
{
    gd_status st;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_split_args args;
    uint64_t inner;
    uint64_t slice_axis;
    uint64_t count;
    if (ctx == NULL || x == NULL || out == NULL || axis >= x->rank || axis_offset < 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_split_validate_slice_shape(ctx, x, out, axis, axis_offset, x->shape[axis]);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split_tensor_factors(ctx, out, axis, &inner, &slice_axis, &count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(out, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split invalid tensor view");
    }
    memset(&args, 0, sizeof(args));
    args.count = count;
    args.inner = inner;
    args.slice_axis = slice_axis;
    args.full_axis = (uint64_t)x->shape[axis];
    args.axis_offset = (uint64_t)axis_offset;
    st = gd_backend_split_from_full(gd_context_backend(ctx), &xv, &yv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend split failed");
    }
    return GD_OK;
}

gd_status gd_split_backward_output_to_full_impl(gd_context *ctx,
                                                const gd_tensor *grad_output,
                                                const gd_tensor *grad_x,
                                                uint32_t axis,
                                                int64_t axis_offset,
                                                int64_t full_axis)
{
    gd_status st;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_backend_split_args args;
    uint64_t inner;
    uint64_t slice_axis;
    uint64_t count;
    if (ctx == NULL || grad_output == NULL || grad_x == NULL || axis >= grad_x->rank ||
        axis_offset < 0 || full_axis <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_split_validate_slice_shape(ctx, grad_x, grad_output, axis, axis_offset, full_axis);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split_tensor_factors(ctx, grad_output, axis, &inner, &slice_axis, &count);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(grad_output, &gv) || !gd_op_tensor_view_from_tensor(grad_x, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split backward invalid tensor view");
    }
    memset(&args, 0, sizeof(args));
    args.count = count;
    args.inner = inner;
    args.slice_axis = slice_axis;
    args.full_axis = (uint64_t)full_axis;
    args.axis_offset = (uint64_t)axis_offset;
    st = gd_backend_split_to_full(gd_context_backend(ctx), &gv, &dxv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend split backward failed");
    }
    return GD_OK;
}

static gd_status gd_split_validate_grad_outputs(gd_context *ctx,
                                                const gd_tensor *x,
                                                const gd_tensor *const *grad_outputs,
                                                const int64_t *sizes,
                                                uint32_t n_outputs,
                                                uint32_t axis)
{
    gd_status st;
    uint32_t i;
    uint32_t d;
    if (ctx == NULL || x == NULL || grad_outputs == NULL || sizes == NULL ||
        n_outputs == 0U || n_outputs > GD_SPLIT_MAX_OUTPUTS || axis >= x->rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < n_outputs; ++i) {
        const gd_tensor *g = grad_outputs[i];
        if (g == NULL) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split grad output is NULL");
        }
        st = gd_tensor_validate(ctx, g);
        if (st != GD_OK) {
            return st;
        }
        if (g->dtype != x->dtype || g->rank != x->rank || !gd_tensor_is_contiguous(g) ||
            g->shape[axis] != sizes[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "split backward grad output mismatch");
        }
        for (d = 0U; d < x->rank; ++d) {
            if (d != axis && g->shape[d] != x->shape[d]) {
                return gd_context_set_error(ctx,
                                            GD_ERR_INVALID_ARGUMENT,
                                            "split backward grad output non-axis shape mismatch");
            }
        }
    }
    return GD_OK;
}

gd_status gd_split(gd_context *ctx,
                   const gd_tensor *x,
                   const int64_t *sizes,
                   uint32_t n_outputs,
                   int32_t axis,
                   gd_tensor *outputs)
{
    gd_status st;
    uint32_t normalized_axis;
    bool needs_grad;
    uint32_t i;
    int64_t axis_offset = 0;
    if (outputs != NULL && n_outputs <= GD_SPLIT_MAX_OUTPUTS) {
        for (i = 0U; i < n_outputs; ++i) {
            memset(&outputs[i], 0, sizeof(outputs[i]));
        }
    }
    if (ctx == NULL || x == NULL || sizes == NULL || outputs == NULL ||
        n_outputs == 0U || n_outputs > GD_SPLIT_MAX_OUTPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_split_validate_sizes(ctx, x, sizes, n_outputs, axis, &normalized_axis, &needs_grad);
    if (st != GD_OK) {
        return st;
    }
    if (needs_grad && !gd_split_autograd_dtype(x->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "split autograd supports f16/f32 tensors only");
    }
    for (i = 0U; i < n_outputs; ++i) {
        int64_t out_shape[GD_MAX_DIMS];
        uint32_t d;
        gd_tensor y;
        for (d = 0U; d < x->rank; ++d) {
            out_shape[d] = x->shape[d];
        }
        out_shape[normalized_axis] = sizes[i];
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             x->dtype,
                             gd_shape_make(x->rank, out_shape),
                             256U,
                             &y);
        if (st != GD_OK) {
            return st;
        }
        y.requires_grad = false;
        y.is_leaf = false;
        st = gd_split_forward_output_impl(ctx, x, normalized_axis, axis_offset, &y);
        if (st != GD_OK) {
            return st;
        }
        outputs[i] = y;
        axis_offset += sizes[i];
    }
    if (needs_grad) {
        gd_tensor *output_ptrs[GD_SPLIT_MAX_OUTPUTS];
        gd_split_attrs attrs;
        for (i = 0U; i < n_outputs; ++i) {
            output_ptrs[i] = &outputs[i];
        }
        memset(&attrs, 0, sizeof(attrs));
        attrs.axis = (int32_t)normalized_axis;
        attrs.n_outputs = n_outputs;
        st = gd_autograd_record(ctx,
                                GD_OP_SPLIT,
                                &x,
                                1U,
                                output_ptrs,
                                (uint16_t)n_outputs,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

gd_status gd_split_backward(gd_context *ctx,
                            const gd_tensor *x,
                            const gd_tensor *const *grad_outputs,
                            const int64_t *sizes,
                            uint32_t n_outputs,
                            int32_t axis,
                            gd_tensor *grad_x)
{
    gd_status st;
    uint32_t normalized_axis;
    uint32_t i;
    int64_t axis_offset = 0;
    gd_tensor dx;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_outputs == NULL || sizes == NULL ||
        n_outputs == 0U || n_outputs > GD_SPLIT_MAX_OUTPUTS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_split_validate_sizes(ctx, x, sizes, n_outputs, axis, &normalized_axis, NULL);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split_validate_grad_outputs(ctx, x, grad_outputs, sizes, n_outputs, normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL) {
        return GD_OK;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         x->dtype,
                         gd_shape_make(x->rank, x->shape),
                         256U,
                         &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.requires_grad = false;
    dx.is_leaf = false;
    for (i = 0U; i < n_outputs; ++i) {
        st = gd_split_backward_output_to_full_impl(ctx,
                                                   grad_outputs[i],
                                                   &dx,
                                                   normalized_axis,
                                                   axis_offset,
                                                   x->shape[normalized_axis]);
        if (st != GD_OK) {
            return st;
        }
        axis_offset += sizes[i];
    }
    *grad_x = dx;
    return GD_OK;
}
