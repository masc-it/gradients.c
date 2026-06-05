#ifndef GD_OPS_SHARED_REDUCE_CORE_H
#define GD_OPS_SHARED_REDUCE_CORE_H

#include <gradients/ops.h>

#include "../../autograd_impl.h"
#include "../../op_common.h"

#include <stddef.h>
#include <string.h>

#define GD_REDUCE_CONTIGUOUS_CHUNK_SIZE 4096U
#define GD_REDUCE_AXIS_ALL (-1)

typedef struct gd_reduce_op_attrs {
    int32_t axis;      /* GD_REDUCE_AXIS_ALL for all-elements reductions; otherwise normalized. */
    uint32_t keepdims; /* 0 or 1. */
    uint32_t reserved0;
    uint32_t reserved1;
} gd_reduce_op_attrs;

static inline gd_status gd_reduce_validate_input(gd_context *ctx, const gd_tensor *x)
{
    gd_status st;
    if (ctx == NULL || x == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 && x->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "reduce currently supports f16/f32 tensors only");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "reduce requires contiguous input");
    }
    return GD_OK;
}

static inline gd_status gd_reduce_tensor_count(gd_context *ctx,
                                               const gd_tensor *x,
                                               size_t *out_count)
{
    int64_t numel;
    gd_status st;
    if (ctx == NULL || x == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "reduce invalid input shape");
    }
    if ((uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "reduce count overflow");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static inline gd_status gd_reduce_normalize_axis(gd_context *ctx,
                                                 const gd_tensor *x,
                                                 int32_t axis,
                                                 uint32_t *out_axis)
{
    int32_t rank;
    int32_t normalized;
    if (ctx == NULL || x == NULL || out_axis == NULL || x->rank == 0U || x->rank > GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    rank = (int32_t)x->rank;
    normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reduce axis out of range");
    }
    *out_axis = (uint32_t)normalized;
    return GD_OK;
}

static inline gd_status gd_reduce_axis_output_shape(gd_context *ctx,
                                                    const gd_tensor *x,
                                                    uint32_t axis,
                                                    bool keepdims,
                                                    uint32_t *out_rank,
                                                    int64_t out_shape[GD_MAX_DIMS])
{
    uint32_t dim;
    uint32_t j = 0U;
    if (ctx == NULL || x == NULL || out_rank == NULL || out_shape == NULL || axis >= x->rank) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out_shape, 0, sizeof(int64_t) * GD_MAX_DIMS);
    if (keepdims) {
        *out_rank = x->rank;
        for (dim = 0U; dim < x->rank; ++dim) {
            out_shape[dim] = dim == axis ? 1 : x->shape[dim];
        }
        return GD_OK;
    }
    *out_rank = x->rank - 1U;
    for (dim = 0U; dim < x->rank; ++dim) {
        if (dim == axis) {
            continue;
        }
        out_shape[j] = x->shape[dim];
        j += 1U;
    }
    return GD_OK;
}

static inline bool gd_reduce_axis_grad_shape_matches(const gd_tensor *x,
                                                     const gd_tensor *grad_out,
                                                     uint32_t axis,
                                                     bool keepdims)
{
    uint32_t expected_rank;
    int64_t expected_shape[GD_MAX_DIMS];
    uint32_t dim;
    if (x == NULL || grad_out == NULL || axis >= x->rank) {
        return false;
    }
    expected_rank = keepdims ? x->rank : x->rank - 1U;
    memset(expected_shape, 0, sizeof(expected_shape));
    if (keepdims) {
        for (dim = 0U; dim < x->rank; ++dim) {
            expected_shape[dim] = dim == axis ? 1 : x->shape[dim];
        }
    } else {
        uint32_t j = 0U;
        for (dim = 0U; dim < x->rank; ++dim) {
            if (dim != axis) {
                expected_shape[j] = x->shape[dim];
                j += 1U;
            }
        }
    }
    if (grad_out->rank != expected_rank) {
        return false;
    }
    for (dim = 0U; dim < expected_rank; ++dim) {
        if (grad_out->shape[dim] != expected_shape[dim]) {
            return false;
        }
    }
    return true;
}

static inline gd_status gd_reduce_contiguous_dispatch(gd_context *ctx,
                                                      const gd_tensor *src,
                                                      const gd_tensor *dst,
                                                      float scale)
{
    gd_backend_tensor_view sv;
    gd_backend_tensor_view dv;
    gd_status st;
    if (ctx == NULL || src == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(src, &sv) || !gd_op_tensor_view_from_tensor(dst, &dv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reduce invalid tensor view");
    }
    st = gd_backend_reduce_contiguous(gd_context_backend(ctx), &sv, &dv, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend reduce contiguous failed");
    }
    return GD_OK;
}

static inline gd_status gd_reduce_axis_dispatch(gd_context *ctx,
                                                const gd_tensor *src,
                                                const gd_tensor *dst,
                                                uint32_t axis,
                                                float scale)
{
    gd_backend_tensor_view sv;
    gd_backend_tensor_view dv;
    gd_status st;
    if (ctx == NULL || src == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(src, &sv) || !gd_op_tensor_view_from_tensor(dst, &dv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reduce axis invalid tensor view");
    }
    st = gd_backend_reduce_axis(gd_context_backend(ctx), &sv, &dv, axis, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend reduce axis failed");
    }
    return GD_OK;
}

static inline gd_status gd_reduce_axis_backward_dispatch(gd_context *ctx,
                                                         const gd_tensor *grad_out,
                                                         const gd_tensor *dst,
                                                         uint32_t axis,
                                                         float scale)
{
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dv;
    gd_status st;
    if (ctx == NULL || grad_out == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(grad_out, &gv) || !gd_op_tensor_view_from_tensor(dst, &dv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "reduce axis backward invalid tensor view");
    }
    st = gd_backend_broadcast_axis(gd_context_backend(ctx), &gv, &dv, axis, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend reduce axis backward broadcast failed");
    }
    return GD_OK;
}

static inline gd_status gd_reduce_all_forward_impl(gd_context *ctx,
                                                   const gd_tensor *x,
                                                   gd_tensor *out,
                                                   gd_op_kind op_kind,
                                                   float final_scale)
{
    gd_status st;
    gd_tensor result;
    gd_tensor current_tensor;
    const gd_tensor *current;
    size_t current_count;
    gd_reduce_op_attrs attrs;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL || !(final_scale == final_scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_tensor_count(ctx, x, &current_count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, 0U, NULL, 256U, &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    current = x;
    memset(&current_tensor, 0, sizeof(current_tensor));
    while (current_count > 1U) {
        size_t next_count = (current_count + GD_REDUCE_CONTIGUOUS_CHUNK_SIZE - 1U) /
                            GD_REDUCE_CONTIGUOUS_CHUNK_SIZE;
        float stage_scale = next_count == 1U ? final_scale : 1.0f;
        if (next_count == 1U) {
            st = gd_reduce_contiguous_dispatch(ctx, current, &result, stage_scale);
            if (st != GD_OK) {
                return st;
            }
            current = &result;
            current_count = 1U;
        } else {
            gd_tensor partial;
            int64_t partial_shape[1];
            if (next_count > (size_t)INT64_MAX) {
                return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "reduce partial count overflow");
            }
            partial_shape[0] = (int64_t)next_count;
            st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, 1U, partial_shape, 256U, &partial);
            if (st != GD_OK) {
                return st;
            }
            partial.is_leaf = false;
            st = gd_reduce_contiguous_dispatch(ctx, current, &partial, stage_scale);
            if (st != GD_OK) {
                return st;
            }
            current_tensor = partial;
            current = &current_tensor;
            current_count = next_count;
        }
    }
    if (current_count == 1U) {
        if (current != &result) {
            st = gd_reduce_contiguous_dispatch(ctx, current, &result, final_scale);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    {
        const gd_tensor *inputs[1];
        gd_tensor *outputs[1];
        memset(&attrs, 0, sizeof(attrs));
        attrs.axis = GD_REDUCE_AXIS_ALL;
        attrs.keepdims = 0U;
        inputs[0] = x;
        outputs[0] = &result;
        st = gd_autograd_record(ctx,
                                op_kind,
                                inputs,
                                1U,
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
    *out = result;
    return GD_OK;
}

static inline gd_status gd_reduce_axis_forward_impl(gd_context *ctx,
                                                    const gd_tensor *x,
                                                    gd_tensor *out,
                                                    gd_op_kind op_kind,
                                                    int32_t axis,
                                                    bool keepdims,
                                                    float scale)
{
    gd_status st;
    gd_tensor result;
    uint32_t normalized_axis;
    uint32_t out_rank;
    int64_t out_shape[GD_MAX_DIMS];
    gd_reduce_op_attrs attrs;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_normalize_axis(ctx, x, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_axis_output_shape(ctx, x, normalized_axis, keepdims, &out_rank, out_shape);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         x->dtype,
                         out_rank,
                         out_rank == 0U ? NULL : out_shape,
                         256U,
                         &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    st = gd_reduce_axis_dispatch(ctx, x, &result, normalized_axis, scale);
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_tensor *inputs[1];
        gd_tensor *outputs[1];
        memset(&attrs, 0, sizeof(attrs));
        attrs.axis = (int32_t)normalized_axis;
        attrs.keepdims = keepdims ? 1U : 0U;
        inputs[0] = x;
        outputs[0] = &result;
        st = gd_autograd_record(ctx,
                                op_kind,
                                inputs,
                                1U,
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
    *out = result;
    return GD_OK;
}

static inline gd_status gd_reduce_axis_backward_impl(gd_context *ctx,
                                                     const gd_tensor *x,
                                                     const gd_tensor *grad_out,
                                                     int32_t axis,
                                                     bool keepdims,
                                                     float scale,
                                                     gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    uint32_t normalized_axis;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reduce_normalize_axis(ctx, x, axis, &normalized_axis);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != x->dtype || !gd_tensor_is_contiguous(grad_out) ||
        !gd_reduce_axis_grad_shape_matches(x, grad_out, normalized_axis, keepdims)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "reduce axis backward grad_out shape mismatch");
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, x->rank, x->shape, 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_reduce_axis_backward_dispatch(ctx, grad_out, &dx, normalized_axis, scale);
    if (st != GD_OK) {
        return st;
    }
    *grad_x = dx;
    return GD_OK;
}

static inline gd_status gd_reduce_all_backward_impl(gd_context *ctx,
                                                    const gd_tensor *x,
                                                    const gd_tensor *grad_out,
                                                    float scale,
                                                    gd_tensor *grad_x)
{
    gd_status st;
    gd_tensor dx;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_reduce_validate_input(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != x->dtype || grad_out->rank != 0U || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "reduce backward requires scalar grad_out with matching dtype");
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, x->dtype, x->rank, x->shape, 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(grad_out, &gv) || !gd_op_tensor_view_from_tensor(&dx, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "reduce backward invalid tensor view");
    }
    st = gd_backend_broadcast_to(gd_context_backend(ctx), &gv, &dxv, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend reduce backward broadcast failed");
    }
    *grad_x = dx;
    return GD_OK;
}

#endif /* GD_OPS_SHARED_REDUCE_CORE_H */
