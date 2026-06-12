#include "permute_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool gd_permute_autograd_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_permute_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
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

gd_status gd_permute_resolve_axes(gd_context *ctx,
                                  uint32_t rank,
                                  const int32_t *axes,
                                  uint32_t n_axes,
                                  uint32_t normalized[GD_PERMUTE_MAX_DIMS])
{
    uint32_t i;
    uint32_t seen = 0U;
    if (normalized != NULL) {
        for (i = 0U; i < GD_PERMUTE_MAX_DIMS; ++i) {
            normalized[i] = 0U;
        }
    }
    if (ctx == NULL || normalized == NULL || rank > GD_PERMUTE_MAX_DIMS || n_axes != rank ||
        (rank > 0U && axes == NULL)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < rank; ++i) {
        int32_t axis = axes[i];
        uint32_t bit;
        if (axis < 0) {
            axis += (int32_t)rank;
        }
        if (axis < 0 || axis >= (int32_t)rank) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute axis out of range");
        }
        bit = 1U << (uint32_t)axis;
        if ((seen & bit) != 0U) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute axes must be unique");
        }
        seen |= bit;
        normalized[i] = (uint32_t)axis;
    }
    return GD_OK;
}

static gd_status gd_permute_compute_inner(gd_context *ctx,
                                          uint32_t rank,
                                          const int64_t out_shape[GD_PERMUTE_MAX_DIMS],
                                          const uint32_t axes[GD_PERMUTE_MAX_DIMS],
                                          uint64_t *out_inner,
                                          uint32_t *out_active_rank)
{
    uint32_t active_rank;
    uint64_t inner = 1U;
    if (ctx == NULL || out_shape == NULL || axes == NULL || out_inner == NULL ||
        out_active_rank == NULL || rank > GD_PERMUTE_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    active_rank = rank;
    while (active_rank > 0U && axes[active_rank - 1U] == active_rank - 1U) {
        uint32_t d = active_rank - 1U;
        if (out_shape[d] <= 0 || !gd_permute_u64_mul(inner, (uint64_t)out_shape[d], &inner)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "permute contiguous suffix overflow");
        }
        --active_rank;
    }
    *out_inner = inner;
    *out_active_rank = active_rank;
    return GD_OK;
}

static gd_status gd_permute_materialize_impl(gd_context *ctx,
                                             const gd_tensor *x,
                                             const uint32_t axes[GD_PERMUTE_MAX_DIMS],
                                             uint32_t n_axes,
                                             bool record_autograd,
                                             gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_permute_args args;
    int64_t out_shape[GD_PERMUTE_MAX_DIMS];
    int64_t numel;
    uint64_t inner;
    uint32_t active_rank;
    uint32_t d;
    bool needs_grad;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || axes == NULL || out == NULL || n_axes > GD_PERMUTE_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (x->rank != n_axes) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute rank/axis count mismatch");
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "permute requires contiguous input");
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "permute invalid input shape");
    }
    if ((uint64_t)numel > UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "permute supports at most UINT32_MAX elements");
    }
    for (d = 0U; d < n_axes; ++d) {
        if (axes[d] >= n_axes) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute normalized axis out of range");
        }
        out_shape[d] = x->shape[axes[d]];
    }
    for (d = n_axes; d < GD_PERMUTE_MAX_DIMS; ++d) {
        out_shape[d] = 0;
    }
    st = gd_permute_compute_inner(ctx, n_axes, out_shape, axes, &inner, &active_rank);
    if (st != GD_OK) {
        return st;
    }
    needs_grad = record_autograd && x->requires_grad;
    if (needs_grad && !gd_permute_autograd_dtype(x->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "permute autograd supports f16/f32 tensors only");
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         x->dtype,
                         gd_shape_make(n_axes, out_shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    y.requires_grad = false;
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute invalid tensor view");
    }
    memset(&args, 0, sizeof(args));
    args.count = (uint64_t)numel;
    args.inner = inner;
    args.rank = n_axes;
    args.active_rank = active_rank;
    for (d = 0U; d < n_axes; ++d) {
        args.axes[d] = axes[d];
    }
    st = gd_backend_permute(gd_context_backend(ctx), &xv, &yv, &args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend permute failed");
    }
    if (needs_grad) {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        gd_permute_attrs attrs;
        memset(&attrs, 0, sizeof(attrs));
        attrs.n_axes = n_axes;
        for (d = 0U; d < n_axes; ++d) {
            attrs.axes[d] = (int32_t)axes[d];
        }
        st = gd_autograd_record(ctx,
                                GD_OP_PERMUTE,
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
    *out = y;
    return GD_OK;
}

gd_status gd_permute(gd_context *ctx,
                     const gd_tensor *x,
                     const int32_t *axes,
                     uint32_t n_axes,
                     gd_tensor *out)
{
    gd_status st;
    uint32_t normalized[GD_PERMUTE_MAX_DIMS];
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_permute_resolve_axes(ctx, x->rank, axes, n_axes, normalized);
    if (st != GD_OK) {
        return st;
    }
    return gd_permute_materialize_impl(ctx, x, normalized, n_axes, true, out);
}

gd_status gd_permute_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              const int32_t *axes,
                              uint32_t n_axes,
                              gd_tensor *grad_x)
{
    gd_status st;
    uint32_t normalized[GD_PERMUTE_MAX_DIMS];
    uint32_t inverse[GD_PERMUTE_MAX_DIMS];
    uint32_t d;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != grad_out->dtype || grad_out->rank != x->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "permute backward dtype/rank mismatch");
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "permute backward requires contiguous tensors");
    }
    st = gd_permute_resolve_axes(ctx, x->rank, axes, n_axes, normalized);
    if (st != GD_OK) {
        return st;
    }
    for (d = 0U; d < n_axes; ++d) {
        if (grad_out->shape[d] != x->shape[normalized[d]]) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "permute backward grad_out shape mismatch");
        }
    }
    if (grad_x == NULL) {
        return GD_OK;
    }
    for (d = 0U; d < n_axes; ++d) {
        inverse[normalized[d]] = d;
    }
    return gd_permute_materialize_impl(ctx, grad_out, inverse, n_axes, false, grad_x);
}
