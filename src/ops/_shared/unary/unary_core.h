#ifndef GD_OPS_SHARED_UNARY_CORE_H
#define GD_OPS_SHARED_UNARY_CORE_H

#include <stdbool.h>
#include <string.h>

#include <gradients/ops.h>

#include "../../autograd_impl.h"
#include "../../op_common.h"

typedef bool (*gd_unary_dtype_supported_fn)(gd_dtype dtype);
typedef gd_status (*gd_unary_backend_fn)(gd_backend *backend,
                                         const gd_backend_tensor_view *x,
                                         const gd_backend_tensor_view *out);
typedef gd_status (*gd_unary_backend_backward_fn)(gd_backend *backend,
                                                  const gd_backend_tensor_view *x,
                                                  const gd_backend_tensor_view *grad_out,
                                                  const gd_backend_tensor_view *grad_x);
typedef gd_status (*gd_unary_backward_api_fn)(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *grad_out,
                                              gd_tensor *grad_x);

typedef struct gd_unary_op_spec {
    const char *name;
    gd_op_kind op;
    gd_unary_dtype_supported_fn dtype_supported;
    gd_unary_backend_fn backend_forward;
    gd_unary_backend_backward_fn backend_backward;
    const char *unsupported_dtype_message;
    const char *contiguous_input_message;
    const char *grad_shape_message;
    const char *grad_contiguous_message;
    const char *forward_view_message;
    const char *backward_view_message;
    const char *backend_forward_failed_message;
    const char *backend_backward_failed_message;
} gd_unary_op_spec;

static inline bool gd_unary_spec_valid(const gd_unary_op_spec *spec)
{
    return spec != NULL && spec->name != NULL && spec->dtype_supported != NULL &&
           spec->backend_forward != NULL && spec->backend_backward != NULL;
}

static inline gd_status gd_unary_validate_input(gd_context *ctx,
                                                const gd_tensor *x,
                                                const gd_unary_op_spec *spec)
{
    gd_status st;
    if (ctx == NULL || x == NULL || !gd_unary_spec_valid(spec)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (!spec->dtype_supported(x->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, spec->unsupported_dtype_message);
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, spec->contiguous_input_message);
    }
    return GD_OK;
}

static inline gd_status gd_unary_validate_like(gd_context *ctx,
                                               const gd_tensor *ref,
                                               const gd_tensor *tensor,
                                               const char *shape_message,
                                               const char *contiguous_message)
{
    gd_status st;
    uint32_t i;
    if (ctx == NULL || ref == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (tensor->dtype != ref->dtype || tensor->rank != ref->rank) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, shape_message);
    }
    for (i = 0U; i < ref->rank; ++i) {
        if (tensor->shape[i] != ref->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, shape_message);
        }
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, contiguous_message);
    }
    return GD_OK;
}

static inline gd_status gd_unary_apply_impl(gd_context *ctx,
                                            const gd_tensor *x,
                                            gd_tensor *out,
                                            const gd_unary_op_spec *spec)
{
    gd_status st;
    gd_tensor y;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL || !gd_unary_spec_valid(spec)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_unary_validate_input(ctx, x, spec);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         x->dtype,
                         gd_shape_make(x->rank, x->shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(&y, &yv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, spec->forward_view_message);
    }
    st = spec->backend_forward(gd_context_backend(ctx), &xv, &yv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, spec->backend_forward_failed_message);
    }
    {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx, spec->op, inputs, 1U, outputs, 1U, NULL, 0U, NULL, 0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

static inline gd_status gd_unary_backward_impl_with_backend(
    gd_context *ctx,
    const gd_tensor *x,
    const gd_tensor *grad_out,
    gd_tensor *grad_x,
    const gd_unary_op_spec *spec,
    gd_unary_backend_backward_fn backend_backward,
    const char *shape_message,
    const char *backward_view_message,
    const char *backend_failed_message)
{
    gd_status st;
    gd_tensor dx;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL || grad_x == NULL ||
        !gd_unary_spec_valid(spec) || backend_backward == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_unary_validate_input(ctx, x, spec);
    if (st != GD_OK) {
        return st;
    }
    st = gd_unary_validate_like(ctx,
                                x,
                                grad_out,
                                shape_message,
                                spec->grad_contiguous_message);
    if (st != GD_OK) {
        return st;
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
    dx.is_leaf = false;
    if (!gd_op_tensor_view_from_tensor(x, &xv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(&dx, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, backward_view_message);
    }
    st = backend_backward(gd_context_backend(ctx), &xv, &gv, &dxv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, backend_failed_message);
    }
    *grad_x = dx;
    return GD_OK;
}

static inline gd_status gd_unary_backward_impl(gd_context *ctx,
                                               const gd_tensor *x,
                                               const gd_tensor *grad_out,
                                               gd_tensor *grad_x,
                                               const gd_unary_op_spec *spec)
{
    if (!gd_unary_spec_valid(spec)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_unary_backward_impl_with_backend(ctx,
                                               x,
                                               grad_out,
                                               grad_x,
                                               spec,
                                               spec->backend_backward,
                                               spec->grad_shape_message,
                                               spec->backward_view_message,
                                               spec->backend_backward_failed_message);
}

static inline gd_status gd_unary_autograd_backward_with_tensor(
    gd_bwd_ctx *bwd,
    const gd_tape_node *node,
    gd_unary_backward_api_fn backward_fn,
    bool use_output_for_backward)
{
    const gd_tensor *x;
    const gd_tensor *out;
    gd_tensor grad_out;
    gd_tensor dx;
    if (bwd == NULL || node == NULL || backward_fn == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    x = gd_tape_input(bwd->tape, node, 0U);
    out = gd_tape_output(bwd->tape, node, 0U);
    if (x == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out) || !x->requires_grad) {
        return GD_OK;
    }
    GD_TRY(backward_fn(bwd->ctx,
                       use_output_for_backward ? out : x,
                       &grad_out,
                       &dx));
    return gd_autograd_accumulate(bwd, x->id, &dx);
}

static inline gd_status gd_unary_autograd_backward(gd_bwd_ctx *bwd,
                                                   const gd_tape_node *node,
                                                   gd_unary_backward_api_fn backward_fn)
{
    return gd_unary_autograd_backward_with_tensor(bwd, node, backward_fn, false);
}

static inline gd_status gd_unary_autograd_backward_from_output(
    gd_bwd_ctx *bwd,
    const gd_tape_node *node,
    gd_unary_backward_api_fn backward_fn)
{
    return gd_unary_autograd_backward_with_tensor(bwd, node, backward_fn, true);
}

#endif /* GD_OPS_SHARED_UNARY_CORE_H */
