#ifndef GD_OPS_SHARED_LOSS_PAIRWISE_LOSS_CORE_H
#define GD_OPS_SHARED_LOSS_PAIRWISE_LOSS_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <gradients/ops.h>

#include "../../autograd_impl.h"
#include "../../op_common.h"

typedef gd_status (*gd_pairwise_loss_attr_validate_fn)(gd_context *ctx,
                                                       const void *attrs);

typedef gd_status (*gd_pairwise_loss_backend_forward_fn)(
    gd_backend *backend,
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *out,
    uint64_t chunk_size,
    float scale,
    const void *attrs);

typedef gd_status (*gd_pairwise_loss_backend_backward_fn)(
    gd_backend *backend,
    const gd_backend_tensor_view *x,
    const gd_backend_tensor_view *y,
    const gd_backend_tensor_view *grad_out,
    const gd_backend_tensor_view *grad_x,
    const gd_backend_tensor_view *grad_y,
    float scale,
    const void *attrs);

typedef gd_status (*gd_pairwise_loss_backward_api_fn)(gd_context *ctx,
                                                     const gd_tensor *x,
                                                     const gd_tensor *y,
                                                     const gd_tensor *grad_out,
                                                     gd_tensor *grad_x,
                                                     gd_tensor *grad_y);

typedef struct gd_pairwise_loss_spec {
    const char *name;
    gd_op_kind op;
    size_t chunk_size;
    float backward_scale;
    gd_pairwise_loss_attr_validate_fn validate_attrs;
    gd_pairwise_loss_backend_forward_fn backend_forward;
    gd_pairwise_loss_backend_backward_fn backend_backward;
    const char *unsupported_dtype_message;
    const char *dtype_mismatch_message;
    const char *shape_mismatch_message;
    const char *contiguous_message;
    const char *invalid_shape_message;
    const char *element_count_message;
    const char *grad_out_message;
    const char *forward_view_message;
    const char *reduce_view_message;
    const char *backward_view_message;
    const char *forward_failed_message;
    const char *reduce_failed_message;
    const char *backward_failed_message;
    const char *partial_overflow_message;
} gd_pairwise_loss_spec;

static inline const char *gd_pairwise_loss_msg(const char *message,
                                               const char *fallback)
{
    return message != NULL ? message : fallback;
}

static inline gd_status gd_pairwise_loss_spec_validate(const gd_pairwise_loss_spec *spec)
{
    if (spec == NULL || spec->name == NULL || spec->chunk_size == 0U ||
        spec->chunk_size > (size_t)UINT32_MAX || !(spec->backward_scale == spec->backward_scale) ||
        spec->backend_forward == NULL || spec->backend_backward == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_validate_inputs(gd_context *ctx,
                                                         const gd_tensor *x,
                                                         const gd_tensor *y,
                                                         const gd_pairwise_loss_spec *spec,
                                                         size_t *count_out)
{
    gd_status st;
    int64_t numel;
    uint32_t i;
    if (ctx == NULL || x == NULL || y == NULL || spec == NULL || count_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *count_out = 0U;
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, y);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != GD_DTYPE_F16 && x->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(
            ctx,
            GD_ERR_UNSUPPORTED,
            gd_pairwise_loss_msg(spec->unsupported_dtype_message,
                                 "pairwise loss supports f16/f32 tensors only"));
    }
    if (y->dtype != x->dtype) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->dtype_mismatch_message,
                                 "pairwise loss input dtypes must match"));
    }
    if (x->rank != y->rank) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->shape_mismatch_message,
                                 "pairwise loss input shapes must match"));
    }
    for (i = 0U; i < x->rank; ++i) {
        if (x->shape[i] != y->shape[i]) {
            return gd_context_set_error(
                ctx,
                GD_ERR_INVALID_ARGUMENT,
                gd_pairwise_loss_msg(spec->shape_mismatch_message,
                                     "pairwise loss input shapes must match"));
        }
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(y)) {
        return gd_context_set_error(
            ctx,
            GD_ERR_UNSUPPORTED,
            gd_pairwise_loss_msg(spec->contiguous_message,
                                 "pairwise loss requires contiguous inputs"));
    }
    st = gd_tensor_numel(x, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(
            ctx,
            st,
            gd_pairwise_loss_msg(spec->invalid_shape_message,
                                 "pairwise loss invalid input shape"));
    }
    if (numel <= 0 || (uint64_t)numel > (uint64_t)UINT32_MAX ||
        (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(
            ctx,
            numel <= 0 ? GD_ERR_INVALID_ARGUMENT : GD_ERR_OUT_OF_MEMORY,
            gd_pairwise_loss_msg(spec->element_count_message,
                                 "pairwise loss element count unsupported"));
    }
    *count_out = (size_t)numel;
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_validate_grad_out(gd_context *ctx,
                                                           const gd_tensor *grad_out,
                                                           const gd_pairwise_loss_spec *spec)
{
    gd_status st;
    if (ctx == NULL || grad_out == NULL || spec == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F32 || grad_out->rank != 0U ||
        !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->grad_out_message,
                                 "pairwise loss backward requires scalar f32 grad_out"));
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_dispatch_forward(gd_context *ctx,
                                                          const gd_tensor *x,
                                                          const gd_tensor *y,
                                                          const gd_tensor *out,
                                                          size_t chunk_size,
                                                          float scale,
                                                          const gd_pairwise_loss_spec *spec,
                                                          const void *attrs)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view ov;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || out == NULL || spec == NULL ||
        chunk_size == 0U || chunk_size > (size_t)UINT32_MAX || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->forward_view_message,
                                 "pairwise loss forward invalid tensor view"));
    }
    st = spec->backend_forward(gd_context_backend(ctx),
                               &xv,
                               &yv,
                               &ov,
                               (uint64_t)chunk_size,
                               scale,
                               attrs);
    if (st != GD_OK) {
        return gd_context_set_error(
            ctx,
            st,
            gd_pairwise_loss_msg(spec->forward_failed_message,
                                 "backend pairwise loss forward failed"));
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_dispatch_reduce(gd_context *ctx,
                                                         const gd_tensor *partial,
                                                         const gd_tensor *out,
                                                         float scale,
                                                         const gd_pairwise_loss_spec *spec)
{
    gd_backend_tensor_view pv;
    gd_backend_tensor_view ov;
    gd_status st;
    if (ctx == NULL || partial == NULL || out == NULL || spec == NULL || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(partial, &pv) || !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->reduce_view_message,
                                 "pairwise loss reduce invalid tensor view"));
    }
    st = gd_backend_reduce_contiguous(gd_context_backend(ctx), &pv, &ov, scale);
    if (st != GD_OK) {
        return gd_context_set_error(
            ctx,
            st,
            gd_pairwise_loss_msg(spec->reduce_failed_message,
                                 "backend pairwise loss reduce failed"));
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_dispatch_backward(gd_context *ctx,
                                                           const gd_tensor *x,
                                                           const gd_tensor *y,
                                                           const gd_tensor *grad_out,
                                                           const gd_tensor *grad_x,
                                                           const gd_tensor *grad_y,
                                                           float scale,
                                                           const gd_pairwise_loss_spec *spec,
                                                           const void *attrs)
{
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_backend_tensor_view dyv;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL || spec == NULL ||
        (grad_x == NULL && grad_y == NULL) || !(scale == scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&dxv, 0, sizeof(dxv));
    memset(&dyv, 0, sizeof(dyv));
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        (grad_x != NULL && !gd_op_tensor_view_from_tensor(grad_x, &dxv)) ||
        (grad_y != NULL && !gd_op_tensor_view_from_tensor(grad_y, &dyv))) {
        return gd_context_set_error(
            ctx,
            GD_ERR_INVALID_ARGUMENT,
            gd_pairwise_loss_msg(spec->backward_view_message,
                                 "pairwise loss backward invalid tensor view"));
    }
    st = spec->backend_backward(gd_context_backend(ctx),
                                &xv,
                                &yv,
                                &gv,
                                grad_x != NULL ? &dxv : NULL,
                                grad_y != NULL ? &dyv : NULL,
                                scale,
                                attrs);
    if (st != GD_OK) {
        return gd_context_set_error(
            ctx,
            st,
            gd_pairwise_loss_msg(spec->backward_failed_message,
                                 "backend pairwise loss backward failed"));
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_forward_mean(gd_context *ctx,
                                                      const gd_tensor *x,
                                                      const gd_tensor *y,
                                                      gd_tensor *out,
                                                      const gd_pairwise_loss_spec *spec,
                                                      const void *attrs)
{
    gd_status st;
    gd_tensor result;
    size_t count = 0U;
    size_t chunk_count;
    float inv_count;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_pairwise_loss_spec_validate(spec);
    if (st != GD_OK) {
        return st;
    }
    if (spec->validate_attrs != NULL) {
        st = spec->validate_attrs(ctx, attrs);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_pairwise_loss_validate_inputs(ctx, x, y, spec, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F32,
                         gd_shape_make(0U, NULL),
                         256U,
                         &result);
    if (st != GD_OK) {
        return st;
    }
    result.is_leaf = false;
    inv_count = 1.0f / (float)count;
    chunk_count = (count + spec->chunk_size - 1U) / spec->chunk_size;
    if (chunk_count <= 1U) {
        st = gd_pairwise_loss_dispatch_forward(ctx, x, y, &result, count, inv_count, spec, attrs);
    } else {
        gd_tensor partial;
        int64_t partial_shape[1];
        if (chunk_count > (size_t)INT64_MAX) {
            return gd_context_set_error(
                ctx,
                GD_ERR_OUT_OF_MEMORY,
                gd_pairwise_loss_msg(spec->partial_overflow_message,
                                     "pairwise loss partial count overflow"));
        }
        partial_shape[0] = (int64_t)chunk_count;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, partial_shape),
                             256U,
                             &partial);
        if (st == GD_OK) {
            partial.is_leaf = false;
            st = gd_pairwise_loss_dispatch_forward(ctx,
                                                   x,
                                                   y,
                                                   &partial,
                                                   spec->chunk_size,
                                                   1.0f,
                                                   spec,
                                                   attrs);
        }
        if (st == GD_OK) {
            st = gd_pairwise_loss_dispatch_reduce(ctx, &partial, &result, inv_count, spec);
        }
    }
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        inputs[0] = x;
        inputs[1] = y;
        outputs[0] = &result;
        st = gd_autograd_record(ctx, spec->op, inputs, 2U, outputs, 1U, NULL, 0U, NULL, 0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = result;
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_backward_mean(gd_context *ctx,
                                                       const gd_tensor *x,
                                                       const gd_tensor *y,
                                                       const gd_tensor *grad_out,
                                                       gd_tensor *grad_x,
                                                       gd_tensor *grad_y,
                                                       const gd_pairwise_loss_spec *spec,
                                                       const void *attrs)
{
    gd_status st;
    gd_tensor dx;
    gd_tensor dy;
    size_t count = 0U;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_y != NULL) {
        memset(grad_y, 0, sizeof(*grad_y));
    }
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_pairwise_loss_spec_validate(spec);
    if (st != GD_OK) {
        return st;
    }
    if (spec->validate_attrs != NULL) {
        st = spec->validate_attrs(ctx, attrs);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_pairwise_loss_validate_inputs(ctx, x, y, spec, &count);
    if (st != GD_OK) {
        return st;
    }
    st = gd_pairwise_loss_validate_grad_out(ctx, grad_out, spec);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x == NULL && grad_y == NULL) {
        return GD_OK;
    }
    memset(&dx, 0, sizeof(dx));
    memset(&dy, 0, sizeof(dy));
    if (grad_x != NULL) {
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
    }
    if (grad_y != NULL) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             y->dtype,
                             gd_shape_make(y->rank, y->shape),
                             256U,
                             &dy);
        if (st != GD_OK) {
            return st;
        }
        dy.is_leaf = false;
    }
    st = gd_pairwise_loss_dispatch_backward(ctx,
                                            x,
                                            y,
                                            grad_out,
                                            grad_x != NULL ? &dx : NULL,
                                            grad_y != NULL ? &dy : NULL,
                                            spec->backward_scale / (float)count,
                                            spec,
                                            attrs);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x != NULL) {
        *grad_x = dx;
    }
    if (grad_y != NULL) {
        *grad_y = dy;
    }
    return GD_OK;
}

static inline gd_status gd_pairwise_loss_autograd_backward(
    gd_bwd_ctx *bwd,
    const gd_tape_node *node,
    gd_pairwise_loss_backward_api_fn backward_fn)
{
    const gd_tensor *x;
    const gd_tensor *y;
    const gd_tensor *out;
    gd_tensor grad_out;
    gd_tensor dx;
    gd_tensor dy;
    bool need_x;
    bool need_y;
    if (bwd == NULL || node == NULL || backward_fn == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    x = gd_tape_input(bwd->tape, node, 0U);
    y = gd_tape_input(bwd->tape, node, 1U);
    out = gd_tape_output(bwd->tape, node, 0U);
    if (x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    need_x = x->requires_grad;
    need_y = y->requires_grad;
    if (!need_x && !need_y) {
        return GD_OK;
    }
    GD_TRY(backward_fn(bwd->ctx,
                       x,
                       y,
                       &grad_out,
                       need_x ? &dx : NULL,
                       need_y ? &dy : NULL));
    if (need_x) {
        GD_TRY(gd_autograd_accumulate(bwd, x->id, &dx));
    }
    if (need_y) {
        GD_TRY(gd_autograd_accumulate(bwd, y->id, &dy));
    }
    return GD_OK;
}

#endif /* GD_OPS_SHARED_LOSS_PAIRWISE_LOSS_CORE_H */
