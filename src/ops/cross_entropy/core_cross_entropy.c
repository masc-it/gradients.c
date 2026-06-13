#include "cross_entropy_impl.h"

#include "../_shared/reduce/reduce_core.h"
#include "../op_common.h"

#include <stddef.h>
#include <string.h>

static gd_status gd_cross_entropy_validate_logits_targets(gd_context *ctx,
                                                          const gd_tensor *logits,
                                                          const gd_tensor *targets)
{
    gd_status st;
    int64_t numel;
    if (ctx == NULL || logits == NULL || targets == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, logits);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, targets);
    if (st != GD_OK) {
        return st;
    }
    if (logits->dtype != GD_DTYPE_F16) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "cross_entropy logits must be f16");
    }
    if (targets->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "cross_entropy targets must be i32 class indices");
    }
    if (logits->rank != 2U || targets->rank != 1U || logits->shape[0] <= 0 ||
        logits->shape[1] <= 1 || targets->shape[0] != logits->shape[0]) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy expects logits [N, C] and targets [N]");
    }
    if (!gd_tensor_is_contiguous(logits) || !gd_tensor_is_contiguous(targets)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "cross_entropy requires contiguous tensors");
    }
    st = gd_tensor_numel(logits, &numel);
    if (st != GD_OK || (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, st != GD_OK ? st : GD_ERR_OUT_OF_MEMORY,
                                    "cross_entropy logits count overflow");
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_validate_row_stats(gd_context *ctx,
                                                     const gd_tensor *logits,
                                                     const gd_tensor *row_max,
                                                     const gd_tensor *row_inv_sum)
{
    gd_status st;
    if (ctx == NULL || logits == NULL || row_max == NULL || row_inv_sum == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, row_max);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, row_inv_sum);
    if (st != GD_OK) {
        return st;
    }
    if (row_max->dtype != GD_DTYPE_F32 || row_inv_sum->dtype != GD_DTYPE_F32 ||
        row_max->rank != 1U || row_inv_sum->rank != 1U ||
        row_max->shape[0] != logits->shape[0] || row_inv_sum->shape[0] != logits->shape[0] ||
        !gd_tensor_is_contiguous(row_max) || !gd_tensor_is_contiguous(row_inv_sum)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy row stats must be contiguous f32 [N]");
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_dispatch_loss(gd_context *ctx,
                                                const gd_tensor *logits,
                                                const gd_tensor *targets,
                                                const gd_tensor *row_loss)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view rv;
    gd_status st;
    if (ctx == NULL || logits == NULL || targets == NULL || row_loss == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(logits, &lv) ||
        !gd_op_tensor_view_from_tensor(targets, &tv) ||
        !gd_op_tensor_view_from_tensor(row_loss, &rv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy invalid tensor view");
    }
    st = gd_backend_cross_entropy_loss(gd_context_backend(ctx), &lv, &tv, &rv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend cross_entropy loss failed");
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_dispatch_loss_stats(gd_context *ctx,
                                                      const gd_tensor *logits,
                                                      const gd_tensor *targets,
                                                      const gd_tensor *row_loss,
                                                      const gd_tensor *row_max,
                                                      const gd_tensor *row_inv_sum)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view rv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_status st;
    if (ctx == NULL || logits == NULL || targets == NULL || row_loss == NULL ||
        row_max == NULL || row_inv_sum == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(logits, &lv) ||
        !gd_op_tensor_view_from_tensor(targets, &tv) ||
        !gd_op_tensor_view_from_tensor(row_loss, &rv) ||
        !gd_op_tensor_view_from_tensor(row_max, &mv) ||
        !gd_op_tensor_view_from_tensor(row_inv_sum, &iv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy stats invalid tensor view");
    }
    st = gd_backend_cross_entropy_loss_stats(gd_context_backend(ctx), &lv, &tv, &rv, &mv, &iv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend cross_entropy stats loss failed");
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_dispatch_backward(gd_context *ctx,
                                                    const gd_tensor *logits,
                                                    const gd_tensor *targets,
                                                    const gd_tensor *grad_loss,
                                                    const gd_tensor *grad_logits,
                                                    float scale)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_status st;
    if (ctx == NULL || logits == NULL || targets == NULL || grad_loss == NULL || grad_logits == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(logits, &lv) ||
        !gd_op_tensor_view_from_tensor(targets, &tv) ||
        !gd_op_tensor_view_from_tensor(grad_loss, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_logits, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy backward invalid tensor view");
    }
    st = gd_backend_cross_entropy_backward(gd_context_backend(ctx), &lv, &tv, &gv, &dxv, scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend cross_entropy backward failed");
    }
    return GD_OK;
}

static gd_status gd_cross_entropy_dispatch_backward_stats(gd_context *ctx,
                                                          const gd_tensor *logits,
                                                          const gd_tensor *targets,
                                                          const gd_tensor *row_max,
                                                          const gd_tensor *row_inv_sum,
                                                          const gd_tensor *grad_loss,
                                                          const gd_tensor *grad_logits,
                                                          float scale)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_status st;
    if (ctx == NULL || logits == NULL || targets == NULL || row_max == NULL ||
        row_inv_sum == NULL || grad_loss == NULL || grad_logits == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(logits, &lv) ||
        !gd_op_tensor_view_from_tensor(targets, &tv) ||
        !gd_op_tensor_view_from_tensor(row_max, &mv) ||
        !gd_op_tensor_view_from_tensor(row_inv_sum, &iv) ||
        !gd_op_tensor_view_from_tensor(grad_loss, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_logits, &dxv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy backward stats invalid tensor view");
    }
    st = gd_backend_cross_entropy_backward_stats(gd_context_backend(ctx),
                                                &lv,
                                                &tv,
                                                &mv,
                                                &iv,
                                                &gv,
                                                &dxv,
                                                scale);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend cross_entropy backward stats failed");
    }
    return GD_OK;
}

gd_status gd_cross_entropy(gd_context *ctx,
                           const gd_tensor *x,
                           const gd_tensor *y,
                           gd_tensor *out)
{
    gd_status st;
    gd_tensor row_loss;
    gd_tensor row_max;
    gd_tensor row_inv_sum;
    int64_t row_shape[1];
    float inv_rows;
    bool need_stats;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || y == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_validate_logits_targets(ctx, x, y);
    if (st != GD_OK) {
        return st;
    }
    row_shape[0] = x->shape[0];
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_loss);
    if (st != GD_OK) {
        return st;
    }
    row_loss.is_leaf = false;
    memset(&row_max, 0, sizeof(row_max));
    memset(&row_inv_sum, 0, sizeof(row_inv_sum));
    need_stats = x->requires_grad && gd_context_scope_mode(ctx) == GD_SCOPE_TRAIN;
    if (need_stats) {
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_max);
        if (st != GD_OK) {
            return st;
        }
        st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_inv_sum);
        if (st != GD_OK) {
            return st;
        }
        row_max.is_leaf = false;
        row_inv_sum.is_leaf = false;
        st = gd_cross_entropy_dispatch_loss_stats(ctx, x, y, &row_loss, &row_max, &row_inv_sum);
    } else {
        st = gd_cross_entropy_dispatch_loss(ctx, x, y, &row_loss);
    }
    if (st != GD_OK) {
        return st;
    }
    inv_rows = 1.0f / (float)x->shape[0];
    st = gd_reduce_all_forward_impl(ctx, &row_loss, out, GD_OP_CROSS_ENTROPY, inv_rows);
    if (st != GD_OK) {
        return st;
    }
    if (x->requires_grad) {
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        const gd_tensor *saved[2];
        uint16_t n_saved = 0U;
        inputs[0] = x;
        inputs[1] = y;
        outputs[0] = out;
        if (need_stats) {
            saved[0] = &row_max;
            saved[1] = &row_inv_sum;
            n_saved = 2U;
        }
        st = gd_autograd_record(ctx,
                                GD_OP_CROSS_ENTROPY,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                n_saved != 0U ? saved : NULL,
                                n_saved);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

gd_status gd_cross_entropy_backward(gd_context *ctx,
                                    const gd_tensor *x,
                                    const gd_tensor *y,
                                    const gd_tensor *grad_out,
                                    gd_tensor *grad_x,
                                    gd_tensor *grad_y)
{
    gd_status st;
    gd_tensor dx;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_y != NULL) {
        memset(grad_y, 0, sizeof(*grad_y));
    }
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (grad_y != NULL) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "cross_entropy targets are non-differentiable");
    }
    st = gd_cross_entropy_validate_logits_targets(ctx, x, y);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F32 || grad_out->rank != 0U || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy backward requires scalar f32 grad_out");
    }
    if (grad_x == NULL) {
        return GD_OK;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(x->rank, x->shape), 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_cross_entropy_dispatch_backward(ctx, x, y, grad_out, &dx, 1.0f / (float)x->shape[0]);
    if (st != GD_OK) {
        return st;
    }
    *grad_x = dx;
    return GD_OK;
}

gd_status gd_cross_entropy_backward_with_stats(gd_context *ctx,
                                               const gd_tensor *logits,
                                               const gd_tensor *targets,
                                               const gd_tensor *row_max,
                                               const gd_tensor *row_inv_sum,
                                               const gd_tensor *grad_out,
                                               gd_tensor *grad_logits)
{
    gd_status st;
    gd_tensor dx;
    if (grad_logits != NULL) {
        memset(grad_logits, 0, sizeof(*grad_logits));
    }
    if (ctx == NULL || logits == NULL || targets == NULL || row_max == NULL ||
        row_inv_sum == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_cross_entropy_validate_logits_targets(ctx, logits, targets);
    if (st != GD_OK) {
        return st;
    }
    st = gd_cross_entropy_validate_row_stats(ctx, logits, row_max, row_inv_sum);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F32 || grad_out->rank != 0U || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "cross_entropy backward requires scalar f32 grad_out");
    }
    if (grad_logits == NULL) {
        return GD_OK;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(logits->rank, logits->shape), 256U, &dx);
    if (st != GD_OK) {
        return st;
    }
    dx.is_leaf = false;
    st = gd_cross_entropy_dispatch_backward_stats(ctx,
                                                 logits,
                                                 targets,
                                                 row_max,
                                                 row_inv_sum,
                                                 grad_out,
                                                 &dx,
                                                 1.0f / (float)logits->shape[0]);
    if (st != GD_OK) {
        return st;
    }
    *grad_logits = dx;
    return GD_OK;
}
