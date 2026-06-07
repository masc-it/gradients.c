#include "lm_cross_entropy_impl.h"

#include "../_shared/reduce/reduce_core.h"
#include "../linear/linear_impl.h"
#include "../op_common.h"
#include "../../core/memory_internal.h"

#include <stddef.h>
#include <string.h>

static gd_status gd_lm_cross_entropy_validate_targets(gd_context *ctx,
                                                      const gd_tensor *targets,
                                                      int64_t rows)
{
    gd_status st;
    if (ctx == NULL || targets == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, targets);
    if (st != GD_OK) {
        return st;
    }
    if (targets->dtype != GD_DTYPE_I32 || targets->rank != 1U || targets->shape[0] != rows ||
        !gd_tensor_is_contiguous(targets)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "lm_cross_entropy expects contiguous i32 targets [rows]");
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_tensor_view(gd_context *ctx,
                                                 const gd_tensor *tensor,
                                                 gd_backend_tensor_view *out)
{
    if (ctx == NULL || tensor == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(tensor, out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid tensor view");
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_free_scratch(gd_context *ctx, const gd_tensor *tensor)
{
    if (ctx == NULL || tensor == NULL || tensor->storage.arena != GD_ARENA_SCRATCH ||
        tensor->storage.nbytes == 0U) {
        return GD_OK;
    }
    return gd_context_free_span(ctx, &tensor->storage);
}

static gd_status gd_lm_cross_entropy_dispatch_loss_stats(gd_context *ctx,
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
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, logits, &lv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, targets, &tv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_loss, &rv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_max, &mv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_inv_sum, &iv));
    st = gd_backend_cross_entropy_loss_stats(gd_context_backend(ctx), &lv, &tv, &rv, &mv, &iv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy loss_stats failed");
    }
    return GD_OK;
}

gd_status gd_lm_cross_entropy(gd_context *ctx,
                              const gd_tensor *hidden,
                              const gd_tensor *weight,
                              const gd_tensor *targets,
                              gd_tensor *loss)
{
    gd_status st;
    gd_linear_shape_info info;
    gd_tensor logits;
    gd_tensor row_loss;
    gd_tensor row_max;
    gd_tensor row_inv_sum;
    gd_backend_matrix_view hv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view lv;
    int64_t row_shape[1];
    float inv_rows;
    if (loss != NULL) {
        memset(loss, 0, sizeof(*loss));
    }
    if (ctx == NULL || hidden == NULL || weight == NULL || targets == NULL || loss == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_linear_transposed_weight_validate_common(ctx, hidden, weight, NULL, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_lm_cross_entropy_validate_targets(ctx, targets, info.rows);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(info.rank, info.out_shape),
                         256U,
                         &logits);
    if (st != GD_OK) {
        return st;
    }
    logits.is_leaf = false;
    row_shape[0] = (int64_t)info.rows;
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_loss);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_max);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F32, gd_shape_make(1U, row_shape), 256U, &row_inv_sum);
    if (st != GD_OK) {
        return st;
    }
    row_loss.is_leaf = false;
    row_max.is_leaf = false;
    row_inv_sum.is_leaf = false;
    if (!gd_linear_flat_matrix_view_from_tensor(hidden, info.rows, info.k, &hv) ||
        !gd_op_matrix_view_from_tensor(weight, &wv) ||
        !gd_linear_flat_matrix_view_from_tensor(&logits, info.rows, info.n, &lv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid matrix view");
    }
    st = gd_backend_matmul_nt(gd_context_backend(ctx), &hv, &wv, &lv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy logits failed");
    }
    st = gd_lm_cross_entropy_dispatch_loss_stats(ctx, &logits, targets, &row_loss, &row_max, &row_inv_sum);
    if (st != GD_OK) {
        return st;
    }
    inv_rows = 1.0f / (float)info.rows;
    st = gd_reduce_all_forward_impl(ctx, &row_loss, loss, GD_OP_LM_CROSS_ENTROPY, inv_rows);
    if (st != GD_OK) {
        return st;
    }
    st = gd_lm_cross_entropy_free_scratch(ctx, &row_loss);
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_tensor *inputs[3];
        gd_tensor *outputs[1];
        const gd_tensor *saved[3];
        inputs[0] = hidden;
        inputs[1] = weight;
        inputs[2] = targets;
        outputs[0] = loss;
        saved[0] = &logits;
        saved[1] = &row_max;
        saved[2] = &row_inv_sum;
        st = gd_autograd_record(ctx,
                                GD_OP_LM_CROSS_ENTROPY,
                                inputs,
                                3U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                saved,
                                3U);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

gd_status gd_lm_cross_entropy_backward_with_stats(gd_context *ctx,
                                                  const gd_tensor *hidden,
                                                  const gd_tensor *weight,
                                                  const gd_tensor *targets,
                                                  const gd_tensor *logits,
                                                  const gd_tensor *row_max,
                                                  const gd_tensor *row_inv_sum,
                                                  const gd_tensor *grad_out,
                                                  gd_tensor *grad_hidden,
                                                  gd_tensor *grad_weight)
{
    gd_status st;
    gd_linear_shape_info info;
    gd_tensor dlogits;
    gd_tensor dhidden;
    gd_tensor dweight;
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dgv;
    gd_backend_matrix_view hv;
    gd_backend_matrix_view wv;
    gd_backend_matrix_view dlogits_mv;
    gd_backend_matrix_view dhv;
    gd_backend_matrix_view dwv;
    bool need_hidden;
    bool need_weight;
    if (grad_hidden != NULL) {
        memset(grad_hidden, 0, sizeof(*grad_hidden));
    }
    if (grad_weight != NULL) {
        memset(grad_weight, 0, sizeof(*grad_weight));
    }
    memset(&dlogits, 0, sizeof(dlogits));
    memset(&dhidden, 0, sizeof(dhidden));
    memset(&dweight, 0, sizeof(dweight));
    if (ctx == NULL || hidden == NULL || weight == NULL || targets == NULL || logits == NULL ||
        row_max == NULL || row_inv_sum == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    need_hidden = grad_hidden != NULL;
    need_weight = grad_weight != NULL;
    if (!need_hidden && !need_weight) {
        return GD_OK;
    }
    st = gd_linear_transposed_weight_validate_common(ctx, hidden, weight, NULL, &info);
    if (st != GD_OK) {
        return st;
    }
    st = gd_lm_cross_entropy_validate_targets(ctx, targets, info.rows);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, logits);
    if (st != GD_OK) {
        return st;
    }
    if (logits->dtype != GD_DTYPE_F16 || logits->rank != info.rank ||
        !gd_tensor_is_contiguous(logits)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy saved logits invalid");
    }
    for (uint32_t i = 0U; i < info.rank; ++i) {
        if (logits->shape[i] != info.out_shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy saved logits shape mismatch");
        }
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
        row_max->shape[0] != (int64_t)info.rows || row_inv_sum->shape[0] != (int64_t)info.rows ||
        !gd_tensor_is_contiguous(row_max) || !gd_tensor_is_contiguous(row_inv_sum)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy row stats invalid");
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_out->dtype != GD_DTYPE_F32 || grad_out->rank != 0U || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy backward needs scalar f32 grad");
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(info.rank, info.out_shape),
                         256U,
                         &dlogits);
    if (st != GD_OK) {
        return st;
    }
    dlogits.is_leaf = false;
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, logits, &lv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, targets, &tv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_max, &mv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_inv_sum, &iv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, grad_out, &gv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, &dlogits, &dgv));
    st = gd_backend_cross_entropy_backward_stats(gd_context_backend(ctx),
                                                &lv,
                                                &tv,
                                                &mv,
                                                &iv,
                                                &gv,
                                                &dgv,
                                                1.0f / (float)info.rows);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy dlogits failed");
    }
    if (!gd_linear_flat_matrix_view_from_tensor(hidden, info.rows, info.k, &hv) ||
        !gd_op_matrix_view_from_tensor(weight, &wv) ||
        !gd_linear_flat_matrix_view_from_tensor(&dlogits, info.rows, info.n, &dlogits_mv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy backward invalid matrix view");
    }
    if (need_hidden) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(hidden->rank, hidden->shape),
                             256U,
                             &dhidden);
        if (st != GD_OK) {
            return st;
        }
        dhidden.is_leaf = false;
        if (!gd_linear_flat_matrix_view_from_tensor(&dhidden, info.rows, info.k, &dhv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid hidden grad view");
        }
        st = gd_backend_matmul(gd_context_backend(ctx), &dlogits_mv, &wv, &dhv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend lm_cross_entropy hidden grad failed");
        }
        *grad_hidden = dhidden;
    }
    if (need_weight) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(weight->rank, weight->shape),
                             256U,
                             &dweight);
        if (st != GD_OK) {
            return st;
        }
        dweight.is_leaf = false;
        if (!gd_op_matrix_view_from_tensor(&dweight, &dwv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid weight grad view");
        }
        st = gd_backend_matmul_tn(gd_context_backend(ctx), &dlogits_mv, &hv, &dwv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend lm_cross_entropy weight grad failed");
        }
        *grad_weight = dweight;
    }
    st = gd_lm_cross_entropy_free_scratch(ctx, &dlogits);
    if (st != GD_OK) {
        return st;
    }
    return GD_OK;
}
