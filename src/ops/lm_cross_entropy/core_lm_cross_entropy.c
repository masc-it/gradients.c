#include "lm_cross_entropy_impl.h"

#include "../_shared/reduce/reduce_core.h"
#include "../linear/linear_impl.h"
#include "../op_common.h"
#include "../../core/memory_internal.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define GD_LM_CROSS_ENTROPY_DEFAULT_CHUNK_CLASSES 1024ULL
#define GD_LM_CROSS_ENTROPY_F32_NEG_INF_PATTERN 0xff800000U

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

static gd_status gd_lm_cross_entropy_validate_softcap(gd_context *ctx, float logits_softcap)
{
    if (logits_softcap == 0.0f) {
        return GD_OK;
    }
    if (!isfinite(logits_softcap) || logits_softcap < 0.0f || !isfinite(1.0f / logits_softcap)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "lm_cross_entropy logits_softcap must be zero or finite positive with finite reciprocal");
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

static uint64_t gd_lm_cross_entropy_chunk_classes(int64_t classes)
{
    const char *env = getenv("GD_LM_CE_CHUNK_SIZE");
    uint64_t chunk = GD_LM_CROSS_ENTROPY_DEFAULT_CHUNK_CLASSES;
    if (env != NULL && env[0] != '\0') {
        char *end = NULL;
        unsigned long long value = strtoull(env, &end, 10);
        if (end != env && *end == '\0' && value != 0ULL) {
            chunk = (uint64_t)value;
        }
    }
    if (classes > 0 && chunk > (uint64_t)classes) {
        chunk = (uint64_t)classes;
    }
    return chunk == 0U ? 1U : chunk;
}

static gd_status gd_lm_cross_entropy_fill_tensor_pattern(gd_context *ctx,
                                                         gd_tensor *tensor,
                                                         uint32_t pattern)
{
    gd_status st;
    int64_t numel;
    size_t count;
    size_t elem_size;
    size_t offset;
    gd_backend *backend;
    if (ctx == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "lm_cross_entropy fill requires contiguous tensor");
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK || numel < 0 || (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, st != GD_OK ? st : GD_ERR_OUT_OF_MEMORY,
                                    "lm_cross_entropy fill invalid shape");
    }
    count = (size_t)numel;
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U || tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy fill invalid descriptor");
    }
    offset = tensor->storage.offset + tensor->view_offset;
    backend = gd_context_backend(ctx);
    if (backend == NULL || tensor->storage.buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "lm_cross_entropy fill missing backend buffer");
    }
    st = gd_backend_fill(backend,
                         (gd_backend_buffer *)tensor->storage.buffer,
                         offset,
                         count,
                         elem_size,
                         pattern);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy fill failed");
    }
    tensor->version += 1U;
    if (tensor->version == 0U) {
        tensor->version = 1U;
    }
    return GD_OK;
}

static bool gd_lm_cross_entropy_weight_chunk_view(const gd_tensor *tensor,
                                                  int64_t rows,
                                                  int64_t cols,
                                                  uint64_t class_start,
                                                  uint64_t chunk_classes,
                                                  gd_backend_matrix_view *out)
{
    gd_backend_matrix_view base;
    size_t delta;
    if (tensor == NULL || out == NULL || rows <= 0 || cols <= 0 ||
        !gd_op_matrix_view_from_tensor(tensor, &base) ||
        base.rows != (uint32_t)rows || base.cols != (uint32_t)cols ||
        class_start > (uint64_t)base.rows || chunk_classes == 0U ||
        chunk_classes > (uint64_t)base.rows - class_start ||
        chunk_classes > (uint64_t)UINT32_MAX ||
        (base.row_bytes != 0U && class_start > (uint64_t)(SIZE_MAX / base.row_bytes))) {
        return false;
    }
    delta = (size_t)class_start * base.row_bytes;
    if (base.offset > SIZE_MAX - delta) {
        return false;
    }
    *out = base;
    out->offset = base.offset + delta;
    out->rows = (uint32_t)chunk_classes;
    return true;
}

static gd_status gd_lm_cross_entropy_make_logits_chunk(gd_context *ctx,
                                                       const gd_linear_shape_info *info,
                                                       uint64_t chunk_classes,
                                                       gd_tensor *chunk,
                                                       gd_backend_matrix_view *chunk_view)
{
    int64_t chunk_shape[2];
    gd_status st;
    if (ctx == NULL || info == NULL || chunk == NULL || chunk_view == NULL ||
        chunk_classes == 0U || chunk_classes > (uint64_t)INT64_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    chunk_shape[0] = info->rows;
    chunk_shape[1] = (int64_t)chunk_classes;
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         GD_DTYPE_F16,
                         gd_shape_make(2U, chunk_shape),
                         256U,
                         chunk);
    if (st != GD_OK) {
        return st;
    }
    chunk->is_leaf = false;
    if (!gd_linear_flat_matrix_view_from_tensor(chunk, info->rows, (int64_t)chunk_classes, chunk_view)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid logits chunk view");
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_dispatch_online_update(gd_context *ctx,
                                                            const gd_tensor *logits_chunk,
                                                            const gd_tensor *targets,
                                                            const gd_tensor *row_loss,
                                                            const gd_tensor *row_max,
                                                            const gd_tensor *row_inv_sum,
                                                            uint64_t class_start,
                                                            uint64_t total_classes,
                                                            float logits_softcap)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view rv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_status st;
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, logits_chunk, &lv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, targets, &tv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_loss, &rv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_max, &mv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_inv_sum, &iv));
    st = gd_backend_lm_cross_entropy_online_update(gd_context_backend(ctx),
                                                   &lv,
                                                   &tv,
                                                   &rv,
                                                   &mv,
                                                   &iv,
                                                   class_start,
                                                   total_classes,
                                                   logits_softcap);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy online update failed");
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_dispatch_finalize(gd_context *ctx,
                                                       const gd_tensor *targets,
                                                       const gd_tensor *row_loss,
                                                       const gd_tensor *row_max,
                                                       const gd_tensor *row_inv_sum,
                                                       uint64_t total_classes)
{
    gd_backend_tensor_view tv;
    gd_backend_tensor_view rv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_status st;
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, targets, &tv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_loss, &rv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_max, &mv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_inv_sum, &iv));
    st = gd_backend_lm_cross_entropy_finalize(gd_context_backend(ctx),
                                              &tv,
                                              &rv,
                                              &mv,
                                              &iv,
                                              total_classes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy finalize failed");
    }
    return GD_OK;
}

static gd_status gd_lm_cross_entropy_dispatch_backward_chunk(gd_context *ctx,
                                                             const gd_tensor *logits_chunk,
                                                             const gd_tensor *targets,
                                                             const gd_tensor *row_max,
                                                             const gd_tensor *row_inv_sum,
                                                             const gd_tensor *grad_out,
                                                             const gd_tensor *grad_logits_chunk,
                                                             uint64_t class_start,
                                                             uint64_t total_classes,
                                                             float scale,
                                                             float logits_softcap)
{
    gd_backend_tensor_view lv;
    gd_backend_tensor_view tv;
    gd_backend_tensor_view mv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dgv;
    gd_status st;
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, logits_chunk, &lv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, targets, &tv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_max, &mv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, row_inv_sum, &iv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, grad_out, &gv));
    GD_TRY(gd_lm_cross_entropy_tensor_view(ctx, grad_logits_chunk, &dgv));
    st = gd_backend_lm_cross_entropy_backward_chunk(gd_context_backend(ctx),
                                                    &lv,
                                                    &tv,
                                                    &mv,
                                                    &iv,
                                                    &gv,
                                                    &dgv,
                                                    class_start,
                                                    total_classes,
                                                    scale,
                                                    logits_softcap);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend lm_cross_entropy backward chunk failed");
    }
    return GD_OK;
}

gd_status gd_lm_cross_entropy(gd_context *ctx,
                              const gd_tensor *hidden,
                              const gd_tensor *weight,
                              const gd_tensor *targets,
                              gd_tensor *loss)
{
    return gd_lm_cross_entropy_softcapped(ctx, hidden, weight, targets, 0.0f, loss);
}

gd_status gd_lm_cross_entropy_softcapped(gd_context *ctx,
                                         const gd_tensor *hidden,
                                         const gd_tensor *weight,
                                         const gd_tensor *targets,
                                         float logits_softcap,
                                         gd_tensor *loss)
{
    gd_status st;
    gd_linear_shape_info info;
    gd_tensor row_loss;
    gd_tensor row_max;
    gd_tensor row_inv_sum;
    gd_backend_matrix_view hv;
    int64_t row_shape[1];
    uint64_t chunk_size;
    uint64_t class_start;
    float inv_rows;
    bool needs_grad;
    if (loss != NULL) {
        memset(loss, 0, sizeof(*loss));
    }
    memset(&row_loss, 0, sizeof(row_loss));
    memset(&row_max, 0, sizeof(row_max));
    memset(&row_inv_sum, 0, sizeof(row_inv_sum));
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
    st = gd_lm_cross_entropy_validate_softcap(ctx, logits_softcap);
    if (st != GD_OK) {
        return st;
    }
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
    GD_TRY(gd_tensor_zero_(ctx, &row_loss));
    GD_TRY(gd_lm_cross_entropy_fill_tensor_pattern(ctx,
                                                   &row_max,
                                                   GD_LM_CROSS_ENTROPY_F32_NEG_INF_PATTERN));
    GD_TRY(gd_tensor_zero_(ctx, &row_inv_sum));
    if (!gd_linear_flat_matrix_view_from_tensor(hidden, info.rows, info.k, &hv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid hidden matrix view");
    }
    chunk_size = gd_lm_cross_entropy_chunk_classes(info.n);
    for (class_start = 0U; class_start < (uint64_t)info.n; class_start += chunk_size) {
        const uint64_t remaining = (uint64_t)info.n - class_start;
        const uint64_t chunk_classes = remaining < chunk_size ? remaining : chunk_size;
        gd_tensor logits_chunk;
        gd_backend_matrix_view wv;
        gd_backend_matrix_view lv;
        st = gd_lm_cross_entropy_make_logits_chunk(ctx, &info, chunk_classes, &logits_chunk, &lv);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_lm_cross_entropy_weight_chunk_view(weight,
                                                   info.n,
                                                   info.k,
                                                   class_start,
                                                   chunk_classes,
                                                   &wv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid weight chunk view");
        }
        st = gd_backend_matmul_nt(gd_context_backend(ctx), &hv, &wv, &lv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend lm_cross_entropy logits chunk failed");
        }
        st = gd_lm_cross_entropy_dispatch_online_update(ctx,
                                                        &logits_chunk,
                                                        targets,
                                                        &row_loss,
                                                        &row_max,
                                                        &row_inv_sum,
                                                        class_start,
                                                        (uint64_t)info.n,
                                                        logits_softcap);
        if (st != GD_OK) {
            return st;
        }
        st = gd_lm_cross_entropy_free_scratch(ctx, &logits_chunk);
        if (st != GD_OK) {
            return st;
        }
    }
    st = gd_lm_cross_entropy_dispatch_finalize(ctx, targets, &row_loss, &row_max, &row_inv_sum, (uint64_t)info.n);
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
    needs_grad = hidden->requires_grad || weight->requires_grad;
    if (needs_grad) {
        const gd_tensor *inputs[3];
        gd_tensor *outputs[1];
        const gd_tensor *saved[2];
        const gd_lm_cross_entropy_attrs attrs = {
            .logits_softcap = logits_softcap,
        };
        inputs[0] = hidden;
        inputs[1] = weight;
        inputs[2] = targets;
        outputs[0] = loss;
        saved[0] = &row_max;
        saved[1] = &row_inv_sum;
        st = gd_autograd_record(ctx,
                                GD_OP_LM_CROSS_ENTROPY,
                                inputs,
                                3U,
                                outputs,
                                1U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                saved,
                                2U);
        if (st != GD_OK) {
            return st;
        }
    } else {
        st = gd_lm_cross_entropy_free_scratch(ctx, &row_max);
        if (st != GD_OK) {
            return st;
        }
        st = gd_lm_cross_entropy_free_scratch(ctx, &row_inv_sum);
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
                                                  const gd_tensor *row_max,
                                                  const gd_tensor *row_inv_sum,
                                                  float logits_softcap,
                                                  const gd_tensor *grad_out,
                                                  gd_tensor *grad_hidden,
                                                  gd_tensor *grad_weight)
{
    gd_status st;
    gd_linear_shape_info info;
    gd_tensor dhidden;
    gd_tensor dweight;
    gd_tensor partial_dhidden;
    gd_backend_matrix_view hv;
    bool need_hidden;
    bool need_weight;
    uint64_t chunk_size;
    uint64_t class_start;
    if (grad_hidden != NULL) {
        memset(grad_hidden, 0, sizeof(*grad_hidden));
    }
    if (grad_weight != NULL) {
        memset(grad_weight, 0, sizeof(*grad_weight));
    }
    memset(&dhidden, 0, sizeof(dhidden));
    memset(&dweight, 0, sizeof(dweight));
    memset(&partial_dhidden, 0, sizeof(partial_dhidden));
    if (ctx == NULL || hidden == NULL || weight == NULL || targets == NULL ||
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
    st = gd_lm_cross_entropy_validate_softcap(ctx, logits_softcap);
    if (st != GD_OK) {
        return st;
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
    if (!gd_linear_flat_matrix_view_from_tensor(hidden, info.rows, info.k, &hv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy backward invalid hidden view");
    }
    if (need_hidden) {
        st = gd_tensor_zeros(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(hidden->rank, hidden->shape),
                             256U,
                             &dhidden);
        if (st != GD_OK) {
            return st;
        }
        dhidden.is_leaf = false;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F16,
                             gd_shape_make(hidden->rank, hidden->shape),
                             256U,
                             &partial_dhidden);
        if (st != GD_OK) {
            return st;
        }
        partial_dhidden.is_leaf = false;
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
    }
    chunk_size = gd_lm_cross_entropy_chunk_classes(info.n);
    for (class_start = 0U; class_start < (uint64_t)info.n; class_start += chunk_size) {
        const uint64_t remaining = (uint64_t)info.n - class_start;
        const uint64_t chunk_classes = remaining < chunk_size ? remaining : chunk_size;
        gd_tensor logits_chunk;
        gd_backend_matrix_view wv;
        gd_backend_matrix_view lv;
        st = gd_lm_cross_entropy_make_logits_chunk(ctx, &info, chunk_classes, &logits_chunk, &lv);
        if (st != GD_OK) {
            return st;
        }
        if (!gd_lm_cross_entropy_weight_chunk_view(weight,
                                                   info.n,
                                                   info.k,
                                                   class_start,
                                                   chunk_classes,
                                                   &wv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid weight chunk view");
        }
        st = gd_backend_matmul_nt(gd_context_backend(ctx), &hv, &wv, &lv);
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend lm_cross_entropy backward logits chunk failed");
        }
        st = gd_lm_cross_entropy_dispatch_backward_chunk(ctx,
                                                         &logits_chunk,
                                                         targets,
                                                         row_max,
                                                         row_inv_sum,
                                                         grad_out,
                                                         &logits_chunk,
                                                         class_start,
                                                         (uint64_t)info.n,
                                                         1.0f / (float)info.rows,
                                                         logits_softcap);
        if (st != GD_OK) {
            return st;
        }
        if (need_hidden) {
            gd_backend_matrix_view dhv;
            gd_backend_matrix_view phv;
            if (!gd_linear_flat_matrix_view_from_tensor(&dhidden, info.rows, info.k, &dhv) ||
                !gd_linear_flat_matrix_view_from_tensor(&partial_dhidden, info.rows, info.k, &phv)) {
                return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid hidden grad view");
            }
            st = gd_backend_matmul(gd_context_backend(ctx), &lv, &wv, &phv);
            if (st != GD_OK) {
                return gd_context_set_error(ctx, st, "backend lm_cross_entropy hidden grad chunk failed");
            }
            st = gd_backend_accumulate(gd_context_backend(ctx),
                                       (gd_backend_buffer *)dhidden.storage.buffer,
                                       dhidden.storage.offset + dhidden.view_offset,
                                       (gd_backend_buffer *)partial_dhidden.storage.buffer,
                                       partial_dhidden.storage.offset + partial_dhidden.view_offset,
                                       (size_t)info.rows * (size_t)info.k,
                                       (uint32_t)GD_DTYPE_F16);
            if (st != GD_OK) {
                return gd_context_set_error(ctx, st, "backend lm_cross_entropy hidden grad accumulate failed");
            }
        }
        if (need_weight) {
            gd_backend_matrix_view dwv;
            if (!gd_lm_cross_entropy_weight_chunk_view(&dweight,
                                                       info.n,
                                                       info.k,
                                                       class_start,
                                                       chunk_classes,
                                                       &dwv)) {
                return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy invalid weight grad view");
            }
            st = gd_backend_matmul_tn(gd_context_backend(ctx), &lv, &hv, &dwv);
            if (st != GD_OK) {
                return gd_context_set_error(ctx, st, "backend lm_cross_entropy weight grad chunk failed");
            }
        }
        st = gd_lm_cross_entropy_free_scratch(ctx, &logits_chunk);
        if (st != GD_OK) {
            return st;
        }
    }
    if (need_hidden) {
        st = gd_lm_cross_entropy_free_scratch(ctx, &partial_dhidden);
        if (st != GD_OK) {
            return st;
        }
        *grad_hidden = dhidden;
    }
    if (need_weight) {
        *grad_weight = dweight;
    }
    return GD_OK;
}
