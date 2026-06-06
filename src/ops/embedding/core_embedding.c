#include "embedding_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static bool gd_embedding_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_embedding_same_shape_dtype(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->dtype != b->dtype || a->rank != b->rank) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_embedding_validate_base(gd_context *ctx,
                                            const gd_tensor *table,
                                            const gd_tensor *ids,
                                            int64_t *out_ids_count,
                                            int64_t *out_vocab,
                                            int64_t *out_dim)
{
    gd_status st;
    int64_t ids_count;
    if (ctx == NULL || table == NULL || ids == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, table);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, ids);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_embedding_dtype_supported(table->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "embedding table must be contiguous f16/f32");
    }
    if (ids->dtype != GD_DTYPE_I32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "embedding ids must be contiguous i32");
    }
    if (table->rank != 2U || table->shape[0] <= 0 || table->shape[1] <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "embedding table must be [vocab, dim]");
    }
    if (ids->rank < 1U || ids->rank + 1U > GD_MAX_DIMS) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "embedding ids rank is out of range");
    }
    if (!gd_tensor_is_contiguous(table) || !gd_tensor_is_contiguous(ids)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "embedding requires contiguous table and ids");
    }
    st = gd_tensor_numel(ids, &ids_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "embedding invalid ids shape");
    }
    if (ids_count <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "embedding ids must be non-empty");
    }
    if (out_ids_count != NULL) {
        *out_ids_count = ids_count;
    }
    if (out_vocab != NULL) {
        *out_vocab = table->shape[0];
    }
    if (out_dim != NULL) {
        *out_dim = table->shape[1];
    }
    return GD_OK;
}

static gd_status gd_embedding_output_shape(gd_context *ctx,
                                           const gd_tensor *table,
                                           const gd_tensor *ids,
                                           int64_t shape[GD_MAX_DIMS])
{
    uint32_t i;
    if (ctx == NULL || table == NULL || ids == NULL || shape == NULL ||
        ids->rank + 1U > GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < GD_MAX_DIMS; ++i) {
        shape[i] = 0;
    }
    for (i = 0U; i < ids->rank; ++i) {
        shape[i] = ids->shape[i];
    }
    shape[ids->rank] = table->shape[1];
    return GD_OK;
}

static gd_status gd_embedding_make_args(gd_context *ctx,
                                        int64_t ids_count,
                                        int64_t vocab,
                                        int64_t dim,
                                        gd_backend_embedding_args *out_args)
{
    uint64_t uids;
    uint64_t uvocab;
    uint64_t udim;
    if (ctx == NULL || out_args == NULL || ids_count <= 0 || vocab <= 0 || dim <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    uids = (uint64_t)ids_count;
    uvocab = (uint64_t)vocab;
    udim = (uint64_t)dim;
    if (uids > UINT64_MAX / udim || uvocab > UINT64_MAX / udim ||
        uids * udim > (uint64_t)UINT32_MAX || uvocab * udim > (uint64_t)UINT32_MAX ||
        udim > (uint64_t)UINT32_MAX || uids > (uint64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "embedding launch dimensions exceed Metal limits");
    }
    memset(out_args, 0, sizeof(*out_args));
    out_args->ids_count = uids;
    out_args->vocab = uvocab;
    out_args->dim = udim;
    return GD_OK;
}

static gd_status gd_embedding_dispatch_forward(gd_context *ctx,
                                               const gd_tensor *table,
                                               const gd_tensor *ids,
                                               const gd_tensor *out,
                                               const gd_backend_embedding_args *args)
{
    gd_backend_tensor_view tv;
    gd_backend_tensor_view iv;
    gd_backend_tensor_view ov;
    gd_status st;
    if (ctx == NULL || table == NULL || ids == NULL || out == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(table, &tv) ||
        !gd_op_tensor_view_from_tensor(ids, &iv) ||
        !gd_op_tensor_view_from_tensor(out, &ov)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "embedding forward invalid tensor view");
    }
    st = gd_backend_embedding_forward(gd_context_backend(ctx), &tv, &iv, &ov, args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend embedding forward failed");
    }
    return GD_OK;
}

static gd_status gd_embedding_dispatch_backward(gd_context *ctx,
                                                const gd_tensor *ids,
                                                const gd_tensor *grad_out,
                                                const gd_tensor *grad_table,
                                                const gd_tensor *scratch,
                                                const gd_backend_embedding_args *args)
{
    gd_backend_tensor_view iv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dwv;
    gd_backend_tensor_view sv;
    gd_status st;
    if (ctx == NULL || ids == NULL || grad_out == NULL || grad_table == NULL || args == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_op_tensor_view_from_tensor(ids, &iv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) ||
        !gd_op_tensor_view_from_tensor(grad_table, &dwv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "embedding backward invalid tensor view");
    }
    if (scratch != NULL) {
        if (!gd_op_tensor_view_from_tensor(scratch, &sv)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "embedding backward invalid scratch view");
        }
        st = gd_backend_embedding_backward(gd_context_backend(ctx), &gv, &iv, &dwv, &sv, args);
    } else {
        st = gd_backend_embedding_backward(gd_context_backend(ctx), &gv, &iv, &dwv, NULL, args);
    }
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend embedding backward failed");
    }
    return GD_OK;
}

gd_status gd_embedding(gd_context *ctx,
                       const gd_tensor *table,
                       const gd_tensor *ids,
                       gd_tensor *out)
{
    gd_status st;
    gd_tensor y;
    gd_backend_embedding_args args;
    int64_t ids_count;
    int64_t vocab;
    int64_t dim;
    int64_t out_shape[GD_MAX_DIMS];
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    memset(&y, 0, sizeof(y));
    if (ctx == NULL || table == NULL || ids == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding_validate_base(ctx, table, ids, &ids_count, &vocab, &dim);
    if (st != GD_OK) {
        return st;
    }
    st = gd_embedding_make_args(ctx, ids_count, vocab, dim, &args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_embedding_output_shape(ctx, table, ids, out_shape);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         table->dtype,
                         gd_shape_make(ids->rank + 1U, out_shape),
                         256U,
                         &y);
    if (st != GD_OK) {
        return st;
    }
    y.is_leaf = false;
    st = gd_embedding_dispatch_forward(ctx, table, ids, &y, &args);
    if (st != GD_OK) {
        return st;
    }
    {
        const gd_tensor *inputs[2];
        gd_tensor *outputs[1];
        inputs[0] = table;
        inputs[1] = ids;
        outputs[0] = &y;
        st = gd_autograd_record(ctx,
                                GD_OP_EMBEDDING,
                                inputs,
                                2U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_embedding_backward_impl(gd_context *ctx,
                                     const gd_tensor *table,
                                     const gd_tensor *ids,
                                     const gd_tensor *grad_out,
                                     gd_tensor *grad_table)
{
    gd_status st;
    gd_tensor dtable;
    gd_tensor scratch;
    gd_backend_embedding_args args;
    int64_t ids_count;
    int64_t vocab;
    int64_t dim;
    int64_t expected_shape[GD_MAX_DIMS];
    if (grad_table != NULL) {
        memset(grad_table, 0, sizeof(*grad_table));
    }
    memset(&dtable, 0, sizeof(dtable));
    memset(&scratch, 0, sizeof(scratch));
    if (ctx == NULL || table == NULL || ids == NULL || grad_out == NULL || grad_table == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding_validate_base(ctx, table, ids, &ids_count, &vocab, &dim);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_embedding_output_shape(ctx, table, ids, expected_shape);
    if (st != GD_OK) {
        return st;
    }
    {
        gd_tensor expected;
        memset(&expected, 0, sizeof(expected));
        expected.dtype = table->dtype;
        expected.device = GD_DEVICE_GPU;
        expected.layout = GD_LAYOUT_STRIDED;
        expected.rank = ids->rank + 1U;
        memcpy(expected.shape, expected_shape, sizeof(expected.shape));
        if (!gd_embedding_same_shape_dtype(&expected, grad_out) ||
            !gd_tensor_is_contiguous(grad_out)) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "embedding backward grad_out shape/dtype mismatch");
        }
    }
    st = gd_embedding_make_args(ctx, ids_count, vocab, dim, &args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_SCRATCH,
                         table->dtype,
                         gd_shape_make(table->rank, table->shape),
                         256U,
                         &dtable);
    if (st != GD_OK) {
        return st;
    }
    dtable.is_leaf = false;
    if (table->dtype == GD_DTYPE_F16) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_SCRATCH,
                             GD_DTYPE_F32,
                             gd_shape_make(table->rank, table->shape),
                             256U,
                             &scratch);
        if (st != GD_OK) {
            return st;
        }
        scratch.is_leaf = false;
        st = gd_embedding_dispatch_backward(ctx, ids, grad_out, &dtable, &scratch, &args);
    } else {
        st = gd_embedding_dispatch_backward(ctx, ids, grad_out, &dtable, NULL, &args);
    }
    if (st != GD_OK) {
        return st;
    }
    *grad_table = dtable;
    return GD_OK;
}

gd_status gd_embedding_backward(gd_context *ctx,
                                const gd_tensor *table,
                                const gd_tensor *ids,
                                const gd_tensor *grad_out,
                                gd_tensor *grad_table)
{
    return gd_embedding_backward_impl(ctx, table, ids, grad_out, grad_table);
}
