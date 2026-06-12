#include "qkv_split_rope_impl.h"

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#define GD_QKV_SPLIT_ROPE_DEFAULT_THETA 10000.0f

static bool gd_qkv_split_rope_same_shape(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->rank != b->rank || a->dtype != b->dtype) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_qkv_split_rope_resolve_config(gd_context *ctx,
                                                   int32_t head_dim,
                                                   const gd_rope_config *config,
                                                   gd_backend_rope_args *backend_args)
{
    float theta;
    int32_t n_dims_i32;
    if (ctx == NULL || backend_args == NULL || head_dim <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    theta = (config != NULL && config->theta > 0.0f) ? config->theta : GD_QKV_SPLIT_ROPE_DEFAULT_THETA;
    n_dims_i32 = (config != NULL && config->n_dims > 0) ? config->n_dims : head_dim;
    if (!isfinite(theta) || theta <= 0.0f || n_dims_i32 <= 0 || (n_dims_i32 & 1) != 0 ||
        n_dims_i32 > head_dim) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope invalid RoPE config");
    }
    if (n_dims_i32 != head_dim) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "qkv_split_rope fast path requires full-head RoPE");
    }
    memset(backend_args, 0, sizeof(*backend_args));
    backend_args->theta = theta;
    backend_args->n_dims = (uint32_t)n_dims_i32;
    backend_args->interleaved = (config != NULL && config->interleaved) ? 1U : 0U;
    return GD_OK;
}

static gd_status gd_qkv_split_rope_validate_qkv(gd_context *ctx,
                                                const gd_tensor *qkv,
                                                const gd_tensor *pos_ids,
                                                int32_t n_heads,
                                                int32_t head_dim,
                                                int64_t *out_tokens,
                                                int64_t *out_width)
{
    gd_status st;
    int64_t width;
    if (ctx == NULL || qkv == NULL || pos_ids == NULL || out_tokens == NULL || out_width == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_tokens = 0;
    *out_width = 0;
    st = gd_tensor_validate(ctx, qkv);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, pos_ids);
    if (st != GD_OK) {
        return st;
    }
    if (qkv->dtype != GD_DTYPE_F16 || qkv->rank != 2U || qkv->shape[0] <= 0 ||
        qkv->shape[1] <= 0 || !gd_tensor_is_contiguous(qkv)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_INVALID_ARGUMENT,
                                    "qkv_split_rope expects contiguous f16 qkv [N,3*H*Dh]");
    }
    if (pos_ids->dtype != GD_DTYPE_I32 || pos_ids->rank != 1U || pos_ids->shape[0] != qkv->shape[0] ||
        !gd_tensor_is_contiguous(pos_ids)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope expects i32 pos_ids [N]");
    }
    if (n_heads <= 0 || head_dim <= 0) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope invalid head shape");
    }
    if ((int64_t)n_heads > INT64_MAX / (int64_t)head_dim) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "qkv_split_rope head shape overflow");
    }
    width = (int64_t)n_heads * (int64_t)head_dim;
    if (width <= 0 || width > INT64_MAX / 3 || qkv->shape[1] != 3 * width) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope qkv width mismatch");
    }
    *out_tokens = qkv->shape[0];
    *out_width = width;
    return GD_OK;
}

static gd_status gd_qkv_split_rope_validate_output(gd_context *ctx,
                                                   const gd_tensor *qkv,
                                                   const gd_tensor *out,
                                                   int32_t n_heads,
                                                   int32_t head_dim,
                                                   const char *name)
{
    gd_status st;
    if (ctx == NULL || qkv == NULL || out == NULL || name == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, out);
    if (st != GD_OK) {
        return st;
    }
    if (out->dtype != qkv->dtype || out->rank != 3U || out->shape[0] != qkv->shape[0] ||
        out->shape[1] != (int64_t)n_heads || out->shape[2] != (int64_t)head_dim ||
        !gd_tensor_is_contiguous(out)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, name);
    }
    return GD_OK;
}

static gd_status gd_qkv_split_rope_dispatch_forward(gd_context *ctx,
                                                    const gd_tensor *qkv,
                                                    const gd_tensor *pos_ids,
                                                    gd_tensor *q,
                                                    gd_tensor *k,
                                                    gd_tensor *v,
                                                    uint32_t n_heads,
                                                    uint32_t head_dim,
                                                    const gd_backend_rope_args *args)
{
    gd_backend_tensor_view qkvv;
    gd_backend_tensor_view posv;
    gd_backend_tensor_view qv;
    gd_backend_tensor_view kv;
    gd_backend_tensor_view vv;
    gd_status st;
    if (!gd_op_tensor_view_from_tensor(qkv, &qkvv) || !gd_op_tensor_view_from_tensor(pos_ids, &posv) ||
        !gd_op_tensor_view_from_tensor(q, &qv) || !gd_op_tensor_view_from_tensor(k, &kv) ||
        !gd_op_tensor_view_from_tensor(v, &vv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope invalid tensor view");
    }
    st = gd_backend_qkv_split_rope_forward(gd_context_backend(ctx),
                                           &qkvv,
                                           &posv,
                                           &qv,
                                           &kv,
                                           &vv,
                                           n_heads,
                                           head_dim,
                                           args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend qkv_split_rope forward failed");
    }
    return GD_OK;
}

static gd_status gd_qkv_split_rope_dispatch_backward(gd_context *ctx,
                                                     const gd_tensor *grad_q,
                                                     const gd_tensor *grad_k,
                                                     const gd_tensor *grad_v,
                                                     const gd_tensor *pos_ids,
                                                     gd_tensor *grad_qkv,
                                                     uint32_t n_heads,
                                                     uint32_t head_dim,
                                                     const gd_backend_rope_args *args)
{
    gd_backend_tensor_view gqv;
    gd_backend_tensor_view gkv;
    gd_backend_tensor_view gvv;
    gd_backend_tensor_view posv;
    gd_backend_tensor_view dqkvv;
    gd_status st;
    if (!gd_op_tensor_view_from_tensor(grad_q, &gqv) || !gd_op_tensor_view_from_tensor(grad_k, &gkv) ||
        !gd_op_tensor_view_from_tensor(grad_v, &gvv) || !gd_op_tensor_view_from_tensor(pos_ids, &posv) ||
        !gd_op_tensor_view_from_tensor(grad_qkv, &dqkvv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope backward invalid tensor view");
    }
    st = gd_backend_qkv_split_rope_backward(gd_context_backend(ctx),
                                            &gqv,
                                            &gkv,
                                            &gvv,
                                            &posv,
                                            &dqkvv,
                                            n_heads,
                                            head_dim,
                                            args);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend qkv_split_rope backward failed");
    }
    return GD_OK;
}

gd_status gd_qkv_split_rope(gd_context *ctx,
                            const gd_tensor *qkv,
                            const gd_tensor *pos_ids,
                            int32_t n_heads,
                            int32_t head_dim,
                            const gd_rope_config *config,
                            gd_tensor *q,
                            gd_tensor *k,
                            gd_tensor *v)
{
    gd_status st;
    gd_backend_rope_args backend_args;
    int64_t tokens;
    int64_t width;
    int64_t out_shape[3];
    gd_tensor q_out;
    gd_tensor k_out;
    gd_tensor v_out;
    if (q != NULL) {
        memset(q, 0, sizeof(*q));
    }
    if (k != NULL) {
        memset(k, 0, sizeof(*k));
    }
    if (v != NULL) {
        memset(v, 0, sizeof(*v));
    }
    if (ctx == NULL || qkv == NULL || pos_ids == NULL || q == NULL || k == NULL || v == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_qkv_split_rope_validate_qkv(ctx, qkv, pos_ids, n_heads, head_dim, &tokens, &width);
    if (st != GD_OK) {
        return st;
    }
    st = gd_qkv_split_rope_resolve_config(ctx, head_dim, config, &backend_args);
    if (st != GD_OK) {
        return st;
    }
    (void)width;
    out_shape[0] = tokens;
    out_shape[1] = (int64_t)n_heads;
    out_shape[2] = (int64_t)head_dim;
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, out_shape), 256U, &q_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, out_shape), 256U, &k_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(3U, out_shape), 256U, &v_out);
    if (st != GD_OK) {
        return st;
    }
    q_out.is_leaf = false;
    k_out.is_leaf = false;
    v_out.is_leaf = false;
    st = gd_qkv_split_rope_dispatch_forward(ctx,
                                            qkv,
                                            pos_ids,
                                            &q_out,
                                            &k_out,
                                            &v_out,
                                            (uint32_t)n_heads,
                                            (uint32_t)head_dim,
                                            &backend_args);
    if (st != GD_OK) {
        return st;
    }
    if (qkv->requires_grad) {
        gd_qkv_split_rope_attrs attrs;
        const gd_tensor *inputs[2];
        gd_tensor *outputs[3];
        attrs.rope.theta = backend_args.theta;
        attrs.rope.n_dims = (int32_t)backend_args.n_dims;
        attrs.rope.interleaved = backend_args.interleaved != 0U;
        attrs.n_heads = n_heads;
        attrs.head_dim = head_dim;
        inputs[0] = qkv;
        inputs[1] = pos_ids;
        outputs[0] = &q_out;
        outputs[1] = &k_out;
        outputs[2] = &v_out;
        st = gd_autograd_record(ctx,
                                GD_OP_QKV_SPLIT_ROPE,
                                inputs,
                                2U,
                                outputs,
                                3U,
                                &attrs,
                                (uint32_t)sizeof(attrs),
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *q = q_out;
    *k = k_out;
    *v = v_out;
    return GD_OK;
}

gd_status gd_qkv_split_rope_backward(gd_context *ctx,
                                      const gd_tensor *qkv,
                                      const gd_tensor *pos_ids,
                                      const gd_tensor *grad_q,
                                      const gd_tensor *grad_k,
                                      const gd_tensor *grad_v,
                                      int32_t n_heads,
                                      int32_t head_dim,
                                      const gd_rope_config *config,
                                      gd_tensor *grad_qkv)
{
    gd_status st;
    gd_backend_rope_args backend_args;
    int64_t tokens;
    int64_t width;
    gd_tensor dqkv;
    if (grad_qkv != NULL) {
        memset(grad_qkv, 0, sizeof(*grad_qkv));
    }
    if (ctx == NULL || qkv == NULL || pos_ids == NULL || grad_q == NULL || grad_k == NULL ||
        grad_v == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_qkv_split_rope_validate_qkv(ctx, qkv, pos_ids, n_heads, head_dim, &tokens, &width);
    if (st != GD_OK) {
        return st;
    }
    st = gd_qkv_split_rope_resolve_config(ctx, head_dim, config, &backend_args);
    if (st != GD_OK) {
        return st;
    }
    st = gd_qkv_split_rope_validate_output(ctx,
                                           qkv,
                                           grad_q,
                                           n_heads,
                                           head_dim,
                                           "qkv_split_rope grad_q shape mismatch");
    if (st != GD_OK) {
        return st;
    }
    st = gd_qkv_split_rope_validate_output(ctx,
                                           qkv,
                                           grad_k,
                                           n_heads,
                                           head_dim,
                                           "qkv_split_rope grad_k shape mismatch");
    if (st != GD_OK) {
        return st;
    }
    st = gd_qkv_split_rope_validate_output(ctx,
                                           qkv,
                                           grad_v,
                                           n_heads,
                                           head_dim,
                                           "qkv_split_rope grad_v shape mismatch");
    if (st != GD_OK) {
        return st;
    }
    if (!gd_qkv_split_rope_same_shape(grad_q, grad_k) || !gd_qkv_split_rope_same_shape(grad_q, grad_v)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "qkv_split_rope grad shape mismatch");
    }
    if (grad_qkv == NULL) {
        return GD_OK;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, GD_DTYPE_F16, gd_shape_make(qkv->rank, qkv->shape), 256U, &dqkv);
    if (st != GD_OK) {
        return st;
    }
    dqkv.is_leaf = false;
    dqkv.requires_grad = false;
    (void)tokens;
    (void)width;
    st = gd_qkv_split_rope_dispatch_backward(ctx,
                                             grad_q,
                                             grad_k,
                                             grad_v,
                                             pos_ids,
                                             &dqkv,
                                             (uint32_t)n_heads,
                                             (uint32_t)head_dim,
                                             &backend_args);
    if (st != GD_OK) {
        return st;
    }
    *grad_qkv = dqkv;
    return GD_OK;
}
