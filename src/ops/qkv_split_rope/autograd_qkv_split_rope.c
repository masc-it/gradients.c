#include "qkv_split_rope_impl.h"

#include "../autograd_impl.h"

#include <stdint.h>

static gd_status gd_qkv_split_rope_zero_like(gd_context *ctx, const gd_tensor *like, gd_tensor *out)
{
    gd_status st;
    if (ctx == NULL || like == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_zeros(ctx,
                         GD_ARENA_SCRATCH,
                         like->dtype,
                         gd_shape_make(like->rank, like->shape),
                         256U,
                         out);
    if (st != GD_OK) {
        return st;
    }
    out->requires_grad = false;
    out->is_leaf = false;
    return GD_OK;
}

static gd_status gd_qkv_split_rope_autograd_backward(gd_bwd_ctx *bwd,
                                                     const gd_tape_node *node)
{
    const gd_qkv_split_rope_attrs *attrs;
    const gd_tensor *qkv;
    const gd_tensor *pos_ids;
    const gd_tensor *q;
    const gd_tensor *k;
    const gd_tensor *v;
    gd_tensor grad_q;
    gd_tensor grad_k;
    gd_tensor grad_v;
    gd_tensor zero_q;
    gd_tensor zero_k;
    gd_tensor zero_v;
    gd_tensor grad_qkv;
    bool has_q;
    bool has_k;
    bool has_v;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    attrs = (const gd_qkv_split_rope_attrs *)gd_tape_attrs(bwd->tape, node, (uint32_t)sizeof(*attrs));
    qkv = gd_tape_input(bwd->tape, node, 0U);
    pos_ids = gd_tape_input(bwd->tape, node, 1U);
    q = gd_tape_output(bwd->tape, node, 0U);
    k = gd_tape_output(bwd->tape, node, 1U);
    v = gd_tape_output(bwd->tape, node, 2U);
    if (attrs == NULL || qkv == NULL || pos_ids == NULL || q == NULL || k == NULL || v == NULL ||
        node->n_inputs != 2U || node->n_outputs != 3U) {
        return GD_ERR_INTERNAL;
    }
    if (!qkv->requires_grad) {
        return GD_OK;
    }
    has_q = gd_autograd_get_grad(bwd, q->id, &grad_q);
    has_k = gd_autograd_get_grad(bwd, k->id, &grad_k);
    has_v = gd_autograd_get_grad(bwd, v->id, &grad_v);
    if (!has_q && !has_k && !has_v) {
        return GD_OK;
    }
    if (!has_q) {
        GD_TRY(gd_qkv_split_rope_zero_like(bwd->ctx, q, &zero_q));
        grad_q = zero_q;
    }
    if (!has_k) {
        GD_TRY(gd_qkv_split_rope_zero_like(bwd->ctx, k, &zero_k));
        grad_k = zero_k;
    }
    if (!has_v) {
        GD_TRY(gd_qkv_split_rope_zero_like(bwd->ctx, v, &zero_v));
        grad_v = zero_v;
    }
    GD_TRY(gd_qkv_split_rope_backward(bwd->ctx,
                                      qkv,
                                      pos_ids,
                                      &grad_q,
                                      &grad_k,
                                      &grad_v,
                                      attrs->n_heads,
                                      attrs->head_dim,
                                      &attrs->rope,
                                      &grad_qkv));
    return gd_autograd_accumulate(bwd, qkv->id, &grad_qkv);
}

const gd_autograd_rule gd_bwd_rule_qkv_split_rope = {
    .kind = GD_OP_QKV_SPLIT_ROPE,
    .name = "qkv_split_rope",
    .backward = gd_qkv_split_rope_autograd_backward,
};
