#include "minimax_m3_sparse_attention_impl.h"

#include "../autograd_impl.h"

#include <stdbool.h>
#include <string.h>

static gd_minimax_m3_sparse_config gd_minimax_m3_sparse_config_from_attrs(
    const gd_minimax_m3_sparse_attention_attrs *attrs)
{
    gd_minimax_m3_sparse_config config;
    memset(&config, 0, sizeof(config));
    if (attrs != NULL) {
        config.scale = attrs->scale;
        config.block_size = attrs->block_size;
        config.topk = attrs->topk;
        config.init_blocks = attrs->init_blocks;
        config.local_blocks = attrs->local_blocks;
        config.max_seqlen = attrs->max_seqlen;
    }
    return config;
}

static gd_status gd_minimax_m3_sparse_attention_autograd_backward(gd_bwd_ctx *bwd,
                                                                  const gd_tape_node *node)
{
    const gd_tensor *q;
    const gd_tensor *k;
    const gd_tensor *v;
    const gd_tensor *cu;
    const gd_tensor *topk;
    const gd_tensor *out;
    const gd_minimax_m3_sparse_attention_attrs *attrs;
    gd_minimax_m3_sparse_config config;
    gd_tensor grad_out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    bool need_q;
    bool need_k;
    bool need_v;
    if (bwd == NULL || bwd->tape == NULL || node == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    q = gd_tape_input(bwd->tape, node, 0U);
    k = gd_tape_input(bwd->tape, node, 1U);
    v = gd_tape_input(bwd->tape, node, 2U);
    cu = gd_tape_input(bwd->tape, node, 3U);
    topk = gd_tape_input(bwd->tape, node, 4U);
    out = gd_tape_output(bwd->tape, node, 0U);
    attrs = (const gd_minimax_m3_sparse_attention_attrs *)gd_tape_attrs(
        bwd->tape, node, (uint32_t)sizeof(gd_minimax_m3_sparse_attention_attrs));
    if (q == NULL || k == NULL || v == NULL || cu == NULL || topk == NULL || out == NULL || attrs == NULL ||
        node->n_inputs != 5U || node->n_outputs != 1U) {
        return GD_ERR_INTERNAL;
    }
    need_q = q->requires_grad;
    need_k = k->requires_grad;
    need_v = v->requires_grad;
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!need_q && !need_k && !need_v) {
        return GD_OK;
    }
    config = gd_minimax_m3_sparse_config_from_attrs(attrs);
    GD_TRY(gd_minimax_m3_sparse_attention_backward_with_topk(bwd->ctx,
                                                             q,
                                                             k,
                                                             v,
                                                             cu,
                                                             topk,
                                                             &grad_out,
                                                             &config,
                                                             need_q ? &dq : NULL,
                                                             need_k ? &dk : NULL,
                                                             need_v ? &dv : NULL));
    if (need_q) {
        GD_TRY(gd_autograd_accumulate(bwd, q->id, &dq));
    }
    if (need_k) {
        GD_TRY(gd_autograd_accumulate(bwd, k->id, &dk));
    }
    if (need_v) {
        GD_TRY(gd_autograd_accumulate(bwd, v->id, &dv));
    }
    return GD_OK;
}

const gd_autograd_rule gd_bwd_rule_minimax_m3_sparse_attention = {
    .kind = GD_OP_MINIMAX_M3_SPARSE_ATTENTION,
    .name = "minimax_m3_sparse_attention",
    .backward = gd_minimax_m3_sparse_attention_autograd_backward,
};
