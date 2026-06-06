#include "sdpa_varlen_impl.h"

#include "../autograd_impl.h"

#include <gradients/ops.h>

static gd_status gd_sdpa_varlen_autograd_backward(gd_bwd_ctx *bwd,
                                                  const gd_tape_node *node)
{
    const gd_tensor *q = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *k = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *v = gd_tape_input(bwd->tape, node, 2U);
    const gd_tensor *cu = gd_tape_input(bwd->tape, node, 3U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    const gd_sdpa_varlen_attrs *attrs;
    gd_sdpa_varlen_config config;
    gd_tensor grad_out;
    gd_tensor dq;
    gd_tensor dk;
    gd_tensor dv;
    const bool need_q = q != NULL && q->requires_grad;
    const bool need_k = k != NULL && k->requires_grad;
    const bool need_v = v != NULL && v->requires_grad;
    if (q == NULL || k == NULL || v == NULL || cu == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!need_q && !need_k && !need_v) {
        return GD_OK;
    }
    attrs = (const gd_sdpa_varlen_attrs *)gd_tape_attrs(bwd->tape,
                                                       node,
                                                       (uint32_t)sizeof(gd_sdpa_varlen_attrs));
    if (attrs == NULL) {
        return GD_ERR_INTERNAL;
    }
    config.scale = attrs->scale;
    config.causal = attrs->causal != 0;
    config.sliding_window = attrs->sliding_window;
    config.prefix_len = attrs->prefix_len;
    config.max_seqlen = attrs->max_seqlen;
    GD_TRY(gd_sdpa_varlen_backward(bwd->ctx,
                                   q,
                                   k,
                                   v,
                                   cu,
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

const gd_autograd_rule gd_bwd_rule_sdpa_varlen = {
    .kind = GD_OP_SDPA_VARLEN,
    .name = "sdpa_varlen",
    .backward = gd_sdpa_varlen_autograd_backward,
};
