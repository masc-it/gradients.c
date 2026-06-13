#include "embedding_impl.h"

#include "../autograd_impl.h"

static gd_status gd_embedding_autograd_backward(gd_bwd_ctx *bwd,
                                                const gd_tape_node *node)
{
    const gd_tensor *table = gd_tape_input(bwd->tape, node, 0U);
    const gd_tensor *ids = gd_tape_input(bwd->tape, node, 1U);
    const gd_tensor *out = gd_tape_output(bwd->tape, node, 0U);
    gd_tensor grad_out;
    gd_tensor dtable;
    if (table == NULL || ids == NULL || out == NULL) {
        return GD_ERR_INTERNAL;
    }
    if (!gd_autograd_get_grad(bwd, out->id, &grad_out)) {
        return GD_OK;
    }
    if (!table->requires_grad) {
        return GD_OK;
    }
    GD_TRY(gd_embedding_backward_impl(bwd->ctx, table, ids, &grad_out, &dtable));
    return gd_autograd_accumulate(bwd, table->id, &dtable);
}

const gd_autograd_rule gd_bwd_rule_embedding = {
    .kind = GD_OP_EMBEDDING,
    .name = "embedding",
    .backward = gd_embedding_autograd_backward,
};
