#include "gradients/ops.h"

#include <stdlib.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "autograd_internal.h"

#include "bwd_registry.inc"

const _gd_bwd_rule *_gd_bwd_rule_for(_gd_op_kind op)
{
    if (op <= _GD_OP_INVALID || op >= _GD_OP_COUNT) {
        return NULL;
    }
    return g_bwd_rules[op];
}

gd_context *_gd_bwd_context(_gd_bwd_ctx *b)
{
    return b == NULL ? NULL : b->ctx;
}

gd_graph *_gd_bwd_graph(_gd_bwd_ctx *b)
{
    return b == NULL ? NULL : b->graph;
}

const gd_tensor_desc *_gd_bwd_value_desc(_gd_bwd_ctx *b, int value_id)
{
    if (b == NULL || b->graph == NULL || value_id < 0 || value_id >= b->graph->n_values) {
        return NULL;
    }
    return &b->graph->values[value_id].desc;
}

gd_tensor *_gd_bwd_grad(_gd_bwd_ctx *b, int value_id)
{
    if (b == NULL || value_id < 0 || value_id >= b->n_values) {
        return NULL;
    }
    return b->grad[value_id];
}

gd_status _gd_bwd_fwd(_gd_bwd_ctx *b, int value_id, gd_tensor **out)
{
    _gd_value *value = &b->graph->values[value_id];

    if (b->fwd[value_id] != NULL) {
        *out = b->fwd[value_id];
        return GD_OK;
    }
    if (value->external != NULL) {
        b->fwd[value_id] = value->external;
        *out = value->external;
        return GD_OK;
    }
    {
        gd_tensor *t = NULL;
        gd_status status = _gd_tensor_create_virtual(b->graph, value_id, &value->desc, &t);
        if (status != GD_OK) {
            return status;
        }
        b->fwd[value_id] = t;
        *out = t;
        return GD_OK;
    }
}

/* Accumulates `contrib` (borrowed) into grad[value_id]. */
gd_status _gd_bwd_accumulate(_gd_bwd_ctx *b, int value_id, gd_tensor *contrib)
{
    gd_status status = GD_OK;

    if (b->grad[value_id] == NULL) {
        status = gd_tensor_retain(contrib);
        if (status != GD_OK) {
            return status;
        }
        b->grad[value_id] = contrib;
        return GD_OK;
    }
    {
        gd_tensor *sum = NULL;
        status = gd_add(b->ctx, b->grad[value_id], contrib, &sum);
        if (status != GD_OK) {
            return status;
        }
        gd_tensor_release(b->grad[value_id]);
        b->grad[value_id] = sum;
        return GD_OK;
    }
}

static bool desc_same_shape(const gd_tensor_desc *a, const gd_tensor_desc *b)
{
    int i = 0;

    if (a->ndim != b->ndim) {
        return false;
    }
    for (i = 0; i < a->ndim; ++i) {
        if (a->sizes[i] != b->sizes[i]) {
            return false;
        }
    }
    return true;
}

/* Accumulates `grad` into input `value_id`, summing broadcasted dims down to the
 * input's shape when `grad` has the (larger) output shape. */
gd_status _gd_bwd_accumulate_broadcast(_gd_bwd_ctx *b, int value_id, gd_tensor *grad)
{
    const gd_tensor_desc *target = &b->graph->values[value_id].desc;
    const gd_tensor_desc *gdesc = _gd_tensor_desc_ptr(grad);
    gd_tensor *reduced = NULL;
    gd_status status = GD_OK;
    _gd_op_attrs attrs = {0};

    if (desc_same_shape(target, gdesc)) {
        return _gd_bwd_accumulate(b, value_id, grad);
    }
    attrs.has_reduce_to_desc = true;
    attrs.reduce_to_desc = *target;
    status = _gd_emit_checked(b->ctx, _GD_OP_REDUCE_TO, &grad, 1, &attrs, &reduced, 1);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_bwd_accumulate(b, value_id, reduced);
    gd_tensor_release(reduced);
    return status;
}

gd_status _gd_bwd_emit(_gd_bwd_ctx *b,
                       _gd_op_kind op,
                       gd_tensor **inputs,
                       int n_inputs,
                       const _gd_op_attrs *attrs,
                       const gd_tensor_desc *out_desc,
                       gd_tensor **out)
{
    return _gd_graph_emit(b->graph, op, inputs, n_inputs, attrs, out_desc, out);
}

gd_status _gd_bwd_emit_multi(_gd_bwd_ctx *b,
                             _gd_op_kind op,
                             gd_tensor **inputs,
                             int n_inputs,
                             const _gd_op_attrs *attrs,
                             const gd_tensor_desc *out_descs,
                             int n_outputs,
                             gd_tensor **outs)
{
    return _gd_graph_emit_multi(b->graph, op, inputs, n_inputs, attrs,
                                out_descs, n_outputs, outs);
}

static gd_status backward_node(_gd_bwd_ctx *b, const _gd_node *node_ref)
{
    /* Snapshot before rule emit: graph node array can realloc during backward. */
    _gd_node snapshot = *node_ref;
    const _gd_bwd_rule *rule = _gd_bwd_rule_for(snapshot.op);

    if (rule == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "op has no backward in v1");
    }
    if (rule->unsupported_reason != NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, rule->unsupported_reason);
    }
    if (rule->fn == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "backward rule fn is NULL");
    }
    return rule->fn(b, &snapshot);
}

static void bwd_cleanup(_gd_bwd_ctx *b)
{
    int i = 0;

    if (b->grad != NULL) {
        for (i = 0; i < b->n_values; ++i) {
            gd_tensor_release(b->grad[i]);
        }
        free(b->grad);
    }
    if (b->fwd != NULL) {
        for (i = 0; i < b->n_values; ++i) {
            if (b->fwd[i] != NULL && _gd_tensor_is_virtual(b->fwd[i])) {
                gd_tensor_release(b->fwd[i]);
            }
        }
        free(b->fwd);
    }
    gd_tensor_release(b->ones);
}

gd_status gd_backward(gd_context *ctx, gd_tensor *loss)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    _gd_bwd_ctx b = {0};
    gd_tensor_desc ones_desc;
    float one = 1.0F;
    int loss_value = 0;
    int loss_producer = 0;
    int ni = 0;
    int i = 0;

    if (ctx == NULL || loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_backward argument is NULL");
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "gd_backward requires an active graph");
    }
    if (!_gd_tensor_is_virtual(loss) || _gd_tensor_graph(loss) != graph) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "loss must be a value of the active graph");
    }
    if (gd_tensor_dtype(loss) != GD_DTYPE_F32 || gd_tensor_ndim(loss) != 0) {
        return _gd_error(GD_ERR_UNSUPPORTED, "loss must be a scalar F32 tensor in v1");
    }

    b.ctx = ctx;
    b.graph = graph;
    b.n_values = graph->n_values;
    b.fwd = calloc((size_t)b.n_values, sizeof(*b.fwd));
    b.grad = calloc((size_t)b.n_values, sizeof(*b.grad));
    if (b.n_values > 0 && (b.fwd == NULL || b.grad == NULL)) {
        bwd_cleanup(&b);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate backward state");
    }

    /* Seed dL/dloss = 1. */
    status = gd_tensor_desc_contiguous(GD_DTYPE_F32, gd_tensor_device(loss), 0, NULL, &ones_desc);
    if (status == GD_OK) {
        status = gd_tensor_empty(ctx, &ones_desc, &b.ones);
    }
    if (status == GD_OK) {
        status = gd_tensor_copy_from_cpu(ctx, b.ones, &one, sizeof(one));
    }
    if (status != GD_OK) {
        bwd_cleanup(&b);
        return status;
    }

    loss_value = _gd_tensor_value_id(loss);
    status = gd_tensor_retain(b.ones);
    if (status != GD_OK) {
        bwd_cleanup(&b);
        return status;
    }
    b.grad[loss_value] = b.ones;

    loss_producer = graph->values[loss_value].producer_node_id;
    for (ni = loss_producer; ni >= 0; --ni) {
        status = backward_node(&b, &graph->nodes[ni]);
        if (status != GD_OK) {
            bwd_cleanup(&b);
            return status;
        }
    }

    /* Write accumulated gradients into leaf grad slots. */
    for (i = 0; i < b.n_values; ++i) {
        gd_tensor *leaf = graph->values[i].external;
        gd_tensor *grad_slot = NULL;
        gd_tensor *grad_to_write = b.grad[i];
        gd_tensor *casted_grad = NULL;

        if (leaf == NULL || b.grad[i] == NULL || !gd_tensor_requires_grad(leaf)) {
            continue;
        }
        status = _gd_tensor_ensure_grad(ctx, leaf, &grad_slot);
        if (status != GD_OK) {
            bwd_cleanup(&b);
            return status;
        }
        if (gd_tensor_dtype(grad_to_write) != GD_DTYPE_F32) {
            status = gd_cast(ctx, grad_to_write, GD_DTYPE_F32, &casted_grad);
            if (status != GD_OK) {
                bwd_cleanup(&b);
                return status;
            }
            grad_to_write = casted_grad;
        }
        status = _gd_graph_emit_to(graph, _GD_OP_COPY, &grad_to_write, 1, NULL, grad_slot);
        gd_tensor_release(casted_grad);
        if (status != GD_OK) {
            bwd_cleanup(&b);
            return status;
        }
    }

    bwd_cleanup(&b);
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params)
{
    gd_status status = GD_OK;
    int i = 0;

    if (ctx == NULL || (params == NULL && n_params > 0) || n_params < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_zero_grad argument is invalid");
    }
    for (i = 0; i < n_params; ++i) {
        gd_tensor *grad = NULL;

        if (params[i] == NULL || !gd_tensor_requires_grad(params[i])) {
            continue;
        }
        status = _gd_tensor_ensure_grad(ctx, params[i], &grad);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_tensor_zero(grad);
        if (status != GD_OK) {
            return status;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
