#include "gradients/ops.h"

#include <stdlib.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../graph/graph_internal.h"
#include "../ops/ops_internal.h"

typedef struct bwd_ctx {
    gd_context *ctx;
    gd_graph *graph;
    int n_values;
    gd_tensor **fwd;   /* cached forward value handles (virtual owned or external borrowed) */
    gd_tensor **grad;  /* accumulated gradient handles (owned) */
    gd_tensor *ones;   /* scalar seed (owned) */
} bwd_ctx;

static gd_status fwd_tensor(bwd_ctx *b, int value_id, gd_tensor **out)
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
static gd_status accumulate(bwd_ctx *b, int value_id, gd_tensor *contrib)
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
static gd_status accumulate_broadcast(bwd_ctx *b, int value_id, gd_tensor *grad)
{
    const gd_tensor_desc *target = &b->graph->values[value_id].desc;
    const gd_tensor_desc *gdesc = _gd_tensor_desc_ptr(grad);
    gd_tensor *reduced = NULL;
    gd_status status = GD_OK;

    if (desc_same_shape(target, gdesc)) {
        return accumulate(b, value_id, grad);
    }
    status = _gd_graph_emit(b->graph, _GD_OP_REDUCE_TO, &grad, 1, NULL, target, &reduced);
    if (status != GD_OK) {
        return status;
    }
    status = accumulate(b, value_id, reduced);
    gd_tensor_release(reduced);
    return status;
}

static gd_status emit_custom(bwd_ctx *b,
                             _gd_op_kind op,
                             gd_tensor **inputs,
                             int n_inputs,
                             const _gd_op_attrs *attrs,
                             const gd_tensor_desc *out_desc,
                             gd_tensor **out)
{
    return _gd_graph_emit(b->graph, op, inputs, n_inputs, attrs, out_desc, out);
}

/* Backward for y = op(a, ta) @ op(b, tb), batched + broadcast. Every input grad
 * is itself a trans-flagged matmul, then reduced down to the operand shape. */
static gd_status matmul_backward(bwd_ctx *b,
                                 gd_tensor *go,
                                 int a_id,
                                 int b_id,
                                 bool ta,
                                 bool tb,
                                 gd_compute_policy compute)
{
    gd_status status = GD_OK;
    gd_tensor *a = NULL;
    gd_tensor *bb = NULL;
    gd_tensor *da = NULL;
    gd_tensor *db = NULL;
    gd_matmul_desc desc = {false, false, compute};

    status = fwd_tensor(b, a_id, &a);
    if (status != GD_OK) {
        return status;
    }
    status = fwd_tensor(b, b_id, &bb);
    if (status != GD_OK) {
        return status;
    }

    /* da */
    if (ta) {
        desc.trans_a = tb;
        desc.trans_b = true;
        status = gd_matmul_ex(b->ctx, &desc, bb, go, &da); /* op(b,tb) @ go^T */
    } else {
        desc.trans_a = false;
        desc.trans_b = !tb;
        status = gd_matmul_ex(b->ctx, &desc, go, bb, &da); /* go @ op(b,tb)^T */
    }
    if (status != GD_OK) {
        return status;
    }
    status = accumulate_broadcast(b, a_id, da);
    gd_tensor_release(da);
    if (status != GD_OK) {
        return status;
    }

    /* db */
    if (tb) {
        desc.trans_a = true;
        desc.trans_b = ta;
        status = gd_matmul_ex(b->ctx, &desc, go, a, &db); /* go^T @ op(a,ta) */
    } else {
        desc.trans_a = !ta;
        desc.trans_b = false;
        status = gd_matmul_ex(b->ctx, &desc, a, go, &db); /* op(a,ta)^T @ go */
    }
    if (status != GD_OK) {
        return status;
    }
    status = accumulate_broadcast(b, b_id, db);
    gd_tensor_release(db);
    return status;
}

static gd_status backward_node(bwd_ctx *b, const _gd_node *node_ref)
{
    gd_status status = GD_OK;
    /* Snapshot: emitting backward ops can realloc the graph node array, which
     * would dangle `node_ref`. The inputs/outputs arrays are separate stable
     * allocations, so a shallow copy is safe to use across emits. */
    _gd_node snapshot = *node_ref;
    const _gd_node *node = &snapshot;
    int vo = node->outputs[0];
    gd_tensor *go = b->grad[vo];
    const gd_tensor_desc *out_desc = &b->graph->values[vo].desc;

    if (go == NULL) {
        return GD_OK;
    }

    switch (node->op) {
    case _GD_OP_ADD: {
        status = accumulate_broadcast(b, node->inputs[0], go);
        if (status != GD_OK) {
            return status;
        }
        return accumulate_broadcast(b, node->inputs[1], go);
    }
    case _GD_OP_MUL: {
        gd_tensor *a = NULL;
        gd_tensor *bb = NULL;
        gd_tensor *da = NULL;
        gd_tensor *db = NULL;

        status = fwd_tensor(b, node->inputs[0], &a);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[1], &bb);
        if (status != GD_OK) {
            return status;
        }
        /* da = reduce(go * b) -> a.shape ; db = reduce(go * a) -> b.shape */
        status = gd_mul(b->ctx, go, bb, &da);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate_broadcast(b, node->inputs[0], da);
        gd_tensor_release(da);
        if (status != GD_OK) {
            return status;
        }
        status = gd_mul(b->ctx, go, a, &db);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate_broadcast(b, node->inputs[1], db);
        gd_tensor_release(db);
        return status;
    }
    case _GD_OP_SCALE: {
        gd_tensor *dx = NULL;

        status = gd_scale(b->ctx, go, node->attrs.scale, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_RELU:
    case _GD_OP_SILU: {
        gd_tensor *x = NULL;
        gd_tensor *dx = NULL;
        gd_tensor *inputs[2];
        _gd_op_kind bwd = node->op == _GD_OP_RELU ? _GD_OP_RELU_BWD : _GD_OP_SILU_BWD;

        status = fwd_tensor(b, node->inputs[0], &x);
        if (status != GD_OK) {
            return status;
        }
        inputs[0] = x;
        inputs[1] = go;
        status = emit_custom(b, bwd, inputs, 2, NULL, _gd_tensor_desc_ptr(x), &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_POWLU: {
        gd_tensor *x1 = NULL;
        gd_tensor *x2 = NULL;
        gd_tensor *grads[2] = {NULL, NULL};
        gd_tensor *inputs[3];
        gd_tensor_desc out_descs[2];

        status = fwd_tensor(b, node->inputs[0], &x1);
        if (status == GD_OK) {
            status = fwd_tensor(b, node->inputs[1], &x2);
        }
        if (status != GD_OK) {
            return status;
        }
        inputs[0] = x1;
        inputs[1] = x2;
        inputs[2] = go;
        out_descs[0] = b->graph->values[node->inputs[0]].desc;
        out_descs[1] = b->graph->values[node->inputs[1]].desc;
        status = _gd_graph_emit_multi(b->graph, _GD_OP_POWLU_BWD, inputs, 3,
                                      &node->attrs, out_descs, 2, grads);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], grads[0]);
        if (status == GD_OK) {
            status = accumulate(b, node->inputs[1], grads[1]);
        }
        gd_tensor_release(grads[0]);
        gd_tensor_release(grads[1]);
        return status;
    }
    case _GD_OP_MATMUL:
        return matmul_backward(b, go, node->inputs[0], node->inputs[1],
                               node->attrs.trans_a, node->attrs.trans_b,
                               node->attrs.compute);
    case _GD_OP_LINEAR: {
        /* y = x @ op(w, trans_w): treat as matmul(a=x, ta=false, b=w, tb=trans_w). */
        status = matmul_backward(b, go, node->inputs[0], node->inputs[1], false,
                                 node->attrs.trans_b, node->attrs.compute);
        if (status != GD_OK) {
            return status;
        }
        if (node->attrs.has_bias) {
            /* bias grad = sum of go over all leading dims -> [out]. */
            status = accumulate_broadcast(b, node->inputs[2], go);
        }
        return status;
    }
    case _GD_OP_SUM:
    case _GD_OP_MEAN: {
        gd_tensor *dx = NULL;
        const gd_tensor_desc *xdesc = &b->graph->values[node->inputs[0]].desc;
        _gd_op_attrs attrs = {0};
        _gd_op_kind bwd = node->op == _GD_OP_SUM ? _GD_OP_SUM_BWD : _GD_OP_MEAN_BWD;

        attrs.dim = node->attrs.dim;
        status = emit_custom(b, bwd, &go, 1, &attrs, xdesc, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_SOFTMAX: {
        gd_tensor *y = NULL;
        gd_tensor *dx = NULL;
        gd_tensor *inputs[2];
        _gd_op_attrs attrs = {0};

        status = fwd_tensor(b, vo, &y);
        if (status != GD_OK) {
            return status;
        }
        attrs.dim = node->attrs.dim;
        inputs[0] = y;
        inputs[1] = go;
        status = emit_custom(b, _GD_OP_SOFTMAX_BWD, inputs, 2, &attrs, out_desc, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_CROSS_ENTROPY: {
        gd_tensor *logits = NULL;
        gd_tensor *targets = NULL;
        gd_tensor *dlogits = NULL;
        gd_tensor *inputs[3];
        _gd_op_attrs attrs = {0};
        const gd_tensor_desc *ldesc = &b->graph->values[node->inputs[0]].desc;

        status = fwd_tensor(b, node->inputs[0], &logits);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[1], &targets);
        if (status != GD_OK) {
            return status;
        }
        attrs.dim = node->attrs.dim;
        inputs[0] = logits;
        inputs[1] = targets;
        inputs[2] = go;
        status = emit_custom(b, _GD_OP_CROSS_ENTROPY_BWD, inputs, 3, &attrs, ldesc, &dlogits);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dlogits);
        gd_tensor_release(dlogits);
        return status;
    }
    case _GD_OP_LM_CROSS_ENTROPY: {
        gd_tensor *hidden = NULL;
        gd_tensor *weight = NULL;
        gd_tensor *targets = NULL;
        gd_tensor *row_max = NULL;
        gd_tensor *row_sum = NULL;
        gd_tensor *grads[2] = {NULL, NULL};
        gd_tensor *inputs[6];
        gd_tensor_desc out_descs[2];

        status = fwd_tensor(b, node->inputs[0], &hidden);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[1], &weight);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[2], &targets);
        if (status != GD_OK) {
            return status;
        }
        if (node->n_outputs != 3) {
            return _gd_error(GD_ERR_INTERNAL, "lm_cross_entropy expects stats outputs");
        }
        status = fwd_tensor(b, node->outputs[1], &row_max);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->outputs[2], &row_sum);
        if (status != GD_OK) {
            return status;
        }
        inputs[0] = hidden;
        inputs[1] = weight;
        inputs[2] = targets;
        inputs[3] = go;
        inputs[4] = row_max;
        inputs[5] = row_sum;
        out_descs[0] = b->graph->values[node->inputs[0]].desc;
        out_descs[1] = b->graph->values[node->inputs[1]].desc;
        status = _gd_graph_emit_multi(b->graph, _GD_OP_LM_CROSS_ENTROPY_BWD,
                                      inputs, 6, NULL, out_descs, 2, grads);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], grads[0]);
        gd_tensor_release(grads[0]);
        if (status != GD_OK) {
            gd_tensor_release(grads[1]);
            return status;
        }
        status = accumulate(b, node->inputs[1], grads[1]);
        gd_tensor_release(grads[1]);
        return status;
    }
    case _GD_OP_COPY: {
        /* Reshape/identity: gradient flows back reshaped to the input shape. */
        const gd_tensor_desc *in_desc = &b->graph->values[node->inputs[0]].desc;
        gd_tensor *din = NULL;

        status = emit_custom(b, _GD_OP_COPY, &go, 1, NULL, in_desc, &din);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], din);
        gd_tensor_release(din);
        return status;
    }
    case _GD_OP_GELU: {
        gd_tensor *x = NULL;
        gd_tensor *dx = NULL;
        gd_tensor *inputs[2];
        _gd_op_attrs attrs = {0};

        status = fwd_tensor(b, node->inputs[0], &x);
        if (status != GD_OK) {
            return status;
        }
        attrs.gelu_tanh = node->attrs.gelu_tanh;
        inputs[0] = x;
        inputs[1] = go;
        status = emit_custom(b, _GD_OP_GELU_BWD, inputs, 2, &attrs,
                             _gd_tensor_desc_ptr(x), &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_TRANSPOSE: {
        /* dx = transpose(go, inverse_perm). */
        const gd_tensor_desc *in_desc = &b->graph->values[node->inputs[0]].desc;
        gd_tensor *dx = NULL;
        _gd_op_attrs attrs = {0};
        int i = 0;

        attrs.perm_ndim = node->attrs.perm_ndim;
        for (i = 0; i < node->attrs.perm_ndim; ++i) {
            attrs.perm[node->attrs.perm[i]] = i;
        }
        status = emit_custom(b, _GD_OP_TRANSPOSE, &go, 1, &attrs, in_desc, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_EMBEDDING: {
        /* Only the table has a gradient (scatter-add of go by id). */
        const gd_tensor_desc *table_desc = &b->graph->values[node->inputs[0]].desc;
        gd_tensor *ids = NULL;
        gd_tensor *dtable = NULL;
        gd_tensor *inputs[2];

        status = fwd_tensor(b, node->inputs[1], &ids);
        if (status != GD_OK) {
            return status;
        }
        inputs[0] = go;
        inputs[1] = ids;
        status = emit_custom(b, _GD_OP_EMBEDDING_BWD, inputs, 2, NULL, table_desc, &dtable);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dtable);
        gd_tensor_release(dtable);
        return status;
    }
    case _GD_OP_ROPE: {
        /* dx = R(-theta) go (transpose rotation): same kernel, conjugated sin. */
        const gd_tensor_desc *in_desc = &b->graph->values[node->inputs[0]].desc;
        gd_tensor *pos = NULL;
        gd_tensor *dx = NULL;
        gd_tensor *inputs[2];
        _gd_op_attrs attrs = {0};

        status = fwd_tensor(b, node->inputs[1], &pos);
        if (status != GD_OK) {
            return status;
        }
        attrs.rope_theta = node->attrs.rope_theta;
        attrs.rope_n_dims = node->attrs.rope_n_dims;
        attrs.rope_interleaved = node->attrs.rope_interleaved;
        inputs[0] = go;
        inputs[1] = pos;
        status = emit_custom(b, _GD_OP_ROPE_BWD, inputs, 2, &attrs, in_desc, &dx);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], dx);
        gd_tensor_release(dx);
        return status;
    }
    case _GD_OP_RMS_NORM: {
        /* inputs: x(0), weight(1). dx and dweight via two reference kernels. */
        gd_tensor *x = NULL;
        gd_tensor *weight = NULL;
        gd_tensor *dx = NULL;
        gd_tensor *dw = NULL;
        gd_tensor *dx_inputs[3];
        gd_tensor *dw_inputs[2];
        _gd_op_attrs attrs = {0};
        /* Snapshot by value: emitting RMS_NORM_BWD grows the values array, which
         * would dangle pointers into graph->values before the second emit. */
        gd_tensor_desc x_desc = b->graph->values[node->inputs[0]].desc;
        gd_tensor_desc w_desc = b->graph->values[node->inputs[1]].desc;

        attrs.eps = node->attrs.eps;
        status = fwd_tensor(b, node->inputs[0], &x);
        if (status == GD_OK) {
            status = fwd_tensor(b, node->inputs[1], &weight);
        }
        if (status != GD_OK) {
            return status;
        }
        dx_inputs[0] = x;
        dx_inputs[1] = weight;
        dx_inputs[2] = go;
        status = emit_custom(b, _GD_OP_RMS_NORM_BWD, dx_inputs, 3, &attrs, &x_desc, &dx);
        if (status == GD_OK) {
            status = accumulate(b, node->inputs[0], dx);
        }
        gd_tensor_release(dx);
        if (status != GD_OK) {
            return status;
        }
        dw_inputs[0] = x;
        dw_inputs[1] = go;
        status = emit_custom(b, _GD_OP_RMS_NORM_WBWD, dw_inputs, 2, &attrs, &w_desc, &dw);
        if (status == GD_OK) {
            status = accumulate(b, node->inputs[1], dw);
        }
        gd_tensor_release(dw);
        return status;
    }
    case _GD_OP_SDPA: {
        /* one multi-output node yields dq, dk, dv (recompute-based reference). */
        gd_tensor *q = NULL, *k = NULL, *v = NULL, *bias = NULL;
        gd_tensor *grads[3] = {0};
        gd_tensor *bwd_inputs[5];
        gd_tensor_desc out_descs[3];
        int n_bwd = 4;

        status = fwd_tensor(b, node->inputs[0], &q);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[1], &k);
        if (status != GD_OK) {
            return status;
        }
        status = fwd_tensor(b, node->inputs[2], &v);
        if (status != GD_OK) {
            return status;
        }
        out_descs[0] = b->graph->values[node->inputs[0]].desc;
        out_descs[1] = b->graph->values[node->inputs[1]].desc;
        out_descs[2] = b->graph->values[node->inputs[2]].desc;
        bwd_inputs[0] = go;
        bwd_inputs[1] = q;
        bwd_inputs[2] = k;
        bwd_inputs[3] = v;
        /* bias (if any) is a constant input: needed to recompute softmax, but
         * not differentiated in v1. */
        if (node->attrs.has_bias) {
            status = fwd_tensor(b, node->inputs[3], &bias);
            if (status != GD_OK) {
                return status;
            }
            bwd_inputs[4] = bias;
            n_bwd = 5;
        }
        status = _gd_graph_emit_multi(b->graph, _GD_OP_SDPA_BWD, bwd_inputs, n_bwd,
                                      &node->attrs, out_descs, 3, grads);
        if (status != GD_OK) {
            return status;
        }
        status = accumulate(b, node->inputs[0], grads[0]);
        if (status == GD_OK) {
            status = accumulate(b, node->inputs[1], grads[1]);
        }
        if (status == GD_OK) {
            status = accumulate(b, node->inputs[2], grads[2]);
        }
        gd_tensor_release(grads[0]);
        gd_tensor_release(grads[1]);
        gd_tensor_release(grads[2]);
        return status;
    }
    case _GD_OP_CAST:
        return _gd_error(GD_ERR_UNSUPPORTED, "cast backward is not supported in v1");
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "op has no backward in v1");
    }
}

static void bwd_cleanup(bwd_ctx *b)
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
    bwd_ctx b = {0};
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

        if (leaf == NULL || b.grad[i] == NULL || !gd_tensor_requires_grad(leaf)) {
            continue;
        }
        status = _gd_tensor_ensure_grad(ctx, leaf, &grad_slot);
        if (status != GD_OK) {
            bwd_cleanup(&b);
            return status;
        }
        status = _gd_graph_emit_to(graph, _GD_OP_COPY, &b.grad[i], 1, NULL, grad_slot);
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
