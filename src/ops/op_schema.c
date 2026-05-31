#include "gradients/ops.h"

#include <math.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../graph/graph_internal.h"
#include "ops_internal.h"

static gd_status require_active_graph(gd_context *ctx, gd_graph **out)
{
    gd_graph *graph = NULL;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "context is NULL");
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "compute op requires an active graph");
    }
    *out = graph;
    return GD_OK;
}

gd_status gd_add(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_add argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_elementwise(a, b, &desc);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = a;
    inputs[1] = b;
    return _gd_graph_emit(graph, _GD_OP_ADD, inputs, 2, NULL, &desc, out);
}

gd_status gd_mul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_mul argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_elementwise(a, b, &desc);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = a;
    inputs[1] = b;
    return _gd_graph_emit(graph, _GD_OP_MUL, inputs, 2, NULL, &desc, out);
}

gd_status gd_scale(gd_context *ctx, gd_tensor *x, float scale, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_scale argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_unary_float(x, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.scale = scale;
    return _gd_graph_emit(graph, _GD_OP_SCALE, &x, 1, &attrs, &desc, out);
}

gd_status gd_matmul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out)
{
    return gd_matmul_ex(ctx, NULL, a, b, out);
}

gd_status gd_matmul_ex(gd_context *ctx,
                       const gd_matmul_desc *desc,
                       gd_tensor *a,
                       gd_tensor *b,
                       gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc out_desc;
    _gd_op_attrs attrs = {0};

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_matmul_ex argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    if (desc != NULL) {
        attrs.trans_a = desc->trans_a;
        attrs.trans_b = desc->trans_b;
        attrs.compute = desc->compute;
    } else {
        attrs.compute = gd_context_compute_policy(ctx);
    }
    status = _gd_infer_matmul(a, b, attrs.trans_a, attrs.trans_b, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = a;
    inputs[1] = b;
    return _gd_graph_emit(graph, _GD_OP_MATMUL, inputs, 2, &attrs, &out_desc, out);
}

gd_status gd_linear(gd_context *ctx,
                    gd_tensor *x,
                    gd_tensor *w,
                    gd_tensor *bias,
                    gd_tensor **out)
{
    return gd_linear_ex(ctx, NULL, x, w, bias, out);
}

gd_status gd_linear_ex(gd_context *ctx,
                       const gd_linear_desc *desc,
                       gd_tensor *x,
                       gd_tensor *w,
                       gd_tensor *bias,
                       gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[3];
    gd_tensor_desc out_desc;
    _gd_op_attrs attrs = {0};
    int n_inputs = 0;

    if (x == NULL || w == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_linear_ex argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    if (desc != NULL) {
        attrs.trans_b = desc->trans_w;
        attrs.compute = desc->compute;
    } else {
        attrs.compute = gd_context_compute_policy(ctx);
    }
    status = _gd_infer_linear(x, w, bias, attrs.trans_b, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.has_bias = bias != NULL;
    inputs[0] = x;
    inputs[1] = w;
    n_inputs = 2;
    if (bias != NULL) {
        inputs[2] = bias;
        n_inputs = 3;
    }
    return _gd_graph_emit(graph, _GD_OP_LINEAR, inputs, n_inputs, &attrs, &out_desc, out);
}

static gd_status emit_unary_float(gd_context *ctx,
                                  _gd_op_kind op,
                                  gd_tensor *x,
                                  gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unary op argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_unary_float(x, &desc);
    if (status != GD_OK) {
        return status;
    }
    return _gd_graph_emit(graph, op, &x, 1, NULL, &desc, out);
}

gd_status gd_relu(gd_context *ctx, gd_tensor *x, gd_tensor **out)
{
    return emit_unary_float(ctx, _GD_OP_RELU, x, out);
}

gd_status gd_silu(gd_context *ctx, gd_tensor *x, gd_tensor **out)
{
    return emit_unary_float(ctx, _GD_OP_SILU, x, out);
}

static gd_status emit_reduce(gd_context *ctx,
                             _gd_op_kind op,
                             gd_tensor *x,
                             int dim,
                             bool keepdim,
                             gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    int norm_dim = 0;

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "reduce op argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_reduce(x, dim, keepdim, &norm_dim, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.dim = norm_dim;
    attrs.keepdim = keepdim;
    return _gd_graph_emit(graph, op, &x, 1, &attrs, &desc, out);
}

gd_status gd_sum(gd_context *ctx, gd_tensor *x, int dim, bool keepdim, gd_tensor **out)
{
    return emit_reduce(ctx, _GD_OP_SUM, x, dim, keepdim, out);
}

gd_status gd_mean(gd_context *ctx, gd_tensor *x, int dim, bool keepdim, gd_tensor **out)
{
    return emit_reduce(ctx, _GD_OP_MEAN, x, dim, keepdim, out);
}

gd_status gd_rms_norm(gd_context *ctx,
                      gd_tensor *x,
                      gd_tensor *weight,
                      float eps,
                      gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    const gd_tensor_desc *dx = NULL;
    const gd_tensor_desc *dw = NULL;

    if (x == NULL || weight == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_rms_norm argument is NULL");
    }
    *out = NULL;
    if (eps <= 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rms_norm eps must be positive");
    }
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    if (gd_tensor_dtype(x) != gd_tensor_dtype(weight)) {
        return _gd_error(GD_ERR_DTYPE, "rms_norm input and weight must share dtype");
    }
    if (!gd_device_equal(gd_tensor_device(x), gd_tensor_device(weight))) {
        return _gd_error(GD_ERR_DEVICE, "rms_norm inputs must share a device");
    }
    status = _gd_infer_unary_float(x, &desc);
    if (status != GD_OK) {
        return status;
    }
    dx = _gd_tensor_desc_ptr(x);
    dw = _gd_tensor_desc_ptr(weight);
    if (dw->ndim != 1 || dw->sizes[0] != dx->sizes[dx->ndim - 1]) {
        return _gd_error(GD_ERR_SHAPE, "rms_norm weight must be [last_dim]");
    }
    attrs.eps = eps;
    inputs[0] = x;
    inputs[1] = weight;
    return _gd_graph_emit(graph, _GD_OP_RMS_NORM, inputs, 2, &attrs, &desc, out);
}

gd_status gd_softmax(gd_context *ctx, gd_tensor *x, int dim, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    int norm_dim = 0;

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_softmax argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_softmax(x, dim, &norm_dim, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.dim = norm_dim;
    return _gd_graph_emit(graph, _GD_OP_SOFTMAX, &x, 1, &attrs, &desc, out);
}

gd_status gd_cross_entropy(gd_context *ctx,
                           gd_tensor *logits,
                           gd_tensor *targets,
                           int class_dim,
                           gd_tensor **loss)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    int norm_dim = 0;

    if (logits == NULL || targets == NULL || loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_cross_entropy argument is NULL");
    }
    *loss = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_cross_entropy(logits, targets, class_dim, &norm_dim, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.dim = norm_dim;
    inputs[0] = logits;
    inputs[1] = targets;
    return _gd_graph_emit(graph, _GD_OP_CROSS_ENTROPY, inputs, 2, &attrs, &desc, loss);
}

gd_status gd_lm_cross_entropy(gd_context *ctx,
                              gd_tensor *hidden,
                              gd_tensor *weight,
                              gd_tensor *targets,
                              gd_tensor **loss)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[3];
    gd_tensor_desc desc;

    if (hidden == NULL || weight == NULL || targets == NULL || loss == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_lm_cross_entropy argument is NULL");
    }
    *loss = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_lm_cross_entropy(hidden, weight, targets, &desc);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = hidden;
    inputs[1] = weight;
    inputs[2] = targets;
    return _gd_graph_emit(graph, _GD_OP_LM_CROSS_ENTROPY, inputs, 3, NULL, &desc, loss);
}

gd_status gd_cast(gd_context *ctx, gd_tensor *x, gd_dtype dtype, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_cast argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_cast(x, dtype, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.cast_dtype = dtype;
    return _gd_graph_emit(graph, _GD_OP_CAST, &x, 1, &attrs, &desc, out);
}

gd_status gd_gelu(gd_context *ctx, gd_tensor *x, bool tanh_approx, gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gelu argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_unary_float(x, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.gelu_tanh = tanh_approx;
    return _gd_graph_emit(graph, _GD_OP_GELU, &x, 1, &attrs, &desc, out);
}

gd_status gd_transpose(gd_context *ctx,
                       gd_tensor *x,
                       const int *perm,
                       int ndim,
                       gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    int i = 0;

    if (x == NULL || perm == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_transpose argument is NULL");
    }
    *out = NULL;
    if (ndim < 0 || ndim > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_transpose ndim is out of range");
    }
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_transpose(x, perm, ndim, &desc);
    if (status != GD_OK) {
        return status;
    }
    attrs.perm_ndim = ndim;
    for (i = 0; i < ndim; ++i) {
        attrs.perm[i] = perm[i];
    }
    return _gd_graph_emit(graph, _GD_OP_TRANSPOSE, &x, 1, &attrs, &desc, out);
}

gd_status gd_embedding(gd_context *ctx,
                       gd_tensor *table,
                       gd_tensor *ids,
                       gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;

    if (table == NULL || ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_embedding argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_embedding(table, ids, &desc);
    if (status != GD_OK) {
        return status;
    }
    inputs[0] = table;
    inputs[1] = ids;
    return _gd_graph_emit(graph, _GD_OP_EMBEDDING, inputs, 2, NULL, &desc, out);
}

gd_status gd_rope(gd_context *ctx,
                  gd_tensor *x,
                  gd_tensor *pos_ids,
                  const gd_rope_config *config,
                  gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    const gd_tensor_desc *dx = NULL;
    int64_t head_dim = 0;
    int n_dims = 0;

    if (x == NULL || pos_ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_rope argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_rope(x, pos_ids, &desc);
    if (status != GD_OK) {
        return status;
    }
    dx = _gd_tensor_desc_ptr(x);
    head_dim = dx->sizes[dx->ndim - 1];
    n_dims = (config != NULL && config->n_dims > 0) ? config->n_dims : (int)head_dim;
    if (n_dims % 2 != 0 || n_dims > (int)head_dim) {
        return _gd_error(GD_ERR_SHAPE, "rope n_dims must be even and <= head_dim");
    }
    attrs.rope_theta = (config != NULL && config->theta > 0.0F) ? config->theta : 10000.0F;
    attrs.rope_n_dims = n_dims;
    attrs.rope_interleaved = (config != NULL && config->interleaved) ? 1 : 0;
    inputs[0] = x;
    inputs[1] = pos_ids;
    return _gd_graph_emit(graph, _GD_OP_ROPE, inputs, 2, &attrs, &desc, out);
}

/* Validates that `bias` broadcasts over the [B, Hq, Tq, Tk] score grid. */
static gd_status sdpa_check_bias(gd_tensor *bias, const gd_tensor_desc *out_desc,
                                 const gd_tensor_desc *k_desc)
{
    const gd_tensor_desc *db = NULL;
    int64_t want[4];
    int i = 0;

    if (gd_tensor_dtype(bias) != out_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa bias must share dtype with q");
    }
    db = _gd_tensor_desc_ptr(bias);
    if (db->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE, "sdpa bias must be 4D [B,Hq,Tq,Tk] (broadcastable)");
    }
    want[0] = out_desc->sizes[0]; /* B */
    want[1] = out_desc->sizes[2]; /* Hq */
    want[2] = out_desc->sizes[1]; /* Tq */
    want[3] = k_desc->sizes[1];   /* Tk */
    for (i = 0; i < 4; ++i) {
        if (db->sizes[i] != 1 && db->sizes[i] != want[i]) {
            return _gd_error(GD_ERR_SHAPE, "sdpa bias is not broadcastable to [B,Hq,Tq,Tk]");
        }
    }
    return GD_OK;
}

gd_status gd_sdpa(gd_context *ctx,
                  gd_tensor *q,
                  gd_tensor *k,
                  gd_tensor *v,
                  gd_tensor *bias,
                  const gd_sdpa_config *config,
                  gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    gd_tensor *inputs[4];
    gd_tensor_desc desc;
    _gd_op_attrs attrs = {0};
    const gd_tensor_desc *dq = NULL;
    const gd_tensor_desc *dk = NULL;
    int head_dim = 0;
    int n_inputs = 3;

    if (q == NULL || k == NULL || v == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_sdpa argument is NULL");
    }
    *out = NULL;
    status = require_active_graph(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_infer_sdpa(q, k, v, &desc);
    if (status != GD_OK) {
        return status;
    }
    dq = _gd_tensor_desc_ptr(q);
    dk = _gd_tensor_desc_ptr(k);
    if (bias != NULL) {
        if (!gd_device_equal(gd_tensor_device(q), gd_tensor_device(bias))) {
            return _gd_error(GD_ERR_DEVICE, "sdpa bias must share a device with q");
        }
        status = sdpa_check_bias(bias, &desc, dk);
        if (status != GD_OK) {
            return status;
        }
    }
    head_dim = (int)dq->sizes[3];
    attrs.head_dim = head_dim;
    attrs.n_q_heads = (int)dq->sizes[2];
    attrs.n_kv_heads = (int)dk->sizes[2];
    attrs.attn_scale = (config != NULL && config->scale > 0.0F)
                           ? config->scale
                           : (float)(1.0 / sqrt((double)head_dim));
    attrs.causal = (config != NULL && config->causal) ? 1 : 0;
    attrs.sliding_window = (config != NULL && config->sliding_window > 0)
                               ? config->sliding_window
                               : 0;
    attrs.has_bias = bias != NULL;
    inputs[0] = q;
    inputs[1] = k;
    inputs[2] = v;
    if (bias != NULL) {
        inputs[3] = bias;
        n_inputs = 4;
    }
    return _gd_graph_emit(graph, _GD_OP_SDPA, inputs, n_inputs, &attrs, &desc, out);
}
