#include "gradients/ops.h"

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
