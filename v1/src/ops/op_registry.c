#include "op_impl.h"

#include <stddef.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../graph/graph_internal.h"

#include "op_registry.inc"

const _gd_op_def *_gd_op_def_for(_gd_op_kind kind)
{
    if (kind <= _GD_OP_INVALID || kind >= _GD_OP_COUNT) {
        return NULL;
    }
    return g_op_defs[kind];
}

const char *_gd_op_kind_name(_gd_op_kind kind)
{
    if (kind >= _GD_OP_INVALID && kind < _GD_OP_COUNT && g_op_kind_names[kind] != NULL) {
        return g_op_kind_names[kind];
    }
    return "unknown";
}

gd_status _gd_op_validate_arity(_gd_op_kind kind, int n_inputs, int n_outputs)
{
    const _gd_op_def *def = _gd_op_def_for(kind);

    if (def == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown op kind");
    }
    if (n_inputs < def->min_inputs || n_inputs > def->max_inputs) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input arity mismatch");
    }
    if (n_outputs != def->n_outputs) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op output arity mismatch");
    }
    return GD_OK;
}

bool _gd_op_is_differentiable(_gd_op_kind kind)
{
    const _gd_op_def *def = _gd_op_def_for(kind);

    return def != NULL && (def->flags & GD_OPF_DIFF) != 0u;
}

gd_status _gd_emit_checked(gd_context *ctx,
                           _gd_op_kind op,
                           gd_tensor **inputs,
                           int n_inputs,
                           _gd_op_attrs *attrs,
                           gd_tensor **outputs,
                           int n_outputs)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;
    const _gd_op_def *def = NULL;
    const gd_tensor_desc *input_descs[_GD_OP_MAX_INPUTS];
    gd_tensor_desc output_descs[_GD_OP_MAX_INPUTS];
    _gd_op_attrs local_attrs = {0};
    int meta_outputs = n_outputs;
    int i = 0;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "context is NULL");
    }
    if (n_inputs < 0 || n_inputs > _GD_OP_MAX_INPUTS ||
        n_outputs < 0 || n_outputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op arity is out of range");
    }
    if (n_inputs > 0 && inputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op inputs array is NULL");
    }
    if (n_outputs > 0 && outputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op outputs array is NULL");
    }
    for (i = 0; i < n_outputs; ++i) {
        outputs[i] = NULL;
    }

    def = _gd_op_def_for(op);
    if (def == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown op kind");
    }
    status = _gd_op_validate_arity(op, n_inputs, n_outputs);
    if (status != GD_OK) {
        return status;
    }

    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "compute op requires an active graph");
    }

    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input tensor is NULL");
        }
        input_descs[i] = _gd_tensor_desc_ptr(inputs[i]);
        if (input_descs[i] == NULL) {
            return _gd_error(GD_ERR_INTERNAL, "op input descriptor is NULL");
        }
    }

    if (attrs != NULL) {
        local_attrs = *attrs;
    }
    if (def->meta == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "op meta function is NULL");
    }
    status = def->meta(input_descs, n_inputs, &local_attrs, output_descs, &meta_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (meta_outputs != n_outputs) {
        return _gd_error(GD_ERR_INTERNAL, "op meta output count mismatch");
    }

    if (n_outputs == 0) {
        return _gd_graph_emit_inplace(graph, op, inputs, n_inputs, &local_attrs);
    }
    if (n_outputs == 1) {
        return _gd_graph_emit(graph, op, inputs, n_inputs, &local_attrs,
                              &output_descs[0], &outputs[0]);
    }
    return _gd_graph_emit_multi(graph, op, inputs, n_inputs, &local_attrs,
                                output_descs, n_outputs, outputs);
}
