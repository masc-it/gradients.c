#include "op_impl.h"

#include "../core/internal.h"

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
