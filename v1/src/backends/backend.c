#include "backend.h"

#include "../core/internal.h"
#include "../graph/graph_internal.h"
#include "../ops/op_impl.h"

gd_status _gd_backend_check_node(_gd_backend *backend,
                                 const gd_graph *graph,
                                 const _gd_node *node)
{
    const _gd_op_def *def = NULL;
    gd_status status = GD_OK;

    if (backend == NULL || backend->vt == NULL || node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "backend check_node argument is NULL");
    }
    def = _gd_op_def_for(node->op);
    if (def == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown op kind");
    }
    if ((def->flags & GD_OPF_PSEUDO) != 0U) {
        return _gd_error(GD_ERR_UNSUPPORTED, "pseudo op is not executable");
    }
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (backend->vt->check_node == NULL) {
        return GD_OK;
    }
    return backend->vt->check_node(backend, graph, node);
}

gd_status _gd_backend_check_graph(_gd_backend *backend,
                                  const gd_graph *graph,
                                  int *bad_node_out)
{
    int i = 0;

    if (bad_node_out != NULL) {
        *bad_node_out = -1;
    }
    if (backend == NULL || graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "backend check_graph argument is NULL");
    }
    for (i = 0; i < graph->n_nodes; ++i) {
        gd_status status = _gd_backend_check_node(backend, graph, &graph->nodes[i]);
        if (status != GD_OK) {
            if (bad_node_out != NULL) {
                *bad_node_out = i;
            }
            return status;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
