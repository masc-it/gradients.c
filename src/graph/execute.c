#include "gradients/graph.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "../backends/backend.h"
#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "graph_internal.h"

void _gd_graph_free_executable(gd_graph *graph)
{
    if (graph->exec != NULL && graph->backend != NULL &&
        graph->backend->vt->executable_free != NULL) {
        graph->backend->vt->executable_free(graph->backend, graph->exec);
    }
    graph->exec = NULL;
    graph->backend = NULL;
    graph->has_run = false;
}

/* Returns the backend that will execute the graph, applying fallback policy when
 * the target backend cannot support every node. */
static gd_status select_exec_backend(gd_graph *graph,
                                     _gd_backend *target_backend,
                                     _gd_backend **out)
{
    gd_status status = GD_OK;
    int bad_node = -1;
    char reason[256] = "";
    const char *last = NULL;

    *out = NULL;
    status = _gd_backend_check_graph(target_backend, graph, &bad_node);
    if (status == GD_OK) {
        *out = target_backend;
        return GD_OK;
    }

    last = gd_last_error();
    if (last != NULL && last[0] != '\0') {
        (void)snprintf(reason, sizeof(reason), "%s", last);
    }
    if (bad_node < 0 || bad_node >= graph->n_nodes) {
        return status;
    }
    if (target_backend->caps.supports_cpu_ref) {
        char msg[384];
        if (reason[0] != '\0') {
            (void)snprintf(msg, sizeof(msg),
                           "backend '%s' cannot execute op '%s' (node %d): %s",
                           target_backend->vt->name,
                           _gd_op_kind_name(graph->nodes[bad_node].op),
                           bad_node,
                           reason);
        } else {
            (void)snprintf(msg, sizeof(msg),
                           "backend '%s' cannot execute op '%s' (node %d)",
                           target_backend->vt->name,
                           _gd_op_kind_name(graph->nodes[bad_node].op),
                           bad_node);
        }
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }

    if (gd_context_fallback_policy(graph->ctx) != GD_FALLBACK_CPU_REF) {
        char msg[384];
        if (reason[0] != '\0') {
            (void)snprintf(msg, sizeof(msg),
                           "backend '%s' does not support op '%s' (node %d): %s; enable CPU_REF fallback",
                           target_backend->vt->name,
                           _gd_op_kind_name(graph->nodes[bad_node].op),
                           bad_node,
                           reason);
        } else {
            (void)snprintf(msg, sizeof(msg),
                           "backend '%s' does not support op '%s' (node %d); enable CPU_REF fallback",
                           target_backend->vt->name,
                           _gd_op_kind_name(graph->nodes[bad_node].op),
                           bad_node);
        }
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }

    /* v1 fallback: run the entire graph on the CPU reference backend. */
    {
        _gd_backend *cpu = _gd_context_backend(graph->ctx, (gd_device){GD_DEVICE_CPU, 0});
        int cpu_bad_node = -1;
        gd_status cpu_status = GD_OK;

        if (cpu == NULL || !cpu->caps.supports_cpu_ref) {
            return _gd_error(GD_ERR_UNSUPPORTED, "CPU reference fallback is not available");
        }
        if (cpu->vt->compile == NULL || cpu->vt->execute == NULL) {
            return _gd_error(GD_ERR_UNSUPPORTED, "CPU reference backend cannot execute");
        }
        cpu_status = _gd_backend_check_graph(cpu, graph, &cpu_bad_node);
        if (cpu_status != GD_OK) {
            char msg[384];
            const char *cpu_reason = gd_last_error();
            const char *op = (cpu_bad_node >= 0 && cpu_bad_node < graph->n_nodes) ?
                _gd_op_kind_name(graph->nodes[cpu_bad_node].op) : "unknown";
            (void)snprintf(msg, sizeof(msg),
                           "CPU reference fallback cannot execute op '%s' (node %d): %s",
                           op,
                           cpu_bad_node,
                           cpu_reason != NULL && cpu_reason[0] != '\0' ? cpu_reason : "unsupported");
            return _gd_error(GD_ERR_UNSUPPORTED, msg);
        }
        *out = cpu;
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_compile(gd_graph *graph, gd_device target)
{
    gd_status status = GD_OK;
    _gd_backend *target_backend = NULL;
    _gd_backend *backend = NULL;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_compile graph is NULL");
    }
    if (graph->state != _GD_GRAPH_FINALIZED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be finalized before compile");
    }

    target_backend = _gd_context_backend(graph->ctx, target);
    if (target_backend == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for compile target");
    }
    status = gd_graph_validate(graph);
    if (status != GD_OK) {
        return status;
    }

    status = select_exec_backend(graph, target_backend, &backend);
    if (status != GD_OK) {
        return status;
    }
    if (backend->vt->compile == NULL || backend->vt->execute == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not implement execution");
    }

    _gd_graph_free_executable(graph);
    {
        uint64_t start = _gd_profile_enabled(graph->ctx) ? _gd_profile_now_ns() : 0U;
        status = backend->vt->compile(backend, graph, &graph->exec);
        if (start != 0U) {
            _gd_profile_record_compile(graph->ctx, backend, _gd_profile_now_ns() - start,
                                       graph->nodes, graph->n_nodes);
        }
    }
    if (status != GD_OK) {
        graph->exec = NULL;
        return status;
    }

    graph->backend = backend;
    graph->target = target;
    graph->has_target = true;
    graph->has_run = false;
    graph->state = _GD_GRAPH_COMPILED;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_run(gd_graph *graph)
{
    gd_status status = GD_OK;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_run graph is NULL");
    }
    if (graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be compiled before run");
    }

    {
        uint64_t start = _gd_profile_enabled(graph->ctx) ? _gd_profile_now_ns() : 0U;
        status = graph->backend->vt->execute(graph->backend, graph->exec);
        if (start != 0U) {
            _gd_profile_record_run(graph->ctx, graph->backend, _gd_profile_now_ns() - start,
                                   graph->nodes, graph->n_nodes);
        }
    }
    if (status != GD_OK) {
        return status;
    }

    graph->has_run = true;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_graph_value_storage(gd_graph *graph,
                                  int value_id,
                                  bool require_run,
                                  gd_storage **storage_out,
                                  size_t *offset_out,
                                  const gd_tensor_desc **desc_out)
{
    gd_status status = GD_OK;

    if (graph == NULL || storage_out == NULL || offset_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "_gd_graph_value_storage argument is NULL");
    }
    *storage_out = NULL;
    *offset_out = 0U;
    if (desc_out != NULL) {
        *desc_out = NULL;
    }
    if (graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not compiled");
    }
    if (require_run && !graph->has_run) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph has not been run");
    }
    if (value_id < 0 || value_id >= graph->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    if (graph->backend->vt->value_storage == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend cannot expose value storage");
    }

    status = graph->backend->vt->value_storage(graph->backend, graph->exec, value_id,
                                               storage_out, offset_out);
    if (status != GD_OK) {
        return status;
    }
    if (desc_out != NULL) {
        *desc_out = &graph->values[value_id].desc;
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_graph_materialize_live_virtuals(gd_graph *graph)
{
    gd_status status = GD_OK;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph is NULL");
    }
    while (graph->n_virtual > 0) {
        gd_tensor *tensor = graph->virtual_tensors[0];

        status = _gd_tensor_materialize_from_graph(graph->ctx, tensor);
        if (status != GD_OK) {
            return status;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_reset(gd_graph *graph)
{
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_reset graph is NULL");
    }
    if (_gd_context_active_graph(graph->ctx) == graph) {
        return _gd_error(GD_ERR_INVALID_STATE, "cannot reset active graph");
    }
    return _gd_graph_clear(graph);
}

gd_status gd_graph_run_immediate(gd_context *ctx,
                                 gd_device target,
                                 gd_immediate_build_fn build,
                                 void *user)
{
    gd_status status = GD_OK;
    gd_status cleanup_status = GD_OK;
    gd_graph *graph = NULL;

    if (ctx == NULL || build == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "gd_graph_run_immediate argument is NULL");
    }

    status = gd_graph_create(ctx, &graph);
    if (status != GD_OK) {
        return status;
    }
    status = gd_graph_begin(ctx, graph);
    if (status == GD_OK) {
        status = build(ctx, user);
    }
    if (status == GD_OK) {
        status = gd_graph_end(ctx);
    } else if (_gd_context_active_graph(ctx) == graph) {
        cleanup_status = _gd_context_set_active_graph(ctx, NULL);
        if (cleanup_status == GD_OK) {
            graph->state = _GD_GRAPH_EMPTY;
        }
    }
    if (status == GD_OK) {
        status = gd_graph_compile(graph, target);
    }
    if (status == GD_OK) {
        status = gd_graph_run(graph);
    }
    if (status == GD_OK) {
        status = _gd_graph_materialize_live_virtuals(graph);
    }

    cleanup_status = gd_graph_destroy(graph);
    if (status != GD_OK) {
        return status;
    }
    return cleanup_status;
}

gd_status gd_graph_run_until(gd_graph *graph, int node_id)
{
    gd_status status = GD_OK;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_run_until graph is NULL");
    }
    if (node_id < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "node id must be nonnegative");
    }
    if (graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be compiled before partial run");
    }
    if (node_id >= graph->n_nodes) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "node id is out of range");
    }
    if (graph->backend->vt->execute_until == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not support partial execution");
    }

    {
        uint64_t start = _gd_profile_enabled(graph->ctx) ? _gd_profile_now_ns() : 0U;
        status = graph->backend->vt->execute_until(graph->backend, graph->exec, node_id);
        if (start != 0U) {
            _gd_profile_record_run(graph->ctx, graph->backend, _gd_profile_now_ns() - start,
                                   graph->nodes, node_id + 1);
        }
    }
    if (status != GD_OK) {
        return status;
    }
    graph->has_run = true;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
