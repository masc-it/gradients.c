#include "gradients/graph.h"

#include <stdlib.h>
#include <string.h>

#include "../backends/backend.h"
#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "graph_internal.h"

static int desc_shape_equal(const gd_tensor_desc *a, const gd_tensor_desc *b)
{
    int i = 0;
    if (a->ndim != b->ndim) {
        return 0;
    }
    for (i = 0; i < a->ndim; ++i) {
        if (a->sizes[i] != b->sizes[i]) {
            return 0;
        }
    }
    return 1;
}

static gd_status validate_binding_tensor(const gd_graph_input *input, const gd_tensor *tensor)
{
    const gd_tensor_desc *want = NULL;
    const gd_tensor_desc *got = NULL;

    if (input == NULL || input->graph == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph runner bind argument is NULL");
    }
    if (input->value_id < 0 || input->value_id >= input->graph->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph input value id is out of range");
    }
    if (gd_tensor_storage(tensor) == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph input binding must be materialized");
    }
    if (!_gd_tensor_is_contiguous(tensor)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "graph input binding must be contiguous");
    }
    want = &input->graph->values[input->value_id].desc;
    got = _gd_tensor_desc_ptr(tensor);
    if (got == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "binding tensor has no descriptor");
    }
    if (got->dtype != want->dtype) {
        return _gd_error(GD_ERR_DTYPE, "graph input binding dtype mismatch");
    }
    if (!gd_device_equal(got->device, want->device)) {
        return _gd_error(GD_ERR_DEVICE, "graph input binding device mismatch");
    }
    if (got->layout != want->layout) {
        return _gd_error(GD_ERR_UNSUPPORTED, "graph input binding layout mismatch");
    }
    if (!desc_shape_equal(got, want)) {
        return _gd_error(GD_ERR_SHAPE, "graph input binding shape mismatch");
    }
    return GD_OK;
}

gd_status gd_graph_runner_create(gd_graph *graph, gd_graph_runner **out)
{
    gd_graph_runner *runner = NULL;

    if (graph == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_runner_create argument is NULL");
    }
    *out = NULL;
    if (graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be compiled before creating runner");
    }
    runner = (gd_graph_runner *)calloc(1U, sizeof(*runner));
    if (runner == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph runner");
    }
    runner->graph = graph;
    runner->n_bindings = graph->n_inputs;
    if (graph->n_inputs > 0) {
        runner->bindings = (gd_tensor **)calloc((size_t)graph->n_inputs,
                                                sizeof(*runner->bindings));
        runner->bound = (unsigned char *)calloc((size_t)graph->n_inputs,
                                                sizeof(*runner->bound));
        if (runner->bindings == NULL || runner->bound == NULL) {
            gd_graph_runner_destroy(runner);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph runner bindings");
        }
    }
    *out = runner;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void gd_graph_runner_destroy(gd_graph_runner *runner)
{
    int i = 0;
    if (runner == NULL) {
        return;
    }
    for (i = 0; i < runner->n_bindings; ++i) {
        gd_tensor_release(runner->bindings[i]);
    }
    free(runner->bindings);
    free(runner->bound);
    free(runner);
    _gd_set_last_error(GD_OK, NULL);
}

gd_status gd_graph_runner_bind(gd_graph_runner *runner,
                               gd_graph_input *input,
                               gd_tensor *tensor)
{
    gd_status status = GD_OK;

    if (runner == NULL || input == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_runner_bind argument is NULL");
    }
    if (input->graph != runner->graph) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph input belongs to different graph");
    }
    if (input->index < 0 || input->index >= runner->n_bindings) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph input index is out of range");
    }
    status = validate_binding_tensor(input, tensor);
    if (status != GD_OK) {
        return status;
    }
    if (runner->bindings[input->index] == tensor && runner->bound[input->index] != 0U) {
        return GD_OK;
    }
    status = gd_tensor_retain(tensor);
    if (status != GD_OK) {
        return status;
    }
    gd_tensor_release(runner->bindings[input->index]);
    runner->bindings[input->index] = tensor;
    runner->bound[input->index] = 1U;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_graph_runner_validate_ready(const gd_graph_runner *runner)
{
    int i = 0;

    if (runner == NULL || runner->graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph runner is NULL");
    }
    if (runner->graph->state != _GD_GRAPH_COMPILED) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be compiled before runner run");
    }
    if (runner->n_bindings != runner->graph->n_inputs) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph runner binding table is stale");
    }
    for (i = 0; i < runner->graph->n_inputs; ++i) {
        if (runner->bound == NULL || runner->bound[i] == 0U || runner->bindings[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "graph runner has unbound inputs");
        }
    }
    return GD_OK;
}

gd_status gd_graph_runner_run(gd_graph_runner *runner)
{
    gd_status status = GD_OK;
    gd_graph *graph = NULL;

    status = _gd_graph_runner_validate_ready(runner);
    if (status != GD_OK) {
        return status;
    }
    graph = runner->graph;
    if (graph->n_inputs == 0) {
        status = gd_graph_run(graph);
        if (status != GD_OK) {
            return status;
        }
        return GD_OK;
    }
    if (graph->backend->vt->execute_bound == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not implement bound execution");
    }
    {
        uint64_t start = _gd_profile_enabled(graph->ctx) ? _gd_profile_now_ns() : 0U;
        status = graph->backend->vt->execute_bound(graph->backend, graph->exec, runner);
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
