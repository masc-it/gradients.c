#include "gradients/graph.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "../backends/backend.h"
#include "graph_internal.h"

const char *_gd_graph_state_name(_gd_graph_state state)
{
    switch (state) {
    case _GD_GRAPH_EMPTY:
        return "empty";
    case _GD_GRAPH_BUILDING:
        return "building";
    case _GD_GRAPH_FINALIZED:
        return "finalized";
    case _GD_GRAPH_COMPILED:
        return "compiled";
    }
    return "unknown";
}

const char *_gd_op_kind_name(_gd_op_kind op)
{
    switch (op) {
    case _GD_OP_INVALID:
        return "invalid";
    case _GD_OP_ADD:
        return "add";
    case _GD_OP_MUL:
        return "mul";
    case _GD_OP_SCALE:
        return "scale";
    case _GD_OP_MATMUL:
        return "matmul";
    case _GD_OP_LINEAR:
        return "linear";
    case _GD_OP_RELU:
        return "relu";
    case _GD_OP_SILU:
        return "silu";
    case _GD_OP_SUM:
        return "sum";
    case _GD_OP_MEAN:
        return "mean";
    case _GD_OP_RMS_NORM:
        return "rms_norm";
    case _GD_OP_SOFTMAX:
        return "softmax";
    case _GD_OP_CROSS_ENTROPY:
        return "cross_entropy";
    case _GD_OP_CAST:
        return "cast";
    case _GD_OP_GELU:
        return "gelu";
    case _GD_OP_TRANSPOSE:
        return "transpose";
    case _GD_OP_EMBEDDING:
        return "embedding";
    case _GD_OP_ROPE:
        return "rope";
    case _GD_OP_SDPA:
        return "sdpa";
    case _GD_OP_BACKWARD:
        return "backward";
    case _GD_OP_ZERO_GRAD:
        return "zero_grad";
    case _GD_OP_OPTIMIZER_STEP:
        return "optimizer_step";
    case _GD_OP_ASSERT_FINITE:
        return "assert_finite";
    case _GD_OP_ASSERT_CLOSE:
        return "assert_close";
    case _GD_OP_COPY:
        return "copy";
    case _GD_OP_RELU_BWD:
        return "relu_bwd";
    case _GD_OP_SILU_BWD:
        return "silu_bwd";
    case _GD_OP_SOFTMAX_BWD:
        return "softmax_bwd";
    case _GD_OP_SUM_BWD:
        return "sum_bwd";
    case _GD_OP_MEAN_BWD:
        return "mean_bwd";
    case _GD_OP_CROSS_ENTROPY_BWD:
        return "cross_entropy_bwd";
    case _GD_OP_GELU_BWD:
        return "gelu_bwd";
    case _GD_OP_EMBEDDING_BWD:
        return "embedding_bwd";
    case _GD_OP_ROPE_BWD:
        return "rope_bwd";
    case _GD_OP_SDPA_BWD:
        return "sdpa_bwd";
    case _GD_OP_RMS_NORM_BWD:
        return "rms_norm_bwd";
    case _GD_OP_RMS_NORM_WBWD:
        return "rms_norm_wbwd";
    case _GD_OP_STEP_INC:
        return "step_inc";
    case _GD_OP_ADAMW_STEP:
        return "adamw_step";
    case _GD_OP_REDUCE_TO:
        return "reduce_to";
    }
    return "unknown";
}

static void free_node(_gd_node *node)
{
    if (node == NULL) {
        return;
    }
    free(node->inputs);
    free(node->outputs);
    free(node->scope);
    free(node->name);
    *node = (_gd_node){0};
}

static char *dup_or_null(const char *s)
{
    size_t len = 0U;
    char *copy = NULL;

    if (s == NULL || s[0] == '\0') {
        return NULL;
    }
    len = strlen(s);
    copy = malloc(len + 1U);
    if (copy != NULL) {
        memcpy(copy, s, len + 1U);
    }
    return copy;
}

gd_status _gd_graph_set_value_name(gd_graph *graph, int value_id, const char *name)
{
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph is NULL");
    }
    if (value_id < 0 || value_id >= graph->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    free(graph->values[value_id].name);
    graph->values[value_id].name = dup_or_null(name);
    return GD_OK;
}

static void graph_free_executable(gd_graph *graph)
{
    if (graph->exec != NULL && graph->backend != NULL &&
        graph->backend->vt->executable_free != NULL) {
        graph->backend->vt->executable_free(graph->backend, graph->exec);
    }
    graph->exec = NULL;
    graph->backend = NULL;
    graph->has_run = false;
}

gd_status _gd_graph_clear(gd_graph *graph)
{
    int i = 0;

    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph is NULL");
    }
    if (graph->n_virtual != 0) {
        return _gd_error(GD_ERR_INVALID_STATE,
                         "graph has live virtual tensors");
    }

    graph_free_executable(graph);
    for (i = 0; i < graph->n_nodes; ++i) {
        free_node(&graph->nodes[i]);
    }
    for (i = 0; i < graph->n_values; ++i) {
        gd_tensor_release(graph->values[i].external);
        graph->values[i].external = NULL;
        free(graph->values[i].name);
        graph->values[i].name = NULL;
    }
    free(graph->nodes);
    free(graph->values);
    free(graph->virtual_tensors);
    graph->nodes = NULL;
    graph->values = NULL;
    graph->virtual_tensors = NULL;
    graph->n_nodes = 0;
    graph->node_cap = 0;
    graph->n_values = 0;
    graph->value_cap = 0;
    graph->n_virtual = 0;
    graph->virtual_cap = 0;
    graph->has_target = false;
    graph->target = (gd_device){GD_DEVICE_CPU, 0};
    graph->state = _GD_GRAPH_EMPTY;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_graph_note_virtual_tensor_create(gd_graph *graph, gd_tensor *tensor)
{
    if (graph == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "graph/tensor is NULL");
    }
    if (graph->n_virtual == graph->virtual_cap) {
        int new_cap = graph->virtual_cap == 0 ? 8 : graph->virtual_cap * 2;
        gd_tensor **grown = NULL;

        if (new_cap <= graph->virtual_cap) {
            return _gd_error(GD_ERR_INTERNAL, "virtual tensor capacity overflow");
        }
        grown = realloc(graph->virtual_tensors, (size_t)new_cap * sizeof(*grown));
        if (grown == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow virtual tensor list");
        }
        graph->virtual_tensors = grown;
        graph->virtual_cap = new_cap;
    }
    graph->virtual_tensors[graph->n_virtual] = tensor;
    graph->n_virtual += 1;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void _gd_graph_note_virtual_tensor_release(gd_graph *graph, gd_tensor *tensor)
{
    int i = 0;

    if (graph == NULL || tensor == NULL) {
        return;
    }
    for (i = 0; i < graph->n_virtual; ++i) {
        if (graph->virtual_tensors[i] == tensor) {
            graph->virtual_tensors[i] = graph->virtual_tensors[graph->n_virtual - 1];
            graph->n_virtual -= 1;
            break;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
}

static gd_status grow_values(gd_graph *graph)
{
    int new_cap = 0;
    _gd_value *grown = NULL;

    if (graph->n_values < graph->value_cap) {
        return GD_OK;
    }
    new_cap = graph->value_cap == 0 ? 8 : graph->value_cap * 2;
    if (new_cap <= graph->value_cap) {
        return _gd_error(GD_ERR_INTERNAL, "value capacity overflow");
    }
    grown = realloc(graph->values, (size_t)new_cap * sizeof(*grown));
    if (grown == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow graph values");
    }
    graph->values = grown;
    graph->value_cap = new_cap;
    return GD_OK;
}

static gd_status grow_nodes(gd_graph *graph)
{
    int new_cap = 0;
    _gd_node *grown = NULL;

    if (graph->n_nodes < graph->node_cap) {
        return GD_OK;
    }
    new_cap = graph->node_cap == 0 ? 8 : graph->node_cap * 2;
    if (new_cap <= graph->node_cap) {
        return _gd_error(GD_ERR_INTERNAL, "node capacity overflow");
    }
    grown = realloc(graph->nodes, (size_t)new_cap * sizeof(*grown));
    if (grown == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow graph nodes");
    }
    graph->nodes = grown;
    graph->node_cap = new_cap;
    return GD_OK;
}

static gd_status add_value(gd_graph *graph,
                           const gd_tensor_desc *desc,
                           int producer_node_id,
                           gd_tensor *external,
                           int *value_id_out)
{
    gd_status status = GD_OK;
    _gd_value *value = NULL;

    status = grow_values(graph);
    if (status != GD_OK) {
        return status;
    }
    value = &graph->values[graph->n_values];
    value->id = graph->n_values;
    value->producer_node_id = producer_node_id;
    value->desc = *desc;
    value->external = external;
    value->name = NULL; /* values grow via realloc; name must be set before use */
    *value_id_out = value->id;
    graph->n_values += 1;
    return GD_OK;
}

static gd_status import_tensor(gd_graph *graph, gd_tensor *tensor, int *value_id_out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *desc = NULL;
    int i = 0;

    if (_gd_tensor_is_virtual(tensor)) {
        if (_gd_tensor_graph(tensor) != graph) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT,
                             "virtual tensor belongs to a different graph");
        }
        *value_id_out = _gd_tensor_value_id(tensor);
        return GD_OK;
    }

    if (gd_tensor_storage(tensor) == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE,
                         "op input tensor is neither materialized nor a graph value");
    }

    for (i = 0; i < graph->n_values; ++i) {
        if (graph->values[i].external == tensor) {
            *value_id_out = graph->values[i].id;
            return GD_OK;
        }
    }

    desc = _gd_tensor_desc_ptr(tensor);
    if (desc == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "tensor has no descriptor");
    }
    status = gd_tensor_retain(tensor);
    if (status != GD_OK) {
        return status;
    }
    status = add_value(graph, desc, -1, tensor, value_id_out);
    if (status != GD_OK) {
        gd_tensor_release(tensor);
        return status;
    }
    return GD_OK;
}

gd_status _gd_graph_emit(gd_graph *graph,
                         _gd_op_kind op,
                         gd_tensor **inputs,
                         int n_inputs,
                         const _gd_op_attrs *attrs,
                         const gd_tensor_desc *out_desc,
                         gd_tensor **out_tensor)
{
    gd_status status = GD_OK;
    int input_ids[_GD_OP_MAX_INPUTS];
    int *node_inputs = NULL;
    int *node_outputs = NULL;
    int node_id = 0;
    int out_value_id = 0;
    int i = 0;
    _gd_node *node = NULL;
    gd_tensor_desc out_desc_copy;

    if (graph == NULL || out_desc == NULL || out_tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "_gd_graph_emit argument is NULL");
    }
    /* Callers may pass an out_desc that aliases graph->values[...].desc (e.g.
     * autograd reusing an input value's descriptor). import_tensor/add_value
     * grow that array via realloc, which would dangle the pointer. Copy it up
     * front so the emit path never reads through a relocated allocation. */
    out_desc_copy = *out_desc;
    out_desc = &out_desc_copy;
    if (n_inputs < 0 || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "too many op inputs");
    }
    if (n_inputs > 0 && inputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op inputs array is NULL");
    }
    *out_tensor = NULL;

    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not capturing nodes");
    }

    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input tensor is NULL");
        }
        status = import_tensor(graph, inputs[i], &input_ids[i]);
        if (status != GD_OK) {
            return status;
        }
    }

    node_id = graph->n_nodes;
    status = add_value(graph, out_desc, node_id, NULL, &out_value_id);
    if (status != GD_OK) {
        return status;
    }

    status = _gd_tensor_create_virtual(graph, out_value_id, out_desc, out_tensor);
    if (status != GD_OK) {
        graph->n_values -= 1;
        return status;
    }

    status = grow_nodes(graph);
    if (status != GD_OK) {
        goto fail;
    }
    if (n_inputs > 0) {
        node_inputs = malloc((size_t)n_inputs * sizeof(*node_inputs));
        if (node_inputs == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node inputs");
            goto fail;
        }
        for (i = 0; i < n_inputs; ++i) {
            node_inputs[i] = input_ids[i];
        }
    }
    node_outputs = malloc(sizeof(*node_outputs));
    if (node_outputs == NULL) {
        status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node outputs");
        goto fail;
    }
    node_outputs[0] = out_value_id;

    node = &graph->nodes[node_id];
    *node = (_gd_node){0};
    node->id = node_id;
    node->op = op;
    node->inputs = node_inputs;
    node->n_inputs = n_inputs;
    node->outputs = node_outputs;
    node->n_outputs = 1;
    if (attrs != NULL) {
        node->attrs = *attrs;
    }
    node->scope = dup_or_null(_gd_context_scope(graph->ctx));
    graph->n_nodes += 1;

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;

fail:
    free(node_inputs);
    free(node_outputs);
    gd_tensor_release(*out_tensor);
    *out_tensor = NULL;
    graph->n_values -= 1;
    return status;
}

gd_status _gd_graph_emit_multi(gd_graph *graph,
                               _gd_op_kind op,
                               gd_tensor **inputs,
                               int n_inputs,
                               const _gd_op_attrs *attrs,
                               const gd_tensor_desc *out_descs,
                               int n_outputs,
                               gd_tensor **out_tensors)
{
    gd_status status = GD_OK;
    int input_ids[_GD_OP_MAX_INPUTS];
    int out_value_ids[_GD_OP_MAX_INPUTS];
    int *node_inputs = NULL;
    int *node_outputs = NULL;
    int node_id = 0;
    int created = 0;
    int i = 0;
    _gd_node *node = NULL;

    if (graph == NULL || out_descs == NULL || out_tensors == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "_gd_graph_emit_multi argument is NULL");
    }
    if (n_inputs < 0 || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "too many op inputs");
    }
    if (n_outputs < 1 || n_outputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unsupported op output count");
    }
    if (n_inputs > 0 && inputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "op inputs array is NULL");
    }
    for (i = 0; i < n_outputs; ++i) {
        out_tensors[i] = NULL;
    }
    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not capturing nodes");
    }

    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input tensor is NULL");
        }
        status = import_tensor(graph, inputs[i], &input_ids[i]);
        if (status != GD_OK) {
            return status;
        }
    }

    node_id = graph->n_nodes;
    for (i = 0; i < n_outputs; ++i) {
        status = add_value(graph, &out_descs[i], node_id, NULL, &out_value_ids[i]);
        if (status != GD_OK) {
            goto fail;
        }
        status = _gd_tensor_create_virtual(graph, out_value_ids[i], &out_descs[i],
                                           &out_tensors[i]);
        if (status != GD_OK) {
            graph->n_values -= 1; /* discard the value with no tensor */
            goto fail;
        }
        created += 1;
    }

    status = grow_nodes(graph);
    if (status != GD_OK) {
        goto fail;
    }
    if (n_inputs > 0) {
        node_inputs = malloc((size_t)n_inputs * sizeof(*node_inputs));
        if (node_inputs == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node inputs");
            goto fail;
        }
        for (i = 0; i < n_inputs; ++i) {
            node_inputs[i] = input_ids[i];
        }
    }
    node_outputs = malloc((size_t)n_outputs * sizeof(*node_outputs));
    if (node_outputs == NULL) {
        status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node outputs");
        goto fail;
    }
    for (i = 0; i < n_outputs; ++i) {
        node_outputs[i] = out_value_ids[i];
    }

    node = &graph->nodes[node_id];
    *node = (_gd_node){0};
    node->id = node_id;
    node->op = op;
    node->inputs = node_inputs;
    node->n_inputs = n_inputs;
    node->outputs = node_outputs;
    node->n_outputs = n_outputs;
    if (attrs != NULL) {
        node->attrs = *attrs;
    }
    node->scope = dup_or_null(_gd_context_scope(graph->ctx));
    graph->n_nodes += 1;

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;

fail:
    free(node_inputs);
    free(node_outputs);
    for (i = 0; i < created; ++i) {
        gd_tensor_release(out_tensors[i]);
        out_tensors[i] = NULL;
    }
    graph->n_values -= created;
    return status;
}

gd_status _gd_graph_emit_to(gd_graph *graph,
                            _gd_op_kind op,
                            gd_tensor **inputs,
                            int n_inputs,
                            const _gd_op_attrs *attrs,
                            gd_tensor *out_external)
{
    gd_status status = GD_OK;
    int input_ids[_GD_OP_MAX_INPUTS];
    int *node_inputs = NULL;
    int *node_outputs = NULL;
    const gd_tensor_desc *out_desc = NULL;
    int node_id = 0;
    int out_value_id = 0;
    int i = 0;
    _gd_node *node = NULL;

    if (graph == NULL || out_external == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "_gd_graph_emit_to argument is NULL");
    }
    if (n_inputs < 0 || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "too many op inputs");
    }
    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not capturing nodes");
    }
    if (gd_tensor_storage(out_external) == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "emit_to target must be materialized");
    }
    if (!_gd_tensor_is_contiguous(out_external)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "emit_to target must be contiguous");
    }

    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input tensor is NULL");
        }
        status = import_tensor(graph, inputs[i], &input_ids[i]);
        if (status != GD_OK) {
            return status;
        }
    }

    out_desc = _gd_tensor_desc_ptr(out_external);
    node_id = graph->n_nodes;
    status = gd_tensor_retain(out_external);
    if (status != GD_OK) {
        return status;
    }
    status = add_value(graph, out_desc, node_id, out_external, &out_value_id);
    if (status != GD_OK) {
        gd_tensor_release(out_external);
        return status;
    }

    status = grow_nodes(graph);
    if (status != GD_OK) {
        goto fail;
    }
    if (n_inputs > 0) {
        node_inputs = malloc((size_t)n_inputs * sizeof(*node_inputs));
        if (node_inputs == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node inputs");
            goto fail;
        }
        for (i = 0; i < n_inputs; ++i) {
            node_inputs[i] = input_ids[i];
        }
    }
    node_outputs = malloc(sizeof(*node_outputs));
    if (node_outputs == NULL) {
        status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node outputs");
        goto fail;
    }
    node_outputs[0] = out_value_id;

    node = &graph->nodes[node_id];
    *node = (_gd_node){0};
    node->id = node_id;
    node->op = op;
    node->inputs = node_inputs;
    node->n_inputs = n_inputs;
    node->outputs = node_outputs;
    node->n_outputs = 1;
    if (attrs != NULL) {
        node->attrs = *attrs;
    }
    node->scope = dup_or_null(_gd_context_scope(graph->ctx));
    graph->n_nodes += 1;

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;

fail:
    free(node_inputs);
    free(node_outputs);
    graph->values[out_value_id].external = NULL;
    gd_tensor_release(out_external);
    graph->n_values -= 1;
    return status;
}

gd_status _gd_graph_emit_inplace(gd_graph *graph,
                                 _gd_op_kind op,
                                 gd_tensor **inputs,
                                 int n_inputs,
                                 const _gd_op_attrs *attrs)
{
    gd_status status = GD_OK;
    int input_ids[_GD_OP_MAX_INPUTS];
    int *node_inputs = NULL;
    int node_id = 0;
    int i = 0;
    _gd_node *node = NULL;

    if (graph == NULL || inputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "_gd_graph_emit_inplace argument is NULL");
    }
    if (n_inputs <= 0 || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "in-place op needs 1..MAX inputs");
    }
    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not capturing nodes");
    }

    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "op input tensor is NULL");
        }
        status = import_tensor(graph, inputs[i], &input_ids[i]);
        if (status != GD_OK) {
            return status;
        }
    }

    node_id = graph->n_nodes;
    status = grow_nodes(graph);
    if (status != GD_OK) {
        return status;
    }
    node_inputs = malloc((size_t)n_inputs * sizeof(*node_inputs));
    if (node_inputs == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate node inputs");
    }
    for (i = 0; i < n_inputs; ++i) {
        node_inputs[i] = input_ids[i];
    }

    node = &graph->nodes[node_id];
    *node = (_gd_node){0};
    node->id = node_id;
    node->op = op;
    node->inputs = node_inputs;
    node->n_inputs = n_inputs;
    node->outputs = NULL;
    node->n_outputs = 0;
    if (attrs != NULL) {
        node->attrs = *attrs;
    }
    node->scope = dup_or_null(_gd_context_scope(graph->ctx));
    graph->n_nodes += 1;

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_create(gd_context *ctx, gd_graph **out)
{
    gd_graph *graph = NULL;

    if (ctx == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_create argument is NULL");
    }
    *out = NULL;

    graph = calloc(1U, sizeof(*graph));
    if (graph == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph");
    }
    graph->ctx = ctx;
    graph->state = _GD_GRAPH_EMPTY;
    graph->target = (gd_device){GD_DEVICE_CPU, 0};

    *out = graph;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_destroy(gd_graph *graph)
{
    gd_status status = GD_OK;

    if (graph == NULL) {
        return GD_OK;
    }
    if (graph->n_virtual != 0) {
        return _gd_error(GD_ERR_INVALID_STATE,
                         "cannot destroy graph with live virtual tensors");
    }
    if (_gd_context_active_graph(graph->ctx) == graph) {
        return _gd_error(GD_ERR_INVALID_STATE, "cannot destroy active graph");
    }

    status = _gd_graph_clear(graph);
    if (status != GD_OK) {
        return status;
    }
    free(graph);
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_graph_begin(gd_context *ctx, gd_graph *graph)
{
    if (ctx == NULL || graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_begin argument is NULL");
    }
    if (graph->ctx != ctx) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "graph was created by a different context");
    }
    if (_gd_context_active_graph(ctx) != NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "another graph is already active");
    }
    if (graph->state != _GD_GRAPH_EMPTY) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph must be empty to begin capture");
    }

    graph->state = _GD_GRAPH_BUILDING;
    return _gd_context_set_active_graph(ctx, graph);
}

gd_status gd_graph_end(gd_context *ctx)
{
    gd_graph *graph = NULL;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_end ctx is NULL");
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "no active graph to end");
    }
    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "active graph is not building");
    }

    graph->state = _GD_GRAPH_FINALIZED;
    return _gd_context_set_active_graph(ctx, NULL);
}

gd_status gd_graph_validate(gd_graph *graph)
{
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_validate graph is NULL");
    }
    if (graph->state == _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "cannot validate graph while building");
    }
    if (graph->n_nodes < 0 || graph->n_values < 0 ||
        graph->n_nodes > graph->node_cap || graph->n_values > graph->value_cap) {
        return _gd_error(GD_ERR_INTERNAL, "graph has invalid counts");
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

/* Returns the backend that will execute the graph, applying fallback policy when
 * the target backend cannot support every node. */
static gd_status select_exec_backend(gd_graph *graph,
                                     _gd_backend *target_backend,
                                     _gd_backend **out)
{
    int bad_node = -1;
    int i = 0;

    *out = target_backend;
    if (target_backend->vt->supports_node == NULL) {
        return GD_OK; /* NULL => supports all (reference backend) */
    }
    for (i = 0; i < graph->n_nodes; ++i) {
        if (!target_backend->vt->supports_node(target_backend, &graph->nodes[i])) {
            bad_node = i;
            break;
        }
    }
    if (bad_node < 0) {
        return GD_OK;
    }

    if (gd_context_fallback_policy(graph->ctx) != GD_FALLBACK_CPU_REF) {
        char msg[128];
        (void)snprintf(msg, sizeof(msg),
                       "backend '%s' does not support op '%s' (node %d); enable CPU_REF fallback",
                       target_backend->vt->name,
                       _gd_op_kind_name(graph->nodes[bad_node].op),
                       bad_node);
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }

    /* v1 fallback: run the entire graph on the CPU reference backend. */
    {
        _gd_backend *cpu = _gd_context_backend(graph->ctx, (gd_device){GD_DEVICE_CPU, 0});
        if (cpu == NULL || !cpu->caps.supports_cpu_ref) {
            return _gd_error(GD_ERR_UNSUPPORTED, "CPU reference fallback is not available");
        }
        if (cpu->vt->compile == NULL || cpu->vt->execute == NULL) {
            return _gd_error(GD_ERR_UNSUPPORTED, "CPU reference backend cannot execute");
        }
        *out = cpu;
    }
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

    graph_free_executable(graph);
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

gd_status gd_scope_push(gd_context *ctx, const char *name)
{
    return _gd_context_scope_push(ctx, name);
}

gd_status gd_scope_pop(gd_context *ctx)
{
    return _gd_context_scope_pop(ctx);
}

gd_status gd_tensor_materialize(gd_context *ctx, gd_tensor *tensor)
{
    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "gd_tensor_materialize argument is NULL");
    }
    if (gd_tensor_storage(tensor) != NULL) {
        _gd_set_last_error(GD_OK, NULL);
        return GD_OK;
    }
    return _gd_tensor_materialize_from_graph(ctx, tensor);
}

gd_status gd_tensor_to_cpu(gd_context *ctx, gd_tensor *tensor, void *dst, size_t nbytes)
{
    return gd_tensor_copy_to_cpu(ctx, tensor, dst, nbytes);
}

gd_status gd_debug_print_tensor(gd_context *ctx, gd_tensor *tensor, int max_elems)
{
    gd_status status = GD_OK;
    gd_dtype dtype = GD_DTYPE_INVALID;
    size_t elem = 0U;
    int ndim = 0;
    int64_t numel = 1;
    int64_t k = 0;
    int64_t i = 0;
    void *buf = NULL;

    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "gd_debug_print_tensor argument is NULL");
    }
    if (max_elems < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "max_elems must be nonnegative");
    }

    dtype = gd_tensor_dtype(tensor);
    elem = gd_dtype_sizeof(dtype);
    if (elem == 0U || (dtype != GD_DTYPE_F32 && dtype != GD_DTYPE_I32)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "debug print supports F32/I32 tensors");
    }
    ndim = gd_tensor_ndim(tensor);
    for (i = 0; i < ndim; ++i) {
        numel *= gd_tensor_size(tensor, (int)i);
    }
    k = (max_elems > 0 && (int64_t)max_elems < numel) ? (int64_t)max_elems : numel;

    buf = malloc((size_t)k * elem);
    if (k > 0 && buf == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate debug buffer");
    }
    status = gd_tensor_copy_to_cpu(ctx, tensor, buf, (size_t)k * elem);
    if (status != GD_OK) {
        free(buf);
        return status;
    }

    printf("tensor dtype=%s ndim=%d shape=[", gd_dtype_name(dtype), ndim);
    for (i = 0; i < ndim; ++i) {
        printf("%s%lld", i == 0 ? "" : ",", (long long)gd_tensor_size(tensor, (int)i));
    }
    printf("] data=[");
    for (i = 0; i < k; ++i) {
        if (dtype == GD_DTYPE_F32) {
            printf("%s%g", i == 0 ? "" : ", ", (double)((const float *)buf)[i]);
        } else {
            printf("%s%d", i == 0 ? "" : ", ", ((const int32_t *)buf)[i]);
        }
    }
    printf("%s]\n", k < numel ? ", ..." : "");

    free(buf);
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_assert_finite(gd_context *ctx, gd_tensor *tensor)
{
    gd_graph *graph = NULL;

    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_assert_finite argument is NULL");
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "gd_assert_finite requires an active graph");
    }
    return _gd_graph_emit_inplace(graph, _GD_OP_ASSERT_FINITE, &tensor, 1, NULL);
}

gd_status gd_assert_close(gd_context *ctx,
                          gd_tensor *a,
                          gd_tensor *b,
                          float atol,
                          float rtol)
{
    gd_graph *graph = NULL;
    gd_tensor *inputs[2];
    _gd_op_attrs attrs = {0};

    if (ctx == NULL || a == NULL || b == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_assert_close argument is NULL");
    }
    if (atol < 0.0F || rtol < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tolerances must be nonnegative");
    }
    graph = _gd_context_active_graph(ctx);
    if (graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "gd_assert_close requires an active graph");
    }
    attrs.atol = atol;
    attrs.rtol = rtol;
    inputs[0] = a;
    inputs[1] = b;
    return _gd_graph_emit_inplace(graph, _GD_OP_ASSERT_CLOSE, inputs, 2, &attrs);
}
