#include "gradients/graph.h"

#include <stdlib.h>
#include <string.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"
#include "graph_internal.h"

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

static gd_status grow_inputs(gd_graph *graph)
{
    int new_cap = 0;
    gd_graph_input **grown = NULL;

    if (graph->n_inputs < graph->input_cap) {
        return GD_OK;
    }
    new_cap = graph->input_cap == 0 ? 4 : graph->input_cap * 2;
    if (new_cap <= graph->input_cap) {
        return _gd_error(GD_ERR_INTERNAL, "graph input capacity overflow");
    }
    grown = realloc(graph->inputs, (size_t)new_cap * sizeof(*grown));
    if (grown == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow graph inputs");
    }
    graph->inputs = grown;
    graph->input_cap = new_cap;
    return GD_OK;
}

gd_status gd_graph_add_input(gd_context *ctx,
                             gd_graph *graph,
                             const char *name,
                             const gd_tensor_desc *desc,
                             gd_tensor **tensor_out,
                             gd_graph_input **input_out)
{
    gd_status status = GD_OK;
    gd_graph_input *input = NULL;
    gd_tensor_desc desc_copy;
    size_t nbytes = 0U;
    size_t alignment = 0U;
    int value_id = -1;

    if (ctx == NULL || graph == NULL || desc == NULL || tensor_out == NULL ||
        input_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_graph_add_input argument is NULL");
    }
    *tensor_out = NULL;
    *input_out = NULL;
    if (graph->ctx != ctx) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "graph was created by a different context");
    }
    if (graph->state != _GD_GRAPH_BUILDING) {
        return _gd_error(GD_ERR_INVALID_STATE, "graph is not capturing nodes");
    }
    desc_copy = *desc;
    status = gd_tensor_desc_nbytes(&desc_copy, &nbytes, &alignment);
    if (status != GD_OK) {
        return status;
    }
    (void)nbytes;
    (void)alignment;

    status = grow_inputs(graph);
    if (status != GD_OK) {
        return status;
    }
    input = (gd_graph_input *)calloc(1U, sizeof(*input));
    if (input == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph input");
    }
    input->graph = graph;
    input->index = graph->n_inputs;
    input->value_id = -1;
    input->name = dup_or_null(name);
    if (name != NULL && name[0] != '\0' && input->name == NULL) {
        free(input);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph input name");
    }

    status = _gd_graph_add_value(graph, &desc_copy, -1, NULL, _GD_VALUE_INPUT,
                                 input->index, &value_id);
    if (status != GD_OK) {
        free(input->name);
        free(input);
        return status;
    }
    input->value_id = value_id;
    graph->inputs[graph->n_inputs] = input;
    graph->n_inputs += 1;
    if (input->name != NULL) {
        graph->values[value_id].name = dup_or_null(input->name);
        if (graph->values[value_id].name == NULL) {
            graph->n_inputs -= 1;
            graph->n_values -= 1;
            free(input->name);
            free(input);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate graph input value name");
        }
    }

    status = _gd_tensor_create_virtual(graph, value_id, &desc_copy, tensor_out);
    if (status != GD_OK) {
        free(graph->values[value_id].name);
        graph->values[value_id].name = NULL;
        graph->n_inputs -= 1;
        graph->n_values -= 1;
        free(input->name);
        free(input);
        return status;
    }
    *input_out = input;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
