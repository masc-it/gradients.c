#include "graph_internal.h"

#include <stdlib.h>
#include <string.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"

static int value_is_arena_candidate(const gd_graph *graph, int value_id)
{
    const _gd_value *v = NULL;

    if (graph == NULL || value_id < 0 || value_id >= graph->n_values) {
        return 0;
    }
    v = &graph->values[value_id];
    return v->kind == _GD_VALUE_PRODUCED && v->external == NULL &&
           v->producer_node_id >= 0 && v->desc.storage_offset_bytes == 0;
}

static void mark_exposed_virtual_values(const gd_graph *graph, unsigned char *exposed)
{
    int i = 0;

    if (graph == NULL || exposed == NULL) {
        return;
    }
    for (i = 0; i < graph->n_virtual; ++i) {
        gd_tensor *t = graph->virtual_tensors[i];
        int value_id = -1;
        if (t == NULL || !_gd_tensor_is_virtual(t) || _gd_tensor_graph(t) != graph) {
            continue;
        }
        value_id = _gd_tensor_value_id(t);
        if (value_id >= 0 && value_id < graph->n_values) {
            exposed[value_id] = 1U;
        }
    }
}

static int find_free_slot(const _gd_memory_plan *plan,
                          const int *slot_active_end,
                          int start)
{
    int s = 0;

    for (s = 0; s < plan->n_slots; ++s) {
        if (slot_active_end[s] < start) {
            return s;
        }
    }
    return -1;
}

static gd_status grow_slots(_gd_memory_plan *plan, int **slot_active_end)
{
    int new_count = plan->n_slots + 1;
    size_t *slot_nbytes = NULL;
    size_t *slot_alignment = NULL;
    int *active = NULL;

    if (new_count <= plan->n_slots) {
        return _gd_error(GD_ERR_INTERNAL, "memory plan slot count overflow");
    }
    slot_nbytes = (size_t *)realloc(plan->slot_nbytes,
                                    (size_t)new_count * sizeof(*slot_nbytes));
    if (slot_nbytes == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow memory plan slots");
    }
    plan->slot_nbytes = slot_nbytes;
    slot_alignment = (size_t *)realloc(plan->slot_alignment,
                                       (size_t)new_count * sizeof(*slot_alignment));
    if (slot_alignment == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow memory plan slot alignments");
    }
    plan->slot_alignment = slot_alignment;
    active = (int *)realloc(*slot_active_end, (size_t)new_count * sizeof(*active));
    if (active == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow memory plan active slots");
    }
    *slot_active_end = active;
    plan->slot_nbytes[plan->n_slots] = 0U;
    plan->slot_alignment[plan->n_slots] = 1U;
    (*slot_active_end)[plan->n_slots] = -1;
    plan->n_slots = new_count;
    return GD_OK;
}

gd_status _gd_memory_plan_build(const gd_graph *graph, _gd_memory_plan *plan)
{
    gd_status status = GD_OK;
    int *last_use = NULL;
    int *slot_active_end = NULL;
    unsigned char *exposed = NULL;
    int i = 0;

    if (graph == NULL || plan == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "memory plan argument is NULL");
    }
    memset(plan, 0, sizeof(*plan));
    plan->n_values = graph->n_values;
    if (graph->n_values <= 0) {
        return GD_OK;
    }

    plan->values = (_gd_memory_plan_value *)calloc((size_t)graph->n_values,
                                                   sizeof(*plan->values));
    last_use = (int *)calloc((size_t)graph->n_values, sizeof(*last_use));
    exposed = (unsigned char *)calloc((size_t)graph->n_values, sizeof(*exposed));
    if (plan->values == NULL || last_use == NULL || exposed == NULL) {
        status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate memory plan");
        goto fail;
    }

    for (i = 0; i < graph->n_values; ++i) {
        plan->values[i].slot = -1;
        plan->values[i].start_node = graph->values[i].producer_node_id;
        plan->values[i].end_node = graph->values[i].producer_node_id;
        last_use[i] = graph->values[i].producer_node_id;
    }

    for (i = 0; i < graph->n_nodes; ++i) {
        const _gd_node *node = &graph->nodes[i];
        int j = 0;
        for (j = 0; j < node->n_inputs; ++j) {
            int value_id = node->inputs[j];
            if (value_id >= 0 && value_id < graph->n_values && last_use[value_id] < i) {
                last_use[value_id] = i;
            }
        }
    }

    if (graph->preserve_all_values) {
        memset(exposed, 1, (size_t)graph->n_values);
    } else {
        mark_exposed_virtual_values(graph, exposed);
    }
    for (i = 0; i < graph->n_values; ++i) {
        size_t nbytes = 0U;
        size_t alignment = 1U;
        int start = graph->values[i].producer_node_id;
        int end = last_use[i];
        int slot = -1;

        if (!value_is_arena_candidate(graph, i)) {
            continue;
        }
        status = gd_tensor_desc_nbytes(&graph->values[i].desc, &nbytes, &alignment);
        if (status != GD_OK) {
            goto fail;
        }
        if (nbytes == 0U) {
            continue;
        }
        if (exposed[i] != 0U) {
            end = graph->n_nodes;
        }
        plan->values[i].nbytes = nbytes;
        plan->values[i].alignment = alignment == 0U ? 1U : alignment;
        plan->values[i].start_node = start;
        plan->values[i].end_node = end;

        slot = find_free_slot(plan, slot_active_end, start);
        if (slot < 0) {
            status = grow_slots(plan, &slot_active_end);
            if (status != GD_OK) {
                goto fail;
            }
            slot = plan->n_slots - 1;
        }
        plan->values[i].slot = slot;
        slot_active_end[slot] = end;
        if (nbytes > plan->slot_nbytes[slot]) {
            plan->slot_nbytes[slot] = nbytes;
        }
        if (alignment > plan->slot_alignment[slot]) {
            plan->slot_alignment[slot] = alignment;
        }
    }

    free(last_use);
    free(slot_active_end);
    free(exposed);
    return GD_OK;

fail:
    free(last_use);
    free(slot_active_end);
    free(exposed);
    _gd_memory_plan_free(plan);
    return status;
}

void _gd_memory_plan_free(_gd_memory_plan *plan)
{
    if (plan == NULL) {
        return;
    }
    free(plan->values);
    free(plan->slot_nbytes);
    free(plan->slot_alignment);
    memset(plan, 0, sizeof(*plan));
}
