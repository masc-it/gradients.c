#import "metal_op.h"

void _gd_metal_executable_free(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;

    if (exe == NULL) {
        return;
    }
    /* Ensure no in-flight command buffer still references these buffers. */
    (void)_gd_metal_synchronize(self);
    if (exe->values != NULL) {
        for (i = 0; i < exe->n_values; ++i) {
            gd_storage_release(exe->values[i].storage);
        }
        free(exe->values);
    }
    /* node_pso* hold unretained __bridge pointers; just free the arrays. */
    free(exe->node_pso);
    free(exe->node_pso2);
    free(exe->node_pso3);
    if (exe->node_mps != NULL) {
        for (i = 0; i < exe->n_plan; ++i) {
            if (exe->node_mps[i] != NULL) {
                CFRelease(exe->node_mps[i]);
            }
        }
        free(exe->node_mps);
    }
    free(exe->node_absorbed);
    free(exe->node_fused_src);
    free(exe->node_scratch_bytes);
    gd_storage_release(exe->scratch_arena);
    free(exe);
}

static bool can_alias_copy_value(const gd_graph *graph, const _gd_node *node)
{
    const _gd_value *in = NULL;
    const _gd_value *out = NULL;
    size_t in_bytes = 0U;
    size_t out_bytes = 0U;
    size_t alignment = 0U;

    if (graph == NULL || node == NULL || node->op != _GD_OP_COPY ||
        node->n_inputs != 1 || node->n_outputs != 1) {
        return false;
    }
    if (node->inputs[0] < 0 || node->inputs[0] >= graph->n_values ||
        node->outputs[0] < 0 || node->outputs[0] >= graph->n_values) {
        return false;
    }
    in = &graph->values[node->inputs[0]];
    out = &graph->values[node->outputs[0]];
    if (out->external != NULL) {
        return false; /* emit_to copy must write the external target. */
    }
    if (in->desc.dtype != out->desc.dtype || !gd_device_equal(in->desc.device, out->desc.device) ||
        in->desc.quant != out->desc.quant ||
        in->desc.storage_offset_bytes != 0 || out->desc.storage_offset_bytes != 0 ||
        in->desc.layout != GD_LAYOUT_CONTIGUOUS || out->desc.layout != GD_LAYOUT_CONTIGUOUS) {
        return false;
    }
    if (gd_tensor_desc_nbytes(&in->desc, &in_bytes, &alignment) != GD_OK ||
        gd_tensor_desc_nbytes(&out->desc, &out_bytes, &alignment) != GD_OK) {
        _gd_set_last_error(GD_OK, NULL);
        return false;
    }
    return in_bytes == out_bytes;
}

static bool node_mutates_input(const gd_graph *graph, const _gd_node *node, int input_index)
{
    switch (node->op) {
    case _GD_OP_STEP_INC:
        return input_index == 0;
    case _GD_OP_CLIP_GRAD_NORM:
        return true;
    case _GD_OP_AMP_CLIP_GRAD_NORM:
        return input_index >= 1;
    case _GD_OP_ADAMW_STEP:
        return input_index == 0 || input_index == 2 || input_index == 3;
    case _GD_OP_ADAMW_STEP_AMP:
        if (input_index == 0 || input_index == 2 || input_index == 3) { return true; }
        if (node->n_inputs == 8) { return input_index == 7; }
        if (node->n_inputs == 7 && input_index == 6) {
            const _gd_value *extra = &graph->values[node->inputs[6]];
            return !(extra->desc.dtype == GD_DTYPE_F32 && extra->desc.ndim == 0);
        }
        return false;
    case _GD_OP_AMP_UNSCALE_GRAD:
        return input_index == 0 || input_index == 2;
    case _GD_OP_AMP_STEP_INC:
        return input_index == 0;
    case _GD_OP_AMP_REFRESH_PARAM:
        return input_index == 0;
    default:
        return false;
    }
}

static void mark_external_writebacks(_gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->graph->n_nodes; ++i) {
        const _gd_node *node = &exe->graph->nodes[i];
        int j = 0;

        for (j = 0; j < node->n_outputs; ++j) {
            int value_id = node->outputs[j];
            if (value_id >= 0 && value_id < exe->n_values &&
                exe->values[value_id].external != NULL && !exe->values[value_id].external_alias) {
                exe->values[value_id].needs_writeback = true;
            }
        }
        for (j = 0; j < node->n_inputs; ++j) {
            int value_id = node->inputs[j];
            if (node_mutates_input(exe->graph, node, j) && value_id >= 0 && value_id < exe->n_values &&
                exe->values[value_id].external != NULL && !exe->values[value_id].external_alias) {
                exe->values[value_id].needs_writeback = true;
            }
        }
    }
}

static void refresh_external_flags(_gd_executable *exe)
{
    int i = 0;

    exe->needs_stage = false;
    exe->needs_writeback = false;
    for (i = 0; i < exe->n_values; ++i) {
        if (exe->values[i].external != NULL && !exe->values[i].external_alias) {
            exe->needs_stage = true;
        }
        if (exe->values[i].needs_writeback) {
            exe->needs_writeback = true;
        }
    }
}

static gd_status alias_copy_values(_gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->graph->n_nodes; ++i) {
        const _gd_node *node = &exe->graph->nodes[i];
        int input = 0;
        int output = 0;
        gd_status status = GD_OK;

        if (!can_alias_copy_value(exe->graph, node)) {
            continue;
        }
        input = node->inputs[0];
        output = node->outputs[0];
        if (exe->values[input].storage == exe->values[output].storage) {
            continue;
        }
        status = gd_storage_retain(exe->values[input].storage);
        if (status != GD_OK) {
            return status;
        }
        gd_storage_release(exe->values[output].storage);
        exe->values[output].storage = exe->values[input].storage;
        exe->values[output].leaf_offset = exe->values[input].leaf_offset;
    }
    return GD_OK;
}

gd_status _gd_metal_compile(_gd_backend *self, gd_graph *graph, _gd_executable **out)
{
    gd_status status = GD_OK;
    _gd_executable *exe = NULL;
    int n = graph->n_values;
    int i = 0;

    *out = NULL;
    exe = calloc(1U, sizeof(*exe));
    if (exe == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal executable");
    }
    exe->graph = graph;
    exe->n_values = n;
    exe->values = calloc((size_t)(n > 0 ? n : 1), sizeof(*exe->values));
    if (exe->values == NULL) {
        free(exe);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal value table");
    }

    for (i = 0; i < n; ++i) {
        const _gd_value *value = &graph->values[i];
        gd_storage_desc sdesc;
        gd_storage *storage = NULL;
        size_t nbytes = 0U;
        size_t alignment = 0U;

        if (value->external != NULL && !_gd_tensor_is_contiguous(value->external)) {
            status = _gd_error(GD_ERR_UNSUPPORTED, "metal requires contiguous leaf inputs");
            goto fail;
        }
        status = gd_tensor_desc_nbytes(&value->desc, &nbytes, &alignment);
        if (status != GD_OK) {
            goto fail;
        }
        if (value->external != NULL &&
            gd_tensor_storage(value->external) != NULL &&
            gd_storage_device(gd_tensor_storage(value->external)).type == GD_DEVICE_METAL &&
            gd_storage_device(gd_tensor_storage(value->external)).index == self->device_index &&
            value->desc.storage_offset_bytes == 0) {
            storage = gd_tensor_storage(value->external);
            status = gd_storage_retain(storage);
            if (status != GD_OK) {
                goto fail;
            }
            exe->values[i].external_alias = true;
        } else {
            sdesc = (gd_storage_desc){{GD_DEVICE_METAL, self->device_index}, GD_MEM_UNIFIED,
                                      nbytes, alignment};
            status = gd_storage_create(graph->ctx, &sdesc, &storage);
            if (status != GD_OK) {
                goto fail;
            }
        }
        exe->values[i].storage = storage;
        exe->values[i].external = value->external; /* borrowed; graph retains it */
        exe->values[i].leaf_offset =
            value->external != NULL ? (size_t)value->desc.storage_offset_bytes : 0U;
    }

    status = alias_copy_values(exe);
    if (status != GD_OK) {
        goto fail;
    }
    mark_external_writebacks(exe);
    refresh_external_flags(exe);

    /* Resolve each node's pipeline once (P8): the per-run encode loop then does
     * no NSDictionary lookups. sdpa_bwd needs a second pipeline (dkv). */
    exe->n_plan = graph->n_nodes;
    if (graph->n_nodes > 0) {
        GDMetalState *st = _gd_metal_state(self);
        int j = 0;
        size_t scratch_max_bytes = 0U;

        exe->node_pso = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso));
        exe->node_pso2 = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso2));
        exe->node_pso3 = calloc((size_t)graph->n_nodes, sizeof(*exe->node_pso3));
        exe->node_mps = calloc((size_t)graph->n_nodes, sizeof(*exe->node_mps));
        exe->node_scratch_bytes = calloc((size_t)graph->n_nodes,
                                         sizeof(*exe->node_scratch_bytes));
        exe->node_absorbed = calloc((size_t)graph->n_nodes, sizeof(*exe->node_absorbed));
        exe->node_fused_src = calloc((size_t)graph->n_nodes, sizeof(*exe->node_fused_src));
        if (exe->node_pso == NULL || exe->node_pso2 == NULL || exe->node_pso3 == NULL ||
            exe->node_mps == NULL || exe->node_scratch_bytes == NULL ||
            exe->node_absorbed == NULL || exe->node_fused_src == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate metal encode plan");
            goto fail;
        }
        memset(exe->node_fused_src, 0xFF, (size_t)graph->n_nodes * sizeof(*exe->node_fused_src)); /* -1 */
        for (j = 0; j < graph->n_nodes; ++j) {
            const _gd_node *node = &graph->nodes[j];
            const _gd_metal_op *op = _gd_metal_op_for(node->op);
            _gd_metal_plan_ctx plan_ctx;

            if (op == NULL || op->encode == NULL) {
                char msg[112];
                (void)snprintf(msg, sizeof(msg), "metal has no host entry for op '%s'",
                               _gd_op_kind_name(node->op));
                status = _gd_error(GD_ERR_UNSUPPORTED, msg);
                goto fail;
            }
            plan_ctx = (_gd_metal_plan_ctx){
                .backend = self,
                .state = st,
                .graph = graph,
                .exe = exe,
                .node = node,
                .node_id = j,
            };
            status = op->support != NULL ? op->support(&plan_ctx)
                                         : _gd_metal_support_default(&plan_ctx);
            if (status != GD_OK) {
                goto fail;
            }
            status = op->plan != NULL ? op->plan(&plan_ctx) : _gd_metal_plan_default(&plan_ctx);
            if (status != GD_OK) {
                goto fail;
            }
            if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                scratch_max_bytes = exe->node_scratch_bytes[j];
            }
        }
        if (scratch_max_bytes > 0U) {
            gd_storage_desc sdesc = {{GD_DEVICE_METAL, self->device_index}, GD_MEM_UNIFIED,
                                     scratch_max_bytes, sizeof(float)};
            status = gd_storage_create(graph->ctx, &sdesc, &exe->scratch_arena);
            if (status != GD_OK) {
                goto fail;
            }
            exe->scratch_arena_bytes = scratch_max_bytes;
        }
        _gd_metal_plan_fusions(self, graph, exe);
    }

    *out = exe;
    return GD_OK;

fail:
    _gd_metal_executable_free(self, exe);
    return status;
}

/* ---- Execute ------------------------------------------------------------- */
