#import "metal_op.h"

#include <stdlib.h>

/* Long GPT graphs run faster when submitted as a stream of small command
 * buffers instead of one monolithic buffer. On M-series GPUs, large command
 * buffers with repeated scratch-buffer hazards (notably long-context SDPA)
 * hit a scheduler/hazard-tracking cliff. Causal SDPA specializations need a
 * small chunks to stay on the fast sustained path for long training runs. The
 * default keeps enough adjacent MPS GEMMs together while avoiding the long-buffer
 * hazard cliff. Override with GD_METAL_CMD_CHUNK=0 to force the old single-buffer
 * path. */
#define GD_METAL_DEFAULT_CMD_CHUNK 8

static int metal_command_chunk_size(void)
{
    const char *v = getenv("GD_METAL_CMD_CHUNK");
    char *end = NULL;
    long n = GD_METAL_DEFAULT_CMD_CHUNK;

    if (v == NULL || v[0] == '\0') {
        return GD_METAL_DEFAULT_CMD_CHUNK;
    }
    n = strtol(v, &end, 10);
    if (end == v || n <= 0) {
        return 0;
    }
    if (n > 1000000L) {
        return 1000000;
    }
    return (int)n;
}

static bool metal_has_more_encoded_nodes(const _gd_executable *exe, int start, int end)
{
    int i = 0;

    if (exe == NULL || exe->graph == NULL) {
        return false;
    }
    for (i = start; i <= end && i < exe->graph->n_nodes; ++i) {
        if (!exe->node_absorbed[i]) {
            return true;
        }
    }
    return false;
}

static bool metal_needs_stage_now(const _gd_executable *exe)
{
    int i = 0;

    for (i = 0; i < exe->n_values; ++i) {
        const gd_metal_value *v = &exe->values[i];
        gd_storage *src = NULL;

        if (v->external == NULL || v->external_alias) {
            continue;
        }
        src = gd_tensor_storage(v->external);
        if (src == NULL || !v->has_staged || v->staged_source != src ||
            v->staged_version != _gd_storage_version(src)) {
            return true;
        }
    }
    return false;
}

static gd_status metal_stage_copy(gd_context *ctx,
                                  gd_storage *dst_storage,
                                  gd_storage *src_storage,
                                  size_t src_offset,
                                  size_t nbytes)
{
    void *dst = NULL;
    void *src = NULL;
    gd_status status = gd_storage_data_cpu(dst_storage, &dst);
    if (status != GD_OK) {
        return status;
    }
    status = gd_storage_data_cpu(src_storage, &src);
    if (status == GD_OK) {
        memcpy(dst, (const unsigned char *)src + src_offset, nbytes);
        return GD_OK;
    }
    return gd_storage_copy_to_cpu(ctx, src_storage, src_offset, dst, nbytes);
}

static gd_status stage_leaves(_gd_backend *self, _gd_executable *exe)
{
    int i = 0;
    uint64_t start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
    size_t bytes = 0U;
    uint64_t count = 0U;

    for (i = 0; i < exe->n_values; ++i) {
        gd_metal_value *v = &exe->values[i];
        gd_storage *src = NULL;
        size_t nbytes = 0U;
        gd_status status = GD_OK;

        if (v->external == NULL || v->external_alias) {
            continue;
        }
        src = gd_tensor_storage(v->external);
        if (src == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "metal leaf input has no storage");
        }
        nbytes = gd_storage_nbytes(v->storage);
        if (v->has_staged && v->staged_source == src &&
            v->staged_version == _gd_storage_version(src)) {
            continue;
        }
        bytes += nbytes;
        count += 1U;
        status = metal_stage_copy(self->ctx, v->storage, src, v->leaf_offset, nbytes);
        if (status != GD_OK) {
            return status;
        }
        v->staged_source = src;
        v->staged_version = _gd_storage_version(src);
        v->has_staged = true;
    }
    if (start != 0U) {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_STAGE_LEAVES,
                                 _gd_profile_now_ns() - start, bytes, count);
    }
    return GD_OK;
}

static gd_status encode_planned_node(_gd_backend *self,
                                     GDMetalState *st,
                                     id<MTLCommandBuffer> cmd,
                                     __strong id<MTLComputeCommandEncoder> *enc,
                                     _gd_executable *exe,
                                     int node_id,
                                     id<MTLComputePipelineState> pso,
                                     id<MTLComputePipelineState> pso2,
                                     id<MTLComputePipelineState> pso3,
                                     id<MTLBuffer> scratch)
{
    const _gd_node *node = &exe->graph->nodes[node_id];
    const _gd_metal_op *op = NULL;
    _gd_metal_encode_ctx ctx;

    if (exe->node_fused_src[node_id] >= 0) {
        return _gd_metal_encode_fused_head(*enc, pso2, exe, node,
                                           &exe->graph->nodes[exe->node_fused_src[node_id]]);
    }

    op = _gd_metal_op_for(node->op);
    if (op == NULL || op->encode == NULL) {
        char msg[112];
        (void)snprintf(msg, sizeof(msg), "metal has no host entry for op '%s'",
                       _gd_op_kind_name(node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    ctx = (_gd_metal_encode_ctx){
        .backend = self,
        .state = st,
        .command_buffer = cmd,
        .encoder = enc,
        .exe = exe,
        .node = node,
        .node_id = node_id,
        .pso = pso,
        .pso2 = pso2,
        .pso3 = pso3,
        .scratch = scratch,
    };
    return op->encode(&ctx);
}


static gd_status metal_set_active_storage(gd_metal_value *value, gd_storage *storage)
{
    gd_status status = GD_OK;

    if (value == NULL || storage == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "metal active storage argument is NULL");
    }
    if (value->storage == storage) {
        return GD_OK;
    }
    status = gd_storage_retain(storage);
    if (status != GD_OK) {
        return status;
    }
    gd_storage_release(value->storage);
    value->storage = storage;
    return GD_OK;
}

static bool metal_can_alias_binding(_gd_backend *self, gd_tensor *tensor)
{
    gd_storage *storage = NULL;
    const gd_tensor_desc *desc = NULL;
    gd_device device;

    if (self == NULL || tensor == NULL) {
        return false;
    }
    storage = gd_tensor_storage(tensor);
    desc = _gd_tensor_desc_ptr(tensor);
    if (storage == NULL || desc == NULL || desc->storage_offset_bytes != 0) {
        return false;
    }
    device = gd_storage_device(storage);
    return device.type == GD_DEVICE_METAL && device.index == self->device_index;
}

static gd_status metal_apply_runner_bindings(_gd_backend *self,
                                             _gd_executable *exe,
                                             const gd_graph_runner *runner)
{
    gd_status status = _gd_graph_runner_validate_ready(runner);
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    if (runner->graph != exe->graph) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "runner graph mismatch");
    }
    for (i = 0; i < exe->graph->n_inputs; ++i) {
        gd_graph_input *input = exe->graph->inputs[i];
        gd_tensor *tensor = runner->bindings[i];
        gd_storage *storage = NULL;
        const gd_tensor_desc *desc = NULL;
        gd_metal_value *value = NULL;
        bool alias = false;

        if (input == NULL || input->value_id < 0 || input->value_id >= exe->n_values ||
            tensor == NULL || gd_tensor_storage(tensor) == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "invalid metal graph input binding");
        }
        value = &exe->values[input->value_id];
        if (!value->is_input) {
            return _gd_error(GD_ERR_INTERNAL, "metal binding target is not an input");
        }
        storage = gd_tensor_storage(tensor);
        desc = _gd_tensor_desc_ptr(tensor);
        if (desc == NULL) {
            return _gd_error(GD_ERR_INTERNAL, "metal binding tensor has no descriptor");
        }
        alias = metal_can_alias_binding(self, tensor);
        if (alias) {
            status = metal_set_active_storage(value, storage);
            if (status != GD_OK) {
                return status;
            }
            value->external_alias = true;
            value->leaf_offset = 0U;
            value->staged_source = storage;
            value->staged_version = _gd_storage_version(storage);
            value->has_staged = true;
        } else {
            if (value->input_staging == NULL) {
                return _gd_error(GD_ERR_INTERNAL, "metal graph input has no staging storage");
            }
            status = metal_set_active_storage(value, value->input_staging);
            if (status != GD_OK) {
                return status;
            }
            value->external_alias = false;
            value->leaf_offset = (size_t)desc->storage_offset_bytes;
        }
        value->external = tensor; /* borrowed; runner owns binding lifetime */
    }
    return GD_OK;
}

static gd_status metal_execute_range(_gd_backend *self, _gd_executable *exe, int last_node)
{
    GDMetalState *st = _gd_metal_state(self);
    gd_status status = GD_OK;
    int i = 0;

    /* If CPU-backed leaves changed, any prior command buffer using their shadow
     * buffers must complete before the CPU mutates those buffers (Metal coherency).
     * If nothing changed, the next command buffer can be queued immediately. */
    if (exe->needs_stage && metal_needs_stage_now(exe)) {
        status = _gd_metal_synchronize(self);
        if (status != GD_OK) {
            return status;
        }
        status = stage_leaves(self, exe);
        if (status != GD_OK) {
            return status;
        }
    } else {
        _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_STAGE_LEAVES,
                                 0U, 0U, 0U);
    }

    /* Trace mode: encode one node per command buffer and wait on each, so the
     * host-measured time is attributed to that op kind. Serializing like this is
     * far slower than normal execution and is for profiling only. */
    if (_gd_profile_trace_enabled(self->ctx)) {
        status = _gd_metal_synchronize(self);
        if (status != GD_OK) {
            return status;
        }
        for (i = 0; i <= last_node && i < exe->graph->n_nodes; ++i) {
            const _gd_node *node = &exe->graph->nodes[i];
            id<MTLComputePipelineState> pso;
            id<MTLComputePipelineState> pso2;
            if (exe->node_absorbed[i]) {
                continue; /* encoded by its fusion group head */
            }
            pso = (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
            pso2 = (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
            id<MTLComputePipelineState> pso3 = (__bridge id<MTLComputePipelineState>)exe->node_pso3[i];
            id<MTLBuffer> scratch = nil;
            if (exe->node_scratch_bytes[i] > 0U) {
                if (exe->scratch_arena == NULL ||
                    exe->node_scratch_bytes[i] > exe->scratch_arena_bytes) {
                    return _gd_error(GD_ERR_BACKEND, "metal scratch arena too small");
                }
                scratch = (__bridge id<MTLBuffer>)_gd_storage_handle(exe->scratch_arena);
            }
            uint64_t op_start = _gd_profile_now_ns();
            @autoreleasepool {
                id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                status = encode_planned_node(self, st, cmd, &enc, exe, i, pso, pso2, pso3,
                                             scratch);
                if (status != GD_OK) {
                    [enc endEncoding];
                    return status;
                }
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
                if (cmd.status == MTLCommandBufferStatusError) {
                    return _gd_error(GD_ERR_BACKEND, "metal trace command buffer failed");
                }
            }
            _gd_profile_record_op_time(self->ctx, self, (int)node->op,
                                       _gd_profile_now_ns() - op_start);
        }
        st.inFlight = nil;
        if (exe->needs_writeback) {
            status = _gd_metal_writeback_externals(self, exe);
            if (status != GD_OK) {
                return status;
            }
        }
        exe->run_id += 1U;
        return GD_OK;
    }

    {
        uint64_t encode_start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
        int chunk_size = metal_command_chunk_size();
        int encoded_in_cmd = 0;
        int capped_last = last_node < exe->graph->n_nodes ? last_node : exe->graph->n_nodes - 1;
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [st.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

            for (i = 0; i <= last_node && i < exe->graph->n_nodes; ++i) {
                id<MTLComputePipelineState> pso;
                if (exe->node_absorbed[i]) {
                    continue; /* encoded by its fusion group head */
                }
                pso = (__bridge id<MTLComputePipelineState>)exe->node_pso[i];
                id<MTLComputePipelineState> pso2 =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso2[i];
                id<MTLComputePipelineState> pso3 =
                    (__bridge id<MTLComputePipelineState>)exe->node_pso3[i];
                id<MTLBuffer> scratch = nil;
                if (exe->node_scratch_bytes[i] > 0U) {
                    if (exe->scratch_arena == NULL ||
                        exe->node_scratch_bytes[i] > exe->scratch_arena_bytes) {
                        [enc endEncoding];
                        return _gd_error(GD_ERR_BACKEND, "metal scratch arena too small");
                    }
                    scratch = (__bridge id<MTLBuffer>)_gd_storage_handle(exe->scratch_arena);
                }
                status = encode_planned_node(self, st, cmd, &enc, exe, i, pso, pso2, pso3,
                                             scratch);
                if (status != GD_OK) {
                    [enc endEncoding];
                    return status;
                }
                encoded_in_cmd++;
                if (chunk_size > 0 && encoded_in_cmd >= chunk_size &&
                    metal_has_more_encoded_nodes(exe, i + 1, capped_last)) {
                    [enc endEncoding];
                    [cmd commit];
                    [st.inFlightBuffers addObject:cmd];
                    st.inFlight = cmd;
                    cmd = [st.queue commandBuffer];
                    enc = [cmd computeCommandEncoder];
                    encoded_in_cmd = 0;
                }
            }
            [enc endEncoding];
            [cmd commit];
            [st.inFlightBuffers addObject:cmd];
            st.inFlight = cmd;
        }
        if (encode_start != 0U) {
            uint64_t count = 0U;
            if (last_node >= 0) {
                int capped = last_node < exe->graph->n_nodes ? last_node : exe->graph->n_nodes - 1;
                count = capped >= 0 ? (uint64_t)capped + 1U : 0U;
            }
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_ENCODE,
                                     _gd_profile_now_ns() - encode_start, 0U, count);
        }
    }
    if (exe->needs_writeback) {
        /* P4: defer the wait + device->host writeback. The mutated CPU-backed
         * leaves are marked pending; a host read (gd_tensor_copy_to_cpu,
         * gd_storage_data_cpu) or gd_synchronize triggers the flush. P5's
         * skip-unchanged staging keeps the shadow buffers authoritative across
         * runs, so deferring does not lose in-place updates. */
        _gd_metal_register_pending_writeback(self, exe);
    }
    exe->run_id += 1U;
    return GD_OK;
}

gd_status _gd_metal_execute(_gd_backend *self, _gd_executable *exe)
{
    return metal_execute_range(self, exe, exe->graph->n_nodes - 1);
}

gd_status _gd_metal_execute_bound(_gd_backend *self,
                                  _gd_executable *exe,
                                  const gd_graph_runner *runner)
{
    gd_status status = metal_apply_runner_bindings(self, exe, runner);
    if (status != GD_OK) {
        return status;
    }
    return metal_execute_range(self, exe, exe->graph->n_nodes - 1);
}

gd_status _gd_metal_execute_until(_gd_backend *self, _gd_executable *exe, int node_id)
{
    gd_status status = metal_execute_range(self, exe, node_id);
    if (status != GD_OK) {
        return status;
    }
    /* Partial execution is a debugging aid; make results immediately readable. */
    return _gd_metal_synchronize(self);
}

gd_status _gd_metal_value_storage(_gd_backend *self, _gd_executable *exe, int value_id,
                                     gd_storage **storage_out, size_t *offset_out)
{
    (void)self;
    if (value_id < 0 || value_id >= exe->n_values) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id is out of range");
    }
    *storage_out = exe->values[value_id].storage;
    *offset_out = 0U;
    return GD_OK;
}

gd_status _gd_metal_check_node(_gd_backend *self, const gd_graph *graph, const _gd_node *node)
{
    return _gd_metal_support_node(self, graph, node);
}
