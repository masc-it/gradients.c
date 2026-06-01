#import "metal_op.h"

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
        if (src == NULL || !v->has_staged || v->staged_version != _gd_storage_version(src)) {
            return true;
        }
    }
    return false;
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
        void *dst = NULL;
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
        if (v->has_staged && v->staged_version == _gd_storage_version(src)) {
            continue;
        }
        bytes += nbytes;
        count += 1U;
        status = gd_storage_data_cpu(v->storage, &dst);
        if (status != GD_OK) {
            return status;
        }
        status = gd_storage_copy_to_cpu(self->ctx, src, v->leaf_offset, dst, nbytes);
        if (status != GD_OK) {
            return status;
        }
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
            return _gd_metal_writeback_externals(self, exe);
        }
        return GD_OK;
    }

    {
        uint64_t encode_start = _gd_profile_enabled(self->ctx) ? _gd_profile_now_ns() : 0U;
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
            }
            [enc endEncoding];
            [cmd commit];
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
    return GD_OK;
}

gd_status _gd_metal_execute(_gd_backend *self, _gd_executable *exe)
{
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

bool _gd_metal_supports_node(_gd_backend *self, const _gd_node *node)
{
    return _gd_metal_support_node(self, node) == GD_OK;
}
