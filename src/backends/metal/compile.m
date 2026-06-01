#import "metal_internal.h"

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

static bool node_mutates_input(const _gd_node *node, int input_index)
{
    switch (node->op) {
    case _GD_OP_STEP_INC:
        return input_index == 0;
    case _GD_OP_CLIP_GRAD_NORM:
        return true;
    case _GD_OP_ADAMW_STEP:
        return input_index == 0 || input_index == 2 || input_index == 3;
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
            if (node_mutates_input(node, j) && value_id >= 0 && value_id < exe->n_values &&
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
            _gd_op_kind op = node->op;
            exe->node_pso[j] = (__bridge void *)_gd_metal_pipeline_for(st, op);
            if (st.useMPS && (op == _GD_OP_LINEAR || op == _GD_OP_MATMUL)) {
                const gd_tensor_desc *a_desc = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *b_desc = &graph->values[node->inputs[1]].desc;
                const gd_tensor_desc *out_desc = &graph->values[node->outputs[0]].desc;
                if (a_desc->dtype == GD_DTYPE_F32 && b_desc->dtype == GD_DTYPE_F32 &&
                    out_desc->dtype == GD_DTYPE_F32 && a_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    b_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    out_desc->layout == GD_LAYOUT_CONTIGUOUS &&
                    a_desc->storage_offset_bytes == 0 && b_desc->storage_offset_bytes == 0 &&
                    out_desc->storage_offset_bytes == 0) {
                    bool ok = false;
                    bool trans_left = false;
                    bool trans_right = node->attrs.trans_b ? true : false;
                    int result_rows = 0;
                    int result_cols = 0;
                    int inner = 0;
                    NSUInteger batch = 1U;
                    NSUInteger a_rows_desc = 0U;
                    NSUInteger a_cols_desc = 0U;
                    NSUInteger b_rows_desc = 0U;
                    NSUInteger b_cols_desc = 0U;
                    NSUInteger out_rows_desc = 0U;
                    NSUInteger out_cols_desc = 0U;
                    if (op == _GD_OP_LINEAR && !node->attrs.has_bias) {
                        inner = (int)a_desc->sizes[a_desc->ndim - 1];
                        result_cols = (int)out_desc->sizes[out_desc->ndim - 1];
                        result_rows = inner > 0 ? (int)(_gd_metal_desc_numel(a_desc) / inner) : 0;
                        a_rows_desc = (NSUInteger)result_rows;
                        a_cols_desc = (NSUInteger)inner;
                        b_rows_desc = (NSUInteger)b_desc->sizes[0];
                        b_cols_desc = (NSUInteger)b_desc->sizes[1];
                        out_rows_desc = (NSUInteger)result_rows;
                        out_cols_desc = (NSUInteger)result_cols;
                        ok = true;
                    } else if (op == _GD_OP_MATMUL && !node->attrs.trans_a && b_desc->ndim == 2) {
                        int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                        int b_rows_i = (int)b_desc->sizes[0];
                        int b_cols_i = (int)b_desc->sizes[1];
                        inner = a_cols_i;
                        result_cols = trans_right ? b_rows_i : b_cols_i;
                        result_rows = inner > 0 ? (int)(_gd_metal_desc_numel(a_desc) / inner) : 0;
                        a_rows_desc = (NSUInteger)result_rows;
                        a_cols_desc = (NSUInteger)inner;
                        b_rows_desc = (NSUInteger)b_rows_i;
                        b_cols_desc = (NSUInteger)b_cols_i;
                        out_rows_desc = (NSUInteger)result_rows;
                        out_cols_desc = (NSUInteger)result_cols;
                        ok = (out_desc->sizes[out_desc->ndim - 1] == result_cols);
                    } else if (op == _GD_OP_MATMUL && out_desc->ndim >= 3 &&
                               a_desc->ndim == out_desc->ndim &&
                               b_desc->ndim == out_desc->ndim) {
                        bool same_batch = true;
                        int batch_ndim = out_desc->ndim - 2;
                        int a_rows_i = (int)a_desc->sizes[a_desc->ndim - 2];
                        int a_cols_i = (int)a_desc->sizes[a_desc->ndim - 1];
                        int b_rows_i = (int)b_desc->sizes[b_desc->ndim - 2];
                        int b_cols_i = (int)b_desc->sizes[b_desc->ndim - 1];
                        int i_batch = 0;
                        for (i_batch = 0; i_batch < batch_ndim; ++i_batch) {
                            if (a_desc->sizes[i_batch] != out_desc->sizes[i_batch] ||
                                b_desc->sizes[i_batch] != out_desc->sizes[i_batch]) {
                                same_batch = false;
                                break;
                            }
                            batch *= (NSUInteger)out_desc->sizes[i_batch];
                        }
                        trans_left = node->attrs.trans_a ? true : false;
                        inner = trans_left ? a_rows_i : a_cols_i;
                        result_rows = trans_left ? a_cols_i : a_rows_i;
                        result_cols = trans_right ? b_rows_i : b_cols_i;
                        a_rows_desc = (NSUInteger)a_rows_i;
                        a_cols_desc = (NSUInteger)a_cols_i;
                        b_rows_desc = (NSUInteger)b_rows_i;
                        b_cols_desc = (NSUInteger)b_cols_i;
                        out_rows_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 2];
                        out_cols_desc = (NSUInteger)out_desc->sizes[out_desc->ndim - 1];
                        ok = same_batch && inner == (trans_right ? b_cols_i : b_rows_i) &&
                             out_desc->sizes[out_desc->ndim - 2] == result_rows &&
                             out_desc->sizes[out_desc->ndim - 1] == result_cols;
                    }
                    if (ok && result_rows > 0 && result_cols > 0 && inner > 0 && batch > 0U) {
                        NSUInteger a_row_bytes = a_cols_desc * sizeof(float);
                        NSUInteger b_row_bytes = b_cols_desc * sizeof(float);
                        NSUInteger out_row_bytes = out_cols_desc * sizeof(float);
                        MPSMatrixDescriptor *ad = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:a_rows_desc
                                             columns:a_cols_desc
                                            matrices:batch
                                            rowBytes:a_row_bytes
                                         matrixBytes:a_rows_desc * a_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor *bd = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:b_rows_desc
                                             columns:b_cols_desc
                                            matrices:batch
                                            rowBytes:b_row_bytes
                                         matrixBytes:b_rows_desc * b_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor *od = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:out_rows_desc
                                             columns:out_cols_desc
                                            matrices:batch
                                            rowBytes:out_row_bytes
                                         matrixBytes:out_rows_desc * out_row_bytes
                                            dataType:MPSDataTypeFloat32];
                        GDMPSGemmPlan *plan = [GDMPSGemmPlan new];
                        plan.kernel = [[MPSMatrixMultiplication alloc]
                            initWithDevice:st.device
                            transposeLeft:trans_left
                            transposeRight:trans_right
                            resultRows:(NSUInteger)result_rows
                            resultColumns:(NSUInteger)result_cols
                            interiorColumns:(NSUInteger)inner
                            alpha:1.0
                            beta:0.0];
                        plan.kernel.batchStart = 0U;
                        plan.kernel.batchSize = batch;
                        plan.left = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[0]].storage)
                                    offset:0
                                descriptor:ad];
                        plan.right = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->inputs[1]].storage)
                                    offset:0
                                descriptor:bd];
                        plan.result = [[MPSMatrix alloc]
                            initWithBuffer:(__bridge id<MTLBuffer>)_gd_storage_handle(exe->values[node->outputs[0]].storage)
                                    offset:0
                                descriptor:od];
                        if (plan.kernel != nil && plan.left != nil && plan.right != nil &&
                            plan.result != nil) {
                            exe->node_mps[j] = (void *)CFBridgingRetain(plan);
                        }
                    }
                }
            }
            if (op == _GD_OP_CLIP_GRAD_NORM) {
                int in_i = 0;
                int64_t total_groups = 0;
                int64_t reduce_groups = 0;
                exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_clip_norm_reduce");
                exe->node_pso3[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_clip_norm_scale");
                if (exe->node_pso2[j] == NULL || exe->node_pso3[j] == NULL) {
                    status = _gd_error(GD_ERR_BACKEND, "metal clip_grad_norm kernels missing");
                    goto fail;
                }
                for (in_i = 0; in_i < node->n_inputs; ++in_i) {
                    const gd_tensor_desc *gd = &graph->values[node->inputs[in_i]].desc;
                    int64_t numel = _gd_metal_desc_numel(gd);
                    total_groups += (numel + GD_METAL_CLIP_NORM_CHUNK - 1) /
                                    GD_METAL_CLIP_NORM_CHUNK;
                }
                if (total_groups <= 0) {
                    status = _gd_error(GD_ERR_SHAPE, "clip_grad_norm has no elements");
                    goto fail;
                }
                reduce_groups = (total_groups + GD_METAL_CLIP_NORM_TG - 1) /
                                GD_METAL_CLIP_NORM_TG;
                exe->node_scratch_bytes[j] = (size_t)(total_groups + reduce_groups + 1) *
                                             sizeof(float);
                if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                    scratch_max_bytes = exe->node_scratch_bytes[j];
                }
            } else if (op == _GD_OP_RMS_NORM_WBWD) {
                const gd_tensor_desc *x_desc = &graph->values[node->inputs[0]].desc;
                int last = (int)x_desc->sizes[x_desc->ndim - 1];
                int rows = last > 0 ? (int)(_gd_metal_desc_numel(x_desc) / last) : 0;
                int row_blocks = (rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
                exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_rms_norm_wbwd_reduce");
                if (row_blocks > 0 && last > 0) {
                    exe->node_scratch_bytes[j] = (size_t)row_blocks * (size_t)last *
                                                 sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_EMBEDDING_BWD) {
                exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_embedding_bwd_scatter");
            } else if (op == _GD_OP_CROSS_ENTROPY) {
                const gd_tensor_desc *logits = &graph->values[node->inputs[0]].desc;
                int dim = node->attrs.dim;
                int64_t positions = 0;
                if (dim < 0) {
                    dim += logits->ndim;
                }
                if (dim >= 0 && dim < logits->ndim && logits->sizes[dim] > 0) {
                    positions = _gd_metal_desc_numel(logits) / logits->sizes[dim];
                }
                exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_cross_entropy_reduce");
                if (positions > 0) {
                    exe->node_scratch_bytes[j] = (size_t)positions * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_LM_CROSS_ENTROPY ||
                       op == _GD_OP_LM_CROSS_ENTROPY_BWD) {
                const gd_tensor_desc *x_desc = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *w_desc = &graph->values[node->inputs[1]].desc;
                int D = (int)x_desc->sizes[x_desc->ndim - 1];
                int V = (int)w_desc->sizes[0];
                int rows = D > 0 ? (int)(_gd_metal_desc_numel(x_desc) / D) : 0;
                int chunk = V < GD_METAL_LMCE_CHUNK ? V : GD_METAL_LMCE_CHUNK;
                if (rows > 0 && chunk > 0) {
                    if (op == _GD_OP_LM_CROSS_ENTROPY) {
                        exe->node_scratch_bytes[j] = _gd_metal_lmce_fwd_scratch_layout_for(rows, chunk).total;
                    } else {
                        exe->node_scratch_bytes[j] = _gd_metal_lmce_bwd_scratch_bytes_for(rows, chunk);
                    }
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_SDPA) {
                /* Long-context forward uses split-K: scratch holds per-split
                 * partial (acc[Dh], m, l) for each (b, hq, query). Only allocated
                 * when the tiled path is eligible (Dh <= DHT) and >1 split. */
                const gd_tensor_desc *qd = &graph->values[node->inputs[0]].desc;
                const gd_tensor_desc *kd = &graph->values[node->inputs[1]].desc;
                int Dh = (int)qd->sizes[3];
                int Tk = (int)kd->sizes[1];
                int nsplit = _gd_metal_sdpa_num_splits(Tk);
                if (Dh <= GD_METAL_SDPA_DHT && nsplit > 1) {
                    int64_t npart = qd->sizes[0] * qd->sizes[2] * qd->sizes[1]
                                    * nsplit * (Dh + 2);
                    exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_sdpa_splitk");
                    exe->node_pso3[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_sdpa_combine");
                    exe->node_scratch_bytes[j] = (size_t)npart * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
            } else if (op == _GD_OP_SDPA_BWD) {
                exe->node_pso2[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_sdpa_bwd_dkv");
                exe->node_pso3[j] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_sdpa_bwd_stats");
                /* Scratch holds the per-query stats [m, l, D] (B*Hq*Tq*3). For
                 * long context the split-K path additionally carves per-split
                 * partial regions out of the same buffer; size it accordingly.
                 * q is input 1: [B, Tq, Hq, Dh]; k is input 2: [B, Tk, Hkv, Dh]. */
                {
                    const gd_tensor_desc *qd = &graph->values[node->inputs[1]].desc;
                    const gd_tensor_desc *kd = &graph->values[node->inputs[2]].desc;
                    int B = (int)qd->sizes[0];
                    int Tq = (int)qd->sizes[1];
                    int Hq = (int)qd->sizes[2];
                    int Dh = (int)qd->sizes[3];
                    int Tk = (int)kd->sizes[1];
                    int Hkv = (int)kd->sizes[2];
                    int nsplit = _gd_metal_sdpa_num_splits(Tk > Tq ? Tk : Tq);
                    int64_t nfloats;
                    if (Dh <= GD_METAL_SDPA_DHT && nsplit > 1) {
                        nfloats = _gd_metal_sdpa_bwd_scratch_layout(B, Hq, Hkv, Tq, Tk, Dh, nsplit).total;
                    } else {
                        nfloats = (int64_t)B * Hq * Tq * 3;
                    }
                    exe->node_scratch_bytes[j] = (size_t)nfloats * sizeof(float);
                    if (exe->node_scratch_bytes[j] > scratch_max_bytes) {
                        scratch_max_bytes = exe->node_scratch_bytes[j];
                    }
                }
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
