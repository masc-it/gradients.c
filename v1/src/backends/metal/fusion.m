#import "metal_internal.h"

static unsigned long g_metal_fusions_applied = 0UL;
unsigned long _gd_metal_fusions_applied(void);
unsigned long _gd_metal_fusions_applied(void) { return g_metal_fusions_applied; }

/* Fusion planner (GPU_FUSED, Option A). Recognizes fusible node groups in the
 * (op-granular, shared) IR and marks absorbed nodes so `execute` skips them,
 * binding each group head to a fused kernel. CPU_REF still runs the unfused
 * nodes as the oracle; parity holds at group boundaries.
 *
 * Phase F0 ships this hook with NO patterns: node_absorbed stays all-zero, so
 * the executable runs every node exactly as the unfused plan (identity pass).
 * Patterns are added incrementally; see docs/metal_gpu_fuse.md. */
/* Returns the lowest node index that consumes `value_id`, or INT_MAX if none.
 * Used as the legality gate when a fused head relocates an intermediate's
 * production to its own (later) position: every consumer must run at or after
 * the head, else it would read the value before the fused kernel writes it. */
static int value_min_consumer(const gd_graph *graph, int value_id)
{
    int best = INT_MAX;
    int i = 0;
    for (i = 0; i < graph->n_nodes; ++i) {
        const _gd_node *node = &graph->nodes[i];
        int j = 0;
        for (j = 0; j < node->n_inputs; ++j) {
            if (node->inputs[j] == value_id) {
                if (i < best) {
                    best = i;
                }
                break;
            }
        }
    }
    return best;
}

/* F1 pattern: silu(gate) -> mul(., up). The mul (head) absorbs the silu and is
 * encoded by gd_silu_mul, which writes both hh = silu(gate)*up and act =
 * silu(gate). act stays materialized (the unfused backward reads it), so the
 * win is one fewer dispatch plus the mul no longer reloading act from DRAM.
 * Legality: head is MUL; its first input is produced by a single-output SILU;
 * silu's act is not consumed by any node before the head (so relocating its
 * production to the head is safe); inputs share the head's shape (no
 * broadcast). */
static bool try_fuse_silu_mul(GDMetalState *st, const gd_graph *graph,
                              _gd_executable *exe, int head_idx)
{
    const _gd_node *head = &graph->nodes[head_idx];
    int act_id = 0;
    int silu_idx = 0;
    const _gd_node *silu = NULL;
    const gd_tensor_desc *out_d = NULL;
    const gd_tensor_desc *up_d = NULL;

    if (head->op != _GD_OP_MUL || head->n_inputs != 2 || head->n_outputs != 1) {
        return false;
    }
    act_id = head->inputs[0];
    if (act_id < 0 || act_id >= exe->n_values) {
        return false;
    }
    silu_idx = exe->graph->values[act_id].producer_node_id;
    if (silu_idx < 0 || silu_idx >= graph->n_nodes) {
        return false;
    }
    silu = &graph->nodes[silu_idx];
    if (silu->op != _GD_OP_SILU || silu->n_inputs != 1 || silu->n_outputs != 1 ||
        silu->outputs[0] != act_id) {
        return false;
    }
    /* No node may read `act` before the head (production is relocated to head). */
    if (value_min_consumer(graph, act_id) < head_idx) {
        return false;
    }
    /* Equal F32 shapes: the fused kernel is contiguous numel, no broadcast.
     * F16 is kept on the unfused typed elementwise kernels until this fusion is
     * explicitly typed. */
    out_d = &exe->graph->values[head->outputs[0]].desc;
    up_d = &exe->graph->values[head->inputs[1]].desc;
    if (out_d->dtype != GD_DTYPE_F32 || up_d->dtype != GD_DTYPE_F32 ||
        exe->graph->values[silu->inputs[0]].desc.dtype != GD_DTYPE_F32) {
        return false;
    }
    if (_gd_metal_desc_numel(out_d) != _gd_metal_desc_numel(up_d) ||
        _gd_metal_desc_numel(out_d) != _gd_metal_desc_numel(&exe->graph->values[silu->inputs[0]].desc)) {
        return false;
    }
    exe->node_pso2[head_idx] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_silu_mul");
    if (exe->node_pso2[head_idx] == NULL) {
        return false; /* kernel missing: fall back to unfused */
    }
    exe->node_fused_src[head_idx] = silu_idx;
    exe->node_absorbed[silu_idx] = 1;
    g_metal_fusions_applied++;
    return true;
}

/* F4 forward: add(a,b) -> rms_norm(sum, weight). RMSNorm (head) absorbs ADD
 * and writes both sum (the residual stream boundary) and normalized output. */
static bool try_fuse_add_rms_norm(GDMetalState *st, const gd_graph *graph,
                                  _gd_executable *exe, int head_idx)
{
    const _gd_node *head = &graph->nodes[head_idx];
    int sum_id = 0;
    int add_idx = 0;
    const _gd_node *add = NULL;
    const gd_tensor_desc *sum_d = NULL;
    const gd_tensor_desc *a_d = NULL;
    const gd_tensor_desc *b_d = NULL;
    int last = 0;

    if (head->op != _GD_OP_RMS_NORM || head->n_inputs != 2 || head->n_outputs != 1) {
        return false;
    }
    sum_id = head->inputs[0];
    if (sum_id < 0 || sum_id >= exe->n_values) {
        return false;
    }
    add_idx = graph->values[sum_id].producer_node_id;
    if (add_idx < 0 || add_idx >= graph->n_nodes || exe->node_absorbed[add_idx]) {
        return false;
    }
    add = &graph->nodes[add_idx];
    if (add->op != _GD_OP_ADD || add->n_inputs != 2 || add->n_outputs != 1 ||
        add->outputs[0] != sum_id) {
        return false;
    }
    if (value_min_consumer(graph, sum_id) < head_idx) {
        return false;
    }
    sum_d = &graph->values[sum_id].desc;
    a_d = &graph->values[add->inputs[0]].desc;
    b_d = &graph->values[add->inputs[1]].desc;
    if (sum_d->dtype != GD_DTYPE_F32 || a_d->dtype != GD_DTYPE_F32 || b_d->dtype != GD_DTYPE_F32 ||
        !_gd_metal_desc_same_shape(sum_d, a_d) || !_gd_metal_desc_same_shape(sum_d, b_d)) {
        return false;
    }
    last = (int)sum_d->sizes[sum_d->ndim - 1];
    if (last <= 0 || last > GD_METAL_FUSED_RMS_MAX) {
        return false;
    }
    exe->node_pso2[head_idx] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_add_rms_norm");
    if (exe->node_pso2[head_idx] == NULL) {
        return false;
    }
    exe->node_fused_src[head_idx] = add_idx;
    exe->node_absorbed[add_idx] = 1;
    g_metal_fusions_applied++;
    return true;
}

/* F4 backward: rms_norm_bwd(x,w,go) -> add(dx, dS_out). ADD (head) absorbs the
 * RMS dx kernel, materializes dx for parity, and writes accumulated dS. */
static bool try_fuse_rms_norm_bwd_add(GDMetalState *st, const gd_graph *graph,
                                      _gd_executable *exe, int head_idx)
{
    const _gd_node *head = &graph->nodes[head_idx];
    int dx_id = -1;
    int src_idx = -1;
    const _gd_node *src = NULL;
    const gd_tensor_desc *out_d = NULL;
    const gd_tensor_desc *dx_d = NULL;
    const gd_tensor_desc *ds_d = NULL;
    int src_slot = -1;
    int ds_slot = -1;
    int last = 0;

    if (head->op != _GD_OP_ADD || head->n_inputs != 2 || head->n_outputs != 1) {
        return false;
    }
    for (src_slot = 0; src_slot < 2; ++src_slot) {
        int v = head->inputs[src_slot];
        int p = (v >= 0 && v < graph->n_values) ? graph->values[v].producer_node_id : -1;
        if (p >= 0 && p < graph->n_nodes && !exe->node_absorbed[p] &&
            graph->nodes[p].op == _GD_OP_RMS_NORM_BWD) {
            dx_id = v;
            src_idx = p;
            ds_slot = 1 - src_slot;
            break;
        }
    }
    if (src_idx < 0) {
        return false;
    }
    src = &graph->nodes[src_idx];
    if (src->n_inputs != 3 || src->n_outputs != 1 || src->outputs[0] != dx_id) {
        return false;
    }
    if (value_min_consumer(graph, dx_id) < head_idx) {
        return false;
    }
    out_d = &graph->values[head->outputs[0]].desc;
    dx_d = &graph->values[dx_id].desc;
    ds_d = &graph->values[head->inputs[ds_slot]].desc;
    if (out_d->dtype != GD_DTYPE_F32 || dx_d->dtype != GD_DTYPE_F32 || ds_d->dtype != GD_DTYPE_F32 ||
        !_gd_metal_desc_same_shape(out_d, dx_d) || !_gd_metal_desc_same_shape(out_d, ds_d)) {
        return false;
    }
    last = (int)dx_d->sizes[dx_d->ndim - 1];
    if (last <= 0 || last > GD_METAL_FUSED_RMS_MAX) {
        return false;
    }
    exe->node_pso2[head_idx] = (__bridge void *)_gd_metal_pipeline_named(st, "gd_rms_norm_bwd_add");
    if (exe->node_pso2[head_idx] == NULL) {
        return false;
    }
    exe->node_fused_src[head_idx] = src_idx;
    exe->node_absorbed[src_idx] = 1;
    g_metal_fusions_applied++;
    return true;
}

void _gd_metal_plan_fusions(_gd_backend *self, const gd_graph *graph,
                               _gd_executable *exe)
{
    GDMetalState *st = _gd_metal_state(self);
    int i = 0;
    for (i = 0; i < graph->n_nodes; ++i) {
        if (exe->node_absorbed[i]) {
            continue;
        }
        if (try_fuse_silu_mul(st, graph, exe, i)) {
            continue;
        }
        if (try_fuse_add_rms_norm(st, graph, exe, i)) {
            continue;
        }
        (void)try_fuse_rms_norm_bwd_add(st, graph, exe, i);
    }
}

static void encode_fused_silu_mul(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  _gd_executable *exe,
                                  const _gd_node *head,
                                  const _gd_node *silu)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[head->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_unary_params params;

    params.numel = (int)numel;
    params.scale = 0.0F;
    params.dtype = GD_METAL_DT_F32;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, silu->inputs[0]) offset:0 atIndex:0];  /* gate */
    [enc setBuffer:_gd_metal_value_buffer(exe, head->inputs[1]) offset:0 atIndex:1];  /* up */
    [enc setBuffer:_gd_metal_value_buffer(exe, head->outputs[0]) offset:0 atIndex:2]; /* hh */
    [enc setBuffer:_gd_metal_value_buffer(exe, silu->outputs[0]) offset:0 atIndex:3]; /* act */
    [enc setBytes:&params length:sizeof(params) atIndex:4];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_fused_add_rms_norm(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      _gd_executable *exe,
                                      const _gd_node *head,
                                      const _gd_node *add)
{
    const gd_tensor_desc *desc = &exe->graph->values[head->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(desc) / p.last) : 0;
    p.eps = head->attrs.eps;
    p.dtype = GD_METAL_DT_F32;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, add->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, add->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, head->inputs[1]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, add->outputs[0]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, head->outputs[0]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (p.rows > 0) {
        _gd_metal_dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

static void encode_fused_rms_norm_bwd_add(id<MTLComputeCommandEncoder> enc,
                                          id<MTLComputePipelineState> pso,
                                          _gd_executable *exe,
                                          const _gd_node *head,
                                          const _gd_node *src)
{
    const gd_tensor_desc *desc = &exe->graph->values[head->outputs[0]].desc;
    int ds_input = head->inputs[0] == src->outputs[0] ? head->inputs[1] : head->inputs[0];
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(desc) / p.last) : 0;
    p.eps = src->attrs.eps;
    p.dtype = GD_METAL_DT_F32;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, src->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, src->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, src->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, ds_input) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, src->outputs[0]) offset:0 atIndex:4];
    [enc setBuffer:_gd_metal_value_buffer(exe, head->outputs[0]) offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    if (p.rows > 0) {
        _gd_metal_dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

gd_status _gd_metal_encode_fused_head(id<MTLComputeCommandEncoder> enc,
                                   id<MTLComputePipelineState> pso,
                                   _gd_executable *exe,
                                   const _gd_node *head,
                                   const _gd_node *src)
{
    if (pso == nil || head == NULL || src == NULL) {
        return _gd_error(GD_ERR_BACKEND, "invalid fused metal encode");
    }
    if (head->op == _GD_OP_MUL && src->op == _GD_OP_SILU) {
        encode_fused_silu_mul(enc, pso, exe, head, src);
        return GD_OK;
    }
    if (head->op == _GD_OP_RMS_NORM && src->op == _GD_OP_ADD) {
        encode_fused_add_rms_norm(enc, pso, exe, head, src);
        return GD_OK;
    }
    if (head->op == _GD_OP_ADD && src->op == _GD_OP_RMS_NORM_BWD) {
        encode_fused_rms_norm_bwd_add(enc, pso, exe, head, src);
        return GD_OK;
    }
    return _gd_error(GD_ERR_BACKEND, "unknown fused metal encode");
}
