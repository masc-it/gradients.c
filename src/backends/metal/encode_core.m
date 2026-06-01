#import "metal_internal.h"

static void encode_binary(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *a_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &exe->graph->values[node->inputs[1]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_ew_params params;

    _gd_metal_build_ew_params(&params, out_desc, a_desc, b_desc);
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&params length:sizeof(params) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_unary(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         _gd_executable *exe,
                         const _gd_node *node,
                         float scale)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_unary_params params;

    params.numel = (int)numel;
    params.scale = scale;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

/* F1 fused SwiGLU: encode gd_silu_mul for a MUL head that absorbed a SILU.
 * gate = silu input, up = head input 1, hh = head output, act = silu output
 * (kept materialized for the unfused backward). */
static void encode_powlu(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         _gd_executable *exe,
                         const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_powlu_params p;

    p.numel = (int)numel;
    p.m = node->attrs.powlu_m;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_powlu_bwd(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_powlu_params p;

    p.numel = (int)numel;
    p.m = node->attrs.powlu_m;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[1]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static gd_status encode_cast(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_cast_params params;
    gd_status status = GD_OK;

    params.numel = (int)numel;
    status = _gd_metal_dtype_code(in_desc->dtype, &params.src_dtype);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_metal_dtype_code(out_desc->dtype, &params.dst_dtype);
    if (status != GD_OK) {
        return status;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&params length:sizeof(params) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

static void encode_matmul(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *a_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &exe->graph->values[node->inputs[1]].desc;
    int a_rows = (int)a_desc->sizes[a_desc->ndim - 2];
    int a_cols = (int)a_desc->sizes[a_desc->ndim - 1];
    int b_rows = (int)b_desc->sizes[b_desc->ndim - 2];
    int b_cols = (int)b_desc->sizes[b_desc->ndim - 1];
    gd_metal_matmul_params p;
    int batch_total = 1;
    int i = 0;

    memset(&p, 0, sizeof(p));
    p.m = (int)out_desc->sizes[out_desc->ndim - 2];
    p.n = (int)out_desc->sizes[out_desc->ndim - 1];
    p.k = node->attrs.trans_a ? a_rows : a_cols;
    p.a_cols = a_cols;
    p.b_cols = b_cols;
    p.a_mat = a_rows * a_cols;
    p.b_mat = b_rows * b_cols;
    p.out_mat = p.m * p.n;
    p.trans_a = node->attrs.trans_a ? 1 : 0;
    p.trans_b = node->attrs.trans_b ? 1 : 0;
    p.batch_ndim = out_desc->ndim - 2;
    p.a_batch_ndim = a_desc->ndim - 2;
    p.b_batch_ndim = b_desc->ndim - 2;
    for (i = 0; i < p.batch_ndim; ++i) {
        p.out_batch_sizes[i] = (int)out_desc->sizes[i];
        batch_total *= p.out_batch_sizes[i];
    }
    for (i = 0; i < p.a_batch_ndim; ++i) {
        p.a_batch_sizes[i] = (int)a_desc->sizes[i];
    }
    for (i = 0; i < p.b_batch_ndim; ++i) {
        p.b_batch_sizes[i] = (int)b_desc->sizes[i];
    }

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.n > 0 && p.m > 0 && batch_total > 0) {
        _gd_metal_dispatch_gemm_tiles(enc, (NSUInteger)p.n, (NSUInteger)p.m, (NSUInteger)batch_total);
    }
}

static void encode_linear(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_linear_params p;
    int bias_input = node->attrs.has_bias ? node->inputs[2] : node->inputs[0];

    p.in_features = (int)x_desc->sizes[x_desc->ndim - 1];
    p.out_features = (int)out_desc->sizes[out_desc->ndim - 1];
    p.rows = p.in_features > 0 ? (int)(_gd_metal_desc_numel(x_desc) / p.in_features) : 0;
    p.trans_w = node->attrs.trans_b ? 1 : 0;
    p.has_bias = node->attrs.has_bias ? 1 : 0;

    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    /* bias is always bound (placeholder when absent; the kernel guards reads). */
    [enc setBuffer:_gd_metal_value_buffer(exe, bias_input) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.out_features > 0 && p.rows > 0) {
        _gd_metal_dispatch_gemm_tiles(enc, (NSUInteger)p.out_features, (NSUInteger)p.rows, 1);
    }
}

static void split_around_dim(const gd_tensor_desc *desc, int dim,
                             int *outer, int *d, int *inner)
{
    int i = 0;
    *outer = 1;
    *inner = 1;
    *d = (int)desc->sizes[dim];
    for (i = 0; i < dim; ++i) {
        *outer *= (int)desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        *inner *= (int)desc->sizes[i];
    }
}

static void encode_reduce(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pso,
                          _gd_executable *exe,
                          const _gd_node *node)
{
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_params p;

    split_around_dim(x_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = (node->op == _GD_OP_MEAN) ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static void encode_softmax(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           _gd_executable *exe,
                           const _gd_node *node)
{
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    split_around_dim(desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static void encode_rms_norm(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pso,
                            _gd_executable *exe,
                            const _gd_node *node)
{
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.rows > 0) {
        _gd_metal_dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

static gd_status encode_cross_entropy(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      id<MTLComputePipelineState> reduce_pso,
                                      id<MTLBuffer> scratch,
                                      _gd_executable *exe,
                                      const _gd_node *node)
{
    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy needs I32 targets in v1");
    }
    split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal cross_entropy scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        if (p.positions > 0) {
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)p.positions, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
        }
    }

    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    {
        NSUInteger tg = reduce_pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return GD_OK;
}

/* relu_bwd/silu_bwd: dx from (x, go). */
static void encode_unary_bwd(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_unary_params p;

    p.numel = (int)numel;
    p.scale = 0.0F;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

/* softmax_bwd: dx from (y, go) along dim; one thread per (outer,inner). */
static void encode_softmax_bwd(id<MTLComputeCommandEncoder> enc,
                               id<MTLComputePipelineState> pso,
                               _gd_executable *exe,
                               const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_softmax_params p;

    split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

/* sum_bwd/mean_bwd: broadcast go back over dim into dx (the output/x shape). */
static void encode_sum_bwd(id<MTLComputeCommandEncoder> enc,
                           id<MTLComputePipelineState> pso,
                           _gd_executable *exe,
                           const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_reduce_params p;

    split_around_dim(out_desc, node->attrs.dim, &p.outer, &p.d, &p.inner);
    p.mean = (node->op == _GD_OP_MEAN_BWD) ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.outer * p.inner > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)(p.outer * p.inner));
    }
}

static gd_status encode_cross_entropy_bwd(id<MTLComputeCommandEncoder> enc,
                                          id<MTLComputePipelineState> pso,
                                          _gd_executable *exe,
                                          const _gd_node *node)
{
    const gd_tensor_desc *logits = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *targets = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_ce_params p;
    int dummy = 0;

    if (targets->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal cross_entropy_bwd needs I32 targets");
    }
    split_around_dim(logits, node->attrs.dim, &p.outer, &dummy, &p.inner);
    p.classes = (int)logits->sizes[node->attrs.dim];
    p.positions = p.outer * p.inner;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    /* One threadgroup per position; threads cooperate over the class dim (see
     * gd_cross_entropy_bwd). Power-of-two threadgroup width for the reductions. */
    if (p.outer * p.inner > 0) {
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        NSUInteger w = 256;
        if (tg < w) {
            w = 1;
            while ((w << 1) <= tg) {
                w <<= 1;
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(p.outer * p.inner), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(w, 1, 1)];
    }
    return GD_OK;
}


static void encode_reduce_to(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *target = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_reduce_to_params p;
    int i = 0;

    memset(&p, 0, sizeof(p));
    p.target_ndim = target->ndim;
    p.target_numel = (int)_gd_metal_desc_numel(target);
    p.go_ndim = go->ndim;
    p.go_numel = (int)_gd_metal_desc_numel(go);
    for (i = 0; i < target->ndim; ++i) {
        p.target_sizes[i] = (int)target->sizes[i];
    }
    for (i = 0; i < go->ndim; ++i) {
        p.go_sizes[i] = (int)go->sizes[i];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.target_numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)p.target_numel);
    }
}



bool _gd_metal_encode_core_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLBuffer> scratch,
                                gd_status *status_out)
{
    *status_out = GD_OK;
    switch (node->op) {
    case _GD_OP_ADD:
    case _GD_OP_MUL:
        encode_binary(enc, pso, exe, node);
        return true;
    case _GD_OP_SCALE:
        encode_unary(enc, pso, exe, node, node->attrs.scale);
        return true;
    case _GD_OP_RELU:
    case _GD_OP_SILU:
        encode_unary(enc, pso, exe, node, 0.0F);
        return true;
    case _GD_OP_POWLU:
        encode_powlu(enc, pso, exe, node);
        return true;
    case _GD_OP_COPY:
        if (node->n_inputs == 1 && node->n_outputs == 1 &&
            exe->values[node->inputs[0]].storage == exe->values[node->outputs[0]].storage) {
            _gd_profile_record_event(self->ctx, self, _GD_PROFILE_EVENT_COPY_ALIAS,
                                     0U, 0U, 1U);
            return true;
        }
        encode_unary(enc, pso, exe, node, 0.0F);
        return true;
    case _GD_OP_CAST:
        *status_out = encode_cast(enc, pso, exe, node);
        return true;
    case _GD_OP_MATMUL:
        encode_matmul(enc, pso, exe, node);
        return true;
    case _GD_OP_LINEAR:
        encode_linear(enc, pso, exe, node);
        return true;
    case _GD_OP_SUM:
    case _GD_OP_MEAN:
        encode_reduce(enc, pso, exe, node);
        return true;
    case _GD_OP_SOFTMAX:
        encode_softmax(enc, pso, exe, node);
        return true;
    case _GD_OP_RMS_NORM:
        encode_rms_norm(enc, pso, exe, node);
        return true;
    case _GD_OP_CROSS_ENTROPY:
        *status_out = encode_cross_entropy(enc, pso, pso2, scratch, exe, node);
        return true;
    case _GD_OP_RELU_BWD:
    case _GD_OP_SILU_BWD:
        encode_unary_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_POWLU_BWD:
        encode_powlu_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_SOFTMAX_BWD:
        encode_softmax_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_SUM_BWD:
    case _GD_OP_MEAN_BWD:
        encode_sum_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_CROSS_ENTROPY_BWD:
        *status_out = encode_cross_entropy_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_REDUCE_TO:
        encode_reduce_to(enc, pso, exe, node);
        return true;
    default:
        return false;
    }
}
