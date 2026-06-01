#import "metal_internal.h"

static void encode_step_inc(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pso,
                            _gd_executable *exe,
                            const _gd_node *node)
{
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    _gd_metal_dispatch_1d(enc, pso, 1);
}

static gd_status encode_clip_grad_norm(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> partial_pso,
                                      id<MTLComputePipelineState> reduce_pso,
                                      id<MTLComputePipelineState> scale_pso,
                                      id<MTLBuffer> scratch,
                                      _gd_executable *exe,
                                      const _gd_node *node)
{
    gd_metal_clip_norm_params p;
    int i = 0;
    int scratch_offset = 0;
    int total_groups = 0;
    int reduce_groups_max = 0;
    int scale_index = 0;
    int src_offset = 0;
    int dst_offset = 0;
    int src_count = 0;

    if (partial_pso == nil || reduce_pso == nil || scale_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal clip_grad_norm pipeline/scratch missing");
    }
    if (node->n_inputs <= 0 || node->n_outputs != 1) {
        return _gd_error(GD_ERR_INTERNAL, "clip_grad_norm expects inputs and one output");
    }
    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        int64_t groups64 = (numel64 + GD_METAL_CLIP_NORM_CHUNK - 1) /
                           GD_METAL_CLIP_NORM_CHUNK;
        if (numel64 > INT_MAX || groups64 > INT_MAX || groups64 <= 0) {
            return _gd_error(GD_ERR_SHAPE, "clip_grad_norm input too large");
        }
        if (total_groups > INT_MAX - (int)groups64) {
            return _gd_error(GD_ERR_SHAPE, "clip_grad_norm has too many groups");
        }
        total_groups += (int)groups64;
    }
    if (total_groups <= 0) {
        return _gd_error(GD_ERR_SHAPE, "clip_grad_norm has no elements");
    }
    reduce_groups_max = (total_groups + GD_METAL_CLIP_NORM_TG - 1) /
                        GD_METAL_CLIP_NORM_TG;
    scale_index = total_groups + reduce_groups_max;

    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        int groups = (int)((numel64 + GD_METAL_CLIP_NORM_CHUNK - 1) /
                           GD_METAL_CLIP_NORM_CHUNK);
        memset(&p, 0, sizeof(p));
        p.numel = (int)numel64;
        p.scratch_offset = scratch_offset;
        p.scale_index = scale_index;
        p.max_norm = node->attrs.scale;
        p.eps = node->attrs.eps;
        [enc setComputePipelineState:partial_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[i]) offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_CLIP_NORM_TG, 1, 1)];
        scratch_offset += groups;
    }

    src_offset = 0;
    dst_offset = total_groups;
    src_count = total_groups;
    for (;;) {
        int groups = (src_count + GD_METAL_CLIP_NORM_TG - 1) / GD_METAL_CLIP_NORM_TG;
        memset(&p, 0, sizeof(p));
        p.numel = src_count;
        p.scratch_offset = src_offset;
        p.dst_offset = dst_offset;
        p.total_groups = groups;
        p.scale_index = scale_index;
        p.finalize = groups == 1 ? 1 : 0;
        p.max_norm = node->attrs.scale;
        p.eps = node->attrs.eps;
        [enc setComputePipelineState:reduce_pso];
        [enc setBuffer:scratch offset:0 atIndex:0];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_CLIP_NORM_TG, 1, 1)];
        if (groups == 1) {
            break;
        }
        src_count = groups;
        src_offset = dst_offset;
        dst_offset = src_offset == 0 ? total_groups : 0;
    }

    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel64 = _gd_metal_desc_numel(desc);
        memset(&p, 0, sizeof(p));
        p.numel = (int)numel64;
        p.scale_index = scale_index;
        [enc setComputePipelineState:scale_pso];
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[i]) offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        _gd_metal_dispatch_1d(enc, scale_pso, (NSUInteger)numel64);
    }
    return GD_OK;
}

/* adamw: in-place update of (param, m, v) using grad and step. */
static void encode_adamw(id<MTLComputeCommandEncoder> enc,
                         id<MTLComputePipelineState> pso,
                         _gd_executable *exe,
                         const _gd_node *node)
{
    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(param);
    gd_metal_adamw_params p;

    p.numel = (int)numel;
    p.use_lr_tensor = node->n_inputs == 6 ? 1 : 0;
    p.lr = node->attrs.lr;
    p.lr_scale = node->attrs.scale;
    p.beta1 = node->attrs.beta1;
    p.beta2 = node->attrs.beta2;
    p.eps = node->attrs.eps;
    p.weight_decay = node->attrs.weight_decay;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[node->n_inputs == 6 ? 5 : 4])
            offset:0
           atIndex:6];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

/* gelu / gelu_bwd. fwd: (x)->out; bwd: (x, go)->dx. */
static void encode_gelu(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> pso,
                        _gd_executable *exe,
                        const _gd_node *node,
                        bool is_bwd)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_gelu_params p;
    int idx = 0;

    p.numel = (int)numel;
    p.tanh_approx = node->attrs.gelu_tanh ? 1 : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    idx = 1;
    if (is_bwd) {
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
        idx = 2;
    }
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:(NSUInteger)idx];
    [enc setBytes:&p length:sizeof(p) atIndex:(NSUInteger)(idx + 1)];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static void encode_transpose(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_transpose_params p;
    int64_t stride = 1;
    int k = 0;

    memset(&p, 0, sizeof(p));
    p.ndim = out_desc->ndim;
    p.numel = (int)numel;
    for (k = out_desc->ndim - 1; k >= 0; --k) {
        p.in_strides[k] = (int)stride;
        stride *= in_desc->sizes[k];
    }
    for (k = 0; k < out_desc->ndim; ++k) {
        p.out_sizes[k] = (int)out_desc->sizes[k];
        p.perm[k] = node->attrs.perm[k];
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
}

static gd_status encode_embedding(id<MTLComputeCommandEncoder> enc,
                                  id<MTLComputePipelineState> pso,
                                  _gd_executable *exe,
                                  const _gd_node *node)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *table = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(numel / p.dim) : 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

static gd_status encode_embedding_bwd(id<MTLComputeCommandEncoder> enc,
                                      id<MTLComputePipelineState> pso,
                                      id<MTLComputePipelineState> scatter_pso,
                                      _gd_executable *exe,
                                      const _gd_node *node)
{
    const gd_tensor_desc *table = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *go = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *ids = &exe->graph->values[node->inputs[1]].desc;
    gd_metal_embedding_params p;

    if (ids->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal embedding_bwd needs I32 ids in v1");
    }
    p.dim = (int)table->sizes[1];
    p.vocab = (int)table->sizes[0];
    p.n = p.dim > 0 ? (int)(_gd_metal_desc_numel(go) / p.dim) : 0;
    if (scatter_pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal embedding_bwd scatter pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:0];
    [enc setBytes:&p length:sizeof(p) atIndex:1];
    {
        int total = p.vocab * p.dim;
        if (total > 0) {
            _gd_metal_dispatch_1d(enc, pso, (NSUInteger)total);
        }
    }
    [enc setComputePipelineState:scatter_pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        int total = p.n * p.dim;
        if (total > 0) {
            _gd_metal_dispatch_1d(enc, scatter_pso, (NSUInteger)total);
        }
    }
    return GD_OK;
}

static gd_status encode_rope(id<MTLComputeCommandEncoder> enc,
                             id<MTLComputePipelineState> pso,
                             _gd_executable *exe,
                             const _gd_node *node,
                             float sin_sign)
{
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    const gd_tensor_desc *pos = &exe->graph->values[node->inputs[1]].desc;
    int64_t head_dim = out_desc->sizes[out_desc->ndim - 1];
    gd_metal_rope_params p;

    if (pos->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal rope needs I32 position ids in v1");
    }
    p.head_dim = (int)head_dim;
    p.heads = (int)out_desc->sizes[out_desc->ndim - 2];
    p.rows = p.head_dim > 0 ? (int)(_gd_metal_desc_numel(out_desc) / head_dim) : 0;
    p.n_dims = node->attrs.rope_n_dims;
    p.interleaved = node->attrs.rope_interleaved;
    p.theta = node->attrs.rope_theta;
    p.sin_sign = sin_sign;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    if (p.rows > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)p.rows);
    }
    return GD_OK;
}

static void encode_rms_norm_bwd(id<MTLComputeCommandEncoder> enc,
                                id<MTLComputePipelineState> pso,
                                _gd_executable *exe,
                                const _gd_node *node)
{
    /* dx: inputs x(0), weight(1), go(2); output dx (x shape). */
    const gd_tensor_desc *desc = &exe->graph->values[node->outputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)desc->sizes[desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    if (p.rows > 0) {
        _gd_metal_dispatch_reduce_groups(enc, (NSUInteger)p.rows);
    }
}

static gd_status encode_rms_norm_wbwd(id<MTLComputeCommandEncoder> enc,
                                       id<MTLComputePipelineState> pso,
                                       id<MTLComputePipelineState> reduce_pso,
                                       id<MTLBuffer> scratch,
                                       _gd_executable *exe,
                                       const _gd_node *node)
{
    /* dweight: inputs x(0), go(1); output dweight [last]. dims from x. */
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    gd_metal_rmsnorm_params p;

    p.last = (int)x_desc->sizes[x_desc->ndim - 1];
    p.rows = p.last > 0 ? (int)(_gd_metal_desc_numel(x_desc) / p.last) : 0;
    p.eps = node->attrs.eps;
    if (p.rows <= 0 || p.last <= 0) {
        return GD_OK;
    }
    if (reduce_pso == nil || scratch == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal rms_norm_wbwd scratch/pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:scratch offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    {
        NSUInteger row_blocks = ((NSUInteger)p.rows + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        NSUInteger channel_blocks = ((NSUInteger)p.last + GD_METAL_RMS_TG - 1) / GD_METAL_RMS_TG;
        [enc dispatchThreadgroups:MTLSizeMake(row_blocks * channel_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(GD_METAL_RMS_TG, 1, 1)];
    }
    [enc setComputePipelineState:reduce_pso];
    [enc setBuffer:scratch offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    _gd_metal_dispatch_1d(enc, reduce_pso, (NSUInteger)p.last);
    return GD_OK;
}


bool _gd_metal_encode_misc_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLComputePipelineState> pso3,
                                id<MTLBuffer> scratch,
                                gd_status *status_out)
{
    (void)self;
    *status_out = GD_OK;
    switch (node->op) {
    case _GD_OP_CLIP_GRAD_NORM:
        *status_out = encode_clip_grad_norm(enc, pso, pso2, pso3, scratch, exe, node);
        return true;
    case _GD_OP_STEP_INC:
        encode_step_inc(enc, pso, exe, node);
        return true;
    case _GD_OP_ADAMW_STEP:
        encode_adamw(enc, pso, exe, node);
        return true;
    case _GD_OP_GELU:
        encode_gelu(enc, pso, exe, node, false);
        return true;
    case _GD_OP_GELU_BWD:
        encode_gelu(enc, pso, exe, node, true);
        return true;
    case _GD_OP_TRANSPOSE:
        encode_transpose(enc, pso, exe, node);
        return true;
    case _GD_OP_EMBEDDING:
        *status_out = encode_embedding(enc, pso, exe, node);
        return true;
    case _GD_OP_EMBEDDING_BWD:
        *status_out = encode_embedding_bwd(enc, pso, pso2, exe, node);
        return true;
    case _GD_OP_ROPE:
        *status_out = encode_rope(enc, pso, exe, node, 1.0F);
        return true;
    case _GD_OP_ROPE_BWD:
        *status_out = encode_rope(enc, pso, exe, node, -1.0F);
        return true;
    case _GD_OP_RMS_NORM_BWD:
        encode_rms_norm_bwd(enc, pso, exe, node);
        return true;
    case _GD_OP_RMS_NORM_WBWD:
        *status_out = encode_rms_norm_wbwd(enc, pso, pso2, scratch, exe, node);
        return true;
    default:
        return false;
    }
}
