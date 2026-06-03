#import "../../backends/metal/metal_op.h"

static bool adamw_has_lr(const _gd_node *node)
{
    return node->attrs.adamw_has_lr ||
           (!node->attrs.adamw_has_found_inf && !node->attrs.adamw_has_refresh &&
            node->n_inputs == 6);
}

static gd_status adamw_step_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_graph *graph = ctx->graph;
    const gd_tensor_desc *param = NULL;
    bool has_found_inf = node->attrs.adamw_has_found_inf;
    bool has_lr = adamw_has_lr(node);
    bool has_refresh = node->attrs.adamw_has_refresh;
    int input = 5;

    if (graph == NULL) {
        return GD_OK;
    }
    param = &graph->values[node->inputs[0]].desc;
    if (param->dtype != GD_DTYPE_F32 ||
        graph->values[node->inputs[1]].desc.dtype != GD_DTYPE_F32 ||
        graph->values[node->inputs[2]].desc.dtype != GD_DTYPE_F32 ||
        graph->values[node->inputs[3]].desc.dtype != GD_DTYPE_F32 ||
        graph->values[node->inputs[4]].desc.dtype != GD_DTYPE_F32) {
        return _gd_error(GD_ERR_DTYPE, "metal adamw_step expects F32 update tensors");
    }
    if (has_found_inf) {
        const gd_tensor_desc *found = &graph->values[node->inputs[input++]].desc;
        if (found->dtype != GD_DTYPE_I32 || found->ndim != 0) {
            return _gd_error(GD_ERR_DTYPE, "metal adamw_step found_inf must be I32 scalar");
        }
    }
    if (has_lr) {
        const gd_tensor_desc *lr = &graph->values[node->inputs[input++]].desc;
        if (lr->dtype != GD_DTYPE_F32 || lr->ndim != 0) {
            return _gd_error(GD_ERR_DTYPE, "metal adamw lr tensor must be F32 scalar");
        }
    }
    if (has_refresh) {
        const gd_tensor_desc *refresh = &graph->values[node->inputs[input++]].desc;
        if ((refresh->dtype != GD_DTYPE_F16 && refresh->dtype != GD_DTYPE_F32) ||
            refresh->ndim != param->ndim) {
            return _gd_error(GD_ERR_DTYPE, "metal adamw_step refresh dtype unsupported");
        }
        for (int d = 0; d < refresh->ndim; ++d) {
            if (refresh->sizes[d] != param->sizes[d]) {
                return _gd_error(GD_ERR_SHAPE, "metal adamw_step refresh shape mismatch");
            }
        }
    }
    if (input != node->n_inputs) {
        return _gd_error(GD_ERR_INTERNAL, "metal adamw_step input/flag mismatch");
    }
    return GD_OK;
}

static gd_status adamw_step_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;

    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    bool has_found_inf = node->attrs.adamw_has_found_inf;
    bool has_lr = adamw_has_lr(node);
    bool has_refresh = node->attrs.adamw_has_refresh;
    int input = 5;
    int found_input = 4;
    int lr_input = 4;
    int refresh_input = 0;
    int64_t numel = _gd_metal_desc_numel(param);
    gd_metal_adamw_params p;

    if (has_found_inf) { found_input = input++; }
    if (has_lr) { lr_input = input++; }
    if (has_refresh) { refresh_input = input++; }
    p.numel = (int)numel;
    p.use_lr_tensor = has_lr ? 1 : 0;
    p.has_found_inf = has_found_inf ? 1 : 0;
    p.lr = node->attrs.lr;
    p.lr_scale = node->attrs.scale;
    p.beta1 = node->attrs.beta1;
    p.beta2 = node->attrs.beta2;
    p.eps = node->attrs.eps;
    p.weight_decay = node->attrs.weight_decay;
    p.refresh_dtype = -1;
    if (has_refresh) {
        const gd_tensor_desc *refresh = &exe->graph->values[node->inputs[refresh_input]].desc;
        gd_status status = _gd_metal_dtype_code(refresh->dtype, &p.refresh_dtype);
        if (status != GD_OK) { return status; }
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[found_input]) offset:0 atIndex:6];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[lr_input]) offset:0 atIndex:7];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[refresh_input]) offset:0 atIndex:8];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_adamw_step = {
    .kind = _GD_OP_ADAMW_STEP,
    .name = "adamw_step",
    .support = adamw_step_support,
    .encode = adamw_step_encode,
};
