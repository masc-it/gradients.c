#import "../../backends/metal/metal_op.h"

static gd_status adamw_step_amp_support(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;

    if (ctx->graph == NULL) {
        return GD_OK;
    }
    if (ctx->graph->values[node->inputs[0]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[1]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[2]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[3]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[4]].desc.dtype != GD_DTYPE_F32 ||
        ctx->graph->values[node->inputs[5]].desc.dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_DTYPE, "metal adamw_step_amp expects F32 tensors and I32 found_inf");
    }
    if (node->n_inputs == 7 || node->n_inputs == 8) {
        int refresh_index = 6;
        const gd_tensor_desc *extra = NULL;
        const gd_tensor_desc *master = &ctx->graph->values[node->inputs[0]].desc;
        if (node->n_inputs == 8) {
            const gd_tensor_desc *lr = &ctx->graph->values[node->inputs[6]].desc;
            if (lr->dtype != GD_DTYPE_F32 || lr->ndim != 0) {
                return _gd_error(GD_ERR_DTYPE, "metal adamw lr tensor must be F32 scalar");
            }
            refresh_index = 7;
        }
        extra = &ctx->graph->values[node->inputs[refresh_index]].desc;
        if (node->n_inputs == 7 && extra->dtype == GD_DTYPE_F32 && extra->ndim == 0) {
            return GD_OK;
        }
        if ((extra->dtype != GD_DTYPE_F16 && extra->dtype != GD_DTYPE_F32) ||
            extra->ndim != master->ndim) {
            return _gd_error(GD_ERR_DTYPE,
                             "metal adamw_step_amp extra input must be lr or refresh param");
        }
        for (int d = 0; d < extra->ndim; ++d) {
            if (extra->sizes[d] != master->sizes[d]) {
                return _gd_error(GD_ERR_SHAPE,
                                 "metal adamw_step_amp refresh shape mismatch");
            }
        }
    }
    return GD_OK;
}

static gd_status adamw_step_amp_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *param = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *extra = node->n_inputs >= 7 ?
        &exe->graph->values[node->inputs[node->n_inputs == 8 ? 7 : 6]].desc : NULL;
    bool has_lr = node->n_inputs == 8 ||
                  (extra != NULL && extra->dtype == GD_DTYPE_F32 && extra->ndim == 0);
    bool refresh = extra != NULL && !(node->n_inputs == 7 && has_lr);
    int lr_input = node->n_inputs == 8 ? 6 : (has_lr ? 6 : 4);
    int refresh_input = refresh ? (node->n_inputs == 8 ? 7 : 6) : 0;
    int64_t numel = _gd_metal_desc_numel(param);
    gd_metal_adamw_params p;

    p.numel = (int)numel;
    p.use_lr_tensor = has_lr ? 1 : 0;
    p.lr = node->attrs.lr;
    p.lr_scale = node->attrs.scale;
    p.beta1 = node->attrs.beta1;
    p.beta2 = node->attrs.beta2;
    p.eps = node->attrs.eps;
    p.weight_decay = node->attrs.weight_decay;
    p.refresh_dtype = -1;
    if (refresh && _gd_metal_dtype_code(extra->dtype, &p.refresh_dtype) != GD_OK) {
        return _gd_error(GD_ERR_DTYPE, "metal adamw_step_amp refresh dtype unsupported");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[1]) offset:0 atIndex:1];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[2]) offset:0 atIndex:2];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[3]) offset:0 atIndex:3];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[4]) offset:0 atIndex:4];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[5]) offset:0 atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[lr_input])
            offset:0
           atIndex:7];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[refresh_input])
            offset:0
           atIndex:8];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_adamw_step_amp = {
    .kind = _GD_OP_ADAMW_STEP_AMP,
    .name = "adamw_step_amp",
    .support = adamw_step_amp_support,
    .encode = adamw_step_amp_encode,
};
