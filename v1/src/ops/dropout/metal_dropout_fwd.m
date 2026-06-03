#import "../../backends/metal/metal_op.h"

static void fill_dropout_params(gd_metal_dropout_params *p,
                                const gd_tensor_desc *desc,
                                float drop_p,
                                uint64_t seed,
                                uint64_t run_id)
{
    int dtype = GD_METAL_DT_F32;

    (void)_gd_metal_dtype_code(desc->dtype, &dtype);
    p->numel = (int)_gd_metal_desc_numel(desc);
    p->p = drop_p;
    p->dtype = dtype;
    p->seed_lo = (unsigned int)(seed & UINT64_C(0xffffffff));
    p->seed_hi = (unsigned int)((seed >> 32) & UINT64_C(0xffffffff));
    p->run_lo = (unsigned int)(run_id & UINT64_C(0xffffffff));
    p->run_hi = (unsigned int)((run_id >> 32) & UINT64_C(0xffffffff));
}

static gd_status dropout_encode(_gd_metal_encode_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    const gd_tensor_desc *out_desc = &ctx->exe->graph->values[node->outputs[0]].desc;
    gd_metal_dropout_params p;

    fill_dropout_params(&p, out_desc, node->attrs.scale, node->attrs.dropout_seed,
                        ctx->exe->run_id);
    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(ctx->exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (p.numel > 0) {
        _gd_metal_dispatch_1d(enc, ctx->pso, (NSUInteger)p.numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_dropout = {
    .kind = _GD_OP_DROPOUT,
    .name = "dropout",
    .support = _gd_metal_support_f32_f16_same_dtype,
    .encode = dropout_encode,
};
