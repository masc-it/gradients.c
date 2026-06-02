#import "../../backends/metal/metal_op.h"

#include <limits.h>

static gd_status fill_slice_params(gd_metal_slice_params *p,
                                   const gd_tensor_desc *full_desc,
                                   const gd_tensor_desc *slice_desc,
                                   const _gd_op_attrs *attrs,
                                   int64_t numel)
{
    size_t elem = gd_dtype_sizeof(full_desc->dtype);
    int i = 0;

    if (p == NULL || full_desc == NULL || slice_desc == NULL || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "metal slice params argument is NULL");
    }
    if (numel < 0 || numel > INT_MAX || elem == 0U || elem > (size_t)INT_MAX ||
        attrs->slice_start < 0 || attrs->slice_start > INT_MAX ||
        attrs->slice_len <= 0 || attrs->slice_len > INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal slice dimensions exceed int32 limits");
    }
    memset(p, 0, sizeof(*p));
    p->ndim = full_desc->ndim;
    p->numel = (int)numel;
    p->elem_size = (int)elem;
    p->dim = attrs->dim;
    p->start = (int)attrs->slice_start;
    p->len = (int)attrs->slice_len;
    for (i = 0; i < full_desc->ndim; ++i) {
        if (full_desc->sizes[i] > INT_MAX || full_desc->strides[i] > INT_MAX ||
            slice_desc->sizes[i] > INT_MAX) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal slice shape exceeds int32 limits");
        }
        p->full_sizes[i] = (int)full_desc->sizes[i];
        p->slice_sizes[i] = (int)slice_desc->sizes[i];
        p->in_strides[i] = (int)full_desc->strides[i];
    }
    return GD_OK;
}

static gd_status slice_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'slice'");
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *x = &ctx->graph->values[ctx->node->inputs[0]].desc;
        const gd_tensor_desc *out = &ctx->graph->values[ctx->node->outputs[0]].desc;
        gd_metal_slice_params p;
        if (x->dtype != out->dtype || gd_dtype_sizeof(x->dtype) == 0U ||
            x->quant != NULL || out->quant != NULL) {
            return _gd_error(GD_ERR_DTYPE, "metal slice requires matching fixed-size dtypes");
        }
        if (out->layout != GD_LAYOUT_CONTIGUOUS) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal slice output must be contiguous");
        }
        return fill_slice_params(&p, x, out, &ctx->node->attrs, _gd_metal_desc_numel(out));
    }
    return GD_OK;
}

static gd_status slice_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *x_desc = &exe->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int64_t numel = _gd_metal_desc_numel(out_desc);
    gd_metal_slice_params p;
    gd_status status = fill_slice_params(&p, x_desc, out_desc, &node->attrs, numel);

    if (status != GD_OK) {
        return status;
    }
    if (pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal slice pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[0]) offset:0 atIndex:0];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    if (numel > 0) {
        _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_slice = {
    .kind = _GD_OP_SLICE,
    .name = "slice",
    .support = slice_support,
    .encode = slice_encode,
};
