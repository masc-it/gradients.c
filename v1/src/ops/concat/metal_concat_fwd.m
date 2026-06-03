#import "../../backends/metal/metal_op.h"

#include <limits.h>

static gd_status concat_dim_product_int(const gd_tensor_desc *desc,
                                        int begin,
                                        int end,
                                        int *out)
{
    int64_t prod = 1;
    int i = 0;

    if (desc == NULL || out == NULL || begin < 0 || end < begin || end > desc->ndim) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "metal concat product arguments are invalid");
    }
    for (i = begin; i < end; ++i) {
        if (desc->sizes[i] <= 0 || prod > (int64_t)INT_MAX / desc->sizes[i]) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal concat shape exceeds int32 limits");
        }
        prod *= desc->sizes[i];
    }
    *out = (int)prod;
    return GD_OK;
}

static gd_status fill_concat_params(gd_metal_concat_params *p,
                                    const gd_tensor_desc *in_desc,
                                    const gd_tensor_desc *out_desc,
                                    int dim,
                                    int64_t dst_start,
                                    int64_t numel)
{
    gd_status status = GD_OK;
    size_t elem = gd_dtype_sizeof(in_desc->dtype);
    int inner = 0;

    if (p == NULL || in_desc == NULL || out_desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "metal concat params argument is NULL");
    }
    if (numel < 0 || numel > INT_MAX || elem == 0U || elem > (size_t)INT_MAX ||
        in_desc->sizes[dim] > INT_MAX || out_desc->sizes[dim] > INT_MAX ||
        dst_start < 0 || dst_start > INT_MAX) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal concat dimensions exceed int32 limits");
    }
    status = concat_dim_product_int(out_desc, dim + 1, out_desc->ndim, &inner);
    if (status != GD_OK) {
        return status;
    }
    memset(p, 0, sizeof(*p));
    p->numel = (int)numel;
    p->elem_size = (int)elem;
    p->inner = inner;
    p->input_dim = (int)in_desc->sizes[dim];
    p->output_dim = (int)out_desc->sizes[dim];
    p->dst_start = (int)dst_start;
    return GD_OK;
}

static gd_status concat_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                             ctx->node->n_outputs);
    int64_t dst_start = 0;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'concat'");
    }
    if (ctx->graph != NULL) {
        const gd_tensor_desc *out = &ctx->graph->values[ctx->node->outputs[0]].desc;
        int dim = ctx->node->attrs.dim;

        if (gd_dtype_sizeof(out->dtype) == 0U || out->quant != NULL) {
            return _gd_error(GD_ERR_DTYPE, "metal concat requires fixed-size dtype");
        }
        if (out->layout != GD_LAYOUT_CONTIGUOUS) {
            return _gd_error(GD_ERR_UNSUPPORTED, "metal concat output must be contiguous");
        }
        for (i = 0; i < ctx->node->n_inputs; ++i) {
            const gd_tensor_desc *in = &ctx->graph->values[ctx->node->inputs[i]].desc;
            gd_metal_concat_params p;

            if (in->dtype != out->dtype || gd_dtype_sizeof(in->dtype) == 0U ||
                in->quant != NULL) {
                return _gd_error(GD_ERR_DTYPE,
                                 "metal concat requires matching fixed-size dtypes");
            }
            if (in->layout != GD_LAYOUT_CONTIGUOUS) {
                return _gd_error(GD_ERR_UNSUPPORTED,
                                 "metal concat inputs must be contiguous");
            }
            status = fill_concat_params(&p, in, out, dim, dst_start,
                                        _gd_metal_desc_numel(in));
            if (status != GD_OK) {
                return status;
            }
            dst_start += in->sizes[dim];
        }
    }
    return GD_OK;
}

static gd_status concat_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *out_desc = &exe->graph->values[node->outputs[0]].desc;
    int dim = node->attrs.dim;
    int64_t dst_start = 0;
    int i = 0;

    if (pso == nil) {
        return _gd_error(GD_ERR_BACKEND, "metal concat pipeline missing");
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:_gd_metal_value_buffer(exe, node->outputs[0]) offset:0 atIndex:1];
    for (i = 0; i < node->n_inputs; ++i) {
        const gd_tensor_desc *in_desc = &exe->graph->values[node->inputs[i]].desc;
        int64_t numel = _gd_metal_desc_numel(in_desc);
        gd_metal_concat_params p;
        gd_status status = fill_concat_params(&p, in_desc, out_desc, dim, dst_start, numel);

        if (status != GD_OK) {
            return status;
        }
        [enc setBuffer:_gd_metal_value_buffer(exe, node->inputs[i]) offset:0 atIndex:0];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        if (numel > 0) {
            _gd_metal_dispatch_1d(enc, pso, (NSUInteger)numel);
        }
        dst_start += in_desc->sizes[dim];
    }
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_concat = {
    .kind = _GD_OP_CONCAT,
    .name = "concat",
    .support = concat_support,
    .encode = concat_encode,
};
