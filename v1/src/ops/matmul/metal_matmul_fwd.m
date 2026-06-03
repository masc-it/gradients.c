#import "../../backends/metal/metal_op.h"

static bool matmul_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16;
}

static bool matmul_node_uses_f16(const _gd_metal_plan_ctx *ctx)
{
    const _gd_node *node = ctx->node;
    const gd_tensor_desc *a_desc = &ctx->graph->values[node->inputs[0]].desc;
    const gd_tensor_desc *b_desc = &ctx->graph->values[node->inputs[1]].desc;
    const gd_tensor_desc *out_desc = &ctx->graph->values[node->outputs[0]].desc;

    return a_desc->dtype == GD_DTYPE_F16 || b_desc->dtype == GD_DTYPE_F16 ||
           out_desc->dtype == GD_DTYPE_F16;
}

static bool matmul_node_mixed_f16_f32(const _gd_node *node,
                                      const gd_tensor_desc *a_desc,
                                      const gd_tensor_desc *b_desc,
                                      const gd_tensor_desc *out_desc)
{
    return a_desc->dtype == GD_DTYPE_F16 && b_desc->dtype == GD_DTYPE_F16 &&
           out_desc->dtype == GD_DTYPE_F32 &&
           node->attrs.compute.compute_dtype == GD_DTYPE_F32 &&
           node->attrs.compute.accum_dtype == GD_DTYPE_INVALID;
}

static gd_status matmul_support(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    const _gd_node *node = NULL;
    const gd_tensor_desc *a_desc = NULL;
    const gd_tensor_desc *b_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal matmul support ctx is NULL");
    }
    node = ctx->node;
    status = _gd_op_validate_arity(node->op, node->n_inputs, node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, node->op) == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op 'matmul'");
    }
    if (ctx->graph == NULL) {
        return GD_OK;
    }
    a_desc = &ctx->graph->values[node->inputs[0]].desc;
    b_desc = &ctx->graph->values[node->inputs[1]].desc;
    out_desc = &ctx->graph->values[node->outputs[0]].desc;
    if (!matmul_dtype_supported(a_desc->dtype) || !matmul_dtype_supported(b_desc->dtype) ||
        !matmul_dtype_supported(out_desc->dtype)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal matmul supports F32/F16 only");
    }
    if ((a_desc->dtype != b_desc->dtype || b_desc->dtype != out_desc->dtype) &&
        !matmul_node_mixed_f16_f32(node, a_desc, b_desc, out_desc)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal matmul requires matching input/output dtypes");
    }
    if ((out_desc->dtype == GD_DTYPE_F16 ||
         matmul_node_mixed_f16_f32(node, a_desc, b_desc, out_desc)) &&
        (ctx->state == nil || !ctx->state.useMPS)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 matmul needs GD_METAL_MPS=1");
    }
    return GD_OK;
}

static gd_status matmul_plan(_gd_metal_plan_ctx *ctx)
{
    gd_status status = _gd_metal_plan_default(ctx);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_metal_plan_mps_gemm(ctx);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->graph != NULL && matmul_node_uses_f16(ctx) &&
        ctx->exe->node_mps[ctx->node_id] == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal F16 matmul requires an MPS GEMM plan");
    }
    return GD_OK;
}

static gd_status matmul_encode(_gd_metal_encode_ctx *ctx)
{
    id<MTLComputeCommandEncoder> enc = *ctx->encoder;
    id<MTLComputePipelineState> pso = ctx->pso;
    _gd_executable *exe = ctx->exe;
    const _gd_node *node = ctx->node;
    GDMPSGemmPlan *mps = exe->node_mps != NULL
                             ? (__bridge GDMPSGemmPlan *)exe->node_mps[ctx->node_id]
                             : nil;

    /* `matmul_plan` prebuilds MPS descriptors for contiguous GEMMs. Use that
     * path when available; otherwise fall back to the portable Metal kernel for
     * broadcasted or offset/non-contiguous F32 shapes. F16 never uses the
     * portable F32 kernel; planning fails if no MPS plan exists. */
    if (mps != nil) {
        return _gd_metal_encode_mps_gemm(ctx->command_buffer, ctx->encoder, exe, mps);
    }

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
    return GD_OK;
}

const _gd_metal_op _gd_metal_op_matmul = {
    .kind = _GD_OP_MATMUL,
    .name = "matmul",
    .support = matmul_support,
    .plan = matmul_plan,
    .encode = matmul_encode,
};
