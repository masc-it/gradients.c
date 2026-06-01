#import "metal_op.h"

#include "metal_registry.inc"

const _gd_metal_op *_gd_metal_op_for(_gd_op_kind kind)
{
    unsigned i = 0;

    for (i = 0u; i < g_metal_op_count; ++i) {
        if (g_metal_ops[i] != NULL && g_metal_ops[i]->kind == kind) {
            return g_metal_ops[i];
        }
    }
    return NULL;
}

static bool metal_dtype_is_non_f32_float(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_BF16 ||
           dtype == GD_DTYPE_FP8_E4M3 || dtype == GD_DTYPE_FP8_E5M2;
}

gd_status _gd_metal_support_default(const _gd_metal_plan_ctx *ctx)
{
    gd_status status = GD_OK;
    int i = 0;

    if (ctx == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal support ctx is NULL");
    }
    status = _gd_op_validate_arity(ctx->node->op, ctx->node->n_inputs,
                                   ctx->node->n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (ctx->state != nil && _gd_metal_pipeline_for(ctx->state, ctx->node->op) == nil) {
        char msg[96];
        (void)snprintf(msg, sizeof(msg), "metal has no kernel for op '%s'",
                       _gd_op_kind_name(ctx->node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    if (ctx->graph != NULL) {
        for (i = 0; i < ctx->node->n_inputs; ++i) {
            gd_dtype dtype = ctx->graph->values[ctx->node->inputs[i]].desc.dtype;
            if (metal_dtype_is_non_f32_float(dtype)) {
                char msg[128];
                (void)snprintf(msg, sizeof(msg),
                               "metal op '%s' supports F32 floating tensors only in v1",
                               _gd_op_kind_name(ctx->node->op));
                return _gd_error(GD_ERR_UNSUPPORTED, msg);
            }
        }
        for (i = 0; i < ctx->node->n_outputs; ++i) {
            gd_dtype dtype = ctx->graph->values[ctx->node->outputs[i]].desc.dtype;
            if (metal_dtype_is_non_f32_float(dtype)) {
                char msg[128];
                (void)snprintf(msg, sizeof(msg),
                               "metal op '%s' supports F32 floating tensors only in v1",
                               _gd_op_kind_name(ctx->node->op));
                return _gd_error(GD_ERR_UNSUPPORTED, msg);
            }
        }
    }
    return GD_OK;
}

gd_status _gd_metal_plan_default(_gd_metal_plan_ctx *ctx)
{
    if (ctx == NULL || ctx->exe == NULL || ctx->node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal plan ctx is NULL");
    }
    ctx->exe->node_pso[ctx->node_id] = (__bridge void *)_gd_metal_pipeline_for(ctx->state,
                                                                                ctx->node->op);
    if (ctx->exe->node_pso[ctx->node_id] == NULL) {
        char msg[96];
        (void)snprintf(msg, sizeof(msg), "metal has no kernel for op '%s'",
                       _gd_op_kind_name(ctx->node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    return GD_OK;
}

gd_status _gd_metal_support_node(_gd_backend *self, const _gd_node *node)
{
    const _gd_metal_op *op = NULL;
    _gd_metal_plan_ctx ctx;

    if (node == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "Metal node is NULL");
    }
    op = _gd_metal_op_for(node->op);
    if (op == NULL) {
        char msg[112];
        (void)snprintf(msg, sizeof(msg), "metal has no host entry for op '%s'",
                       _gd_op_kind_name(node->op));
        return _gd_error(GD_ERR_UNSUPPORTED, msg);
    }
    ctx = (_gd_metal_plan_ctx){
        .backend = self,
        .state = self != NULL ? _gd_metal_state(self) : nil,
        .graph = NULL,
        .exe = NULL,
        .node = node,
        .node_id = -1,
    };
    return op->support != NULL ? op->support(&ctx) : _gd_metal_support_default(&ctx);
}
