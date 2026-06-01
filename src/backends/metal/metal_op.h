#ifndef GD_METAL_OP_H
#define GD_METAL_OP_H

#include "metal_internal.h"

typedef struct _gd_metal_plan_ctx {
    _gd_backend *backend;
    GDMetalState *state;
    gd_graph *graph;
    _gd_executable *exe;
    const _gd_node *node;
    int node_id;
} _gd_metal_plan_ctx;

typedef struct _gd_metal_encode_ctx {
    _gd_backend *backend;
    GDMetalState *state;
    id<MTLCommandBuffer> command_buffer;
    __strong id<MTLComputeCommandEncoder> *encoder;
    _gd_executable *exe;
    const _gd_node *node;
    int node_id;
    id<MTLComputePipelineState> pso;
    id<MTLComputePipelineState> pso2;
    id<MTLComputePipelineState> pso3;
    id<MTLBuffer> scratch;
} _gd_metal_encode_ctx;

typedef gd_status (*_gd_metal_support_fn)(const _gd_metal_plan_ctx *ctx);
typedef gd_status (*_gd_metal_plan_fn)(_gd_metal_plan_ctx *ctx);
typedef gd_status (*_gd_metal_encode_fn)(_gd_metal_encode_ctx *ctx);

typedef struct _gd_metal_op {
    _gd_op_kind kind;
    const char *name;
    _gd_metal_support_fn support;
    _gd_metal_plan_fn plan;
    _gd_metal_encode_fn encode;
} _gd_metal_op;

const _gd_metal_op *_gd_metal_op_for(_gd_op_kind kind);
gd_status _gd_metal_support_default(const _gd_metal_plan_ctx *ctx);
gd_status _gd_metal_support_f32_f16_same_dtype(const _gd_metal_plan_ctx *ctx);
gd_status _gd_metal_plan_default(_gd_metal_plan_ctx *ctx);
gd_status _gd_metal_plan_mps_gemm(_gd_metal_plan_ctx *ctx);
gd_status _gd_metal_support_node(_gd_backend *self, const _gd_node *node);

void _gd_metal_split_around_dim(const gd_tensor_desc *desc, int dim,
                                int *outer, int *d, int *inner);
gd_status _gd_metal_encode_binary(_gd_metal_encode_ctx *ctx);
gd_status _gd_metal_encode_unary(_gd_metal_encode_ctx *ctx, float scale);
gd_status _gd_metal_encode_unary_bwd(_gd_metal_encode_ctx *ctx);
gd_status _gd_metal_encode_reduce(_gd_metal_encode_ctx *ctx, bool mean);
gd_status _gd_metal_encode_softmax(_gd_metal_encode_ctx *ctx);
gd_status _gd_metal_encode_softmax_bwd(_gd_metal_encode_ctx *ctx);
gd_status _gd_metal_encode_sum_bwd(_gd_metal_encode_ctx *ctx, bool mean);

#endif /* GD_METAL_OP_H */
