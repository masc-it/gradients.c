#include "../_shared/binary/metal_binary_common.h"
#include "metal_mul_types.h"

static id<MTLComputePipelineState> gd_mul_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_pso[GD_OP_MUL];
}

static id<MTLComputePipelineState> gd_mul_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_bcast_pso[GD_OP_MUL];
}

gd_status gd_backend_mul(gd_backend *backend,
                         const gd_backend_tensor_view *x,
                         const gd_backend_tensor_view *y,
                         const gd_backend_tensor_view *out)
{
    return gd_metal_binary_dispatch(backend, x, y, out, gd_mul_pso(backend), gd_mul_bcast_pso(backend));
}
