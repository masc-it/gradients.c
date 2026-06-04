#include "../_shared/binary/metal_binary_common.h"
#include "metal_add_types.h"

static id<MTLComputePipelineState> gd_add_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_pso[GD_OP_ADD];
}

static id<MTLComputePipelineState> gd_add_bcast_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->binary_bcast_pso[GD_OP_ADD];
}

gd_status gd_backend_add(gd_backend *backend,
                         const gd_backend_tensor_view *x,
                         const gd_backend_tensor_view *y,
                         const gd_backend_tensor_view *out)
{
    return gd_metal_binary_dispatch(backend, x, y, out, gd_add_pso(backend), gd_add_bcast_pso(backend));
}
