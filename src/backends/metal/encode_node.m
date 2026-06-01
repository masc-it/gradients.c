#import "metal_internal.h"

gd_status _gd_metal_encode_node(_gd_backend *self,
                                id<MTLComputeCommandEncoder> enc,
                                _gd_executable *exe,
                                const _gd_node *node,
                                id<MTLComputePipelineState> pso,
                                id<MTLComputePipelineState> pso2,
                                id<MTLComputePipelineState> pso3,
                                id<MTLBuffer> scratch)
{
    gd_status status = GD_OK;

    if (pso == nil) {
        return _gd_error(GD_ERR_UNSUPPORTED, "metal has no kernel for op");
    }
    if (_gd_metal_encode_core_node(self, enc, exe, node, pso, pso2, scratch, &status)) {
        return status;
    }
    if (_gd_metal_encode_misc_node(self, enc, exe, node, pso, pso2, pso3,
                                   scratch, &status)) {
        return status;
    }

    switch (node->op) {
    case _GD_OP_SDPA:
        _gd_metal_encode_sdpa(self, enc, pso, pso2, pso3, scratch, exe, node);
        return GD_OK;
    case _GD_OP_SDPA_BWD:
        return _gd_metal_encode_sdpa_bwd(self, enc, pso, pso2, pso3, scratch, exe, node);
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "metal op not implemented yet");
    }
}
