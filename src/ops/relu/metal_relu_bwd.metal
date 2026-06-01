#include "metal_common.metal"

kernel void gd_relu_bwd(device const float *x             [[buffer(0)]],
                        device const float *go            [[buffer(1)]],
                        device float *dx                  [[buffer(2)]],
                        constant gd_metal_unary_params &p [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    dx[gid] = x[gid] > 0.0f ? go[gid] : 0.0f;
}
