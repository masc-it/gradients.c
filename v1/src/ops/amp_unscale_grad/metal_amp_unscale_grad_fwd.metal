#include "metal_common.metal"

kernel void gd_amp_unscale_grad(device float *grad                 [[buffer(0)]],
                                device const float *scale          [[buffer(1)]],
                                device atomic_uint *found_inf      [[buffer(2)]],
                                constant gd_metal_amp_params &p    [[buffer(3)]],
                                uint gid                           [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float s = scale[0];
    float g = grad[gid];
    if (!isfinite(s) || s <= 0.0f || !isfinite(g)) {
        atomic_store_explicit(found_inf, 1u, memory_order_relaxed);
        return;
    }
    grad[gid] = g / s;
}
