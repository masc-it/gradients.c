#include "metal_common.metal"

kernel void gd_amp_refresh_param(device uchar *param              [[buffer(0)]],
                                 device const float *master       [[buffer(1)]],
                                 device const atomic_uint *found  [[buffer(2)]],
                                 constant gd_metal_amp_params &p  [[buffer(3)]],
                                 uint gid                         [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel || atomic_load_explicit(found, memory_order_relaxed) != 0u) {
        return;
    }
    gd_store_float(param, p.dtype, gid, master[gid]);
}
