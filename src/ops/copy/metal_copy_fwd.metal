#include "metal_common.metal"

kernel void gd_copy(device const uchar *x            [[buffer(0)]],
                    device uchar *out                [[buffer(1)]],
                    constant gd_metal_unary_params &p [[buffer(2)]],
                    uint gid                         [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    out[gid] = x[gid];
}
