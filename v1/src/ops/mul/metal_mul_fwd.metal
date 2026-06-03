#include "metal_common.metal"

kernel void gd_mul(device const uchar *a          [[buffer(0)]],
                   device const uchar *b          [[buffer(1)]],
                   device uchar *out              [[buffer(2)]],
                   constant gd_metal_ew_params &p [[buffer(3)]],
                   uint gid                       [[thread_position_in_grid]])
{
    gd_binary(a, b, out, p, gid, 1);
}
