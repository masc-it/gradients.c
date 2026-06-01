#include "metal_common.metal"

kernel void gd_silu(device const uchar *x            [[buffer(0)]],
                    device uchar *out                [[buffer(1)]],
                    constant gd_metal_unary_params &p [[buffer(2)]],
                    uint gid                         [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = gd_load_float(x, p.dtype, gid);
    gd_store_float(out, p.dtype, gid, v / (1.0f + exp(-v)));
}
kernel void gd_silu_mul(device const float *gate         [[buffer(0)]],
                        device const float *up           [[buffer(1)]],
                        device float *hh                 [[buffer(2)]],
                        device float *act                [[buffer(3)]],
                        constant gd_metal_unary_params &p [[buffer(4)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = gate[gid];
    float a = v / (1.0f + exp(-v));
    act[gid] = a;
    hh[gid] = a * up[gid];
}
