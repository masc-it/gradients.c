#include "metal_common.metal"

kernel void gd_silu_bwd(device const uchar *x             [[buffer(0)]],
                        device const uchar *go            [[buffer(1)]],
                        device uchar *dx                  [[buffer(2)]],
                        constant gd_metal_unary_params &p [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = gd_load_float(x, p.dtype, gid);
    float gv = gd_load_float(go, p.dtype, gid);
    float s = 1.0f / (1.0f + exp(-xv));
    float grad = s * (1.0f + xv * (1.0f - s));
    gd_store_float(dx, p.dtype, gid, gv * grad);
}
