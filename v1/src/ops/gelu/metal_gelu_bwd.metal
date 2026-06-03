#include "metal_common.metal"

kernel void gd_gelu_bwd(device const uchar *x             [[buffer(0)]],
                        device const uchar *go            [[buffer(1)]],
                        device uchar *dx                  [[buffer(2)]],
                        constant gd_metal_gelu_params &p   [[buffer(3)]],
                        uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = gd_load_float(x, p.dtype, gid);
    float gv = gd_load_float(go, p.dtype, gid);
    if (p.tanh_approx) {
        const float c = 0.7978845608028654f;
        float u = c * (xv + 0.044715f * xv * xv * xv);
        float t = tanh(u);
        float du = c * (1.0f + 3.0f * 0.044715f * xv * xv);
        float g = 0.5f * (1.0f + t) + 0.5f * xv * (1.0f - t * t) * du;
        gd_store_float(dx, p.dtype, gid, gv * g);
    } else {
        const float inv_sqrt2 = 0.7071067811865476f;
        const float inv_sqrt2pi = 0.3989422804014327f;
        float cdf = 0.5f * (1.0f + gd_erff(xv * inv_sqrt2));
        float pdf = inv_sqrt2pi * exp(-0.5f * xv * xv);
        gd_store_float(dx, p.dtype, gid, gv * (cdf + xv * pdf));
    }
}
