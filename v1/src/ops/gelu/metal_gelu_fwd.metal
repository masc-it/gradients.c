#include "metal_common.metal"

kernel void gd_gelu(device const uchar *x             [[buffer(0)]],
                    device uchar *out                 [[buffer(1)]],
                    constant gd_metal_gelu_params &p   [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v = gd_load_float(x, p.dtype, gid);
    float y;
    if (p.tanh_approx) {
        const float c = 0.7978845608028654f;
        float u = c * (v + 0.044715f * v * v * v);
        y = 0.5f * v * (1.0f + tanh(u));
    } else {
        y = 0.5f * v * (1.0f + gd_erff(v * 0.7071067811865476f));
    }
    gd_store_float(out, p.dtype, gid, y);
}
