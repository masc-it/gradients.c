#include "metal_common.metal"

kernel void gd_gelu(device const float *x             [[buffer(0)]],
                    device float *out                 [[buffer(1)]],
                    constant gd_metal_gelu_params &p   [[buffer(2)]],
                    uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float xv = x[gid];
    if (p.tanh_approx) {
        const float c = 0.7978845608028654f; /* sqrt(2/pi) */
        float inner = c * (xv + 0.044715f * xv * xv * xv);
        out[gid] = 0.5f * xv * (1.0f + tanh(inner));
    } else {
        const float inv_sqrt2 = 0.7071067811865476f;
        out[gid] = 0.5f * xv * (1.0f + gd_erff(xv * inv_sqrt2));
    }
}
