#include "metal_common.metal"

static inline float gd_powlu_gate(float z, float m)
{
    float s = gd_sigmoid_stable(z);
    if (z <= 0.0f) {
        return z * s;
    }
    float r = sqrt(max(z, 0.0f));
    float a = m / (r + 1.0f);
    return pow(z, a) * s;
}
kernel void gd_powlu(device const float *x1            [[buffer(0)]],
                     device const float *x2            [[buffer(1)]],
                     device float *out                 [[buffer(2)]],
                     constant gd_metal_powlu_params &p  [[buffer(3)]],
                     uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    out[gid] = x1[gid] * gd_powlu_gate(x2[gid], p.m);
}
