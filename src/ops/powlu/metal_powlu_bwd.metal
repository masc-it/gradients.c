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
static inline float gd_powlu_gate_grad(float z, float m)
{
    float s = gd_sigmoid_stable(z);
    if (z <= 0.0f) {
        return s * (1.0f + z * (1.0f - s));
    }
    float r = sqrt(max(z, 0.0f));
    float rp1 = r + 1.0f;
    float a = m / rp1;
    float g = pow(z, a);
    float da = -m / (2.0f * r * rp1 * rp1);
    float lz = log(max(z, 0x1p-126f));
    return g * s * (a / z + da * lz + (1.0f - s));
}
kernel void gd_powlu_bwd(device const float *x1            [[buffer(0)]],
                         device const float *x2            [[buffer(1)]],
                         device const float *go            [[buffer(2)]],
                         device float *dx1                 [[buffer(3)]],
                         device float *dx2                 [[buffer(4)]],
                         constant gd_metal_powlu_params &p  [[buffer(5)]],
                         uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float gate = gd_powlu_gate(x2[gid], p.m);
    float grad = gd_powlu_gate_grad(x2[gid], p.m);
    dx1[gid] = go[gid] * gate;
    dx2[gid] = go[gid] * x1[gid] * grad;
}
