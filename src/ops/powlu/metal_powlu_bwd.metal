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
kernel void gd_powlu_bwd(device const uchar *x1            [[buffer(0)]],
                         device const uchar *x2            [[buffer(1)]],
                         device const uchar *go            [[buffer(2)]],
                         device uchar *dx1                 [[buffer(3)]],
                         device uchar *dx2                 [[buffer(4)]],
                         constant gd_metal_powlu_params &p  [[buffer(5)]],
                         uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v1 = gd_load_float(x1, p.dtype, gid);
    float v2 = gd_load_float(x2, p.dtype, gid);
    float g = gd_load_float(go, p.dtype, gid);
    float gate = gd_powlu_gate(v2, p.m);
    float grad = gd_powlu_gate_grad(v2, p.m);
    gd_store_float(dx1, p.dtype, gid, g * gate);
    gd_store_float(dx2, p.dtype, gid, g * v1 * grad);
}
