#include "metal_common.metal"

static inline float powlu_gate(float z, float m)
{
    float s = gd_sigmoid_stable(z);
    if (z <= 0.0f) {
        return z * s;
    }
    float r = sqrt(max(z, 0.0f));
    float a = m / (r + 1.0f);
    return pow(z, a) * s;
}

kernel void gd_powlu(device const uchar *x1            [[buffer(0)]],
                     device const uchar *x2            [[buffer(1)]],
                     device uchar *out                 [[buffer(2)]],
                     constant gd_metal_powlu_params &p [[buffer(3)]],
                     uint gid                          [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float v1 = gd_load_float(x1, p.dtype, gid);
    float v2 = gd_load_float(x2, p.dtype, gid);
    gd_store_float(out, p.dtype, gid, v1 * powlu_gate(v2, p.m));
}
