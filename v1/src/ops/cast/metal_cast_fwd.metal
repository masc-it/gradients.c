#include "metal_common.metal"

static inline float gd_cast_load(device const uchar *x, int dtype, uint gid)
{
    if (dtype == GD_METAL_DT_F32) {
        return ((device const float *)x)[gid];
    }
    if (dtype == GD_METAL_DT_F16) {
        return (float)((device const half *)x)[gid];
    }
    return (float)((device const int *)x)[gid];
}

static inline void gd_cast_store(device uchar *out, int dtype, uint gid, float value)
{
    if (dtype == GD_METAL_DT_F32) {
        ((device float *)out)[gid] = value;
    } else if (dtype == GD_METAL_DT_F16) {
        ((device half *)out)[gid] = (half)value;
    } else {
        ((device int *)out)[gid] = (int)value;
    }
}

kernel void gd_cast(device const uchar *x            [[buffer(0)]],
                    device uchar *out                [[buffer(1)]],
                    constant gd_metal_cast_params &p [[buffer(2)]],
                    uint gid                         [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    if (p.src_dtype == p.dst_dtype) {
        if (p.src_dtype == GD_METAL_DT_F32 || p.src_dtype == GD_METAL_DT_I32) {
            ((device uint *)out)[gid] = ((device const uint *)x)[gid];
        } else {
            ((device ushort *)out)[gid] = ((device const ushort *)x)[gid];
        }
        return;
    }
    gd_cast_store(out, p.dst_dtype, gid, gd_cast_load(x, p.src_dtype, gid));
}

kernel void gd_cast_f16_to_f32_x4(device const half *x           [[buffer(0)]],
                                  device float *out              [[buffer(1)]],
                                  constant gd_metal_cast_params &p [[buffer(2)]],
                                  uint gid                       [[thread_position_in_grid]])
{
    uint base = gid * 4U;
    if ((int)base >= p.numel) {
        return;
    }
    if ((int)(base + 3U) < p.numel) {
        out[base] = (float)x[base];
        out[base + 1U] = (float)x[base + 1U];
        out[base + 2U] = (float)x[base + 2U];
        out[base + 3U] = (float)x[base + 3U];
        return;
    }
    for (uint i = base; i < (uint)p.numel; ++i) {
        out[i] = (float)x[i];
    }
}

kernel void gd_cast_f32_to_f16_x4(device const float *x          [[buffer(0)]],
                                  device half *out               [[buffer(1)]],
                                  constant gd_metal_cast_params &p [[buffer(2)]],
                                  uint gid                       [[thread_position_in_grid]])
{
    uint base = gid * 4U;
    if ((int)base >= p.numel) {
        return;
    }
    if ((int)(base + 3U) < p.numel) {
        out[base] = (half)x[base];
        out[base + 1U] = (half)x[base + 1U];
        out[base + 2U] = (half)x[base + 2U];
        out[base + 3U] = (half)x[base + 3U];
        return;
    }
    for (uint i = base; i < (uint)p.numel; ++i) {
        out[i] = (half)x[i];
    }
}
