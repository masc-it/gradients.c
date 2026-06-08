#include <metal_stdlib>
#include "metal_powlu_types.h"

using namespace metal;

static inline float gd_powlu_exp(const float x)
{
    return fast::exp(x);
}

static inline float gd_powlu_log(const float x)
{
    return fast::log(x);
}

static inline float gd_powlu_sqrt(const float x)
{
    return fast::sqrt(x);
}

static inline float gd_powlu_sigmoid_stable(const float x)
{
    if (x >= 0.0f) {
        const float e = gd_powlu_exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = gd_powlu_exp(x);
    return e / (1.0f + e);
}

static inline float gd_powlu_positive_pow(const float z, const float exponent)
{
    return gd_powlu_exp(exponent * gd_powlu_log(max(z, 0x1p-126f)));
}

static inline float gd_powlu_gate(const float z, const float m)
{
    const float s = gd_powlu_sigmoid_stable(z);
    if (z <= 0.0f) {
        return z * s;
    }
    const float r = gd_powlu_sqrt(z);
    const float a = m / (r + 1.0f);
    return gd_powlu_positive_pow(z, a) * s;
}

static inline float gd_powlu_gate_grad(const float z, const float m)
{
    const float s = gd_powlu_sigmoid_stable(z);
    if (z <= 0.0f) {
        return s * (1.0f + z * (1.0f - s));
    }
    const float r = gd_powlu_sqrt(z);
    const float rp1 = r + 1.0f;
    const float a = m / rp1;
    const float lz = gd_powlu_log(max(z, 0x1p-126f));
    const float g = gd_powlu_exp(a * lz);
    const float da = -m / (2.0f * r * rp1 * rp1);
    return g * s * (a / z + da * lz + (1.0f - s));
}

static inline float2 gd_powlu_gate_and_grad(const float z, const float m)
{
    const float s = gd_powlu_sigmoid_stable(z);
    if (z <= 0.0f) {
        return float2(z * s, s * (1.0f + z * (1.0f - s)));
    }
    const float r = gd_powlu_sqrt(z);
    const float rp1 = r + 1.0f;
    const float a = m / rp1;
    const float lz = gd_powlu_log(max(z, 0x1p-126f));
    const float g = gd_powlu_exp(a * lz);
    const float da = -m / (2.0f * r * rp1 * rp1);
    const float gate = g * s;
    return float2(gate, gate * (a / z + da * lz + (1.0f - s)));
}

kernel void gd_powlu_forward_f16_kernel(device const uchar *x1buf [[buffer(0)]],
                                        device const uchar *x2buf [[buffer(1)]],
                                        device uchar *outbuf [[buffer(2)]],
                                        constant gd_metal_powlu_fwd_args &args [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x1 = reinterpret_cast<device const half *>(x1buf + args.x1_offset);
    device const half *x2 = reinterpret_cast<device const half *>(x2buf + args.x2_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    for (ulong i = base; i < args.count && i < base + ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD); ++i) {
        out[i] = half(float(x1[i]) * gd_powlu_gate(float(x2[i]), args.m));
    }
}

kernel void gd_powlu_backward_f16_kernel(device const uchar *x1buf [[buffer(0)]],
                                         device const uchar *x2buf [[buffer(1)]],
                                         device const uchar *gradbuf [[buffer(2)]],
                                         device uchar *dx1buf [[buffer(3)]],
                                         device uchar *dx2buf [[buffer(4)]],
                                         constant gd_metal_powlu_bwd_args &args [[buffer(5)]],
                                         uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x1 = reinterpret_cast<device const half *>(x1buf + args.x1_offset);
    device const half *x2 = reinterpret_cast<device const half *>(x2buf + args.x2_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx1 = reinterpret_cast<device half *>(dx1buf + args.dx1_offset);
    device half *dx2 = reinterpret_cast<device half *>(dx2buf + args.dx2_offset);
    for (ulong i = base; i < args.count && i < base + ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD); ++i) {
        const float z = float(x2[i]);
        const float go = float(grad[i]);
        if (args.write_x1 != 0u && args.write_x2 != 0u) {
            const float2 gate_grad = gd_powlu_gate_and_grad(z, args.m);
            dx1[i] = half(go * gate_grad.x);
            dx2[i] = half(go * float(x1[i]) * gate_grad.y);
        } else if (args.write_x1 != 0u) {
            dx1[i] = half(go * gd_powlu_gate(z, args.m));
        } else if (args.write_x2 != 0u) {
            dx2[i] = half(go * float(x1[i]) * gd_powlu_gate_grad(z, args.m));
        }
    }
}

kernel void gd_powlu_split_forward_f16_kernel(device const uchar *x12buf [[buffer(0)]],
                                              device uchar *outbuf [[buffer(1)]],
                                              constant gd_metal_powlu_split_fwd_args &args [[buffer(2)]],
                                              uint2 gid [[thread_position_in_grid]])
{
    const ulong col_base = ulong(gid.x) * ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD);
    if (col_base >= args.half_width) {
        return;
    }
    const ulong row = ulong(gid.y);
    const ulong out_row_base = row * args.half_width;
    const ulong x_row_base = out_row_base * 2ul;
    device const half *x12 = reinterpret_cast<device const half *>(x12buf + args.x12_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    for (ulong lane = 0ul; lane < ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD); ++lane) {
        const ulong col = col_base + lane;
        if (col >= args.half_width) {
            break;
        }
        const ulong out_index = out_row_base + col;
        const ulong x1_index = x_row_base + col;
        const ulong x2_index = x1_index + args.half_width;
        out[out_index] = half(float(x12[x1_index]) * gd_powlu_gate(float(x12[x2_index]), args.m));
    }
}

kernel void gd_powlu_split_backward_f16_kernel(device const uchar *x12buf [[buffer(0)]],
                                               device const uchar *gradbuf [[buffer(1)]],
                                               device uchar *dx12buf [[buffer(2)]],
                                               constant gd_metal_powlu_split_bwd_args &args [[buffer(3)]],
                                               uint2 gid [[thread_position_in_grid]])
{
    const ulong col_base = ulong(gid.x) * ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD);
    if (col_base >= args.half_width) {
        return;
    }
    const ulong row = ulong(gid.y);
    const ulong out_row_base = row * args.half_width;
    const ulong x_row_base = out_row_base * 2ul;
    device const half *x12 = reinterpret_cast<device const half *>(x12buf + args.x12_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx12 = reinterpret_cast<device half *>(dx12buf + args.dx12_offset);
    for (ulong lane = 0ul; lane < ulong(GD_METAL_POWLU_ELEMENTS_PER_THREAD); ++lane) {
        const ulong col = col_base + lane;
        if (col >= args.half_width) {
            break;
        }
        const ulong out_index = out_row_base + col;
        const ulong x1_index = x_row_base + col;
        const ulong x2_index = x1_index + args.half_width;
        const float x1 = float(x12[x1_index]);
        const float x2 = float(x12[x2_index]);
        const float go = float(grad[out_index]);
        const float2 gate_grad = gd_powlu_gate_and_grad(x2, args.m);
        dx12[x1_index] = half(go * gate_grad.x);
        dx12[x2_index] = half(go * x1 * gate_grad.y);
    }
}
