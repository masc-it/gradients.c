#include <metal_stdlib>
#include "metal_swiglu_types.h"

using namespace metal;

static inline float gd_swiglu_exp(const float x)
{
    return fast::exp(x);
}

static inline float gd_swiglu_sigmoid(const float x)
{
    const float e = gd_swiglu_exp(-fabs(x));
    const float inv = 1.0f / (1.0f + e);
    return select(e * inv, inv, x >= 0.0f);
}

static inline float2 gd_swiglu_gate_and_grad(const float z)
{
    const float s = gd_swiglu_sigmoid(z);
    const float gate = z * s;
    const float grad = s * (1.0f + z * (1.0f - s));
    return float2(gate, grad);
}

static inline void gd_swiglu_forward_f16_one(device const half *x1,
                                             device const half *x2,
                                             device half *out,
                                             ulong i)
{
    const float gate = gd_swiglu_gate_and_grad(float(x2[i])).x;
    out[i] = half(float(x1[i]) * gate);
}

static inline void gd_swiglu_backward_f16_one(device const half *x1,
                                              device const half *x2,
                                              device const half *grad,
                                              device half *dx1,
                                              device half *dx2,
                                              constant gd_metal_swiglu_bwd_args &args,
                                              ulong i)
{
    const float z = float(x2[i]);
    const float go = float(grad[i]);
    const float2 gate_grad = gd_swiglu_gate_and_grad(z);
    if (args.write_x1 != 0u) {
        dx1[i] = half(go * gate_grad.x);
    }
    if (args.write_x2 != 0u) {
        dx2[i] = half(go * float(x1[i]) * gate_grad.y);
    }
}

kernel void gd_swiglu_forward_f16_kernel(device const uchar *x1buf [[buffer(0)]],
                                         device const uchar *x2buf [[buffer(1)]],
                                         device uchar *outbuf [[buffer(2)]],
                                         constant gd_metal_swiglu_fwd_args &args [[buffer(3)]],
                                         uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x1 = reinterpret_cast<device const half *>(x1buf + args.x1_offset);
    device const half *x2 = reinterpret_cast<device const half *>(x2buf + args.x2_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    for (ulong i = base; i < args.count && i < base + ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD); ++i) {
        gd_swiglu_forward_f16_one(x1, x2, out, i);
    }
}

kernel void gd_swiglu_backward_f16_kernel(device const uchar *x1buf [[buffer(0)]],
                                          device const uchar *x2buf [[buffer(1)]],
                                          device const uchar *gradbuf [[buffer(2)]],
                                          device uchar *dx1buf [[buffer(3)]],
                                          device uchar *dx2buf [[buffer(4)]],
                                          constant gd_metal_swiglu_bwd_args &args [[buffer(5)]],
                                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x1 = reinterpret_cast<device const half *>(x1buf + args.x1_offset);
    device const half *x2 = reinterpret_cast<device const half *>(x2buf + args.x2_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx1 = reinterpret_cast<device half *>(dx1buf + args.dx1_offset);
    device half *dx2 = reinterpret_cast<device half *>(dx2buf + args.dx2_offset);
    for (ulong i = base; i < args.count && i < base + ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD); ++i) {
        gd_swiglu_backward_f16_one(x1, x2, grad, dx1, dx2, args, i);
    }
}

kernel void gd_swiglu_split_forward_f16_kernel(device const uchar *x12buf [[buffer(0)]],
                                               device uchar *outbuf [[buffer(1)]],
                                               constant gd_metal_swiglu_split_fwd_args &args [[buffer(2)]],
                                               uint2 gid [[thread_position_in_grid]])
{
    const ulong col_base = ulong(gid.x) * ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD);
    if (col_base >= args.half_width) {
        return;
    }
    const ulong row = ulong(gid.y);
    const ulong out_row_base = row * args.half_width;
    const ulong x_row_base = out_row_base * 2ul;
    device const half *x12 = reinterpret_cast<device const half *>(x12buf + args.x12_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    for (ulong lane = 0ul; lane < ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD); ++lane) {
        const ulong col = col_base + lane;
        if (col >= args.half_width) {
            break;
        }
        const ulong out_index = out_row_base + col;
        const ulong x1_index = x_row_base + col;
        const ulong x2_index = x1_index + args.half_width;
        out[out_index] = half(float(x12[x1_index]) * gd_swiglu_gate_and_grad(float(x12[x2_index])).x);
    }
}

kernel void gd_swiglu_split_backward_f16_kernel(device const uchar *x12buf [[buffer(0)]],
                                                device const uchar *gradbuf [[buffer(1)]],
                                                device uchar *dx12buf [[buffer(2)]],
                                                constant gd_metal_swiglu_split_bwd_args &args [[buffer(3)]],
                                                uint2 gid [[thread_position_in_grid]])
{
    const ulong col_base = ulong(gid.x) * ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD);
    if (col_base >= args.half_width) {
        return;
    }
    const ulong row = ulong(gid.y);
    const ulong out_row_base = row * args.half_width;
    const ulong x_row_base = out_row_base * 2ul;
    device const half *x12 = reinterpret_cast<device const half *>(x12buf + args.x12_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx12 = reinterpret_cast<device half *>(dx12buf + args.dx12_offset);
    for (ulong lane = 0ul; lane < ulong(GD_METAL_SWIGLU_ELEMENTS_PER_THREAD); ++lane) {
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
        const float2 gate_grad = gd_swiglu_gate_and_grad(x2);
        dx12[x1_index] = half(go * gate_grad.x);
        dx12[x2_index] = half(go * x1 * gate_grad.y);
    }
}
