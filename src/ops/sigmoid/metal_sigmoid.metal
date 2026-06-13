#include <metal_stdlib>
#include "metal_sigmoid_types.h"

using namespace metal;

static inline float gd_sigmoid_forward_f32(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

static inline float gd_sigmoid_backward_from_x_f32(float x, float grad)
{
    const float y = gd_sigmoid_forward_f32(x);
    return grad * y * (1.0f - y);
}

static inline float gd_sigmoid_backward_from_y_f32(float y, float grad)
{
    return grad * y * (1.0f - y);
}

kernel void gd_sigmoid_kernel(device const uchar *xbuf [[buffer(0)]],
                              device uchar *ybuf [[buffer(1)]],
                              constant gd_metal_sigmoid_args &args [[buffer(2)]],
                              uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + args.y_offset);
    if (base + 3ul < args.count) {
        y[base + 0ul] = half(gd_sigmoid_forward_f32(float(x[base + 0ul])));
        y[base + 1ul] = half(gd_sigmoid_forward_f32(float(x[base + 1ul])));
        y[base + 2ul] = half(gd_sigmoid_forward_f32(float(x[base + 2ul])));
        y[base + 3ul] = half(gd_sigmoid_forward_f32(float(x[base + 3ul])));
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        y[i] = half(gd_sigmoid_forward_f32(float(x[i])));
    }
}

kernel void gd_sigmoid_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                  device uchar *ybuf [[buffer(1)]],
                                  constant gd_metal_sigmoid_args &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device float *y = reinterpret_cast<device float *>(ybuf + args.y_offset);
    if (base + 3ul < args.count) {
        y[base + 0ul] = gd_sigmoid_forward_f32(x[base + 0ul]);
        y[base + 1ul] = gd_sigmoid_forward_f32(x[base + 1ul]);
        y[base + 2ul] = gd_sigmoid_forward_f32(x[base + 2ul]);
        y[base + 3ul] = gd_sigmoid_forward_f32(x[base + 3ul]);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        y[i] = gd_sigmoid_forward_f32(x[i]);
    }
}

kernel void gd_sigmoid_backward_kernel(device const uchar *xbuf [[buffer(0)]],
                                       device const uchar *gradbuf [[buffer(1)]],
                                       device uchar *dxbuf [[buffer(2)]],
                                       constant gd_metal_sigmoid_args &args [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        dx[base + 0ul] = half(gd_sigmoid_backward_from_x_f32(float(x[base + 0ul]), float(grad[base + 0ul])));
        dx[base + 1ul] = half(gd_sigmoid_backward_from_x_f32(float(x[base + 1ul]), float(grad[base + 1ul])));
        dx[base + 2ul] = half(gd_sigmoid_backward_from_x_f32(float(x[base + 2ul]), float(grad[base + 2ul])));
        dx[base + 3ul] = half(gd_sigmoid_backward_from_x_f32(float(x[base + 3ul]), float(grad[base + 3ul])));
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        dx[i] = half(gd_sigmoid_backward_from_x_f32(float(x[i]), float(grad[i])));
    }
}

kernel void gd_sigmoid_backward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                           device const uchar *gradbuf [[buffer(1)]],
                                           device uchar *dxbuf [[buffer(2)]],
                                           constant gd_metal_sigmoid_args &args [[buffer(3)]],
                                           uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        dx[base + 0ul] = gd_sigmoid_backward_from_x_f32(x[base + 0ul], grad[base + 0ul]);
        dx[base + 1ul] = gd_sigmoid_backward_from_x_f32(x[base + 1ul], grad[base + 1ul]);
        dx[base + 2ul] = gd_sigmoid_backward_from_x_f32(x[base + 2ul], grad[base + 2ul]);
        dx[base + 3ul] = gd_sigmoid_backward_from_x_f32(x[base + 3ul], grad[base + 3ul]);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        dx[i] = gd_sigmoid_backward_from_x_f32(x[i], grad[i]);
    }
}

kernel void gd_sigmoid_backward_saved_f16_kernel(device const uchar *ybuf [[buffer(0)]],
                                                 device const uchar *gradbuf [[buffer(1)]],
                                                 device uchar *dxbuf [[buffer(2)]],
                                                 constant gd_metal_sigmoid_args &args [[buffer(3)]],
                                                 uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.x_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        dx[base + 0ul] = half(gd_sigmoid_backward_from_y_f32(float(y[base + 0ul]), float(grad[base + 0ul])));
        dx[base + 1ul] = half(gd_sigmoid_backward_from_y_f32(float(y[base + 1ul]), float(grad[base + 1ul])));
        dx[base + 2ul] = half(gd_sigmoid_backward_from_y_f32(float(y[base + 2ul]), float(grad[base + 2ul])));
        dx[base + 3ul] = half(gd_sigmoid_backward_from_y_f32(float(y[base + 3ul]), float(grad[base + 3ul])));
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        dx[i] = half(gd_sigmoid_backward_from_y_f32(float(y[i]), float(grad[i])));
    }
}

kernel void gd_sigmoid_backward_saved_f32_kernel(device const uchar *ybuf [[buffer(0)]],
                                                 device const uchar *gradbuf [[buffer(1)]],
                                                 device uchar *dxbuf [[buffer(2)]],
                                                 constant gd_metal_sigmoid_args &args [[buffer(3)]],
                                                 uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_SIGMOID_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const float *y = reinterpret_cast<device const float *>(ybuf + args.x_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        dx[base + 0ul] = gd_sigmoid_backward_from_y_f32(y[base + 0ul], grad[base + 0ul]);
        dx[base + 1ul] = gd_sigmoid_backward_from_y_f32(y[base + 1ul], grad[base + 1ul]);
        dx[base + 2ul] = gd_sigmoid_backward_from_y_f32(y[base + 2ul], grad[base + 2ul]);
        dx[base + 3ul] = gd_sigmoid_backward_from_y_f32(y[base + 3ul], grad[base + 3ul]);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        dx[i] = gd_sigmoid_backward_from_y_f32(y[i], grad[i]);
    }
}
