#include <metal_stdlib>
#include "metal_relu_types.h"

using namespace metal;

static inline half gd_relu_forward_value(half v)
{
    const half zero = half(0.0h);
    /* v < 0 keeps NaNs and signed zero in the forward output. */
    return v < zero ? zero : v;
}

static inline half gd_relu_backward_value(half x, half grad)
{
    const half zero = half(0.0h);
    /* x <= 0 gives zero gradient at +/-0; NaNs pass grad_out through. */
    return x <= zero ? zero : grad;
}

kernel void gd_relu_kernel(device const uchar *xbuf [[buffer(0)]],
                           device uchar *ybuf [[buffer(1)]],
                           constant gd_metal_relu_args &args [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_RELU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + args.y_offset);
    if (base + 3ul < args.count) {
        const half x0 = x[base + 0ul];
        const half x1 = x[base + 1ul];
        const half x2 = x[base + 2ul];
        const half x3 = x[base + 3ul];
        y[base + 0ul] = gd_relu_forward_value(x0);
        y[base + 1ul] = gd_relu_forward_value(x1);
        y[base + 2ul] = gd_relu_forward_value(x2);
        y[base + 3ul] = gd_relu_forward_value(x3);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        y[i] = gd_relu_forward_value(x[i]);
    }
}

kernel void gd_relu_backward_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *gbuf [[buffer(1)]],
                                    device uchar *dxbuf [[buffer(2)]],
                                    constant gd_metal_relu_args &args [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_RELU_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *g = reinterpret_cast<device const half *>(gbuf + args.grad_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        const half x0 = x[base + 0ul];
        const half x1 = x[base + 1ul];
        const half x2 = x[base + 2ul];
        const half x3 = x[base + 3ul];
        dx[base + 0ul] = gd_relu_backward_value(x0, g[base + 0ul]);
        dx[base + 1ul] = gd_relu_backward_value(x1, g[base + 1ul]);
        dx[base + 2ul] = gd_relu_backward_value(x2, g[base + 2ul]);
        dx[base + 3ul] = gd_relu_backward_value(x3, g[base + 3ul]);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        dx[i] = gd_relu_backward_value(x[i], g[i]);
    }
}
