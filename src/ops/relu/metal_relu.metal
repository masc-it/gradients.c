#include <metal_stdlib>
#include "metal_relu_types.h"

using namespace metal;

kernel void gd_relu_kernel(device const uchar *xbuf [[buffer(0)]],
                           device uchar *ybuf [[buffer(1)]],
                           constant gd_metal_relu_args &args [[buffer(2)]],
                           uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    if (args.dtype == 1u) {
        device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
        device half *y = reinterpret_cast<device half *>(ybuf + args.y_offset);
        const half v = x[i];
        y[i] = v > half(0.0h) ? v : half(0.0h);
    } else if (args.dtype == 3u) {
        device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
        device float *y = reinterpret_cast<device float *>(ybuf + args.y_offset);
        const float v = x[i];
        y[i] = max(v, 0.0f);
    }
}

kernel void gd_relu_backward_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *gbuf [[buffer(1)]],
                                    device uchar *dxbuf [[buffer(2)]],
                                    constant gd_metal_relu_args &args [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    if (args.dtype == 1u) {
        device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
        device const half *g = reinterpret_cast<device const half *>(gbuf + args.grad_offset);
        device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
        dx[i] = x[i] > half(0.0h) ? g[i] : half(0.0h);
    } else if (args.dtype == 3u) {
        device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
        device const float *g = reinterpret_cast<device const float *>(gbuf + args.grad_offset);
        device float *dx = reinterpret_cast<device float *>(dxbuf + args.y_offset);
        dx[i] = x[i] > 0.0f ? g[i] : 0.0f;
    }
}
