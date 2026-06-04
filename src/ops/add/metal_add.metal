#include <metal_stdlib>
#include "metal_add_types.h"
#include "../_shared/binary/metal_binary_metal.h"

using namespace metal;

kernel void gd_add_kernel(device const uchar *xbuf [[buffer(0)]],
                          device const uchar *ybuf [[buffer(1)]],
                          device uchar *outbuf [[buffer(2)]],
                          constant gd_metal_binary_args &args [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    if (args.dtype == 1u) {
        device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
        device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
        device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
        out[i] = x[i] + y[i];
    } else if (args.dtype == 3u) {
        device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
        device const float *y = reinterpret_cast<device const float *>(ybuf + args.y_offset);
        device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
        out[i] = x[i] + y[i];
    }
}

kernel void gd_add_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                device const uchar *ybuf [[buffer(1)]],
                                device uchar *outbuf [[buffer(2)]],
                                constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    const ulong xi = gd_binary_bcast_offset(i, args, args.x_strides);
    const ulong yi = gd_binary_bcast_offset(i, args, args.y_strides);
    if (args.dtype == 1u) {
        device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
        device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
        device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
        out[i] = x[xi] + y[yi];
    } else if (args.dtype == 3u) {
        device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
        device const float *y = reinterpret_cast<device const float *>(ybuf + args.y_offset);
        device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
        out[i] = x[xi] + y[yi];
    }
}

kernel void gd_add_row_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *ybuf [[buffer(1)]],
                                    device uchar *outbuf [[buffer(2)]],
                                    constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    const ulong col = ulong(gid.x);
    const ulong row = ulong(gid.y);
    const ulong cols = ulong(args.out_shape[1]);
    if (row >= args.out_shape[0] || col >= cols) {
        return;
    }
    const ulong out_i = row * cols + col;
    const ulong xi = gd_binary_row_bcast_offset(row, col, args.x_strides);
    const ulong yi = gd_binary_row_bcast_offset(row, col, args.y_strides);
    if (args.dtype == 1u) {
        device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
        device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
        device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
        out[out_i] = x[xi] + y[yi];
    } else if (args.dtype == 3u) {
        device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
        device const float *y = reinterpret_cast<device const float *>(ybuf + args.y_offset);
        device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
        out[out_i] = x[xi] + y[yi];
    }
}
