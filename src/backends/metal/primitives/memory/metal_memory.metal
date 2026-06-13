#include <metal_stdlib>
#include "metal_memory_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

kernel void gd_fill_kernel(device uchar *dst [[buffer(0)]],
                           constant gd_metal_fill_args &args [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    gd_write_pattern(dst, args.byte_offset + i * ulong(args.elem_size), args.elem_size, args.pattern);
}

kernel void gd_accumulate_kernel(device uchar *dst [[buffer(0)]],
                                 device const uchar *src [[buffer(1)]],
                                 constant gd_metal_accumulate_args &args [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    if (args.dtype == 1u) {
        device half *d = reinterpret_cast<device half *>(dst + args.dst_offset);
        device const half *s = reinterpret_cast<device const half *>(src + args.src_offset);
        d[i] = half(d[i] + s[i]);
    } else if (args.dtype == 3u) {
        device float *d = reinterpret_cast<device float *>(dst + args.dst_offset);
        device const float *s = reinterpret_cast<device const float *>(src + args.src_offset);
        d[i] += s[i];
    }
}

kernel void gd_scale_kernel(device uchar *dst [[buffer(0)]],
                            device const uchar *src [[buffer(1)]],
                            constant gd_metal_scale_args &args [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    if (args.dtype == 1u) {
        device half *d = reinterpret_cast<device half *>(dst + args.dst_offset);
        device const half *s = reinterpret_cast<device const half *>(src + args.src_offset);
        d[i] = args.scale == 1.0f ? s[i] : half(float(s[i]) * args.scale);
    } else if (args.dtype == 3u) {
        device float *d = reinterpret_cast<device float *>(dst + args.dst_offset);
        device const float *s = reinterpret_cast<device const float *>(src + args.src_offset);
        d[i] = args.scale == 1.0f ? s[i] : s[i] * args.scale;
    }
}
