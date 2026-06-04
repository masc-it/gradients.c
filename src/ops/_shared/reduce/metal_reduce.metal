#include <metal_stdlib>
#include "metal_reduce_types.h"

using namespace metal;

static inline ulong gd_broadcast_to_offset(ulong linear,
                                           constant gd_metal_broadcast_to_args &args)
{
    ulong rem = linear;
    ulong offset = 0ul;
    for (int dim = int(args.rank) - 1; dim >= 0; --dim) {
        const ulong size = ulong(args.dst_shape[dim]);
        const ulong coord = size > 1ul ? rem % size : 0ul;
        if (size > 1ul) {
            rem /= size;
        }
        offset += coord * ulong(args.src_strides[dim]);
    }
    return offset;
}

kernel void gd_reduce_contiguous_kernel(device const uchar *srcbuf [[buffer(0)]],
                                        device uchar *dstbuf [[buffer(1)]],
                                        constant gd_metal_reduce_contiguous_args &args [[buffer(2)]],
                                        uint3 tgpos [[threadgroup_position_in_grid]],
                                        uint simd_lane [[thread_index_in_simdgroup]],
                                        uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong dst_i = ulong(tgpos.x) * ulong(GD_METAL_REDUCE_CONTIGUOUS_SIMDGROUPS) + ulong(simdgroup_id);
    if (dst_i >= args.dst_count) {
        return;
    }
    const ulong chunk = (args.src_count + args.dst_count - 1ul) / args.dst_count;
    const ulong begin = dst_i * chunk;
    const ulong end = min(begin + chunk, ulong(args.src_count));
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong i = begin + ulong(simd_lane); i < end; i += 32ul) {
            acc += float(src[i]);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = half(acc * args.scale);
        }
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong i = begin + ulong(simd_lane); i < end; i += 32ul) {
            acc += src[i];
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = acc * args.scale;
        }
    }
}

kernel void gd_reduce_axis_kernel(device const uchar *srcbuf [[buffer(0)]],
                                  device uchar *dstbuf [[buffer(1)]],
                                  constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                  uint3 tgpos [[threadgroup_position_in_grid]],
                                  uint simd_lane [[thread_index_in_simdgroup]],
                                  uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong dst_i = ulong(tgpos.x) * ulong(GD_METAL_REDUCE_AXIS_SIMDGROUPS) + ulong(simdgroup_id);
    if (dst_i >= args.dst_count) {
        return;
    }
    const ulong inner_i = args.inner_count > 1ul ? dst_i % ulong(args.inner_count) : 0ul;
    const ulong outer_i = args.inner_count > 1ul ? dst_i / ulong(args.inner_count) : dst_i;
    const ulong base = (outer_i * ulong(args.reduce_count) * ulong(args.inner_count)) + inner_i;
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong k = ulong(simd_lane); k < args.reduce_count; k += 32ul) {
            acc += float(src[base + k * ulong(args.inner_count)]);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = half(acc * args.scale);
        }
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong k = ulong(simd_lane); k < args.reduce_count; k += 32ul) {
            acc += src[base + k * ulong(args.inner_count)];
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = acc * args.scale;
        }
    }
}

kernel void gd_broadcast_axis_kernel(device const uchar *srcbuf [[buffer(0)]],
                                     device uchar *dstbuf [[buffer(1)]],
                                     constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                     uint3 gid [[thread_position_in_grid]])
{
    const ulong inner_i = ulong(gid.x);
    const ulong reduce_i = ulong(gid.y);
    const ulong outer_i = ulong(gid.z);
    if (inner_i >= args.inner_count || reduce_i >= args.reduce_count || outer_i >= args.outer_count) {
        return;
    }
    const ulong src_i = outer_i * ulong(args.inner_count) + inner_i;
    const ulong dst_i = (outer_i * ulong(args.reduce_count) + reduce_i) * ulong(args.inner_count) + inner_i;
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        dst[dst_i] = half(float(src[src_i]) * args.scale);
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        dst[dst_i] = src[src_i] * args.scale;
    }
}

kernel void gd_broadcast_to_kernel(device const uchar *srcbuf [[buffer(0)]],
                                   device uchar *dstbuf [[buffer(1)]],
                                   constant gd_metal_broadcast_to_args &args [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.dst_count) {
        return;
    }
    const ulong src_i = gd_broadcast_to_offset(i, args);
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        dst[i] = half(float(src[src_i]) * args.scale);
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        dst[i] = src[src_i] * args.scale;
    }
}
