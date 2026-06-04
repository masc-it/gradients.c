#include <metal_stdlib>
#include "metal_binary_types.h"

using namespace metal;

static inline ulong gd_binary_reduce_base_offset(ulong dst_linear,
                                                 constant gd_metal_binary_reduce_args &args)
{
    ulong rem = dst_linear;
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

static inline ulong gd_binary_reduce_repeat_count(constant gd_metal_binary_reduce_args &args)
{
    ulong count = 1ul;
    for (uint dim = 0u; dim < args.rank; ++dim) {
        if (args.dst_shape[dim] == 1ul) {
            count *= ulong(args.src_shape[dim]);
        }
    }
    return count;
}

static inline ulong gd_binary_reduce_repeat_offset(ulong repeat_linear,
                                                   constant gd_metal_binary_reduce_args &args)
{
    ulong rem = repeat_linear;
    ulong offset = 0ul;
    for (int dim = int(args.rank) - 1; dim >= 0; --dim) {
        if (args.dst_shape[dim] == 1ul) {
            const ulong size = ulong(args.src_shape[dim]);
            const ulong coord = size > 1ul ? rem % size : 0ul;
            if (size > 1ul) {
                rem /= size;
            }
            offset += coord * ulong(args.src_strides[dim]);
        }
    }
    return offset;
}

kernel void gd_binary_reduce_broadcast_kernel(device const uchar *srcbuf [[buffer(0)]],
                                              device uchar *dstbuf [[buffer(1)]],
                                              constant gd_metal_binary_reduce_args &args [[buffer(2)]],
                                              uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.dst_count) {
        return;
    }
    const ulong base = gd_binary_reduce_base_offset(i, args);
    const ulong repeats = gd_binary_reduce_repeat_count(args);
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong r = 0ul; r < repeats; ++r) {
            acc += float(src[base + gd_binary_reduce_repeat_offset(r, args)]);
        }
        dst[i] = half(acc * args.scale);
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong r = 0ul; r < repeats; ++r) {
            acc += src[base + gd_binary_reduce_repeat_offset(r, args)];
        }
        dst[i] = acc * args.scale;
    }
}

kernel void gd_binary_reduce_broadcast_suffix_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                     device uchar *dstbuf [[buffer(1)]],
                                                     constant gd_metal_binary_reduce_args &args [[buffer(2)]],
                                                     uint3 tgpos [[threadgroup_position_in_grid]],
                                                     uint simd_lane [[thread_index_in_simdgroup]],
                                                     uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong dst_i = ulong(tgpos.x) * ulong(GD_METAL_BINARY_REDUCE_SUFFIX_SIMDGROUPS) + ulong(simdgroup_id);
    if (dst_i >= args.dst_count) {
        return;
    }
    const ulong repeats = args.src_count / args.dst_count;
    if (args.dtype == 1u) {
        device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
        device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong r = ulong(simd_lane); r < repeats; r += 32ul) {
            acc += float(src[r * args.dst_count + dst_i]);
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = half(acc * args.scale);
        }
    } else if (args.dtype == 3u) {
        device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
        device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
        float acc = 0.0f;
        for (ulong r = ulong(simd_lane); r < repeats; r += 32ul) {
            acc += src[r * args.dst_count + dst_i];
        }
        acc = simd_sum(acc);
        if (simd_lane == 0u) {
            dst[dst_i] = acc * args.scale;
        }
    }
}
