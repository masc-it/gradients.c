#include <metal_stdlib>
#include "metal_reduce_types.h"

using namespace metal;

static inline uint gd_reduce_sg_count(const uint requested)
{
    if (requested == 0u) {
        return 1u;
    }
    return requested > uint(GD_METAL_REDUCE_MAX_SIMDGROUPS) ? uint(GD_METAL_REDUCE_MAX_SIMDGROUPS) : requested;
}

static inline float gd_reduce_finish(float simd_acc,
                                     threadgroup float *partials,
                                     uint simd_lane,
                                     uint simdgroup_id,
                                     uint simdgroups)
{
    if (simd_lane == 0u) {
        partials[simdgroup_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (simdgroup_id == 0u) {
        total = simd_lane < simdgroups ? partials[simd_lane] : 0.0f;
        total = simd_sum(total);
    }
    return total;
}

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

kernel void gd_reduce_contiguous_f16_to_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                   device uchar *dstbuf [[buffer(1)]],
                                                   constant gd_metal_reduce_contiguous_args &args [[buffer(2)]],
                                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong chunk = (args.src_count + args.dst_count - 1ul) / args.dst_count;
    const ulong begin = dst_i * chunk;
    const ulong end = min(begin + chunk, ulong(args.src_count));
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        acc += float(src[i]);
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = half(acc * args.scale);
    }
}

kernel void gd_reduce_contiguous_f16_to_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                   device uchar *dstbuf [[buffer(1)]],
                                                   constant gd_metal_reduce_contiguous_args &args [[buffer(2)]],
                                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong chunk = (args.src_count + args.dst_count - 1ul) / args.dst_count;
    const ulong begin = dst_i * chunk;
    const ulong end = min(begin + chunk, ulong(args.src_count));
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        acc += float(src[i]);
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = acc * args.scale;
    }
}

kernel void gd_reduce_contiguous_f32_to_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                   device uchar *dstbuf [[buffer(1)]],
                                                   constant gd_metal_reduce_contiguous_args &args [[buffer(2)]],
                                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong chunk = (args.src_count + args.dst_count - 1ul) / args.dst_count;
    const ulong begin = dst_i * chunk;
    const ulong end = min(begin + chunk, ulong(args.src_count));
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        acc += src[i];
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = acc * args.scale;
    }
}

kernel void gd_reduce_contiguous_f32_to_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                   device uchar *dstbuf [[buffer(1)]],
                                                   constant gd_metal_reduce_contiguous_args &args [[buffer(2)]],
                                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong chunk = (args.src_count + args.dst_count - 1ul) / args.dst_count;
    const ulong begin = dst_i * chunk;
    const ulong end = min(begin + chunk, ulong(args.src_count));
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        acc += src[i];
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = half(acc * args.scale);
    }
}

kernel void gd_reduce_axis_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                      device uchar *dstbuf [[buffer(1)]],
                                      constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                      uint3 tgpos [[threadgroup_position_in_grid]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong inner_i = args.inner_count > 1ul ? dst_i % ulong(args.inner_count) : 0ul;
    const ulong outer_i = args.inner_count > 1ul ? dst_i / ulong(args.inner_count) : dst_i;
    const ulong base = (outer_i * ulong(args.reduce_count) * ulong(args.inner_count)) + inner_i;
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong k = ulong(simdgroup_id) * 32ul + ulong(simd_lane); k < args.reduce_count;
         k += ulong(simdgroups) * 32ul) {
        acc += float(src[base + k * ulong(args.inner_count)]);
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = half(acc * args.scale);
    }
}

kernel void gd_reduce_axis_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                      device uchar *dstbuf [[buffer(1)]],
                                      constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                      uint3 tgpos [[threadgroup_position_in_grid]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong inner_i = args.inner_count > 1ul ? dst_i % ulong(args.inner_count) : 0ul;
    const ulong outer_i = args.inner_count > 1ul ? dst_i / ulong(args.inner_count) : dst_i;
    const ulong base = (outer_i * ulong(args.reduce_count) * ulong(args.inner_count)) + inner_i;
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong k = ulong(simdgroup_id) * 32ul + ulong(simd_lane); k < args.reduce_count;
         k += ulong(simdgroups) * 32ul) {
        acc += src[base + k * ulong(args.inner_count)];
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = acc * args.scale;
    }
}

kernel void gd_reduce_axis_last_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                           device uchar *dstbuf [[buffer(1)]],
                                           constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                           uint3 tgpos [[threadgroup_position_in_grid]],
                                           uint simd_lane [[thread_index_in_simdgroup]],
                                           uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong base = dst_i * ulong(args.reduce_count);
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong k = ulong(simdgroup_id) * 32ul + ulong(simd_lane); k < args.reduce_count;
         k += ulong(simdgroups) * 32ul) {
        acc += float(src[base + k]);
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = half(acc * args.scale);
    }
}

kernel void gd_reduce_axis_last_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                           device uchar *dstbuf [[buffer(1)]],
                                           constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                           uint3 tgpos [[threadgroup_position_in_grid]],
                                           uint simd_lane [[thread_index_in_simdgroup]],
                                           uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_REDUCE_MAX_SIMDGROUPS];
    const ulong dst_i = ulong(tgpos.x);
    if (dst_i >= args.dst_count) {
        return;
    }
    const uint simdgroups = gd_reduce_sg_count(args.simdgroups_per_output);
    const ulong base = dst_i * ulong(args.reduce_count);
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong k = ulong(simdgroup_id) * 32ul + ulong(simd_lane); k < args.reduce_count;
         k += ulong(simdgroups) * 32ul) {
        acc += src[base + k];
    }
    acc = simd_sum(acc);
    acc = gd_reduce_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        dst[dst_i] = acc * args.scale;
    }
}

kernel void gd_broadcast_axis_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
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
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    dst[dst_i] = half(float(src[src_i]) * args.scale);
}

kernel void gd_broadcast_axis_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
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
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    dst[dst_i] = src[src_i] * args.scale;
}

kernel void gd_broadcast_axis_last_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                              device uchar *dstbuf [[buffer(1)]],
                                              constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                              uint3 gid [[thread_position_in_grid]])
{
    const ulong col = ulong(gid.x) * ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH);
    const ulong row = ulong(gid.y);
    if (row >= args.outer_count || col >= args.reduce_count) {
        return;
    }
    const ulong dst_base = row * ulong(args.reduce_count) + col;
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    const half value = half(float(src[row]) * args.scale);
    if (col + 3ul < args.reduce_count) {
        device half4 *dst4 = reinterpret_cast<device half4 *>(dstbuf + args.dst_offset);
        dst4[dst_base / ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH)] = half4(value, value, value, value);
        return;
    }
    for (ulong lane = 0ul; lane < ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH); ++lane) {
        if (col + lane < args.reduce_count) {
            dst[dst_base + lane] = value;
        }
    }
}

kernel void gd_broadcast_axis_last_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                              device uchar *dstbuf [[buffer(1)]],
                                              constant gd_metal_reduce_axis_args &args [[buffer(2)]],
                                              uint3 gid [[thread_position_in_grid]])
{
    const ulong col = ulong(gid.x) * ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH);
    const ulong row = ulong(gid.y);
    if (row >= args.outer_count || col >= args.reduce_count) {
        return;
    }
    const ulong dst_base = row * ulong(args.reduce_count) + col;
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    const float value = src[row] * args.scale;
    if (col + 3ul < args.reduce_count) {
        device float4 *dst4 = reinterpret_cast<device float4 *>(dstbuf + args.dst_offset);
        dst4[dst_base / ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH)] = float4(value, value, value, value);
        return;
    }
    for (ulong lane = 0ul; lane < ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH); ++lane) {
        if (col + lane < args.reduce_count) {
            dst[dst_base + lane] = value;
        }
    }
}

kernel void gd_broadcast_to_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                       device uchar *dstbuf [[buffer(1)]],
                                       constant gd_metal_broadcast_to_args &args [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.dst_count) {
        return;
    }
    const ulong src_i = gd_broadcast_to_offset(i, args);
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    dst[i] = half(float(src[src_i]) * args.scale);
}

kernel void gd_broadcast_to_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                       device uchar *dstbuf [[buffer(1)]],
                                       constant gd_metal_broadcast_to_args &args [[buffer(2)]],
                                       uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.dst_count) {
        return;
    }
    const ulong src_i = gd_broadcast_to_offset(i, args);
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    dst[i] = src[src_i] * args.scale;
}

kernel void gd_broadcast_scalar_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                           device uchar *dstbuf [[buffer(1)]],
                                           constant gd_metal_broadcast_scalar_args &args [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]])
{
    device const half *src = reinterpret_cast<device const half *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    const half value = half(float(src[0]) * args.scale);
    if (args.vector_width == uint(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH)) {
        const ulong vec_count = args.dst_count / ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH);
        if (ulong(gid) < vec_count) {
            device half4 *dst4 = reinterpret_cast<device half4 *>(dstbuf + args.dst_offset);
            dst4[gid] = half4(value, value, value, value);
            return;
        }
        const ulong i = vec_count * ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) + (ulong(gid) - vec_count);
        if (i < args.dst_count) {
            dst[i] = value;
        }
        return;
    }
    if (ulong(gid) < args.dst_count) {
        dst[gid] = value;
    }
}

kernel void gd_broadcast_scalar_f32_kernel(device const uchar *srcbuf [[buffer(0)]],
                                           device uchar *dstbuf [[buffer(1)]],
                                           constant gd_metal_broadcast_scalar_args &args [[buffer(2)]],
                                           uint gid [[thread_position_in_grid]])
{
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device float *dst = reinterpret_cast<device float *>(dstbuf + args.dst_offset);
    const float value = src[0] * args.scale;
    if (args.vector_width == uint(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH)) {
        const ulong vec_count = args.dst_count / ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH);
        if (ulong(gid) < vec_count) {
            device float4 *dst4 = reinterpret_cast<device float4 *>(dstbuf + args.dst_offset);
            dst4[gid] = float4(value, value, value, value);
            return;
        }
        const ulong i = vec_count * ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) + (ulong(gid) - vec_count);
        if (i < args.dst_count) {
            dst[i] = value;
        }
        return;
    }
    if (ulong(gid) < args.dst_count) {
        dst[gid] = value;
    }
}

kernel void gd_broadcast_scalar_f32_to_f16_kernel(device const uchar *srcbuf [[buffer(0)]],
                                                  device uchar *dstbuf [[buffer(1)]],
                                                  constant gd_metal_broadcast_scalar_args &args [[buffer(2)]],
                                                  uint gid [[thread_position_in_grid]])
{
    device const float *src = reinterpret_cast<device const float *>(srcbuf + args.src_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    const half value = half(src[0] * args.scale);
    if (args.vector_width == uint(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH)) {
        const ulong vec_count = args.dst_count / ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH);
        if (ulong(gid) < vec_count) {
            device half4 *dst4 = reinterpret_cast<device half4 *>(dstbuf + args.dst_offset);
            dst4[gid] = half4(value, value, value, value);
            return;
        }
        const ulong i = vec_count * ulong(GD_METAL_REDUCE_SCALAR_VECTOR_WIDTH) + (ulong(gid) - vec_count);
        if (i < args.dst_count) {
            dst[i] = value;
        }
        return;
    }
    if (ulong(gid) < args.dst_count) {
        dst[gid] = value;
    }
}
