#include <metal_stdlib>
#include "metal_mse_types.h"

using namespace metal;

static inline uint gd_mse_sg_count(const uint requested)
{
    if (requested == 0u) {
        return 1u;
    }
    return requested > uint(GD_METAL_MSE_MAX_SIMDGROUPS) ?
           uint(GD_METAL_MSE_MAX_SIMDGROUPS) : requested;
}

static inline float gd_mse_finish(float simd_acc,
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

kernel void gd_mse_forward_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                      device const uchar *ybuf [[buffer(1)]],
                                      device uchar *outbuf [[buffer(2)]],
                                      constant gd_metal_mse_args &args [[buffer(3)]],
                                      uint3 tgpos [[threadgroup_position_in_grid]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_MSE_MAX_SIMDGROUPS];
    const ulong chunk_id = ulong(tgpos.x);
    const uint simdgroups = gd_mse_sg_count(args.simdgroups);
    const ulong begin = chunk_id * args.chunk_size;
    const ulong end = min(begin + args.chunk_size, args.count);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        const float d = float(x[i]) - float(y[i]);
        acc += d * d;
    }
    acc = simd_sum(acc);
    acc = gd_mse_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        out[chunk_id] = acc * args.scale;
    }
}

kernel void gd_mse_forward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                      device const uchar *ybuf [[buffer(1)]],
                                      device uchar *outbuf [[buffer(2)]],
                                      constant gd_metal_mse_args &args [[buffer(3)]],
                                      uint3 tgpos [[threadgroup_position_in_grid]],
                                      uint simd_lane [[thread_index_in_simdgroup]],
                                      uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partials[GD_METAL_MSE_MAX_SIMDGROUPS];
    const ulong chunk_id = ulong(tgpos.x);
    const uint simdgroups = gd_mse_sg_count(args.simdgroups);
    const ulong begin = chunk_id * args.chunk_size;
    const ulong end = min(begin + args.chunk_size, args.count);
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *y = reinterpret_cast<device const float *>(ybuf + args.y_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    float acc = 0.0f;
    for (ulong i = begin + ulong(simdgroup_id) * 32ul + ulong(simd_lane); i < end;
         i += ulong(simdgroups) * 32ul) {
        const float d = x[i] - y[i];
        acc += d * d;
    }
    acc = simd_sum(acc);
    acc = gd_mse_finish(acc, partials, simd_lane, simdgroup_id, simdgroups);
    if (simdgroup_id == 0u && simd_lane == 0u) {
        out[chunk_id] = acc * args.scale;
    }
}

kernel void gd_mse_backward_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                       device const uchar *ybuf [[buffer(1)]],
                                       device const uchar *gradbuf [[buffer(2)]],
                                       device uchar *dxbuf [[buffer(3)]],
                                       device uchar *dybuf [[buffer(4)]],
                                       constant gd_metal_mse_args &args [[buffer(5)]],
                                       uint gid [[thread_position_in_grid]])
{
    if (ulong(gid) >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    const float g = grad[0] * args.scale;
    const float d = (float(x[gid]) - float(y[gid])) * g;
    if (args.write_x != 0u) {
        device half *dx = reinterpret_cast<device half *>(dxbuf + args.out_offset);
        dx[gid] = half(d);
    }
    if (args.write_y != 0u) {
        device half *dy = reinterpret_cast<device half *>(dybuf + args.dy_offset);
        dy[gid] = half(-d);
    }
}

kernel void gd_mse_backward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                       device const uchar *ybuf [[buffer(1)]],
                                       device const uchar *gradbuf [[buffer(2)]],
                                       device uchar *dxbuf [[buffer(3)]],
                                       device uchar *dybuf [[buffer(4)]],
                                       constant gd_metal_mse_args &args [[buffer(5)]],
                                       uint gid [[thread_position_in_grid]])
{
    if (ulong(gid) >= args.count) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *y = reinterpret_cast<device const float *>(ybuf + args.y_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    const float d = (x[gid] - y[gid]) * (grad[0] * args.scale);
    if (args.write_x != 0u) {
        device float *dx = reinterpret_cast<device float *>(dxbuf + args.out_offset);
        dx[gid] = d;
    }
    if (args.write_y != 0u) {
        device float *dy = reinterpret_cast<device float *>(dybuf + args.dy_offset);
        dy[gid] = -d;
    }
}
