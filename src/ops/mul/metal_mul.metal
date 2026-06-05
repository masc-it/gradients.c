#include <metal_stdlib>
#include "metal_mul_types.h"
#include "../_shared/binary/metal_binary_metal.h"

using namespace metal;

#define GD_MUL_VEC ulong(GD_METAL_MUL_F16_VECTOR_WIDTH)

static inline half4 gd_mul_load4(device const half *ptr, ulong base, ulong stride)
{
    if (stride == 1ul) {
        return *reinterpret_cast<device const half4 *>(ptr + base);
    }
    if (stride == 0ul) {
        return half4(ptr[base]);
    }
    return half4(ptr[base], ptr[base + stride], ptr[base + stride * 2ul], ptr[base + stride * 3ul]);
}

kernel void gd_mul_kernel(device const uchar *xbuf [[buffer(0)]],
                          device const uchar *ybuf [[buffer(1)]],
                          device uchar *outbuf [[buffer(2)]],
                          constant gd_metal_binary_args &args [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_MUL_VEC;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (base >= args.count) {
        return;
    }
    if (base + 3ul < args.count) {
        const half4 xv = *reinterpret_cast<device const half4 *>(x + base);
        const half4 yv = *reinterpret_cast<device const half4 *>(y + base);
        *reinterpret_cast<device half4 *>(out + base) = xv * yv;
    } else {
        for (ulong lane = 0ul; lane < GD_MUL_VEC && base + lane < args.count; ++lane) {
            out[base + lane] = x[base + lane] * y[base + lane];
        }
    }
}

kernel void gd_mul_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                device const uchar *ybuf [[buffer(1)]],
                                device uchar *outbuf [[buffer(2)]],
                                constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_MUL_VEC;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (base >= args.count) {
        return;
    }
    const uint last_dim = args.rank - 1u;
    const ulong inner = base % ulong(args.out_shape[last_dim]);
    const ulong x_last_stride = ulong(args.x_strides[last_dim]);
    const ulong y_last_stride = ulong(args.y_strides[last_dim]);
    if (base + 3ul < args.count && inner + 3ul < ulong(args.out_shape[last_dim]) &&
        (x_last_stride == 0ul || x_last_stride == 1ul) &&
        (y_last_stride == 0ul || y_last_stride == 1ul)) {
        const ulong xi = gd_binary_bcast_offset(base, args, args.x_strides);
        const ulong yi = gd_binary_bcast_offset(base, args, args.y_strides);
        const half4 xv = gd_mul_load4(x, xi, x_last_stride);
        const half4 yv = gd_mul_load4(y, yi, y_last_stride);
        *reinterpret_cast<device half4 *>(out + base) = xv * yv;
    } else {
        for (ulong lane = 0ul; lane < GD_MUL_VEC && base + lane < args.count; ++lane) {
            const ulong i = base + lane;
            const ulong xi = gd_binary_bcast_offset(i, args, args.x_strides);
            const ulong yi = gd_binary_bcast_offset(i, args, args.y_strides);
            out[i] = x[xi] * y[yi];
        }
    }
}

kernel void gd_mul_row_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *ybuf [[buffer(1)]],
                                    device uchar *outbuf [[buffer(2)]],
                                    constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    const ulong base_col = ulong(gid.x) * GD_MUL_VEC;
    const ulong row = ulong(gid.y);
    const ulong rows = ulong(args.out_shape[0]);
    const ulong cols = ulong(args.out_shape[1]);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (row >= rows || base_col >= cols) {
        return;
    }
    const ulong out_i = row * cols + base_col;
    const ulong x_col_stride = ulong(args.x_strides[1]);
    const ulong y_col_stride = ulong(args.y_strides[1]);
    const ulong x_base = row * ulong(args.x_strides[0]) + base_col * x_col_stride;
    const ulong y_base = row * ulong(args.y_strides[0]) + base_col * y_col_stride;
    if (base_col + 3ul < cols &&
        (x_col_stride == 0ul || x_col_stride == 1ul) &&
        (y_col_stride == 0ul || y_col_stride == 1ul)) {
        const half4 xv = gd_mul_load4(x, x_base, x_col_stride);
        const half4 yv = gd_mul_load4(y, y_base, y_col_stride);
        *reinterpret_cast<device half4 *>(out + out_i) = xv * yv;
    } else {
        for (ulong lane = 0ul; lane < GD_MUL_VEC && base_col + lane < cols; ++lane) {
            const ulong col = base_col + lane;
            const ulong xi = gd_binary_row_bcast_offset(row, col, args.x_strides);
            const ulong yi = gd_binary_row_bcast_offset(row, col, args.y_strides);
            out[row * cols + col] = x[xi] * y[yi];
        }
    }
}

kernel void gd_mul_backward_direct_kernel(device const uchar *xbuf [[buffer(0)]],
                                          device const uchar *ybuf [[buffer(1)]],
                                          device const uchar *gradbuf [[buffer(2)]],
                                          device uchar *grad_xbuf [[buffer(3)]],
                                          device uchar *grad_ybuf [[buffer(4)]],
                                          constant gd_metal_mul_backward_direct_args &args [[buffer(5)]],
                                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_MUL_VEC;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device half *grad_x = reinterpret_cast<device half *>(grad_xbuf + args.grad_x_offset);
    device half *grad_y = reinterpret_cast<device half *>(grad_ybuf + args.grad_y_offset);
    if (base >= args.count) {
        return;
    }
    if (base + 3ul < args.count) {
        const half4 xv = *reinterpret_cast<device const half4 *>(x + base);
        const half4 yv = *reinterpret_cast<device const half4 *>(y + base);
        const half4 gv = *reinterpret_cast<device const half4 *>(grad + base);
        *reinterpret_cast<device half4 *>(grad_x + base) = gv * yv;
        *reinterpret_cast<device half4 *>(grad_y + base) = gv * xv;
    } else {
        for (ulong lane = 0ul; lane < GD_MUL_VEC && base + lane < args.count; ++lane) {
            const ulong i = base + lane;
            grad_x[i] = grad[i] * y[i];
            grad_y[i] = grad[i] * x[i];
        }
    }
}

kernel void gd_mul_reduce_suffix_kernel(device const uchar *gradbuf [[buffer(0)]],
                                        device const uchar *otherbuf [[buffer(1)]],
                                        device uchar *dstbuf [[buffer(2)]],
                                        constant gd_metal_mul_reduce_suffix_args &args [[buffer(3)]],
                                        uint3 tgpos [[threadgroup_position_in_grid]],
                                        uint simd_lane [[thread_index_in_simdgroup]],
                                        uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong dst_i = ulong(tgpos.x) * ulong(GD_METAL_MUL_REDUCE_SUFFIX_SIMDGROUPS) + ulong(simdgroup_id);
    if (dst_i >= args.dst_count) {
        return;
    }
    const ulong repeats = args.src_count / args.dst_count;
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device const half *other = reinterpret_cast<device const half *>(otherbuf + args.other_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    float acc = 0.0f;
    for (ulong r = ulong(simd_lane); r < repeats; r += 32ul) {
        const ulong i = r * args.dst_count + dst_i;
        acc += float(grad[i]) * float(other[i]);
    }
    acc = simd_sum(acc);
    if (simd_lane == 0u) {
        dst[dst_i] = half(acc);
    }
}

kernel void gd_mul_reduce_suffix_small_kernel(device const uchar *gradbuf [[buffer(0)]],
                                              device const uchar *otherbuf [[buffer(1)]],
                                              device uchar *dstbuf [[buffer(2)]],
                                              constant gd_metal_mul_reduce_suffix_args &args [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_MUL_VEC;
    if (base >= args.dst_count) {
        return;
    }
    const ulong repeats = args.src_count / args.dst_count;
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_offset);
    device const half *other = reinterpret_cast<device const half *>(otherbuf + args.other_offset);
    device half *dst = reinterpret_cast<device half *>(dstbuf + args.dst_offset);
    if (base + 3ul < args.dst_count) {
        float4 acc = float4(0.0f);
        for (ulong r = 0ul; r < repeats; ++r) {
            const ulong i = r * args.dst_count + base;
            const half4 gv = *reinterpret_cast<device const half4 *>(grad + i);
            const half4 ov = *reinterpret_cast<device const half4 *>(other + i);
            acc += float4(gv) * float4(ov);
        }
        *reinterpret_cast<device half4 *>(dst + base) = half4(acc);
    } else {
        for (ulong lane = 0ul; lane < GD_MUL_VEC && base + lane < args.dst_count; ++lane) {
            const ulong dst_i = base + lane;
            float acc = 0.0f;
            for (ulong r = 0ul; r < repeats; ++r) {
                const ulong i = r * args.dst_count + dst_i;
                acc += float(grad[i]) * float(other[i]);
            }
            dst[dst_i] = half(acc);
        }
    }
}
