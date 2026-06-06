#include <metal_stdlib>
#include "metal_rms_norm_types.h"

using namespace metal;

static inline float gd_rms_norm_threadgroup_sum(float local,
                                                threadgroup float *partial,
                                                threadgroup float *shared,
                                                uint simd_lane,
                                                uint simdgroup_id,
                                                uint simdgroups)
{
    float v = simd_sum(local);
    if (simd_lane == 0U) {
        partial[simdgroup_id] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0U) {
        v = simd_lane < simdgroups ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0U) {
            shared[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

static inline void gd_rms_norm_forward_f16_impl(device const uchar *xbuf,
                                                device const uchar *weightbuf,
                                                device uchar *outbuf,
                                                device uchar *invbuf,
                                                constant gd_metal_rms_norm_args &args,
                                                bool write_inv,
                                                uint row,
                                                uint simd_lane,
                                                uint simdgroup_id,
                                                threadgroup float *partial,
                                                threadgroup float *shared)
{
    const ulong rows = args.rows;
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *weight = reinterpret_cast<device const half *>(weightbuf + args.weight_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    device float *inv_out = write_inv ? reinterpret_cast<device float *>(invbuf + args.inv_rms_offset) : nullptr;
    if (ulong(row) >= rows) {
        return;
    }
    const ulong base = ulong(row) * cols;
    float local_ss = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = float(x[base + c]);
        local_ss += xv * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float inv = rsqrt(ss / float(cols) + args.eps);
    if (write_inv && thread_i == 0ul) {
        inv_out[row] = inv;
    }
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        out[base + c] = half(float(x[base + c]) * float(weight[c]) * inv);
    }
}

static inline void gd_rms_norm_forward_f32_impl(device const uchar *xbuf,
                                                device const uchar *weightbuf,
                                                device uchar *outbuf,
                                                device uchar *invbuf,
                                                constant gd_metal_rms_norm_args &args,
                                                bool write_inv,
                                                uint row,
                                                uint simd_lane,
                                                uint simdgroup_id,
                                                threadgroup float *partial,
                                                threadgroup float *shared)
{
    const ulong rows = args.rows;
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *weight = reinterpret_cast<device const float *>(weightbuf + args.weight_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    device float *inv_out = write_inv ? reinterpret_cast<device float *>(invbuf + args.inv_rms_offset) : nullptr;
    if (ulong(row) >= rows) {
        return;
    }
    const ulong base = ulong(row) * cols;
    float local_ss = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = x[base + c];
        local_ss += xv * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float inv = rsqrt(ss / float(cols) + args.eps);
    if (write_inv && thread_i == 0ul) {
        inv_out[row] = inv;
    }
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        out[base + c] = x[base + c] * weight[c] * inv;
    }
}

kernel void gd_rms_norm_forward_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                           device const uchar *weightbuf [[buffer(1)]],
                                           device uchar *outbuf [[buffer(2)]],
                                           constant gd_metal_rms_norm_args &args [[buffer(3)]],
                                           uint3 tgpos [[threadgroup_position_in_grid]],
                                           uint simd_lane [[thread_index_in_simdgroup]],
                                           uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_forward_f16_impl(xbuf, weightbuf, outbuf, nullptr, args, false, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_forward_stats_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                                 device const uchar *weightbuf [[buffer(1)]],
                                                 device uchar *outbuf [[buffer(2)]],
                                                 device uchar *invbuf [[buffer(3)]],
                                                 constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                                 uint3 tgpos [[threadgroup_position_in_grid]],
                                                 uint simd_lane [[thread_index_in_simdgroup]],
                                                 uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_forward_f16_impl(xbuf, weightbuf, outbuf, invbuf, args, true, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_forward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                           device const uchar *weightbuf [[buffer(1)]],
                                           device uchar *outbuf [[buffer(2)]],
                                           constant gd_metal_rms_norm_args &args [[buffer(3)]],
                                           uint3 tgpos [[threadgroup_position_in_grid]],
                                           uint simd_lane [[thread_index_in_simdgroup]],
                                           uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_forward_f32_impl(xbuf, weightbuf, outbuf, nullptr, args, false, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_forward_stats_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                                 device const uchar *weightbuf [[buffer(1)]],
                                                 device uchar *outbuf [[buffer(2)]],
                                                 device uchar *invbuf [[buffer(3)]],
                                                 constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                                 uint3 tgpos [[threadgroup_position_in_grid]],
                                                 uint simd_lane [[thread_index_in_simdgroup]],
                                                 uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_forward_f32_impl(xbuf, weightbuf, outbuf, invbuf, args, true, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_inv_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                       device uchar *invbuf [[buffer(1)]],
                                       constant gd_metal_rms_norm_args &args [[buffer(2)]],
                                       uint3 tgpos [[threadgroup_position_in_grid]],
                                       uint simd_lane [[thread_index_in_simdgroup]],
                                       uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device float *inv = reinterpret_cast<device float *>(invbuf + args.inv_rms_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * cols;
    float local_ss = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = float(x[base + c]);
        local_ss += xv * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        inv[row] = rsqrt(ss / float(cols) + args.eps);
    }
}

kernel void gd_rms_norm_inv_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                       device uchar *invbuf [[buffer(1)]],
                                       constant gd_metal_rms_norm_args &args [[buffer(2)]],
                                       uint3 tgpos [[threadgroup_position_in_grid]],
                                       uint simd_lane [[thread_index_in_simdgroup]],
                                       uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device float *inv = reinterpret_cast<device float *>(invbuf + args.inv_rms_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * cols;
    float local_ss = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = x[base + c];
        local_ss += xv * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        inv[row] = rsqrt(ss / float(cols) + args.eps);
    }
}

static inline void gd_rms_norm_backward_stats_f16_impl(device const uchar *xbuf,
                                                       device const uchar *weightbuf,
                                                       device const uchar *invbuf,
                                                       device const uchar *gradbuf,
                                                       device uchar *dxbuf,
                                                       constant gd_metal_rms_norm_args &args,
                                                       uint row,
                                                       uint simd_lane,
                                                       uint simdgroup_id,
                                                       threadgroup float *partial,
                                                       threadgroup float *shared)
{
    const ulong rows = args.rows;
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *weight = reinterpret_cast<device const half *>(weightbuf + args.weight_offset);
    device const float *invbuf_f = reinterpret_cast<device const float *>(invbuf + args.inv_rms_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_out_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.out_offset);
    if (ulong(row) >= rows) {
        return;
    }
    const ulong base = ulong(row) * cols;
    float local_a = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        local_a += float(grad[base + c]) * float(weight[c]) * float(x[base + c]);
    }
    const float a = gd_rms_norm_threadgroup_sum(local_a, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float inv = invbuf_f[row];
    const float inv3_over_cols = inv * inv * inv / float(cols);
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = float(x[base + c]);
        const float d = inv * float(grad[base + c]) * float(weight[c]) - xv * inv3_over_cols * a;
        dx[base + c] = half(d);
    }
}

static inline void gd_rms_norm_backward_stats_f32_impl(device const uchar *xbuf,
                                                       device const uchar *weightbuf,
                                                       device const uchar *invbuf,
                                                       device const uchar *gradbuf,
                                                       device uchar *dxbuf,
                                                       constant gd_metal_rms_norm_args &args,
                                                       uint row,
                                                       uint simd_lane,
                                                       uint simdgroup_id,
                                                       threadgroup float *partial,
                                                       threadgroup float *shared)
{
    const ulong rows = args.rows;
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *weight = reinterpret_cast<device const float *>(weightbuf + args.weight_offset);
    device const float *invbuf_f = reinterpret_cast<device const float *>(invbuf + args.inv_rms_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.out_offset);
    if (ulong(row) >= rows) {
        return;
    }
    const ulong base = ulong(row) * cols;
    float local_a = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        local_a += grad[base + c] * weight[c] * x[base + c];
    }
    const float a = gd_rms_norm_threadgroup_sum(local_a, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float inv = invbuf_f[row];
    const float inv3_over_cols = inv * inv * inv / float(cols);
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        dx[base + c] = inv * grad[base + c] * weight[c] - x[base + c] * inv3_over_cols * a;
    }
}

kernel void gd_rms_norm_backward_stats_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                                  device const uchar *weightbuf [[buffer(1)]],
                                                  device const uchar *invbuf [[buffer(2)]],
                                                  device const uchar *gradbuf [[buffer(3)]],
                                                  device uchar *dxbuf [[buffer(4)]],
                                                  constant gd_metal_rms_norm_args &args [[buffer(5)]],
                                                  uint3 tgpos [[threadgroup_position_in_grid]],
                                                  uint simd_lane [[thread_index_in_simdgroup]],
                                                  uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_backward_stats_f16_impl(xbuf, weightbuf, invbuf, gradbuf, dxbuf, args, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_backward_stats_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                                  device const uchar *weightbuf [[buffer(1)]],
                                                  device const uchar *invbuf [[buffer(2)]],
                                                  device const uchar *gradbuf [[buffer(3)]],
                                                  device uchar *dxbuf [[buffer(4)]],
                                                  constant gd_metal_rms_norm_args &args [[buffer(5)]],
                                                  uint3 tgpos [[threadgroup_position_in_grid]],
                                                  uint simd_lane [[thread_index_in_simdgroup]],
                                                  uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    gd_rms_norm_backward_stats_f32_impl(xbuf, weightbuf, invbuf, gradbuf, dxbuf, args, tgpos.x, simd_lane, simdgroup_id, partial, shared);
}

kernel void gd_rms_norm_backward_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                            device const uchar *weightbuf [[buffer(1)]],
                                            device const uchar *gradbuf [[buffer(2)]],
                                            device uchar *dxbuf [[buffer(3)]],
                                            constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                            uint3 tgpos [[threadgroup_position_in_grid]],
                                            uint simd_lane [[thread_index_in_simdgroup]],
                                            uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[2];
    const ulong row = ulong(tgpos.x);
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *weight = reinterpret_cast<device const half *>(weightbuf + args.weight_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_out_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * cols;
    float local_ss = 0.0f;
    float local_a = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = float(x[base + c]);
        local_ss += xv * xv;
        local_a += float(grad[base + c]) * float(weight[c]) * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float a = gd_rms_norm_threadgroup_sum(local_a, partial, shared + 1, simd_lane, simdgroup_id, simdgroups);
    const float inv = rsqrt(ss / float(cols) + args.eps);
    const float inv3_over_cols = inv * inv * inv / float(cols);
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = float(x[base + c]);
        dx[base + c] = half(inv * float(grad[base + c]) * float(weight[c]) - xv * inv3_over_cols * a);
    }
}

kernel void gd_rms_norm_backward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                            device const uchar *weightbuf [[buffer(1)]],
                                            device const uchar *gradbuf [[buffer(2)]],
                                            device uchar *dxbuf [[buffer(3)]],
                                            constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                            uint3 tgpos [[threadgroup_position_in_grid]],
                                            uint simd_lane [[thread_index_in_simdgroup]],
                                            uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[2];
    const ulong row = ulong(tgpos.x);
    const ulong cols = args.cols;
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *weight = reinterpret_cast<device const float *>(weightbuf + args.weight_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * cols;
    float local_ss = 0.0f;
    float local_a = 0.0f;
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        const float xv = x[base + c];
        local_ss += xv * xv;
        local_a += grad[base + c] * weight[c] * xv;
    }
    const float ss = gd_rms_norm_threadgroup_sum(local_ss, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float a = gd_rms_norm_threadgroup_sum(local_a, partial, shared + 1, simd_lane, simdgroup_id, simdgroups);
    const float inv = rsqrt(ss / float(cols) + args.eps);
    const float inv3_over_cols = inv * inv * inv / float(cols);
    for (ulong c = thread_i; c < cols; c += thread_stride) {
        dx[base + c] = inv * grad[base + c] * weight[c] - x[base + c] * inv3_over_cols * a;
    }
}

kernel void gd_rms_norm_wgrad_stage_stats_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                                     device const uchar *invbuf [[buffer(1)]],
                                                     device const uchar *gradbuf [[buffer(2)]],
                                                     device uchar *partialbuf [[buffer(3)]],
                                                     constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                                     uint3 tgpos [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float inv_sh[GD_METAL_RMS_NORM_WGRAD_ROW_BLOCK];
    const ulong cb = ulong(tgpos.x);
    const ulong rb = ulong(tgpos.y);
    const ulong c = cb * ulong(GD_METAL_RMS_NORM_WGRAD_CHANNELS) + ulong(tid);
    const ulong row0 = rb * ulong(args.wgrad_row_block);
    ulong tile = args.rows > row0 ? args.rows - row0 : 0ul;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const float *inv = reinterpret_cast<device const float *>(invbuf + args.inv_rms_offset);
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + args.grad_out_offset);
    device float *partial = reinterpret_cast<device float *>(partialbuf + args.partial_offset);
    if (tile > ulong(args.wgrad_row_block)) {
        tile = ulong(args.wgrad_row_block);
    }
    if (ulong(tid) < tile) {
        inv_sh[tid] = inv[row0 + ulong(tid)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < args.cols && rb < args.row_blocks) {
        float acc = 0.0f;
        for (ulong i = 0ul; i < tile; ++i) {
            const ulong idx = (row0 + i) * args.cols + c;
            acc += float(grad[idx]) * float(x[idx]) * inv_sh[i];
        }
        partial[rb * args.cols + c] = acc;
    }
}

kernel void gd_rms_norm_wgrad_stage_stats_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                                     device const uchar *invbuf [[buffer(1)]],
                                                     device const uchar *gradbuf [[buffer(2)]],
                                                     device uchar *partialbuf [[buffer(3)]],
                                                     constant gd_metal_rms_norm_args &args [[buffer(4)]],
                                                     uint3 tgpos [[threadgroup_position_in_grid]],
                                                     uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float inv_sh[GD_METAL_RMS_NORM_WGRAD_ROW_BLOCK];
    const ulong cb = ulong(tgpos.x);
    const ulong rb = ulong(tgpos.y);
    const ulong c = cb * ulong(GD_METAL_RMS_NORM_WGRAD_CHANNELS) + ulong(tid);
    const ulong row0 = rb * ulong(args.wgrad_row_block);
    ulong tile = args.rows > row0 ? args.rows - row0 : 0ul;
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const float *inv = reinterpret_cast<device const float *>(invbuf + args.inv_rms_offset);
    device const float *grad = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device float *partial = reinterpret_cast<device float *>(partialbuf + args.partial_offset);
    if (tile > ulong(args.wgrad_row_block)) {
        tile = ulong(args.wgrad_row_block);
    }
    if (ulong(tid) < tile) {
        inv_sh[tid] = inv[row0 + ulong(tid)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c < args.cols && rb < args.row_blocks) {
        float acc = 0.0f;
        for (ulong i = 0ul; i < tile; ++i) {
            const ulong idx = (row0 + i) * args.cols + c;
            acc += grad[idx] * x[idx] * inv_sh[i];
        }
        partial[rb * args.cols + c] = acc;
    }
}

kernel void gd_rms_norm_wgrad_reduce_f16_kernel(device const uchar *partialbuf [[buffer(0)]],
                                                device uchar *dwbuf [[buffer(1)]],
                                                constant gd_metal_rms_norm_args &args [[buffer(2)]],
                                                uint3 tgpos [[threadgroup_position_in_grid]],
                                                uint simd_lane [[thread_index_in_simdgroup]],
                                                uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_sh[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong c = ulong(tgpos.x);
    const uint simdgroups = args.wgrad_simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *partial = reinterpret_cast<device const float *>(partialbuf + args.partial_offset);
    device half *dw = reinterpret_cast<device half *>(dwbuf + args.out_offset);
    if (c >= args.cols) {
        return;
    }
    float local = 0.0f;
    for (ulong rb = thread_i; rb < args.row_blocks; rb += thread_stride) {
        local += partial[rb * args.cols + c];
    }
    const float sum = gd_rms_norm_threadgroup_sum(local, partial_sh, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        dw[c] = half(sum);
    }
}

kernel void gd_rms_norm_wgrad_reduce_f32_kernel(device const uchar *partialbuf [[buffer(0)]],
                                                device uchar *dwbuf [[buffer(1)]],
                                                constant gd_metal_rms_norm_args &args [[buffer(2)]],
                                                uint3 tgpos [[threadgroup_position_in_grid]],
                                                uint simd_lane [[thread_index_in_simdgroup]],
                                                uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_sh[GD_METAL_RMS_NORM_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong c = ulong(tgpos.x);
    const uint simdgroups = args.wgrad_simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const float *partial = reinterpret_cast<device const float *>(partialbuf + args.partial_offset);
    device float *dw = reinterpret_cast<device float *>(dwbuf + args.out_offset);
    if (c >= args.cols) {
        return;
    }
    float local = 0.0f;
    for (ulong rb = thread_i; rb < args.row_blocks; rb += thread_stride) {
        local += partial[rb * args.cols + c];
    }
    const float sum = gd_rms_norm_threadgroup_sum(local, partial_sh, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        dw[c] = sum;
    }
}
