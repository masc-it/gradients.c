#include <metal_stdlib>
#include "metal_cross_entropy_types.h"

using namespace metal;

#define GD_CE_NEG_INF (-3.402823466e38f)

static inline float gd_ce_nan(void)
{
    return as_type<float>(0x7fc00000u);
}

static inline float gd_ce_threadgroup_max(float local,
                                          threadgroup float *partial,
                                          threadgroup float *shared,
                                          uint simd_lane,
                                          uint simdgroup_id,
                                          uint simdgroups)
{
    float v = simd_max(local);
    if (simd_lane == 0u) {
        partial[simdgroup_id] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0u) {
        v = simd_lane < simdgroups ? partial[simd_lane] : GD_CE_NEG_INF;
        v = simd_max(v);
        if (simd_lane == 0u) {
            shared[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

static inline float gd_ce_threadgroup_sum(float local,
                                          threadgroup float *partial,
                                          threadgroup float *shared,
                                          uint simd_lane,
                                          uint simdgroup_id,
                                          uint simdgroups)
{
    float v = simd_sum(local);
    if (simd_lane == 0u) {
        partial[simdgroup_id] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_id == 0u) {
        v = simd_lane < simdgroups ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0u) {
            shared[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

kernel void gd_cross_entropy_loss_f16_kernel(device const uchar *logitsbuf [[buffer(0)]],
                                             device const uchar *targetbuf [[buffer(1)]],
                                             device uchar *lossbuf [[buffer(2)]],
                                             constant gd_metal_cross_entropy_args &args [[buffer(3)]],
                                             uint3 tgpos [[threadgroup_position_in_grid]],
                                             uint simd_lane [[thread_index_in_simdgroup]],
                                             uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_CROSS_ENTROPY_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device float *loss = reinterpret_cast<device float *>(lossbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const int target = targets[row];
    if (target < 0 || ulong(target) >= classes) {
        if (thread_i == 0ul) {
            loss[row] = gd_ce_nan();
        }
        return;
    }
    const ulong base = row * classes;
    float local_max = GD_CE_NEG_INF;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_max = max(local_max, float(logits[base + c]));
    }
    const float row_max = gd_ce_threadgroup_max(local_max, partial, shared, simd_lane, simdgroup_id, simdgroups);
    float local_sum = 0.0f;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_sum += exp(float(logits[base + c]) - row_max);
    }
    const float row_sum = gd_ce_threadgroup_sum(local_sum, partial, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        loss[row] = log(row_sum) + row_max - float(logits[base + ulong(target)]);
    }
}

kernel void gd_cross_entropy_loss_stats_f16_kernel(device const uchar *logitsbuf [[buffer(0)]],
                                                   device const uchar *targetbuf [[buffer(1)]],
                                                   device uchar *lossbuf [[buffer(2)]],
                                                   device uchar *row_maxbuf [[buffer(3)]],
                                                   device uchar *row_inv_sumbuf [[buffer(4)]],
                                                   constant gd_metal_cross_entropy_args &args [[buffer(5)]],
                                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                                   uint simd_lane [[thread_index_in_simdgroup]],
                                                   uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_CROSS_ENTROPY_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device float *loss = reinterpret_cast<device float *>(lossbuf + args.out_offset);
    device float *row_max_out = reinterpret_cast<device float *>(row_maxbuf + args.row_max_offset);
    device float *row_inv_sum_out = reinterpret_cast<device float *>(row_inv_sumbuf + args.row_inv_sum_offset);
    if (row >= args.rows) {
        return;
    }
    const int target = targets[row];
    if (target < 0 || ulong(target) >= classes) {
        if (thread_i == 0ul) {
            loss[row] = gd_ce_nan();
            row_max_out[row] = 0.0f;
            row_inv_sum_out[row] = 0.0f;
        }
        return;
    }
    const ulong base = row * classes;
    float local_max = GD_CE_NEG_INF;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_max = max(local_max, float(logits[base + c]));
    }
    const float row_max = gd_ce_threadgroup_max(local_max, partial, shared, simd_lane, simdgroup_id, simdgroups);
    float local_sum = 0.0f;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_sum += exp(float(logits[base + c]) - row_max);
    }
    const float row_sum = gd_ce_threadgroup_sum(local_sum, partial, shared, simd_lane, simdgroup_id, simdgroups);
    if (thread_i == 0ul) {
        const float inv_sum = 1.0f / row_sum;
        loss[row] = log(row_sum) + row_max - float(logits[base + ulong(target)]);
        row_max_out[row] = row_max;
        row_inv_sum_out[row] = inv_sum;
    }
}

kernel void gd_cross_entropy_backward_f16_kernel(device const uchar *logitsbuf [[buffer(0)]],
                                                 device const uchar *targetbuf [[buffer(1)]],
                                                 device const uchar *gradbuf [[buffer(2)]],
                                                 device uchar *grad_logitsbuf [[buffer(3)]],
                                                 constant gd_metal_cross_entropy_args &args [[buffer(4)]],
                                                 uint3 tgpos [[threadgroup_position_in_grid]],
                                                 uint simd_lane [[thread_index_in_simdgroup]],
                                                 uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_CROSS_ENTROPY_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device const float *grad_loss = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device half *grad_logits = reinterpret_cast<device half *>(grad_logitsbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * classes;
    const int target = targets[row];
    if (target < 0 || ulong(target) >= classes) {
        for (ulong c = thread_i; c < classes; c += thread_stride) {
            grad_logits[base + c] = half(0.0f);
        }
        return;
    }
    float local_max = GD_CE_NEG_INF;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_max = max(local_max, float(logits[base + c]));
    }
    const float row_max = gd_ce_threadgroup_max(local_max, partial, shared, simd_lane, simdgroup_id, simdgroups);
    float local_sum = 0.0f;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_sum += exp(float(logits[base + c]) - row_max);
    }
    const float row_sum = gd_ce_threadgroup_sum(local_sum, partial, shared, simd_lane, simdgroup_id, simdgroups);
    const float seed = grad_loss[0] * args.scale;
    const float inv_sum = 1.0f / row_sum;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        float p = exp(float(logits[base + c]) - row_max) * inv_sum;
        if (c == ulong(target)) {
            p -= 1.0f;
        }
        grad_logits[base + c] = half(p * seed);
    }
}

kernel void gd_cross_entropy_backward_stats_f16_kernel(device const uchar *logitsbuf [[buffer(0)]],
                                                       device const uchar *targetbuf [[buffer(1)]],
                                                       device const uchar *gradbuf [[buffer(2)]],
                                                       device const uchar *row_maxbuf [[buffer(3)]],
                                                       device const uchar *row_inv_sumbuf [[buffer(4)]],
                                                       device uchar *grad_logitsbuf [[buffer(5)]],
                                                       constant gd_metal_cross_entropy_args &args [[buffer(6)]],
                                                       uint3 tgpos [[threadgroup_position_in_grid]],
                                                       uint simd_lane [[thread_index_in_simdgroup]],
                                                       uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device const float *grad_loss = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device const float *row_max = reinterpret_cast<device const float *>(row_maxbuf + args.row_max_offset);
    device const float *row_inv_sum = reinterpret_cast<device const float *>(row_inv_sumbuf + args.row_inv_sum_offset);
    device half *grad_logits = reinterpret_cast<device half *>(grad_logitsbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * classes;
    const int target = targets[row];
    if (target < 0 || ulong(target) >= classes) {
        for (ulong c = thread_i; c < classes; c += thread_stride) {
            grad_logits[base + c] = half(0.0f);
        }
        return;
    }
    const float seed = grad_loss[0] * args.scale;
    const float rm = row_max[row];
    const float inv_sum = row_inv_sum[row];
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        float p = exp(float(logits[base + c]) - rm) * inv_sum;
        if (c == ulong(target)) {
            p -= 1.0f;
        }
        grad_logits[base + c] = half(p * seed);
    }
}
