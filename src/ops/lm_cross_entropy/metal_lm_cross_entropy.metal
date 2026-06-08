#include <metal_stdlib>
#include "metal_lm_cross_entropy_types.h"

using namespace metal;

#define GD_LM_CE_NEG_INF (-3.402823466e38f)

static inline float gd_lm_ce_nan(void)
{
    return as_type<float>(0x7fc00000u);
}

static inline float gd_lm_ce_threadgroup_max(float local,
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
        v = simd_lane < simdgroups ? partial[simd_lane] : GD_LM_CE_NEG_INF;
        v = simd_max(v);
        if (simd_lane == 0u) {
            shared[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

static inline float gd_lm_ce_threadgroup_sum(float local,
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

kernel void gd_lm_cross_entropy_online_update_f16_kernel(
    device const uchar *logitsbuf [[buffer(0)]],
    device const uchar *targetbuf [[buffer(1)]],
    device uchar *row_lossbuf [[buffer(2)]],
    device uchar *row_maxbuf [[buffer(3)]],
    device uchar *row_inv_sumbuf [[buffer(4)]],
    constant gd_metal_lm_cross_entropy_args &args [[buffer(5)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[GD_METAL_LM_CROSS_ENTROPY_MAX_SIMDGROUPS];
    threadgroup float shared[1];
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.chunk_classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device float *target_logit = reinterpret_cast<device float *>(row_lossbuf + args.row_loss_offset);
    device float *row_max = reinterpret_cast<device float *>(row_maxbuf + args.row_max_offset);
    device float *row_sum = reinterpret_cast<device float *>(row_inv_sumbuf + args.row_inv_sum_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * classes;
    float local_max = GD_LM_CE_NEG_INF;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_max = max(local_max, float(logits[base + c]));
    }
    const float chunk_max = gd_lm_ce_threadgroup_max(local_max,
                                                     partial,
                                                     shared,
                                                     simd_lane,
                                                     simdgroup_id,
                                                     simdgroups);
    float local_sum = 0.0f;
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        local_sum += exp(float(logits[base + c]) - chunk_max);
    }
    const float chunk_sum = gd_lm_ce_threadgroup_sum(local_sum,
                                                     partial,
                                                     shared,
                                                     simd_lane,
                                                     simdgroup_id,
                                                     simdgroups);
    if (thread_i == 0ul) {
        const int target = targets[row];
        if (target >= 0) {
            const ulong target_u = ulong(target);
            const ulong chunk_start = ulong(args.class_start);
            if (target_u >= chunk_start && target_u < chunk_start + classes &&
                target_u < ulong(args.total_classes)) {
                target_logit[row] = float(logits[base + (target_u - chunk_start)]);
            }
        }
        const float old_max = row_max[row];
        const float old_sum = row_sum[row];
        const float new_max = max(old_max, chunk_max);
        const float new_sum = old_sum * exp(old_max - new_max) +
                              chunk_sum * exp(chunk_max - new_max);
        row_max[row] = new_max;
        row_sum[row] = new_sum;
    }
}

kernel void gd_lm_cross_entropy_finalize_f32_kernel(
    device const uchar *targetbuf [[buffer(0)]],
    device uchar *row_lossbuf [[buffer(1)]],
    device uchar *row_maxbuf [[buffer(2)]],
    device uchar *row_inv_sumbuf [[buffer(3)]],
    constant gd_metal_lm_cross_entropy_args &args [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (ulong(row) >= args.rows) {
        return;
    }
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device float *row_loss = reinterpret_cast<device float *>(row_lossbuf + args.row_loss_offset);
    device float *row_max = reinterpret_cast<device float *>(row_maxbuf + args.row_max_offset);
    device float *row_inv_sum = reinterpret_cast<device float *>(row_inv_sumbuf + args.row_inv_sum_offset);
    const int target = targets[row];
    if (target < 0 || ulong(target) >= ulong(args.total_classes)) {
        row_loss[row] = gd_lm_ce_nan();
        row_max[row] = 0.0f;
        row_inv_sum[row] = 0.0f;
        return;
    }
    const float sum = row_inv_sum[row];
    const float inv_sum = 1.0f / sum;
    row_loss[row] = log(sum) + row_max[row] - row_loss[row];
    row_inv_sum[row] = inv_sum;
}

kernel void gd_lm_cross_entropy_backward_chunk_f16_kernel(
    device const uchar *logitsbuf [[buffer(0)]],
    device const uchar *targetbuf [[buffer(1)]],
    device const uchar *row_maxbuf [[buffer(2)]],
    device const uchar *row_inv_sumbuf [[buffer(3)]],
    device const uchar *gradbuf [[buffer(4)]],
    device uchar *grad_logitsbuf [[buffer(5)]],
    constant gd_metal_lm_cross_entropy_args &args [[buffer(6)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const ulong row = ulong(tgpos.x);
    const ulong classes = ulong(args.chunk_classes);
    const uint simdgroups = args.simdgroups;
    const ulong thread_i = ulong(simdgroup_id) * 32ul + ulong(simd_lane);
    const ulong thread_stride = 32ul * ulong(simdgroups);
    device const half *logits = reinterpret_cast<device const half *>(logitsbuf + args.logits_offset);
    device const int *targets = reinterpret_cast<device const int *>(targetbuf + args.target_offset);
    device const float *row_max = reinterpret_cast<device const float *>(row_maxbuf + args.row_max_offset);
    device const float *row_inv_sum = reinterpret_cast<device const float *>(row_inv_sumbuf + args.row_inv_sum_offset);
    device const float *grad_loss = reinterpret_cast<device const float *>(gradbuf + args.grad_out_offset);
    device half *grad_logits = reinterpret_cast<device half *>(grad_logitsbuf + args.out_offset);
    if (row >= args.rows) {
        return;
    }
    const ulong base = row * classes;
    const int target = targets[row];
    if (target < 0 || ulong(target) >= ulong(args.total_classes)) {
        for (ulong c = thread_i; c < classes; c += thread_stride) {
            grad_logits[base + c] = half(0.0f);
        }
        return;
    }
    const float seed = grad_loss[0] * args.scale;
    const float rm = row_max[row];
    const float inv_sum = row_inv_sum[row];
    const ulong target_u = ulong(target);
    const ulong chunk_start = ulong(args.class_start);
    for (ulong c = thread_i; c < classes; c += thread_stride) {
        const ulong global_c = chunk_start + c;
        float p = exp(float(logits[base + c]) - rm) * inv_sum;
        if (global_c == target_u) {
            p -= 1.0f;
        }
        grad_logits[base + c] = half(p * seed);
    }
}
