#include "metal_common.metal"

kernel void gd_amp_clip_norm_partial(device float *grad                    [[buffer(0)]],
                                     device float *scratch                 [[buffer(1)]],
                                     device const float *scale             [[buffer(2)]],
                                     device atomic_uint *found_inf         [[buffer(3)]],
                                     constant gd_metal_clip_norm_params &p [[buffer(4)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint tg  [[threadgroup_position_in_grid]])
{
    threadgroup float partial[GD_CLIP_NORM_TG];
    const int base = (int)tg * GD_CLIP_NORM_TG * GD_CLIP_NORM_ITEMS + (int)tid;
    float s = scale[0];
    float sum = 0.0f;

    if (!isfinite(s) || s <= 0.0f) {
        atomic_store_explicit(found_inf, 1u, memory_order_relaxed);
    } else {
        for (int k = 0; k < GD_CLIP_NORM_ITEMS; ++k) {
            int idx = base + k * GD_CLIP_NORM_TG;
            if (idx < p.numel) {
                float g = grad[idx];
                if (!isfinite(g)) {
                    atomic_store_explicit(found_inf, 1u, memory_order_relaxed);
                } else {
                    float u = g / s;
                    grad[idx] = u;
                    sum += u * u;
                }
            }
        }
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GD_CLIP_NORM_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        scratch[p.scratch_offset + (int)tg] = partial[0];
    }
}

kernel void gd_amp_clip_norm_reduce(device float *scratch                  [[buffer(0)]],
                                    device float *norm_out                 [[buffer(1)]],
                                    device const atomic_uint *found_inf    [[buffer(2)]],
                                    constant gd_metal_clip_norm_params &p  [[buffer(3)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint tg  [[threadgroup_position_in_grid]])
{
    threadgroup float partial[GD_CLIP_NORM_TG];
    int idx = (int)tg * GD_CLIP_NORM_TG + (int)tid;
    float sum = 0.0f;
    if (idx < p.numel) {
        sum = scratch[p.scratch_offset + idx];
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = GD_CLIP_NORM_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        if (p.finalize != 0 && p.total_groups == 1) {
            if (atomic_load_explicit(found_inf, memory_order_relaxed) != 0u) {
                scratch[p.scale_index] = 1.0f;
                norm_out[0] = INFINITY;
            } else {
                float norm = sqrt(partial[0]);
                float clip = 1.0f;
                if (norm > p.max_norm) {
                    clip = p.max_norm / (norm + p.eps);
                }
                scratch[p.scale_index] = clip;
                norm_out[0] = norm;
            }
        } else {
            scratch[p.dst_offset + (int)tg] = partial[0];
        }
    }
}

kernel void gd_amp_clip_norm_scale(device float *grad                       [[buffer(0)]],
                                   device const float *scratch              [[buffer(1)]],
                                   device const atomic_uint *found_inf      [[buffer(2)]],
                                   constant gd_metal_clip_norm_params &p    [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel ||
        atomic_load_explicit(found_inf, memory_order_relaxed) != 0u) {
        return;
    }
    float clip = scratch[p.scale_index];
    if (clip == 1.0f) {
        return;
    }
    grad[gid] *= clip;
}
