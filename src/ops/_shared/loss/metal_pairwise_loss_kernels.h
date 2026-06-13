#ifndef GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_KERNELS_H
#define GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_KERNELS_H

static inline uint gd_pairwise_loss_sg_count(const uint requested, const uint max_simdgroups)
{
    if (requested == 0u) {
        return 1u;
    }
    return requested > max_simdgroups ? max_simdgroups : requested;
}

static inline float gd_pairwise_loss_finish(float simd_acc,
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

#endif /* GD_OPS_SHARED_LOSS_METAL_PAIRWISE_LOSS_KERNELS_H */
