#include "metal_common.metal"

kernel void gd_amp_step_inc(device float *step              [[buffer(0)]],
                            device const atomic_uint *found [[buffer(1)]],
                            uint gid                        [[thread_position_in_grid]])
{
    if (gid > 0) {
        return;
    }
    if (atomic_load_explicit(found, memory_order_relaxed) == 0u) {
        step[0] += 1.0f;
    }
}
