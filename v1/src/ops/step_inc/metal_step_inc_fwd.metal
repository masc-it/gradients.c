#include "metal_common.metal"

kernel void gd_step_inc(device float *step [[buffer(0)]],
                        uint gid           [[thread_position_in_grid]])
{
    if ((int)gid != 0) {
        return;
    }
    step[0] += 1.0f;
}
