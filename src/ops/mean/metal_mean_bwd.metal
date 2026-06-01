#include "metal_common.metal"

kernel void gd_mean_bwd(device const float *go              [[buffer(0)]],
                       device float *dx                    [[buffer(1)]],
                       constant gd_metal_reduce_params &p   [[buffer(2)]],
                       uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float scale = p.mean ? (1.0f / (float)p.d) : 1.0f;
    float g = go[o * p.inner + in] * scale;
    for (int c = 0; c < p.d; ++c) {
        dx[(o * p.d + c) * p.inner + in] = g;
    }
}
