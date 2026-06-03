#include "metal_common.metal"

kernel void gd_softmax_bwd(device const float *y               [[buffer(0)]],
                           device const float *go              [[buffer(1)]],
                           device float *dx                    [[buffer(2)]],
                           constant gd_metal_softmax_params &p  [[buffer(3)]],
                           uint gid                            [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float dot = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        int idx = (o * p.d + c) * p.inner + in;
        dot += go[idx] * y[idx];
    }
    for (int c = 0; c < p.d; ++c) {
        int idx = (o * p.d + c) * p.inner + in;
        dx[idx] = y[idx] * (go[idx] - dot);
    }
}
