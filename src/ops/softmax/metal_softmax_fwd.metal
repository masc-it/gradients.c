#include "metal_common.metal"

kernel void gd_softmax(device const uchar *x              [[buffer(0)]],
                       device uchar *out                  [[buffer(1)]],
                       constant gd_metal_softmax_params &p [[buffer(2)]],
                       uint gid                           [[thread_position_in_grid]])
{
    int total = p.outer * p.inner;
    if ((int)gid >= total) {
        return;
    }
    int o = (int)gid / p.inner;
    int in = (int)gid % p.inner;
    float maxv = -INFINITY;
    for (int c = 0; c < p.d; ++c) {
        uint idx = (uint)((o * p.d + c) * p.inner + in);
        float v = gd_load_float(x, p.dtype, idx);
        if (v > maxv) {
            maxv = v;
        }
    }
    float sum = 0.0f;
    for (int c = 0; c < p.d; ++c) {
        uint idx = (uint)((o * p.d + c) * p.inner + in);
        sum += exp(gd_load_float(x, p.dtype, idx) - maxv);
    }
    for (int c = 0; c < p.d; ++c) {
        uint idx = (uint)((o * p.d + c) * p.inner + in);
        float e = exp(gd_load_float(x, p.dtype, idx) - maxv);
        gd_store_float(out, p.dtype, idx, e / sum);
    }
}
