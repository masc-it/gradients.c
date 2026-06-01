#include "metal_common.metal"

kernel void gd_rope(device const float *x                 [[buffer(0)]],
                    device const int *pos                 [[buffer(1)]],
                    device float *out                     [[buffer(2)]],
                    constant gd_metal_rope_params &p       [[buffer(3)]],
                    uint gid                              [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) {
        return;
    }
    int pos_idx = (int)gid / p.heads;
    int pp = pos[pos_idx];
    int base = (int)gid * p.head_dim;
    int rd_half = p.n_dims / 2;
    for (int d = p.n_dims; d < p.head_dim; ++d) {
        out[base + d] = x[base + d];
    }
    for (int i = 0; i < rd_half; ++i) {
        float inv = pow(p.theta, -((float)(2 * i)) / (float)p.n_dims);
        float angle = (float)pp * inv;
        float c = cos(angle);
        float s = sin(angle) * p.sin_sign;
        int a = p.interleaved ? (2 * i) : i;
        int bb = p.interleaved ? (2 * i + 1) : (i + rd_half);
        float x1 = x[base + a];
        float x2 = x[base + bb];
        out[base + a] = x1 * c - x2 * s;
        out[base + bb] = x1 * s + x2 * c;
    }
}
