#include "metal_common.metal"

kernel void gd_rope(device const uchar *x                [[buffer(0)]],
                    device const int *pos                [[buffer(1)]],
                    device uchar *out                    [[buffer(2)]],
                    constant gd_metal_rope_params &p      [[buffer(3)]],
                    uint gid                             [[thread_position_in_grid]])
{
    if ((int)gid >= p.rows) {
        return;
    }
    int pos_idx = (int)gid / p.heads;
    int pp = pos[pos_idx];
    int base = (int)gid * p.head_dim;
    int rd_half = p.n_dims / 2;
    for (int d = p.n_dims; d < p.head_dim; ++d) {
        float v = gd_load_float(x, p.dtype, (uint)(base + d));
        gd_store_float(out, p.dtype, (uint)(base + d), v);
    }
    for (int i = 0; i < rd_half; ++i) {
        float inv = pow(p.theta, -((float)(2 * i)) / (float)p.n_dims);
        float angle = (float)pp * inv;
        float c = cos(angle);
        float s = sin(angle) * p.sin_sign;
        int a = p.interleaved ? (2 * i) : i;
        int bb = p.interleaved ? (2 * i + 1) : (i + rd_half);
        float x1 = gd_load_float(x, p.dtype, (uint)(base + a));
        float x2 = gd_load_float(x, p.dtype, (uint)(base + bb));
        gd_store_float(out, p.dtype, (uint)(base + a), x1 * c - x2 * s);
        gd_store_float(out, p.dtype, (uint)(base + bb), x1 * s + x2 * c);
    }
}
