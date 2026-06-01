#include "metal_common.metal"

kernel void gd_transpose(device const uchar *in               [[buffer(0)]],
                         device uchar *out                    [[buffer(1)]],
                         constant gd_metal_transpose_params &p [[buffer(2)]],
                         uint gid                             [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    int out_index[GD_METAL_MAX_DIMS];
    int lin = (int)gid;
    for (int k = p.ndim - 1; k >= 0; --k) {
        out_index[k] = lin % p.out_sizes[k];
        lin /= p.out_sizes[k];
    }
    int in_off = 0;
    for (int k = 0; k < p.ndim; ++k) {
        in_off += out_index[k] * p.in_strides[p.perm[k]];
    }
    int out_b = (int)gid * p.elem_size;
    int in_b = in_off * p.elem_size;
    for (int b = 0; b < p.elem_size; ++b) {
        out[out_b + b] = in[in_b + b];
    }
}
