#include "metal_common.metal"

kernel void gd_transpose(device const uint *in                [[buffer(0)]],
                         device uint *out                     [[buffer(1)]],
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
    out[gid] = in[in_off];
}
