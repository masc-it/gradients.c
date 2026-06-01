#include "metal_common.metal"

kernel void gd_embedding(device const float *table           [[buffer(0)]],
                         device const int *ids               [[buffer(1)]],
                         device float *out                   [[buffer(2)]],
                         constant gd_metal_embedding_params &p [[buffer(3)]],
                         uint gid                            [[thread_position_in_grid]])
{
    int total = p.n * p.dim;
    if ((int)gid >= total) {
        return;
    }
    int row = (int)gid / p.dim;
    int c = (int)gid % p.dim;
    int id = ids[row];
    out[gid] = table[id * p.dim + c];
}
