#include "metal_common.metal"

kernel void gd_embedding_bwd(device float *dtable                 [[buffer(0)]],
                             constant gd_metal_embedding_params &p [[buffer(1)]],
                             uint gid                            [[thread_position_in_grid]])
{
    int idx = (int)gid;
    if (idx < p.vocab * p.dim) {
        dtable[idx] = 0.0f;
    }
}
kernel void gd_embedding_bwd_scatter(device const float *go               [[buffer(0)]],
                                     device const int *ids                [[buffer(1)]],
                                     device atomic_float *dtable          [[buffer(2)]],
                                     constant gd_metal_embedding_params &p [[buffer(3)]],
                                     uint gid                            [[thread_position_in_grid]])
{
    int idx = (int)gid;
    int total = p.n * p.dim;
    if (idx >= total) {
        return;
    }
    int row = idx / p.dim;
    int c = idx % p.dim;
    int v = ids[row];
    if (v >= 0 && v < p.vocab) {
        atomic_fetch_add_explicit(&dtable[v * p.dim + c], go[idx], memory_order_relaxed);
    }
}
