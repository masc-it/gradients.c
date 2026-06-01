#include "metal_common.metal"

kernel void gd_embedding(device const uchar *table          [[buffer(0)]],
                         device const int *ids              [[buffer(1)]],
                         device uchar *out                  [[buffer(2)]],
                         constant gd_metal_embedding_params &p [[buffer(3)]],
                         uint gid                           [[thread_position_in_grid]])
{
    int total = p.n * p.dim;
    if ((int)gid >= total) {
        return;
    }
    int row = (int)gid / p.dim;
    int c = (int)gid % p.dim;
    int id = ids[row];
    float value = gd_load_float(table, p.dtype, (uint)(id * p.dim + c));
    gd_store_float(out, p.dtype, gid, value);
}
