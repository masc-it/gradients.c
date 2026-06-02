#include "metal_common.metal"

kernel void gd_kv_cache_append(device uchar *k_cache                          [[buffer(0)]],
                               device uchar *v_cache                          [[buffer(1)]],
                               device const int *cache_pos                    [[buffer(2)]],
                               device const uchar *k_new                      [[buffer(3)]],
                               device const uchar *v_new                      [[buffer(4)]],
                               constant gd_metal_kv_cache_append_params &p    [[buffer(5)]],
                               uint gid [[thread_position_in_grid]])
{
    uint row_bytes = (uint)p.row_bytes;
    uint Tnew = (uint)p.Tnew;
    uint row = gid / row_bytes;
    uint within = gid - row * row_bytes;
    uint b = row / Tnew;
    uint t = row - b * Tnew;
    int pos = cache_pos[0];
    if (pos < 0 || pos + p.Tnew > p.Tmax) {
        return;
    }
    uint dst = ((b * (uint)p.Tmax + (uint)(pos + (int)t)) * row_bytes) + within;
    k_cache[dst] = k_new[gid];
    v_cache[dst] = v_new[gid];
}
