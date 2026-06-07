#include <metal_stdlib>
#include "metal_kv_cache_append_types.h"

using namespace metal;

static inline void gd_kv_cache_copy_unit(device uchar *dst,
                                         device const uchar *src,
                                         uint copy_unit)
{
    if (copy_unit == 16u) {
        *reinterpret_cast<device uint4 *>(dst) = *reinterpret_cast<device const uint4 *>(src);
    } else if (copy_unit == 4u) {
        *reinterpret_cast<device uint *>(dst) = *reinterpret_cast<device const uint *>(src);
    } else {
        *dst = *src;
    }
}

kernel void gd_kv_cache_append_kernel(device uchar *k_cache [[buffer(0)]],
                                      device uchar *v_cache [[buffer(1)]],
                                      device const uchar *k_new [[buffer(2)]],
                                      device const uchar *v_new [[buffer(3)]],
                                      constant gd_metal_kv_cache_append_args &p [[buffer(4)]],
                                      uint gid [[thread_position_in_grid]])
{
    if (ulong(gid) >= p.total_units || p.copy_unit == 0u || p.row_bytes == 0u ||
        p.tnew == 0u || p.tmax == 0u || p.cache_pos + p.tnew > p.tmax) {
        return;
    }

    ulong src_byte = ulong(gid) * ulong(p.copy_unit);
    ulong row = src_byte / ulong(p.row_bytes);
    ulong within = src_byte - row * ulong(p.row_bytes);
    ulong b = row / ulong(p.tnew);
    ulong t = row - b * ulong(p.tnew);
    ulong dst_byte = ((b * ulong(p.tmax) + ulong(p.cache_pos) + t) * ulong(p.row_bytes)) + within;

    device uchar *k_dst = k_cache + p.k_cache_offset + dst_byte;
    device uchar *v_dst = v_cache + p.v_cache_offset + dst_byte;
    device const uchar *k_src = k_new + p.k_new_offset + src_byte;
    device const uchar *v_src = v_new + p.v_new_offset + src_byte;
    gd_kv_cache_copy_unit(k_dst, k_src, p.copy_unit);
    gd_kv_cache_copy_unit(v_dst, v_src, p.copy_unit);
}
