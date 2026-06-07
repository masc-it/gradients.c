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

static inline int gd_kv_cache_pos_for(device const uchar *pos_buf,
                                      constant gd_metal_kv_cache_append_args &p,
                                      uint b)
{
    if (p.pos_mode == GD_METAL_KV_CACHE_POS_VECTOR) {
        device const int *pos = reinterpret_cast<device const int *>(pos_buf + p.pos_offset);
        return pos[b];
    }
    return int(p.cache_pos);
}

kernel void gd_kv_cache_append_kernel(device uchar *k_cache [[buffer(0)]],
                                      device uchar *v_cache [[buffer(1)]],
                                      device const uchar *k_new [[buffer(2)]],
                                      device const uchar *v_new [[buffer(3)]],
                                      device const uchar *pos_buf [[buffer(4)]],
                                      constant gd_metal_kv_cache_append_args &p [[buffer(5)]],
                                      uint gid [[thread_position_in_grid]])
{
    if (ulong(gid) >= p.total_units || p.copy_unit == 0u || p.row_bytes == 0u ||
        p.tnew == 0u || p.tmax == 0u) {
        return;
    }

    ulong src_byte = ulong(gid) * ulong(p.copy_unit);
    ulong row = src_byte / ulong(p.row_bytes);
    ulong within = src_byte - row * ulong(p.row_bytes);
    ulong b = row / ulong(p.tnew);
    ulong t = row - b * ulong(p.tnew);
    if (b >= ulong(p.batch)) {
        return;
    }
    int pos = gd_kv_cache_pos_for(pos_buf, p, uint(b));
    if (pos < 0 || pos + int(p.tnew) > int(p.tmax)) {
        return;
    }
    ulong dst_byte = ((b * ulong(p.tmax) + ulong(pos) + t) * ulong(p.row_bytes)) + within;

    device uchar *k_dst = k_cache + p.k_cache_offset + dst_byte;
    device uchar *v_dst = v_cache + p.v_cache_offset + dst_byte;
    device const uchar *k_src = k_new + p.k_new_offset + src_byte;
    device const uchar *v_src = v_new + p.v_new_offset + src_byte;
    gd_kv_cache_copy_unit(k_dst, k_src, p.copy_unit);
    gd_kv_cache_copy_unit(v_dst, v_src, p.copy_unit);
}

static inline uint gd_kv_cache_find_batch(device const int *cu, uint batch, uint row)
{
    uint lo = 0u;
    uint hi = batch;
    while (lo + 1u < hi) {
        uint mid = (lo + hi) >> 1u;
        if (uint(cu[mid]) <= row) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

kernel void gd_kv_cache_append_packed_kernel(device uchar *k_cache [[buffer(0)]],
                                             device uchar *v_cache [[buffer(1)]],
                                             device const uchar *k_new [[buffer(2)]],
                                             device const uchar *v_new [[buffer(3)]],
                                             device const uchar *pos_buf [[buffer(4)]],
                                             device const uchar *cu_buf [[buffer(5)]],
                                             constant gd_metal_kv_cache_append_args &p [[buffer(6)]],
                                             uint gid [[thread_position_in_grid]])
{
    if (ulong(gid) >= p.total_units || p.copy_unit == 0u || p.row_bytes == 0u ||
        p.tmax == 0u || p.batch == 0u) {
        return;
    }
    uint units_per_row = p.row_bytes / p.copy_unit;
    if (units_per_row == 0u) {
        return;
    }
    uint row = gid / units_per_row;
    uint unit = gid - row * units_per_row;
    ulong within = ulong(unit) * ulong(p.copy_unit);
    device const int *cu = reinterpret_cast<device const int *>(cu_buf + p.cu_offset);
    device const int *pos_vec = reinterpret_cast<device const int *>(pos_buf + p.pos_offset);
    if (cu[0] != 0 || int(row) >= cu[p.batch]) {
        return;
    }
    uint b = gd_kv_cache_find_batch(cu, p.batch, row);
    int seq_start = cu[b];
    int seq_end = cu[b + 1u];
    int t = int(row) - seq_start;
    int pos = pos_vec[b];
    if (seq_start < 0 || seq_end < seq_start || t < 0 || pos < 0 ||
        pos + (seq_end - seq_start) > int(p.tmax)) {
        return;
    }
    ulong src_byte = ulong(row) * ulong(p.row_bytes) + within;
    ulong dst_byte = ((ulong(b) * ulong(p.tmax) + ulong(pos + t)) * ulong(p.row_bytes)) + within;
    gd_kv_cache_copy_unit(k_cache + p.k_cache_offset + dst_byte,
                          k_new + p.k_new_offset + src_byte,
                          p.copy_unit);
    gd_kv_cache_copy_unit(v_cache + p.v_cache_offset + dst_byte,
                          v_new + p.v_new_offset + src_byte,
                          p.copy_unit);
}
