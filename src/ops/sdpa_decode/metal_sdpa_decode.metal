#include <metal_stdlib>
#include "metal_sdpa_decode_types.h"

using namespace metal;

static inline float gd_sdpa_decode_load(device const uchar *buf,
                                        ulong byte_offset,
                                        uint dtype,
                                        ulong idx)
{
    if (dtype == uint(GD_METAL_SDPA_DECODE_DTYPE_F16)) {
        return float(reinterpret_cast<device const half *>(buf + byte_offset)[idx]);
    }
    return reinterpret_cast<device const float *>(buf + byte_offset)[idx];
}

static inline void gd_sdpa_decode_store(device uchar *buf,
                                        ulong byte_offset,
                                        uint dtype,
                                        ulong idx,
                                        float value)
{
    if (dtype == uint(GD_METAL_SDPA_DECODE_DTYPE_F16)) {
        reinterpret_cast<device half *>(buf + byte_offset)[idx] = half(value);
    } else {
        reinterpret_cast<device float *>(buf + byte_offset)[idx] = value;
    }
}

static inline bool gd_sdpa_decode_allowed(int qpos, int j, int window, int prefix_len)
{
    if (prefix_len > 0) {
        if (qpos < prefix_len) {
            if (j >= prefix_len) {
                return false;
            }
        } else if (j > qpos) {
            return false;
        }
    } else if (j > qpos) {
        return false;
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (qpos >= prefix_len && j >= prefix_len && (qpos - j) >= window) {
                return false;
            }
        } else if ((qpos - j) >= window) {
            return false;
        }
    }
    return true;
}

static inline int gd_sdpa_decode_kb_start(int q0pos, int window, int prefix_len, int live_len)
{
    if (window <= 0) {
        return 0;
    }
    int qpos = q0pos;
    if (prefix_len > 0) {
        if (qpos < prefix_len) {
            qpos = prefix_len;
        }
        int first = qpos - window + 1;
        if (first < prefix_len) {
            first = prefix_len;
        }
        return first > live_len ? live_len : first;
    }
    int first = qpos - window + 1;
    if (first <= 0) {
        return 0;
    }
    return first > live_len ? live_len : first;
}

static inline int gd_sdpa_decode_kb_prefix_end(int live_len, int window, int prefix_len)
{
    if (window <= 0 || prefix_len <= 0) {
        return 0;
    }
    return prefix_len < live_len ? prefix_len : live_len;
}

kernel void gd_sdpa_decode_kernel(device const uchar *qbuf [[buffer(0)]],
                                  device const uchar *kbuf [[buffer(1)]],
                                  device const uchar *vbuf [[buffer(2)]],
                                  device const uchar *posbuf [[buffer(3)]],
                                  device uchar *outbuf [[buffer(4)]],
                                  constant gd_metal_sdpa_decode_args &p [[buffer(5)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_DECODE_BK * GD_METAL_SDPA_DECODE_DHT];
    threadgroup float vsh[GD_METAL_SDPA_DECODE_BK * GD_METAL_SDPA_DECODE_DHT];
    device const int *cache_pos = reinterpret_cast<device const int *>(posbuf + p.pos_offset);

    const int bq = int(GD_METAL_SDPA_DECODE_BQ);
    const int bk = int(GD_METAL_SDPA_DECODE_BK);
    int n_qb = (int(p.tq) + bq - 1) / bq;
    int qb = int(tgid) % n_qb;
    int r = int(tgid) / n_qb;
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    if (b >= int(p.batch) || p.dh > uint(GD_METAL_SDPA_DECODE_DHT)) {
        return;
    }
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * bq + int(tid);
    bool active = i < int(p.tq);
    int pos = p.pos_mode == 0u ? int(p.cache_pos) : cache_pos[p.pos_mode == 2u ? b : 0];
    if (pos < 0) {
        return;
    }
    int live_len = pos + int(p.tq);
    if (live_len > int(p.tmax)) {
        live_len = int(p.tmax);
    }
    int qpos = pos + i;
    ulong qbase = active ? ulong(((b * int(p.tq) + i) * int(p.hq) + hq) * int(p.dh)) : 0ul;

    float qreg[GD_METAL_SDPA_DECODE_DHT];
    float acc[GD_METAL_SDPA_DECODE_DHT];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_sdpa_decode_load(qbuf, p.q_offset, p.dtype, qbase + ulong(c)) : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0pos = pos + qb * bq;
    int qmax = q0pos + bq - 1;
    int kb_end = qmax + 1;
    if (p.prefix_len > 0u && qmax < int(p.prefix_len)) {
        kb_end = int(p.prefix_len);
    }
    if (kb_end > live_len) {
        kb_end = live_len;
    }
    int kb_start = gd_sdpa_decode_kb_start(q0pos, int(p.window), int(p.prefix_len), live_len);
    int kb_prefix_end = gd_sdpa_decode_kb_prefix_end(live_len, int(p.window), int(p.prefix_len));

    for (int kb0 = 0; kb0 < kb_end; kb0 += bk) {
        if (kb0 < kb_start && (kb_prefix_end == 0 || kb0 >= kb_prefix_end)) {
            kb0 = kb_start - bk;
            continue;
        }
        int tile = kb_end - kb0;
        if (tile > bk) {
            tile = bk;
        }
        for (int idx = int(tid); idx < tile * int(p.dh); idx += bq) {
            int jj = idx / int(p.dh);
            int c = idx % int(p.dh);
            ulong kbase = ulong(((b * int(p.tmax) + (kb0 + jj)) * int(p.hkv) + hkv) * int(p.dh));
            ksh[jj * int(p.dh) + c] = gd_sdpa_decode_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c));
            vsh[jj * int(p.dh) + c] = gd_sdpa_decode_load(vbuf, p.v_offset, p.dtype, kbase + ulong(c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb0 + jj;
                if (gd_sdpa_decode_allowed(qpos, j, int(p.window), int(p.prefix_len))) {
                    float s = 0.0f;
                    for (int c = 0; c < int(p.dh); ++c) {
                        s += qreg[c] * ksh[jj * int(p.dh) + c];
                    }
                    s *= p.scale;
                    float mnew = s > m ? s : m;
                    float corr = exp(m - mnew);
                    float e = exp(s - mnew);
                    l = l * corr + e;
                    for (int c = 0; c < int(p.dh); ++c) {
                        acc[c] = acc[c] * corr + e * vsh[jj * int(p.dh) + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < int(p.dh); ++c) {
            gd_sdpa_decode_store(outbuf, p.out_offset, p.dtype, qbase + ulong(c), acc[c] * inv_l);
        }
    }
}
