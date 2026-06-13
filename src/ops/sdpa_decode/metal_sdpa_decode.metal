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

kernel void gd_sdpa_decode_tq1_dh64_f16_kernel(device const uchar *qbuf [[buffer(0)]],
                                               device const uchar *kbuf [[buffer(1)]],
                                               device const uchar *vbuf [[buffer(2)]],
                                               device const uchar *posbuf [[buffer(3)]],
                                               device uchar *outbuf [[buffer(4)]],
                                               constant gd_metal_sdpa_decode_args &p [[buffer(5)]],
                                               uint tgid [[threadgroup_position_in_grid]],
                                               uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_METAL_SDPA_DECODE_DHT];
    threadgroup float scores[GD_METAL_SDPA_DECODE_TQ1_KEYS];
    threadgroup float red[GD_METAL_SDPA_DECODE_TQ1_KEYS];
    device const int *cache_pos = reinterpret_cast<device const int *>(posbuf + p.pos_offset);

    const uint hq = tgid % p.hq;
    const uint b = tgid / p.hq;
    if (b >= p.batch || p.tq != 1u || p.dh != uint(GD_METAL_SDPA_DECODE_DHT) ||
        p.dtype != uint(GD_METAL_SDPA_DECODE_DTYPE_F16) || p.prefix_len != 0u ||
        p.window == 0u || p.window > uint(GD_METAL_SDPA_DECODE_TQ1_KEYS)) {
        return;
    }
    const uint group = p.hq / p.hkv;
    const uint hkv = hq / group;
    const int pos = p.pos_mode == 0u ? int(p.cache_pos) : cache_pos[p.pos_mode == 2u ? b : 0u];
    if (pos < 0 || pos >= int(p.tmax)) {
        return;
    }
    const int qpos = pos;
    int start = qpos - int(p.window) + 1;
    if (start < 0) {
        start = 0;
    }
    const int len = qpos + 1 - start;
    const ulong qbase = ulong((b * p.hq + hq) * p.dh);
    const ulong outbase = qbase;

    if (tid < p.dh) {
        qsh[tid] = gd_sdpa_decode_load(qbuf, p.q_offset, p.dtype, qbase + ulong(tid));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float score = -INFINITY;
    if (tid < uint(len)) {
        const int j = start + int(tid);
        const ulong kbase = ulong(((int(b) * int(p.tmax) + j) * int(p.hkv) + int(hkv)) * int(p.dh));
        float dot = 0.0f;
        for (uint c = 0u; c < uint(GD_METAL_SDPA_DECODE_DHT); ++c) {
            dot += qsh[c] * gd_sdpa_decode_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c));
        }
        score = dot * p.scale;
    }
    scores[tid] = score;
    red[tid] = score;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GD_METAL_SDPA_DECODE_TQ1_KEYS >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other = red[tid + stride];
            red[tid] = other > red[tid] ? other : red[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    float e = 0.0f;
    if (tid < uint(len)) {
        e = exp(scores[tid] - max_score);
    }
    scores[tid] = e;
    red[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = GD_METAL_SDPA_DECODE_TQ1_KEYS >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            red[tid] += red[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;

    if (tid < p.dh) {
        const uint c = tid;
        float acc = 0.0f;
        for (int n = 0; n < len; ++n) {
            const int j = start + n;
            const ulong vbase = ulong(((int(b) * int(p.tmax) + j) * int(p.hkv) + int(hkv)) * int(p.dh));
            acc += scores[n] * gd_sdpa_decode_load(vbuf, p.v_offset, p.dtype, vbase + ulong(c));
        }
        gd_sdpa_decode_store(outbuf, p.out_offset, p.dtype, outbase + ulong(c), acc * inv_sum);
    }
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
