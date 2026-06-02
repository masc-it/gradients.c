#include "metal_common.metal"

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
        if (first > live_len) {
            first = live_len;
        }
        return first;
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

kernel void gd_sdpa_decode(device const uchar *q                         [[buffer(0)]],
                           device const uchar *k_cache                   [[buffer(1)]],
                           device const uchar *v_cache                   [[buffer(2)]],
                           device const int *cache_pos                   [[buffer(3)]],
                           device uchar *out                             [[buffer(4)]],
                           constant gd_metal_sdpa_decode_params &p       [[buffer(5)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    if (b >= p.B || p.Dh > GD_SDPA_DHT) {
        return;
    }
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int pos = cache_pos[0];
    if (pos < 0) {
        return;
    }
    int live_len = pos + p.Tq;
    if (live_len > p.Tmax) {
        live_len = p.Tmax;
    }
    int qpos = pos + i;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? gd_load_float(q, p.dtype, (uint)(qbase + c)) : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0pos = pos + qb * GD_SDPA_BQ;
    int qmax = q0pos + GD_SDPA_BQ - 1;
    int kb_end = qmax + 1;
    if (p.prefix_len > 0 && qmax < p.prefix_len) {
        kb_end = p.prefix_len;
    }
    if (kb_end > live_len) {
        kb_end = live_len;
    }
    int kb_start = gd_sdpa_decode_kb_start(q0pos, p.window, p.prefix_len, live_len);
    int kb_prefix_end = gd_sdpa_decode_kb_prefix_end(live_len, p.window, p.prefix_len);

    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        if (kb < kb_start && (kb_prefix_end == 0 || kb >= kb_prefix_end)) {
            kb = kb_start - GD_SDPA_BK;
            continue;
        }
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tmax + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = gd_load_float(k_cache, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v_cache, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_decode_allowed(qpos, j, p.window, p.prefix_len)) {
                    float s = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    s *= p.scale;
                    float mnew = s > m ? s : m;
                    float corr = exp(m - mnew);
                    float e = exp(s - mnew);
                    l = l * corr + e;
                    for (int c = 0; c < p.Dh; ++c) {
                        acc[c] = acc[c] * corr + e * vsh[jj * p.Dh + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < p.Dh; ++c) {
            gd_store_float(out, p.dtype, (uint)(qbase + c), acc[c] * inv_l);
        }
    }
}
