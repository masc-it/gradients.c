#include "metal_common.metal"

static inline bool gd_sdpa_varlen_allowed(int i, int j, int causal,
                                          int window, int prefix_len)
{
    if (causal) {
        if (prefix_len > 0) {
            if (i < prefix_len) {
                if (j >= prefix_len) {
                    return false;
                }
            } else if (j > i) {
                return false;
            }
        } else if (j > i) {
            return false;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (i >= prefix_len && j >= prefix_len && (i - j) >= window) {
                return false;
            }
        } else if ((i - j) >= window) {
            return false;
        }
    }
    return true;
}

static inline int gd_sdpa_varlen_kb_end(int q0, int bq, int T,
                                        int causal, int prefix_len)
{
    if (!causal) {
        return T;
    }
    int qmax = q0 + bq - 1;
    if (qmax > T - 1) {
        qmax = T - 1;
    }
    int lim = qmax + 1;
    if (prefix_len > 0 && qmax < prefix_len) {
        lim = prefix_len;
    }
    if (lim < 0) {
        return 0;
    }
    return lim > T ? T : lim;
}

static inline int gd_sdpa_varlen_kb_start(int q0, int T,
                                          int window, int prefix_len)
{
    if (window <= 0) {
        return 0;
    }
    if (prefix_len > 0) {
        int qpos = q0 < prefix_len ? prefix_len : q0;
        int first = qpos - window + 1;
        if (first < prefix_len) {
            first = prefix_len;
        }
        return first > T ? T : first;
    }
    int first = q0 - window + 1;
    if (first <= 0) {
        return 0;
    }
    return first > T ? T : first;
}

static inline int gd_sdpa_varlen_kb_prefix_end(int T, int window, int prefix_len)
{
    if (window <= 0 || prefix_len <= 0) {
        return 0;
    }
    return prefix_len < T ? prefix_len : T;
}

kernel void gd_sdpa_varlen(device const uchar *q                         [[buffer(0)]],
                           device const uchar *k                         [[buffer(1)]],
                           device const uchar *v                         [[buffer(2)]],
                           device const int *cu                          [[buffer(3)]],
                           device uchar *out                             [[buffer(4)]],
                           constant gd_metal_sdpa_varlen_params &p        [[buffer(5)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int qb = (int)tgid % p.n_qb_max;
    int r = (int)tgid / p.n_qb_max;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    if (b >= p.B || p.Dh > GD_SDPA_DHT) {
        return;
    }
    int start = cu[b];
    int end = cu[b + 1];
    int T = end - start;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? gd_load_float(q, p.dtype, (uint)(qbase + c)) : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * GD_SDPA_BQ;
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, p.window, p.prefix_len);
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, p.window, p.prefix_len);
    int kb_end = gd_sdpa_varlen_kb_end(q0, GD_SDPA_BQ, T, p.causal, p.prefix_len);
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
            int kg = start + kb + jj;
            int kbase = ((kg * p.Hkv + hkv) * p.Dh);
            ksh[jj * p.Dh + c] = gd_load_float(k, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, p.causal, p.window, p.prefix_len)) {
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

#if GD_SDPA_DHT != 64
#error "varlen DH64 f16 kernels require GD_SDPA_DHT == 64"
#endif
#if GD_SDPA_DKV_LANES != 8
#error "varlen DH64 f16 kernels assume 8 channel lanes"
#endif

#define GD_SDPA_VARLEN_DH64 64
#define GD_SDPA_VARLEN_DH64_CHANS (GD_SDPA_VARLEN_DH64 / GD_SDPA_DKV_LANES)

static inline float gd_sdpa_varlen_sum_lanes8(float x)
{
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 4);
    return x;
}

kernel void gd_sdpa_varlen_prefix_window_lane8_dh64_f16(
    device const half *q                         [[buffer(0)]],
    device const half *k                         [[buffer(1)]],
    device const half *v                         [[buffer(2)]],
    device const int *cu                         [[buffer(3)]],
    device half *out                             [[buffer(4)]],
    constant gd_metal_sdpa_varlen_params &p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];

    int n_qb = (p.max_seqlen + GD_SDPA_CAUSAL_QROWS - 1) / GD_SDPA_CAUSAL_QROWS;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int local_q = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_q * GD_SDPA_DKV_LANES;
    if (b >= p.B) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_CAUSAL_QROWS + local_q;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * p.Hq + hq) * GD_SDPA_VARLEN_DH64) : 0;

    float qreg[GD_SDPA_VARLEN_DH64_CHANS];
    float acc[GD_SDPA_VARLEN_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_LANES;
        qreg[x] = active ? (float)q[qbase + c] : 0.0f;
        acc[x] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * GD_SDPA_CAUSAL_QROWS;
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, p.window, p.prefix_len);
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, p.window, p.prefix_len);
    int kb_end = gd_sdpa_varlen_kb_end(q0, GD_SDPA_CAUSAL_QROWS, T,
                                       p.causal, p.prefix_len);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        if (kb < kb_start && (kb_prefix_end == 0 || kb >= kb_prefix_end)) {
            kb = kb_start - GD_SDPA_BK;
            continue;
        }
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * GD_SDPA_VARLEN_DH64;
             idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / GD_SDPA_VARLEN_DH64;
            int c = idx % GD_SDPA_VARLEN_DH64;
            int kg = start + kb + jj;
            int kbase = ((kg * p.Hkv + hkv) * GD_SDPA_VARLEN_DH64);
            ksh[jj * GD_SDPA_VARLEN_DH64 + c] = (float)k[kbase + c];
            vsh[jj * GD_SDPA_VARLEN_DH64 + c] = (float)v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, p.causal, p.window, p.prefix_len)) {
                    float ss = 0.0f;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        ss += qreg[x] * ksh[jj * GD_SDPA_VARLEN_DH64 + c];
                    }
                    ss = gd_sdpa_varlen_sum_lanes8(ss) * p.scale;
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        acc[x] = acc[x] * corr + e * vsh[jj * GD_SDPA_VARLEN_DH64 + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_LANES;
            out[qbase + c] = (half)(acc[x] * inv_l);
        }
    }
}
