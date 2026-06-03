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

static inline int gd_sdpa_varlen_qb_start(int k0, int qblk, int causal,
                                          int prefix_len)
{
    if (!causal) {
        return 0;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return 0;
    }
    if (k0 <= 0) {
        return 0;
    }
    return (k0 / qblk) * qblk;
}

static inline int gd_sdpa_varlen_qb_end(int k0, int key_count, int T,
                                        int window, int prefix_len)
{
    if (window <= 0) {
        return T;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return T;
    }
    int kend = k0 + key_count;
    if (kend > T) {
        kend = T;
    }
    int lim = kend + window - 1;
    if (lim < 0) {
        return 0;
    }
    return lim > T ? T : lim;
}

kernel void gd_sdpa_varlen_bwd_stats(device const uchar *go                  [[buffer(0)]],
                                     device const uchar *q                   [[buffer(1)]],
                                     device const uchar *k                   [[buffer(2)]],
                                     device const uchar *v                   [[buffer(3)]],
                                     device const int *cu                    [[buffer(4)]],
                                     device float *stats                     [[buffer(5)]],
                                     constant gd_metal_sdpa_varlen_params &p [[buffer(6)]],
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
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? gd_load_float(q, p.dtype, (uint)(qbase + c)) : 0.0f;
        goreg[c] = active ? gd_load_float(go, p.dtype, (uint)(qbase + c)) : 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

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
            int kbase = (kg * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = gd_load_float(k, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, p.causal, p.window, p.prefix_len)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s *= p.scale;
                    float mnew = s > m ? s : m;
                    float corr = exp(m - mnew);
                    float e = exp(s - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int sbase = (qg * p.Hq + hq) * 3;
        stats[sbase + 0] = m;
        stats[sbase + 1] = l;
        stats[sbase + 2] = l > 0.0f ? raw / l : 0.0f;
    }
}

kernel void gd_sdpa_varlen_bwd(device const uchar *go                        [[buffer(0)]],
                               device const uchar *q                         [[buffer(1)]],
                               device const uchar *k                         [[buffer(2)]],
                               device const uchar *v                         [[buffer(3)]],
                               device const int *cu                          [[buffer(4)]],
                               device uchar *dq                              [[buffer(5)]],
                               constant gd_metal_sdpa_varlen_params &p       [[buffer(6)]],
                               device const float *stats                     [[buffer(7)]],
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
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * p.Hq + hq) * p.Dh) : 0;
    int sbase = active ? ((qg * p.Hq + hq) * 3) : 0;
    float m = active ? stats[sbase + 0] : 0.0f;
    float l = active ? stats[sbase + 1] : 0.0f;
    float D = active ? stats[sbase + 2] : 0.0f;
    bool run = active && l > 0.0f;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    float dqacc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? gd_load_float(q, p.dtype, (uint)(qbase + c)) : 0.0f;
        goreg[c] = active ? gd_load_float(go, p.dtype, (uint)(qbase + c)) : 0.0f;
        dqacc[c] = 0.0f;
    }

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
            int kbase = (kg * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = gd_load_float(k, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (run) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, p.causal, p.window, p.prefix_len)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s *= p.scale;
                    float pj = exp(s - m) / l;
                    float ds = pj * (dp - D);
                    for (int c = 0; c < p.Dh; ++c) {
                        dqacc[c] += p.scale * ds * ksh[jj * p.Dh + c];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (int c = 0; c < p.Dh; ++c) {
            gd_store_float(dq, p.dtype, (uint)(qbase + c), dqacc[c]);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv(device const uchar *go                    [[buffer(0)]],
                                   device const uchar *q                     [[buffer(1)]],
                                   device const uchar *k                     [[buffer(2)]],
                                   device const uchar *v                     [[buffer(3)]],
                                   device const int *cu                      [[buffer(4)]],
                                   device uchar *dk                          [[buffer(5)]],
                                   device uchar *dv                          [[buffer(6)]],
                                   constant gd_metal_sdpa_varlen_params &p   [[buffer(7)]],
                                   device const float *stats                 [[buffer(8)]],
                                   uint tgid [[threadgroup_position_in_grid]],
                                   uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int kb = (int)tgid % p.n_qb_max;
    int r = (int)tgid / p.n_qb_max;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    if (b >= p.B || p.Dh > GD_SDPA_DHT) {
        return;
    }
    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int j = kb * GD_SDPA_BQ + (int)tid;
    bool active = j < T;
    int kg = start + j;
    int kbase = active ? ((kg * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DHT];
    float vreg[GD_SDPA_DHT];
    float dkacc[GD_SDPA_DHT];
    float dvacc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        kreg[c] = active ? gd_load_float(k, p.dtype, (uint)(kbase + c)) : 0.0f;
        vreg[c] = active ? gd_load_float(v, p.dtype, (uint)(kbase + c)) : 0.0f;
        dkacc[c] = 0.0f;
        dvacc[c] = 0.0f;
    }

    int k0 = kb * GD_SDPA_BQ;
    int qb_start = gd_sdpa_varlen_qb_start(k0, GD_SDPA_BK, p.causal, p.prefix_len);
    int qb_end = gd_sdpa_varlen_qb_end(k0, GD_SDPA_BQ, T, p.window, p.prefix_len);
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = qb_start; qb < qb_end; qb += GD_SDPA_BK) {
            int tile = qb_end - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
                int ii = idx / p.Dh;
                int c = idx % p.Dh;
                int qg = start + qb + ii;
                int qbase = (qg * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = gd_load_float(q, p.dtype, (uint)(qbase + c));
                gsh[ii * p.Dh + c] = gd_load_float(go, p.dtype, (uint)(qbase + c));
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_BQ) {
                int qg = start + qb + ii;
                int sb = (qg * p.Hq + hq) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    if (!gd_sdpa_varlen_allowed(i, j, p.causal, p.window, p.prefix_len)) {
                        continue;
                    }
                    float l = lsh[ii];
                    if (l <= 0.0f) {
                        continue;
                    }
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qsh[ii * p.Dh + c] * kreg[c];
                        dp += gsh[ii * p.Dh + c] * vreg[c];
                    }
                    s *= p.scale;
                    float pj = exp(s - msh[ii]) / l;
                    float ds = pj * (dp - dsh[ii]);
                    for (int c = 0; c < p.Dh; ++c) {
                        dvacc[c] += pj * gsh[ii * p.Dh + c];
                        dkacc[c] += p.scale * ds * qsh[ii * p.Dh + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        for (int c = 0; c < p.Dh; ++c) {
            gd_store_float(dk, p.dtype, (uint)(kbase + c), dkacc[c]);
            gd_store_float(dv, p.dtype, (uint)(kbase + c), dvacc[c]);
        }
    }
}

#if GD_SDPA_DHT != 64
#error "varlen DH64 f16 backward kernels require GD_SDPA_DHT == 64"
#endif
#if GD_SDPA_DKV_LANES != 8
#error "varlen DH64 f16 backward kernels assume 8 channel lanes"
#endif
#if GD_SDPA_DKV_WIDE_LANES != 8
#error "varlen DH64 f16 backward dK/dV kernel assumes 8 channel lanes"
#endif

#define GD_SDPA_VARLEN_DH64 64
#define GD_SDPA_VARLEN_DH64_CHANS (GD_SDPA_VARLEN_DH64 / GD_SDPA_DKV_LANES)
#define GD_SDPA_VARLEN_DH64_WIDE_CHANS (GD_SDPA_VARLEN_DH64 / GD_SDPA_DKV_WIDE_LANES)

static inline float gd_sdpa_varlen_sum_lanes8(float x)
{
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 4);
    return x;
}

/* VLM training hot path: packed fp16 prefix-causal + suffix-window backward,
 * Dh=64. One 8-lane subgroup owns one query/key row, so channel reductions are
 * SIMD shuffles instead of serial 64-wide loops. The stats+dq pass computes dq
 * in a single key sweep via:
 *   dq = scale * (sum(e * (go.v) * k) / l - D * sum(e * k) / l)
 * where D=sum(p * go.v). Stats are still written for the dK/dV pass. */
kernel void gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16(
    device const half *go                       [[buffer(0)]],
    device const half *q                        [[buffer(1)]],
    device const half *k                        [[buffer(2)]],
    device const half *v                        [[buffer(3)]],
    device const int *cu                        [[buffer(4)]],
    device half *dq                             [[buffer(5)]],
    constant gd_metal_sdpa_varlen_params &p     [[buffer(6)]],
    device float *stats                         [[buffer(7)]],
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
    float goreg[GD_SDPA_VARLEN_DH64_CHANS];
    float acc[GD_SDPA_VARLEN_DH64_CHANS];
    float ksum[GD_SDPA_VARLEN_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_LANES;
        qreg[x] = active ? (float)q[qbase + c] : 0.0f;
        goreg[x] = active ? (float)go[qbase + c] : 0.0f;
        acc[x] = 0.0f;
        ksum[x] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

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
                    float dp = 0.0f;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        ss += qreg[x] * ksh[jj * GD_SDPA_VARLEN_DH64 + c];
                        dp += goreg[x] * vsh[jj * GD_SDPA_VARLEN_DH64 + c];
                    }
                    ss = gd_sdpa_varlen_sum_lanes8(ss) * p.scale;
                    dp = gd_sdpa_varlen_sum_lanes8(dp);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        float kc = ksh[jj * GD_SDPA_VARLEN_DH64 + c];
                        acc[x] = acc[x] * corr + e * dp * kc;
                        ksum[x] = ksum[x] * corr + e * kc;
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        float D = raw * inv_l;
        int sbase = (qg * p.Hq + hq) * 3;
        if (lane == 0) {
            stats[sbase + 0] = m;
            stats[sbase + 1] = l;
            stats[sbase + 2] = D;
        }
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_LANES;
            dq[qbase + c] = (half)(p.scale * (acc[x] - D * ksum[x]) * inv_l);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16(
    device const half *go                       [[buffer(0)]],
    device const half *q                        [[buffer(1)]],
    device const half *k                        [[buffer(2)]],
    device const half *v                        [[buffer(3)]],
    device const int *cu                        [[buffer(4)]],
    device half *dk                             [[buffer(5)]],
    device half *dv                             [[buffer(6)]],
    constant gd_metal_sdpa_varlen_params &p     [[buffer(7)]],
    device const float *stats                   [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.max_seqlen + GD_SDPA_DKV_WIDE_KEYS - 1) / GD_SDPA_DKV_WIDE_KEYS;
    int kblk = (int)tgid % n_kb;
    int r = (int)tgid / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_WIDE_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_WIDE_LANES;
    if (b >= p.B) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int j = kblk * GD_SDPA_DKV_WIDE_KEYS + local_key;
    bool active = j < T;
    int kg = start + j;
    int kbase = active ? ((kg * p.Hkv + hkv) * GD_SDPA_VARLEN_DH64) : 0;

    float kreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
        kreg[x] = active ? (float)k[kbase + c] : 0.0f;
        vreg[x] = active ? (float)v[kbase + c] : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
    }

    int k0 = kblk * GD_SDPA_DKV_WIDE_KEYS;
    int qb_start = gd_sdpa_varlen_qb_start(k0, GD_SDPA_BK, p.causal, p.prefix_len);
    int qb_end = gd_sdpa_varlen_qb_end(k0, GD_SDPA_DKV_WIDE_KEYS, T,
                                       p.window, p.prefix_len);
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = qb_start; qb < qb_end; qb += GD_SDPA_BK) {
            int tile = qb_end - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * GD_SDPA_VARLEN_DH64;
                 idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / GD_SDPA_VARLEN_DH64;
                int c = idx % GD_SDPA_VARLEN_DH64;
                int qg = start + qb + ii;
                int qbase = (qg * p.Hq + hq) * GD_SDPA_VARLEN_DH64;
                qsh[ii * GD_SDPA_VARLEN_DH64 + c] = (float)q[qbase + c];
                gsh[ii * GD_SDPA_VARLEN_DH64 + c] = (float)go[qbase + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_DKV_WIDE_THREADS) {
                int qg = start + qb + ii;
                int sb = (qg * p.Hq + hq) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int ii = 0; ii < tile; ++ii) {
                int i = qb + ii;
                float ss = 0.0f;
                float dp = 0.0f;
                float pj = 0.0f;
                float ds = 0.0f;
                for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                    int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                    ss += qsh[ii * GD_SDPA_VARLEN_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_VARLEN_DH64 + c] * vreg[x];
                }
                ss = gd_sdpa_varlen_sum_lanes8(ss);
                dp = gd_sdpa_varlen_sum_lanes8(dp);
                if (active && gd_sdpa_varlen_allowed(i, j, p.causal, p.window,
                                                      p.prefix_len)) {
                    float l = lsh[ii];
                    if (l > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / l;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                        dvacc[x] += pj * gsh[ii * GD_SDPA_VARLEN_DH64 + c];
                        dkacc[x] += p.scale * ds * qsh[ii * GD_SDPA_VARLEN_DH64 + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
            dk[kbase + c] = (half)dkacc[x];
            dv[kbase + c] = (half)dvacc[x];
        }
    }
}

static inline int gd_sdpa_varlen_split_len(int T, int n)
{
    int len = (T + n - 1) / n;
    len = ((len + GD_SDPA_BK - 1) / GD_SDPA_BK) * GD_SDPA_BK;
    return (len < GD_SDPA_BK) ? GD_SDPA_BK : len;
}

kernel void gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16(
    device const half *go                       [[buffer(0)]],
    device const half *q                        [[buffer(1)]],
    device const half *k                        [[buffer(2)]],
    device const half *v                        [[buffer(3)]],
    device const int *cu                        [[buffer(4)]],
    device float *part                          [[buffer(5)]],
    constant gd_metal_sdpa_varlen_params &p     [[buffer(6)]],
    device const float *stats                   [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.max_seqlen + GD_SDPA_DKV_WIDE_KEYS - 1) / GD_SDPA_DKV_WIDE_KEYS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_WIDE_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_WIDE_LANES;
    if (b >= p.B) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = p.Hq / p.Hkv;
    int j = kblk * GD_SDPA_DKV_WIDE_KEYS + local_key;
    bool active = j < T;
    int kg = start + j;
    int kbase = active ? ((kg * p.Hkv + hkv) * GD_SDPA_VARLEN_DH64) : 0;

    float kreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
        kreg[x] = active ? (float)k[kbase + c] : 0.0f;
        vreg[x] = active ? (float)v[kbase + c] : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
    }

    int k0 = kblk * GD_SDPA_DKV_WIDE_KEYS;
    int qb_start = gd_sdpa_varlen_qb_start(k0, GD_SDPA_BK, p.causal, p.prefix_len);
    int qb_end = gd_sdpa_varlen_qb_end(k0, GD_SDPA_DKV_WIDE_KEYS, T,
                                       p.window, p.prefix_len);
    int slen = gd_sdpa_varlen_split_len(T, p.n_splits);
    int q_lo = s * slen;
    int q_hi = q_lo + slen;
    if (q_lo < qb_start) {
        q_lo = qb_start;
    }
    if (q_hi > qb_end) {
        q_hi = qb_end;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = q_lo; qb < q_hi; qb += GD_SDPA_BK) {
            int tile = q_hi - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * GD_SDPA_VARLEN_DH64;
                 idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / GD_SDPA_VARLEN_DH64;
                int c = idx % GD_SDPA_VARLEN_DH64;
                int qg = start + qb + ii;
                int qbase = (qg * p.Hq + hq) * GD_SDPA_VARLEN_DH64;
                qsh[ii * GD_SDPA_VARLEN_DH64 + c] = (float)q[qbase + c];
                gsh[ii * GD_SDPA_VARLEN_DH64 + c] = (float)go[qbase + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_DKV_WIDE_THREADS) {
                int qg = start + qb + ii;
                int sb = (qg * p.Hq + hq) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int ii = 0; ii < tile; ++ii) {
                int i = qb + ii;
                float ss = 0.0f;
                float dp = 0.0f;
                float pj = 0.0f;
                float ds = 0.0f;
                for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                    int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                    ss += qsh[ii * GD_SDPA_VARLEN_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_VARLEN_DH64 + c] * vreg[x];
                }
                ss = gd_sdpa_varlen_sum_lanes8(ss);
                dp = gd_sdpa_varlen_sum_lanes8(dp);
                if (active && gd_sdpa_varlen_allowed(i, j, p.causal, p.window,
                                                      p.prefix_len)) {
                    float l = lsh[ii];
                    if (l > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / l;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                        dvacc[x] += pj * gsh[ii * GD_SDPA_VARLEN_DH64 + c];
                        dkacc[x] += p.scale * ds * qsh[ii * GD_SDPA_VARLEN_DH64 + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int pbase = ((kg * p.Hkv + hkv) * p.n_splits + s) * 2 * GD_SDPA_VARLEN_DH64;
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
            part[pbase + c] = dkacc[x];
            part[pbase + GD_SDPA_VARLEN_DH64 + c] = dvacc[x];
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv_reduce_f16(
    device const float *part                    [[buffer(0)]],
    device half *dk                             [[buffer(1)]],
    device half *dv                             [[buffer(2)]],
    constant gd_metal_sdpa_varlen_params &p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int total = p.total_tokens * p.Hkv * GD_SDPA_VARLEN_DH64;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % GD_SDPA_VARLEN_DH64;
    int row = (int)gid / GD_SDPA_VARLEN_DH64;
    int base = row * GD_SDPA_VARLEN_DH64 + c;
    int pbase = row * p.n_splits * 2 * GD_SDPA_VARLEN_DH64;
    float adk = 0.0f;
    float adv = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        int off = pbase + s * 2 * GD_SDPA_VARLEN_DH64;
        adk += part[off + c];
        adv += part[off + GD_SDPA_VARLEN_DH64 + c];
    }
    dk[base] = (half)adk;
    dv[base] = (half)adv;
}
