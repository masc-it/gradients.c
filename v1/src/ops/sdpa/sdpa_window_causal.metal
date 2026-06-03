#include "metal_common.metal"

static inline int gd_sdpa_window_kb_start(int q0, int Tq, int Tk, int window)
{
    int first = q0 + (Tk - Tq) - window + 1;
    if (first <= 0) {
        return 0;
    }
    if (first > Tk) {
        return Tk;
    }
    return (first / GD_SDPA_BK) * GD_SDPA_BK;
}
static inline int gd_sdpa_window_kb_end(int q0, int bq, int Tq, int Tk)
{
    int qmax = q0 + bq - 1;
    if (qmax > Tq - 1) {
        qmax = Tq - 1;
    }
    int lim = qmax + (Tk - Tq) + 1;
    if (lim < 0) {
        lim = 0;
    }
    if (lim > Tk) {
        lim = Tk;
    }
    return lim;
}
static inline int gd_sdpa_window_qb_start(int k0, int qblk, int Tk, int Tq)
{
    int qmin = k0 - (Tk - Tq);
    if (qmin <= 0) {
        return 0;
    }
    if (qmin > Tq) {
        return Tq;
    }
    return (qmin / qblk) * qblk;
}
static inline int gd_sdpa_window_qb_end(int k0, int key_count, int Tq, int Tk, int window)
{
    int kend = k0 + key_count;
    if (kend > Tk) {
        kend = Tk;
    }
    int lim = kend + window - 1 - (Tk - Tq);
    if (lim < 0) {
        return 0;
    }
    return lim > Tq ? Tq : lim;
}
static inline bool gd_sdpa_window_allowed(int i, int j, int Tq, int Tk, int window)
{
    int qpos = i + (Tk - Tq);
    return j <= qpos && (qpos - j) < window;
}
#if GD_SDPA_DKV_LANES != 8
#error "causal window SDPA kernels assume 8 channel lanes"
#endif
static inline float gd_sdpa_sum_lanes8(float x)
{
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 4);
    return x;
}
static inline int gd_sdpa_split_len(int T, int n)
{
    int len = (T + n - 1) / n;
    len = ((len + GD_SDPA_BK - 1) / GD_SDPA_BK) * GD_SDPA_BK;
    return (len < GD_SDPA_BK) ? GD_SDPA_BK : len;
}

kernel void gd_sdpa_splitk_causal_window_lane8(device const float *q              [[buffer(0)]],
                                                device const float *k              [[buffer(1)]],
                                                device const float *v              [[buffer(2)]],
                                                device const float *bias           [[buffer(3)]],
                                                device float *partials             [[buffer(4)]],
                                                constant gd_metal_sdpa_params &p    [[buffer(5)]],
                                                uint tgid [[threadgroup_position_in_grid]],
                                                uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_CAUSAL_QROWS - 1) / GD_SDPA_CAUSAL_QROWS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int local_q = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_q * GD_SDPA_DKV_LANES;
    int i = qb * GD_SDPA_CAUSAL_QROWS + local_q;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DKV_CMAX];
    float acc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        qreg[nchan] = active ? q[qbase + c] : 0.0f;
        acc[nchan] = 0.0f;
        nchan++;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * GD_SDPA_CAUSAL_QROWS;
    int kb_start = gd_sdpa_window_kb_start(q0, p.Tq, p.Tk, p.window);
    int kb_end = gd_sdpa_window_kb_end(q0, GD_SDPA_CAUSAL_QROWS, p.Tq, p.Tk);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    if (k_lo < kb_start) {
        k_lo = kb_start;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float ss = 0.0f;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        ss += qreg[x] * ksh[jj * p.Dh + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        acc[x] = acc[x] * corr + e * vsh[jj * p.Dh + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            partials[base + c] = acc[x];
        }
        if (lane == 0) {
            partials[base + p.Dh] = m;
            partials[base + p.Dh + 1] = l;
        }
    }
}

kernel void gd_sdpa_splitk_causal_window_lane8_f16(device const half *q              [[buffer(0)]],
                                                    device const half *k              [[buffer(1)]],
                                                    device const half *v              [[buffer(2)]],
                                                    device const half *bias           [[buffer(3)]],
                                                    device float *partials             [[buffer(4)]],
                                                    constant gd_metal_sdpa_params &p    [[buffer(5)]],
                                                    uint tgid [[threadgroup_position_in_grid]],
                                                    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_CAUSAL_QROWS - 1) / GD_SDPA_CAUSAL_QROWS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int local_q = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_q * GD_SDPA_DKV_LANES;
    int i = qb * GD_SDPA_CAUSAL_QROWS + local_q;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DKV_CMAX];
    float acc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        qreg[nchan] = active ? (float)q[qbase + c] : 0.0f;
        acc[nchan] = 0.0f;
        nchan++;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * GD_SDPA_CAUSAL_QROWS;
    int kb_start = gd_sdpa_window_kb_start(q0, p.Tq, p.Tk, p.window);
    int kb_end = gd_sdpa_window_kb_end(q0, GD_SDPA_CAUSAL_QROWS, p.Tq, p.Tk);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    if (k_lo < kb_start) {
        k_lo = kb_start;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = (float)k[kbase + c];
            vsh[jj * p.Dh + c] = (float)v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float ss = 0.0f;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        ss += qreg[x] * ksh[jj * p.Dh + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        acc[x] = acc[x] * corr + e * vsh[jj * p.Dh + c];
                    }
                    m = mnew;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            partials[base + c] = acc[x];
        }
        if (lane == 0) {
            partials[base + p.Dh] = m;
            partials[base + p.Dh + 1] = l;
        }
    }
    (void)bias;
}

kernel void gd_sdpa_bwd_stats_dq_split_causal_window_lane8(device const float *go           [[buffer(0)]],
                                                            device const float *q            [[buffer(1)]],
                                                            device const float *k            [[buffer(2)]],
                                                            device const float *v            [[buffer(3)]],
                                                            device const float *bias         [[buffer(4)]],
                                                            device float *stats_part         [[buffer(5)]],
                                                            device float *dq_part            [[buffer(6)]],
                                                            constant gd_metal_sdpa_params &p [[buffer(7)]],
                                                            uint tgid [[threadgroup_position_in_grid]],
                                                            uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_CAUSAL_QROWS - 1) / GD_SDPA_CAUSAL_QROWS;
    int sp = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int local_q = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_q * GD_SDPA_DKV_LANES;
    int i = qb * GD_SDPA_CAUSAL_QROWS + local_q;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DKV_CMAX];
    float goreg[GD_SDPA_DKV_CMAX];
    float acc[GD_SDPA_DKV_CMAX];
    float ksum[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        qreg[nchan] = active ? q[qbase + c] : 0.0f;
        goreg[nchan] = active ? go[qbase + c] : 0.0f;
        acc[nchan] = 0.0f;
        ksum[nchan] = 0.0f;
        nchan++;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int q0 = qb * GD_SDPA_CAUSAL_QROWS;
    int kb_start = gd_sdpa_window_kb_start(q0, p.Tq, p.Tk, p.window);
    int kb_end = gd_sdpa_window_kb_end(q0, GD_SDPA_CAUSAL_QROWS, p.Tq, p.Tk);
    int slen = gd_sdpa_split_len(p.Tk, p.n_splits);
    int k_lo = sp * slen;
    int k_hi = k_lo + slen;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    if (k_lo < kb_start) {
        k_lo = kb_start;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = k[kbase + c];
            vsh[jj * p.Dh + c] = v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        ss += qreg[x] * ksh[jj * p.Dh + c];
                        dp += goreg[x] * vsh[jj * p.Dh + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    dp = gd_sdpa_sum_lanes8(dp);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        float kc = ksh[jj * p.Dh + c];
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
        int sbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 3;
        int vbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 2 * p.Dh;
        if (lane == 0) {
            stats_part[sbase + 0] = m;
            stats_part[sbase + 1] = l;
            stats_part[sbase + 2] = raw;
        }
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            dq_part[vbase + c] = acc[x];
            dq_part[vbase + p.Dh + c] = ksum[x];
        }
    }
}

kernel void gd_sdpa_bwd_dkv_split_causal_window_k16(device const float *go           [[buffer(0)]],
                                                     device const float *q            [[buffer(1)]],
                                                     device const float *k            [[buffer(2)]],
                                                     device const float *v            [[buffer(3)]],
                                                     device const float *bias         [[buffer(4)]],
                                                     device float *part               [[buffer(5)]],
                                                     constant gd_metal_sdpa_params &p  [[buffer(6)]],
                                                     device const float *stats        [[buffer(7)]],
                                                     uint tgid [[threadgroup_position_in_grid]],
                                                     uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.Tk + GD_SDPA_DKV_WIDE_KEYS - 1) / GD_SDPA_DKV_WIDE_KEYS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_WIDE_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_WIDE_LANES;
    int j = kblk * GD_SDPA_DKV_WIDE_KEYS + local_key;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DKV_CMAX];
    float vreg[GD_SDPA_DKV_CMAX];
    float dkacc[GD_SDPA_DKV_CMAX];
    float dvacc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_WIDE_LANES) {
        kreg[nchan] = active ? k[kbase + c] : 0.0f;
        vreg[nchan] = active ? v[kbase + c] : 0.0f;
        dkacc[nchan] = 0.0f;
        dvacc[nchan] = 0.0f;
        nchan++;
    }

    int k0 = kblk * GD_SDPA_DKV_WIDE_KEYS;
    int qb_start = gd_sdpa_window_qb_start(k0, GD_SDPA_BK, p.Tk, p.Tq);
    int q_cap = gd_sdpa_window_qb_end(k0, GD_SDPA_DKV_WIDE_KEYS, p.Tq, p.Tk, p.window);
    int slen = gd_sdpa_split_len(p.Tq, p.n_splits);
    int q_lo = s * slen;
    int q_hi = q_lo + slen;
    if (q_lo < qb_start) {
        q_lo = qb_start;
    }
    if (q_hi > q_cap) {
        q_hi = q_cap;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = q_lo; qb < q_hi; qb += GD_SDPA_BK) {
            int tile = q_hi - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / p.Dh;
                int c = idx % p.Dh;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = q[qb2 + c];
                gsh[ii * p.Dh + c] = go[qb2 + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_DKV_WIDE_THREADS) {
                int sb = ((b * p.Hq + hq) * p.Tq + (qb + ii)) * 3;
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

                for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
                    ss += qsh[ii * p.Dh + c] * kreg[x];
                    dp += gsh[ii * p.Dh + c] * vreg[x];
                }
                ss = gd_sdpa_sum_lanes8(ss);
                dp = gd_sdpa_sum_lanes8(dp);
                if (active && gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float l = lsh[ii];
                    if (l > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / l;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
                        dvacc[x] += pj * gsh[ii * p.Dh + c];
                        dkacc[x] += p.scale * ds * qsh[ii * p.Dh + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int base = (((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits + s) * 2 * p.Dh;
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
            part[base + c] = dkacc[x];
            part[base + p.Dh + c] = dvacc[x];
        }
    }
}

kernel void gd_sdpa_bwd_stats_dq_split_causal_window_lane8_f16(device const half *go            [[buffer(0)]],
                                                               device const half *q             [[buffer(1)]],
                                                               device const half *k             [[buffer(2)]],
                                                               device const half *v             [[buffer(3)]],
                                                               device const half *bias          [[buffer(4)]],
                                                               device float *stats_part         [[buffer(5)]],
                                                               device float *dq_part            [[buffer(6)]],
                                                               constant gd_metal_sdpa_params &p [[buffer(7)]],
                                                               uint tgid [[threadgroup_position_in_grid]],
                                                               uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_CAUSAL_QROWS - 1) / GD_SDPA_CAUSAL_QROWS;
    int sp = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int local_q = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_q * GD_SDPA_DKV_LANES;
    int i = qb * GD_SDPA_CAUSAL_QROWS + local_q;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DKV_CMAX];
    float goreg[GD_SDPA_DKV_CMAX];
    float acc[GD_SDPA_DKV_CMAX];
    float ksum[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        qreg[nchan] = active ? (float)q[qbase + c] : 0.0f;
        goreg[nchan] = active ? (float)go[qbase + c] : 0.0f;
        acc[nchan] = 0.0f;
        ksum[nchan] = 0.0f;
        nchan++;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int q0 = qb * GD_SDPA_CAUSAL_QROWS;
    int kb_start = gd_sdpa_window_kb_start(q0, p.Tq, p.Tk, p.window);
    int kb_end = gd_sdpa_window_kb_end(q0, GD_SDPA_CAUSAL_QROWS, p.Tq, p.Tk);
    int slen = gd_sdpa_split_len(p.Tk, p.n_splits);
    int k_lo = sp * slen;
    int k_hi = k_lo + slen;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    if (k_lo < kb_start) {
        k_lo = kb_start;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / p.Dh;
            int c = idx % p.Dh;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = (float)k[kbase + c];
            vsh[jj * p.Dh + c] = (float)v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        ss += qreg[x] * ksh[jj * p.Dh + c];
                        dp += goreg[x] * vsh[jj * p.Dh + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    dp = gd_sdpa_sum_lanes8(dp);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                        float kc = ksh[jj * p.Dh + c];
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
        int sbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 3;
        int vbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) * 2 * p.Dh;
        if (lane == 0) {
            stats_part[sbase + 0] = m;
            stats_part[sbase + 1] = l;
            stats_part[sbase + 2] = raw;
        }
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            dq_part[vbase + c] = acc[x];
            dq_part[vbase + p.Dh + c] = ksum[x];
        }
    }
    (void)bias;
}

kernel void gd_sdpa_bwd_dkv_split_causal_window_k16_f16(device const half *go            [[buffer(0)]],
                                                        device const half *q             [[buffer(1)]],
                                                        device const half *k             [[buffer(2)]],
                                                        device const half *v             [[buffer(3)]],
                                                        device const half *bias          [[buffer(4)]],
                                                        device float *part               [[buffer(5)]],
                                                        constant gd_metal_sdpa_params &p  [[buffer(6)]],
                                                        device const float *stats        [[buffer(7)]],
                                                        uint tgid [[threadgroup_position_in_grid]],
                                                        uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.Tk + GD_SDPA_DKV_WIDE_KEYS - 1) / GD_SDPA_DKV_WIDE_KEYS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_WIDE_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_WIDE_LANES;
    int j = kblk * GD_SDPA_DKV_WIDE_KEYS + local_key;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DKV_CMAX];
    float vreg[GD_SDPA_DKV_CMAX];
    float dkacc[GD_SDPA_DKV_CMAX];
    float dvacc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_WIDE_LANES) {
        kreg[nchan] = active ? (float)k[kbase + c] : 0.0f;
        vreg[nchan] = active ? (float)v[kbase + c] : 0.0f;
        dkacc[nchan] = 0.0f;
        dvacc[nchan] = 0.0f;
        nchan++;
    }

    int k0 = kblk * GD_SDPA_DKV_WIDE_KEYS;
    int qb_start = gd_sdpa_window_qb_start(k0, GD_SDPA_BK, p.Tk, p.Tq);
    int q_cap = gd_sdpa_window_qb_end(k0, GD_SDPA_DKV_WIDE_KEYS, p.Tq, p.Tk, p.window);
    int slen = gd_sdpa_split_len(p.Tq, p.n_splits);
    int q_lo = s * slen;
    int q_hi = q_lo + slen;
    if (q_lo < qb_start) {
        q_lo = qb_start;
    }
    if (q_hi > q_cap) {
        q_hi = q_cap;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = q_lo; qb < q_hi; qb += GD_SDPA_BK) {
            int tile = q_hi - qb;
            if (tile > GD_SDPA_BK) {
                tile = GD_SDPA_BK;
            }
            for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / p.Dh;
                int c = idx % p.Dh;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = (float)q[qb2 + c];
                gsh[ii * p.Dh + c] = (float)go[qb2 + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_DKV_WIDE_THREADS) {
                int sb = ((b * p.Hq + hq) * p.Tq + (qb + ii)) * 3;
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

                for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
                    ss += qsh[ii * p.Dh + c] * kreg[x];
                    dp += gsh[ii * p.Dh + c] * vreg[x];
                }
                ss = gd_sdpa_sum_lanes8(ss);
                dp = gd_sdpa_sum_lanes8(dp);
                if (active && gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float l = lsh[ii];
                    if (l > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / l;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
                        dvacc[x] += pj * gsh[ii * p.Dh + c];
                        dkacc[x] += p.scale * ds * qsh[ii * p.Dh + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int base = (((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits + s) * 2 * p.Dh;
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_WIDE_LANES) {
            part[base + c] = dkacc[x];
            part[base + p.Dh + c] = dvacc[x];
        }
    }
    (void)bias;
}
