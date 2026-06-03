#include "metal_common.metal"

static inline bool gd_sdpa_allowed(int i, int j, int Tq, int Tk,
                                   int causal, int window, int prefix_len)
{
    int qpos = i + (Tk - Tq);

    if (causal) {
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
static inline int gd_sdpa_kb_end(int q0, int bq, int Tq, int Tk,
                                  int causal, int prefix_len)
{
    if (!causal) {
        return Tk;
    }
    int qmax = q0 + bq - 1;
    if (qmax > Tq - 1) {
        qmax = Tq - 1;
    }
    int qmaxpos = qmax + (Tk - Tq);
    int lim = qmaxpos + 1; /* keys j < lim may be attended */
    if (prefix_len > 0 && qmaxpos < prefix_len) {
        lim = prefix_len;
    }
    if (lim < 0) {
        lim = 0;
    }
    if (lim > Tk) {
        lim = Tk;
    }
    return lim;
}
static inline int gd_sdpa_kb_start(int q0, int Tq, int Tk,
                                    int window, int prefix_len)
{
    if (window <= 0) {
        return 0;
    }
    int qpos = q0 + (Tk - Tq);
    if (prefix_len > 0) {
        if (qpos < prefix_len) {
            qpos = prefix_len;
        }
        int first = qpos - window + 1;
        if (first < prefix_len) {
            first = prefix_len;
        }
        if (first > Tk) {
            first = Tk;
        }
        return first;
    }
    int first = qpos - window + 1;
    if (first <= 0) {
        return 0;
    }
    if (first > Tk) {
        return Tk;
    }
    return first;
}
static inline int gd_sdpa_kb_prefix_end(int Tk, int window, int prefix_len)
{
    if (window <= 0 || prefix_len <= 0) {
        return 0;
    }
    return prefix_len < Tk ? prefix_len : Tk;
}
static inline int gd_sdpa_qb_start(int k0, int qblk, int Tk, int Tq,
                                    int causal, int prefix_len)
{
    if (!causal) {
        return 0;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return 0;
    }
    int qmin = k0 - (Tk - Tq);
    if (qmin <= 0) {
        return 0;
    }
    if (qmin > Tq) {
        return Tq;
    }
    return (qmin / qblk) * qblk;
}
static inline int gd_sdpa_qb_end(int k0, int key_count, int Tk, int Tq,
                                  int window, int prefix_len)
{
    if (window <= 0) {
        return Tq;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return Tq;
    }
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
static inline float gd_sdpa_bias_at(device const float *bias,
                                    constant gd_metal_sdpa_params &p,
                                    int b, int hq, int i, int j)
{
    if (!p.has_bias) {
        return 0.0f;
    }
    int bb = (p.Bb == 1) ? 0 : b;
    int hb = (p.Hb == 1) ? 0 : hq;
    int ib = (p.Tqb == 1) ? 0 : i;
    int jb = (p.Tkb == 1) ? 0 : j;
    return bias[((bb * p.Hb + hb) * p.Tqb + ib) * p.Tkb + jb];
}
static inline float gd_sdpa_bias_at_typed(device const uchar *bias,
                                          constant gd_metal_sdpa_params &p,
                                          int b, int hq, int i, int j)
{
    if (!p.has_bias) {
        return 0.0f;
    }
    int bb = (p.Bb == 1) ? 0 : b;
    int hb = (p.Hb == 1) ? 0 : hq;
    int ib = (p.Tqb == 1) ? 0 : i;
    int jb = (p.Tkb == 1) ? 0 : j;
    return gd_load_float(bias, p.dtype, (uint)(((bb * p.Hb + hb) * p.Tqb + ib) * p.Tkb + jb));
}
static inline int gd_sdpa_split_len(int T, int n)
{
    int len = (T + n - 1) / n;
    len = ((len + GD_SDPA_BK - 1) / GD_SDPA_BK) * GD_SDPA_BK;
    return (len < GD_SDPA_BK) ? GD_SDPA_BK : len;
}
kernel void gd_sdpa_bwd_stats(device const uchar *go            [[buffer(0)]],
                              device const uchar *q             [[buffer(1)]],
                              device const uchar *k             [[buffer(2)]],
                              device const uchar *v             [[buffer(3)]],
                              device const uchar *bias          [[buffer(4)]],
                              device float *stats               [[buffer(5)]],
                              constant gd_metal_sdpa_params &p   [[buffer(6)]],
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
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

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
    int kb_start = gd_sdpa_kb_start(q0, p.Tq, p.Tk, p.window, p.prefix_len);
    int kb_prefix_end = gd_sdpa_kb_prefix_end(p.Tk, p.window, p.prefix_len);
    int kb_end = gd_sdpa_kb_end(q0, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = gd_load_float(k, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at_typed(bias, p, b, hq, i, j);
                    float mnew = (s > m) ? s : m;
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
        int sbase = ((b * p.Hq + hq) * p.Tq + i) * 3;
        stats[sbase + 0] = m;
        stats[sbase + 1] = l;
        stats[sbase + 2] = (l > 0.0f) ? (raw / l) : 0.0f;
    }
}
kernel void gd_sdpa_bwd_dq(device const uchar *go            [[buffer(0)]],
                           device const uchar *q             [[buffer(1)]],
                           device const uchar *k             [[buffer(2)]],
                           device const uchar *v             [[buffer(3)]],
                           device const uchar *bias          [[buffer(4)]],
                           device uchar *dq                  [[buffer(5)]],
                           constant gd_metal_sdpa_params &p   [[buffer(6)]],
                           device const float *stats         [[buffer(7)]],
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
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;
    int sbase = active ? (((b * p.Hq + hq) * p.Tq + i) * 3) : 0;
    float m = active ? stats[sbase + 0] : 0.0f;
    float l = active ? stats[sbase + 1] : 0.0f;
    float D = active ? stats[sbase + 2] : 0.0f;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    float dqacc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? gd_load_float(q, p.dtype, (uint)(qbase + c)) : 0.0f;
        goreg[c] = active ? gd_load_float(go, p.dtype, (uint)(qbase + c)) : 0.0f;
        dqacc[c] = 0.0f;
    }
    bool run = active && l > 0.0f;

    int q0 = qb * GD_SDPA_BQ;
    int kb_start = gd_sdpa_kb_start(q0, p.Tq, p.Tk, p.window, p.prefix_len);
    int kb_prefix_end = gd_sdpa_kb_prefix_end(p.Tk, p.window, p.prefix_len);
    int kb_end = gd_sdpa_kb_end(q0, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
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
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * p.Dh;
            ksh[jj * p.Dh + c] = gd_load_float(k, p.dtype, (uint)(kbase + c));
            vsh[jj * p.Dh + c] = gd_load_float(v, p.dtype, (uint)(kbase + c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (run) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at_typed(bias, p, b, hq, i, j);
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
kernel void gd_sdpa_bwd_dkv(device const uchar *go            [[buffer(0)]],
                            device const uchar *q             [[buffer(1)]],
                            device const uchar *k             [[buffer(2)]],
                            device const uchar *v             [[buffer(3)]],
                            device const uchar *bias          [[buffer(4)]],
                            device uchar *dk                  [[buffer(5)]],
                            device uchar *dv                  [[buffer(6)]],
                            constant gd_metal_sdpa_params &p   [[buffer(7)]],
                            device const float *stats         [[buffer(8)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];

    int n_kb = (p.Tk + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int kblk = (int)tgid % n_kb;
    int r = (int)tgid / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int j = kblk * GD_SDPA_BQ + (int)tid;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

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

    int k0 = kblk * GD_SDPA_BQ;
    int qb_start = gd_sdpa_qb_start(k0, GD_SDPA_BK, p.Tk, p.Tq, p.causal, p.prefix_len);
    int qb_end = gd_sdpa_qb_end(k0, GD_SDPA_BQ, p.Tk, p.Tq, p.window, p.prefix_len);
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
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = gd_load_float(q, p.dtype, (uint)(qb2 + c));
                gsh[ii * p.Dh + c] = gd_load_float(go, p.dtype, (uint)(qb2 + c));
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_BQ) {
                int sb = ((b * p.Hq + hq) * p.Tq + (qb + ii)) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    if (!gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                        continue;
                    }
                    float l = lsh[ii];
                    if (l <= 0.0f) {
                        continue;
                    }
                    float mm = msh[ii];
                    float D = dsh[ii];
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qsh[ii * p.Dh + c] * kreg[c];
                        dp += gsh[ii * p.Dh + c] * vreg[c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at_typed(bias, p, b, hq, i, j);
                    float pj = exp(s - mm) / l;
                    float ds = pj * (dp - D);
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
kernel void gd_sdpa_bwd_stats_dq_split(device const float *go          [[buffer(0)]],
                                       device const float *q           [[buffer(1)]],
                                       device const float *k           [[buffer(2)]],
                                       device const float *v           [[buffer(3)]],
                                       device const float *bias        [[buffer(4)]],
                                       device float *stats_part        [[buffer(5)]],
                                       device float *dq_part           [[buffer(6)]],
                                       constant gd_metal_sdpa_params &p [[buffer(7)]],
                                       uint tgid [[threadgroup_position_in_grid]],
                                       uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int sp = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int qb = t2 % n_qb;
    int r = t2 / n_qb;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float goreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    float ksum[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        goreg[c] = active ? go[qbase + c] : 0.0f;
        acc[c] = 0.0f;
        ksum[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int q0 = qb * GD_SDPA_BQ;
    int kb_start = gd_sdpa_kb_start(q0, p.Tq, p.Tk, p.window, p.prefix_len);
    int kb_prefix_end = gd_sdpa_kb_prefix_end(p.Tk, p.window, p.prefix_len);
    int kb_end = gd_sdpa_kb_end(q0, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
    int slen = gd_sdpa_split_len(p.Tk, p.n_splits);
    int k_lo = sp * slen;
    int k_hi = k_lo + slen;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
        if (kb < kb_start && (kb_prefix_end == 0 || kb >= kb_prefix_end)) {
            kb = kb_start - GD_SDPA_BK;
            continue;
        }
        int tile = k_hi - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
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
                if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        ss += qreg[c] * ksh[jj * p.Dh + c];
                        dp += goreg[c] * vsh[jj * p.Dh + c];
                    }
                    ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int c = 0; c < p.Dh; ++c) {
                        float kc = ksh[jj * p.Dh + c];
                        acc[c] = acc[c] * corr + e * dp * kc;
                        ksum[c] = ksum[c] * corr + e * kc;
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
        stats_part[sbase + 0] = m;
        stats_part[sbase + 1] = l;
        stats_part[sbase + 2] = raw;
        for (int c = 0; c < p.Dh; ++c) {
            dq_part[vbase + c] = acc[c];
            dq_part[vbase + p.Dh + c] = ksum[c];
        }
    }
}
kernel void gd_sdpa_bwd_stats_dq_combine(device const float *stats_part [[buffer(0)]],
                                         device const float *dq_part    [[buffer(1)]],
                                         device float *stats            [[buffer(2)]],
                                         device float *dq               [[buffer(3)]],
                                         constant gd_metal_sdpa_params &p [[buffer(4)]],
                                         uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int i = row % p.Tq;
    int r = row / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;

    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;
    float acc = 0.0f;
    float ksum = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        int sbase = (row * p.n_splits + sp) * 3;
        float ls = stats_part[sbase + 1];
        if (ls <= 0.0f) {
            continue;
        }
        float ms = stats_part[sbase + 0];
        float mnew = (ms > m) ? ms : m;
        float corr_o = exp(m - mnew);
        float corr_s = exp(ms - mnew);
        int vbase = (row * p.n_splits + sp) * 2 * p.Dh;
        l = l * corr_o + ls * corr_s;
        raw = raw * corr_o + stats_part[sbase + 2] * corr_s;
        acc = acc * corr_o + dq_part[vbase + c] * corr_s;
        ksum = ksum * corr_o + dq_part[vbase + p.Dh + c] * corr_s;
        m = mnew;
    }
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float D = raw * inv_l;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    dq[qbase + c] = p.scale * (acc * inv_l - D * ksum * inv_l);
    if (c == 0) {
        int sb = ((b * p.Hq + hq) * p.Tq + i) * 3;
        stats[sb + 0] = m;
        stats[sb + 1] = l;
        stats[sb + 2] = D;
    }
}

kernel void gd_sdpa_bwd_stats_dq_combine_f16(device const float *stats_part [[buffer(0)]],
                                             device const float *dq_part    [[buffer(1)]],
                                             device float *stats            [[buffer(2)]],
                                             device half *dq                [[buffer(3)]],
                                             constant gd_metal_sdpa_params &p [[buffer(4)]],
                                             uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int i = row % p.Tq;
    int r = row / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;

    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;
    float acc = 0.0f;
    float ksum = 0.0f;
    for (int sp = 0; sp < p.n_splits; ++sp) {
        int sbase = (row * p.n_splits + sp) * 3;
        float ls = stats_part[sbase + 1];
        if (ls <= 0.0f) {
            continue;
        }
        float ms = stats_part[sbase + 0];
        float mnew = (ms > m) ? ms : m;
        float corr_o = exp(m - mnew);
        float corr_s = exp(ms - mnew);
        int vbase = (row * p.n_splits + sp) * 2 * p.Dh;
        l = l * corr_o + ls * corr_s;
        raw = raw * corr_o + stats_part[sbase + 2] * corr_s;
        acc = acc * corr_o + dq_part[vbase + c] * corr_s;
        ksum = ksum * corr_o + dq_part[vbase + p.Dh + c] * corr_s;
        m = mnew;
    }
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    float D = raw * inv_l;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    dq[qbase + c] = (half)(p.scale * (acc * inv_l - D * ksum * inv_l));
    if (c == 0) {
        int sb = ((b * p.Hq + hq) * p.Tq + i) * 3;
        stats[sb + 0] = m;
        stats[sb + 1] = l;
        stats[sb + 2] = D;
    }
}
kernel void gd_sdpa_bwd_dkv_split(device const float *go            [[buffer(0)]],
                                  device const float *q             [[buffer(1)]],
                                  device const float *k             [[buffer(2)]],
                                  device const float *v             [[buffer(3)]],
                                  device const float *bias          [[buffer(4)]],
                                  device float *part                [[buffer(5)]],
                                  constant gd_metal_sdpa_params &p   [[buffer(6)]],
                                  device const float *stats         [[buffer(7)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float msh[GD_SDPA_BK];
    threadgroup float lsh[GD_SDPA_BK];
    threadgroup float dsh[GD_SDPA_BK];
    threadgroup float ss_part[GD_SDPA_BK * GD_SDPA_DKV_KEYS * GD_SDPA_DKV_LANES];
    threadgroup float dp_part[GD_SDPA_BK * GD_SDPA_DKV_KEYS * GD_SDPA_DKV_LANES];
    threadgroup float pjsh[GD_SDPA_BK * GD_SDPA_DKV_KEYS];
    threadgroup float dssh[GD_SDPA_BK * GD_SDPA_DKV_KEYS];

    int n_kb = (p.Tk + GD_SDPA_DKV_KEYS - 1) / GD_SDPA_DKV_KEYS;
    int s = (int)tgid % p.n_splits;
    int t2 = (int)tgid / p.n_splits;
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int group = p.Hq / p.Hkv;
    int local_key = (int)tid / GD_SDPA_DKV_LANES;
    int lane = (int)tid - local_key * GD_SDPA_DKV_LANES;
    int j = kblk * GD_SDPA_DKV_KEYS + local_key;
    bool active = j < p.Tk;
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * p.Dh) : 0;

    float kreg[GD_SDPA_DKV_CMAX];
    float vreg[GD_SDPA_DKV_CMAX];
    float dkacc[GD_SDPA_DKV_CMAX];
    float dvacc[GD_SDPA_DKV_CMAX];
    int nchan = 0;
    for (int c = lane; c < p.Dh; c += GD_SDPA_DKV_LANES) {
        kreg[nchan] = active ? k[kbase + c] : 0.0f;
        vreg[nchan] = active ? v[kbase + c] : 0.0f;
        dkacc[nchan] = 0.0f;
        dvacc[nchan] = 0.0f;
        nchan++;
    }

    int k0 = kblk * GD_SDPA_DKV_KEYS;
    int qb_start = gd_sdpa_qb_start(k0, GD_SDPA_BK, p.Tk, p.Tq, p.causal, p.prefix_len);
    int qb_end = gd_sdpa_qb_end(k0, GD_SDPA_DKV_KEYS, p.Tk, p.Tq, p.window, p.prefix_len);
    int slen = gd_sdpa_split_len(p.Tq, p.n_splits);
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
            for (int idx = (int)tid; idx < tile * p.Dh; idx += GD_SDPA_BQ) {
                int ii = idx / p.Dh;
                int c = idx % p.Dh;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * p.Dh;
                qsh[ii * p.Dh + c] = q[qb2 + c];
                gsh[ii * p.Dh + c] = go[qb2 + c];
            }
            for (int ii = (int)tid; ii < tile; ii += GD_SDPA_BQ) {
                int sb = ((b * p.Hq + hq) * p.Tq + (qb + ii)) * 3;
                msh[ii] = stats[sb + 0];
                lsh[ii] = stats[sb + 1];
                dsh[ii] = stats[sb + 2];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int ii = 0; ii < tile; ++ii) {
                float ss = 0.0f;
                float dp = 0.0f;
                for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
                    ss += qsh[ii * p.Dh + c] * kreg[x];
                    dp += gsh[ii * p.Dh + c] * vreg[x];
                }
                int rb = (ii * GD_SDPA_DKV_KEYS + local_key) * GD_SDPA_DKV_LANES + lane;
                ss_part[rb] = ss;
                dp_part[rb] = dp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    int ob = ii * GD_SDPA_DKV_KEYS + local_key;
                    float pj = 0.0f;
                    float ds = 0.0f;
                    if (active && gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
                        float l = lsh[ii];
                        if (l > 0.0f) {
                            int rb = ob * GD_SDPA_DKV_LANES;
                            float ss = 0.0f;
                            float dp = 0.0f;
                            for (int ln = 0; ln < GD_SDPA_DKV_LANES; ++ln) {
                                ss += ss_part[rb + ln];
                                dp += dp_part[rb + ln];
                            }
                            ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                            pj = exp(ss - msh[ii]) / l;
                            ds = pj * (dp - dsh[ii]);
                        }
                    }
                    pjsh[ob] = pj;
                    dssh[ob] = ds;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int ob = ii * GD_SDPA_DKV_KEYS + local_key;
                    float pj = pjsh[ob];
                    float ds = dssh[ob];
                    for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
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
        for (int x = 0, c = lane; x < nchan; ++x, c += GD_SDPA_DKV_LANES) {
            part[base + c] = dkacc[x];
            part[base + p.Dh + c] = dvacc[x];
        }
    }
}
kernel void gd_sdpa_bwd_dkv_reduce(device const float *part         [[buffer(0)]],
                                   device float *dk                 [[buffer(1)]],
                                   device float *dv                 [[buffer(2)]],
                                   constant gd_metal_sdpa_params &p  [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hkv * p.Tk * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int j = row % p.Tk;
    int r = row / p.Tk;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
    int pbase = ((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits * 2 * p.Dh;
    float adk = 0.0f;
    float adv = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        adk += part[pbase + s * 2 * p.Dh + c];
        adv += part[pbase + s * 2 * p.Dh + p.Dh + c];
    }
    dk[kbase + c] = adk;
    dv[kbase + c] = adv;
}

kernel void gd_sdpa_bwd_dkv_reduce_f16(device const float *part         [[buffer(0)]],
                                       device half *dk                  [[buffer(1)]],
                                       device half *dv                  [[buffer(2)]],
                                       constant gd_metal_sdpa_params &p  [[buffer(3)]],
                                       uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hkv * p.Tk * p.Dh;
    if ((int)gid >= total) {
        return;
    }
    int c = (int)gid % p.Dh;
    int row = (int)gid / p.Dh;
    int j = row % p.Tk;
    int r = row / p.Tk;
    int hkv = r % p.Hkv;
    int b = r / p.Hkv;
    int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
    int pbase = ((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits * 2 * p.Dh;
    float adk = 0.0f;
    float adv = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        adk += part[pbase + s * 2 * p.Dh + c];
        adv += part[pbase + s * 2 * p.Dh + p.Dh + c];
    }
    dk[kbase + c] = (half)adk;
    dv[kbase + c] = (half)adv;
}
