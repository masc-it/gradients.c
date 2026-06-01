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
static inline int gd_sdpa_causal_kb_end(int q0, int bq, int Tq, int Tk)
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
static inline bool gd_sdpa_causal_allowed(int i, int j, int Tq, int Tk)
{
    return j <= i + (Tk - Tq);
}
#if GD_SDPA_DKV_LANES != 8
#error "gd_sdpa_splitk_causal_lane8 assumes 8 channel lanes per query"
#endif
static inline float gd_sdpa_sum_lanes8(float x)
{
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 4);
    return x;
}
static inline float gd_sdpa_dot(device const float *a, device const float *b, int n)
{
    float s = 0.0f;
    for (int c = 0; c < n; ++c) {
        s += a[c] * b[c];
    }
    return s;
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
kernel void gd_sdpa_tiled(device const float *q              [[buffer(0)]],
                          device const float *k              [[buffer(1)]],
                          device const float *v              [[buffer(2)]],
                          device const float *bias           [[buffer(3)]],
                          device float *out                  [[buffer(4)]],
                          constant gd_metal_sdpa_params &p    [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb; /* b*Hq + hq */
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        /* Cooperatively stage this key tile (all threads participate). */
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
                    float s = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    s = p.scale * s + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (s > m) ? s : m;
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
        if (l > 0.0f) {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = acc[c] / l;
            }
        } else {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = 0.0f;
            }
        }
    }
}
kernel void gd_sdpa_tiled_causal(device const float *q              [[buffer(0)]],
                                  device const float *k              [[buffer(1)]],
                                  device const float *v              [[buffer(2)]],
                                  device const float *bias           [[buffer(3)]],
                                  device float *out                  [[buffer(4)]],
                                  constant gd_metal_sdpa_params &p    [[buffer(5)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DHT];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DHT];

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int qb = (int)tgid % n_qb;
    int r = (int)tgid / n_qb; /* b*Hq + hq */
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int i = qb * GD_SDPA_BQ + (int)tid;
    bool active = i < p.Tq;
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * p.Dh) : 0;

    float qreg[GD_SDPA_DHT];
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_causal_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk);
    for (int kb = 0; kb < kb_end; kb += GD_SDPA_BK) {
        int tile = kb_end - kb;
        if (tile > GD_SDPA_BK) {
            tile = GD_SDPA_BK;
        }
        /* Cooperatively stage this key tile (all threads participate). */
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
                if (gd_sdpa_causal_allowed(i, j, p.Tq, p.Tk)) {
                    float s = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        s += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    s *= p.scale;
                    float mnew = (s > m) ? s : m;
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
        if (l > 0.0f) {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = acc[c] / l;
            }
        } else {
            for (int c = 0; c < p.Dh; ++c) {
                out[qbase + c] = 0.0f;
            }
        }
    }
}
kernel void gd_sdpa_splitk(device const float *q              [[buffer(0)]],
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

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int s = (int)tgid % p.n_splits;
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
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk, p.causal, p.prefix_len);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
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
                    for (int c = 0; c < p.Dh; ++c) {
                        ss += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    ss = p.scale * ss + gd_sdpa_bias_at(bias, p, b, hq, i, j);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
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
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        for (int c = 0; c < p.Dh; ++c) {
            partials[base + c] = acc[c];
        }
        partials[base + p.Dh] = m;
        partials[base + p.Dh + 1] = l;
    }
}
kernel void gd_sdpa_splitk_causal(device const float *q              [[buffer(0)]],
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

    int n_qb = (p.Tq + GD_SDPA_BQ - 1) / GD_SDPA_BQ;
    int s = (int)tgid % p.n_splits;
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
    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        qreg[c] = active ? q[qbase + c] : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int kb_end = gd_sdpa_causal_kb_end(qb * GD_SDPA_BQ, GD_SDPA_BQ, p.Tq, p.Tk);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
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
                if (gd_sdpa_causal_allowed(i, j, p.Tq, p.Tk)) {
                    float ss = 0.0f;
                    for (int c = 0; c < p.Dh; ++c) {
                        ss += qreg[c] * ksh[jj * p.Dh + c];
                    }
                    ss *= p.scale;
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
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
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        for (int c = 0; c < p.Dh; ++c) {
            partials[base + c] = acc[c];
        }
        partials[base + p.Dh] = m;
        partials[base + p.Dh + 1] = l;
    }
}
kernel void gd_sdpa_splitk_causal_lane8(device const float *q              [[buffer(0)]],
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

    int n_qb = (p.Tq + GD_SDPA_FWD_CAUSAL_QROWS - 1) / GD_SDPA_FWD_CAUSAL_QROWS;
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
    int i = qb * GD_SDPA_FWD_CAUSAL_QROWS + local_q;
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

    int kb_end = gd_sdpa_causal_kb_end(qb * GD_SDPA_FWD_CAUSAL_QROWS,
                                        GD_SDPA_FWD_CAUSAL_QROWS,
                                        p.Tq,
                                        p.Tk);
    int k_lo = s * p.split_len;
    int k_hi = k_lo + p.split_len;
    if (k_hi > kb_end) {
        k_hi = kb_end;
    }
    for (int kb = k_lo; kb < k_hi; kb += GD_SDPA_BK) {
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
                if (gd_sdpa_causal_allowed(i, j, p.Tq, p.Tk)) {
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
kernel void gd_sdpa_combine(device const float *partials       [[buffer(0)]],
                            device float *out                  [[buffer(1)]],
                            constant gd_metal_sdpa_params &p    [[buffer(2)]],
                            uint gid [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq;
    if ((int)gid >= total) {
        return;
    }
    int i = (int)gid % p.Tq;
    int r = (int)gid / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;

    float acc[GD_SDPA_DHT];
    for (int c = 0; c < p.Dh; ++c) {
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    for (int s = 0; s < p.n_splits; ++s) {
        int base = ((((b * p.Hq + hq) * p.Tq + i) * p.n_splits) + s) * (p.Dh + 2);
        float ms = partials[base + p.Dh];
        float ls = partials[base + p.Dh + 1];
        if (ls <= 0.0f) {
            continue;
        }
        float mnew = (ms > m) ? ms : m;
        float corr_o = exp(m - mnew);
        float corr_s = exp(ms - mnew);
        l = l * corr_o + ls * corr_s;
        for (int c = 0; c < p.Dh; ++c) {
            acc[c] = acc[c] * corr_o + partials[base + c] * corr_s;
        }
        m = mnew;
    }
    if (l > 0.0f) {
        for (int c = 0; c < p.Dh; ++c) {
            out[qbase + c] = acc[c] / l;
        }
    } else {
        for (int c = 0; c < p.Dh; ++c) {
            out[qbase + c] = 0.0f;
        }
    }
}
kernel void gd_sdpa(device const float *q              [[buffer(0)]],
                    device const float *k              [[buffer(1)]],
                    device const float *v              [[buffer(2)]],
                    device const float *bias           [[buffer(3)]],
                    device float *out                  [[buffer(4)]],
                    constant gd_metal_sdpa_params &p    [[buffer(5)]],
                    uint gid                           [[thread_position_in_grid]])
{
    int total = p.B * p.Hq * p.Tq;
    if ((int)gid >= total) {
        return;
    }
    int i = (int)gid % p.Tq;
    int r = (int)gid / p.Tq;
    int hq = r % p.Hq;
    int b = r / p.Hq;
    int group = p.Hq / p.Hkv;
    int hkv = hq / group;
    int qbase = ((b * p.Tq + i) * p.Hq + hq) * p.Dh;
    int obase = qbase;

    float m = -INFINITY;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            if (s > m) {
                m = s;
            }
        }
    }
    for (int c = 0; c < p.Dh; ++c) {
        out[obase + c] = 0.0f;
    }
    float sum = 0.0f;
    for (int j = 0; j < p.Tk; ++j) {
        if (gd_sdpa_allowed(i, j, p.Tq, p.Tk, p.causal, p.window, p.prefix_len)) {
            int kbase = ((b * p.Tk + j) * p.Hkv + hkv) * p.Dh;
            int vbase = kbase;
            float s = p.scale * gd_sdpa_dot(q + qbase, k + kbase, p.Dh)
                      + gd_sdpa_bias_at(bias, p, b, hq, i, j);
            float e = exp(s - m);
            sum += e;
            for (int c = 0; c < p.Dh; ++c) {
                out[obase + c] += e * v[vbase + c];
            }
        }
    }
    if (sum > 0.0f) {
        for (int c = 0; c < p.Dh; ++c) {
            out[obase + c] /= sum;
        }
    }
}
