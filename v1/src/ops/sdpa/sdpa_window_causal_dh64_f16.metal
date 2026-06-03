#include "metal_common.metal"

#if GD_SDPA_DHT != 64
#error "DH64 f16 window kernels require GD_SDPA_DHT == 64"
#endif
#if GD_SDPA_DKV_LANES != 8
#error "DH64 f16 window kernels assume 8 channel lanes"
#endif
#if GD_SDPA_DKV_WIDE_LANES != 8
#error "DH64 f16 window dK/dV kernel assumes 8 wide channel lanes"
#endif

/* GPT hot path: fp16 causal sliding-window backward with head_dim=64.
 *
 * Generic fp16 window kernels in sdpa_window_causal.metal keep `p.Dh` dynamic,
 * so the compiler must carry runtime channel loops and dynamic threadgroup
 * offsets through the two heaviest SDPA backward passes. GPT bench uses
 * `head_dim=64`; these variants keep the same numerical contract (fp16 storage,
 * fp32 math) but make channel count compile-time constant. Host dispatch guards
 * on `p.Dh == 64` and falls back to generic fp16 window kernels otherwise. */
#define GD_SDPA_DH64 64
#define GD_SDPA_DH64_CHANS (GD_SDPA_DH64 / GD_SDPA_DKV_LANES)
#define GD_SDPA_DH64_WIDE_CHANS (GD_SDPA_DH64 / GD_SDPA_DKV_WIDE_LANES)

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

kernel void gd_sdpa_bwd_stats_dq_split_causal_window_lane8_dh64_f16(
    device const half *go            [[buffer(0)]],
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
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DH64];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DH64];

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
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * GD_SDPA_DH64) : 0;

    float qreg[GD_SDPA_DH64_CHANS];
    float goreg[GD_SDPA_DH64_CHANS];
    float acc[GD_SDPA_DH64_CHANS];
    float ksum[GD_SDPA_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
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
        for (int idx = (int)tid; idx < tile * GD_SDPA_DH64; idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / GD_SDPA_DH64;
            int c = idx % GD_SDPA_DH64;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * GD_SDPA_DH64;
            ksh[jj * GD_SDPA_DH64 + c] = (float)k[kbase + c];
            vsh[jj * GD_SDPA_DH64 + c] = (float)v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_window_allowed(i, j, p.Tq, p.Tk, p.window)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        ss += qreg[x] * ksh[jj * GD_SDPA_DH64 + c];
                        dp += goreg[x] * vsh[jj * GD_SDPA_DH64 + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    dp = gd_sdpa_sum_lanes8(dp);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        float kc = ksh[jj * GD_SDPA_DH64 + c];
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
        int vbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) *
                    2 * GD_SDPA_DH64;
        if (lane == 0) {
            stats_part[sbase + 0] = m;
            stats_part[sbase + 1] = l;
            stats_part[sbase + 2] = raw;
        }
        for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_LANES;
            dq_part[vbase + c] = acc[x];
            dq_part[vbase + GD_SDPA_DH64 + c] = ksum[x];
        }
    }
    (void)bias;
}

kernel void gd_sdpa_bwd_dkv_split_causal_window_k16_dh64_f16(
    device const half *go             [[buffer(0)]],
    device const half *q              [[buffer(1)]],
    device const half *k              [[buffer(2)]],
    device const half *v              [[buffer(3)]],
    device const half *bias           [[buffer(4)]],
    device float *part                [[buffer(5)]],
    constant gd_metal_sdpa_params &p  [[buffer(6)]],
    device const float *stats         [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DH64];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DH64];
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
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * GD_SDPA_DH64) : 0;

    float kreg[GD_SDPA_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
        kreg[x] = active ? (float)k[kbase + c] : 0.0f;
        vreg[x] = active ? (float)v[kbase + c] : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
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
            for (int idx = (int)tid; idx < tile * GD_SDPA_DH64;
                 idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / GD_SDPA_DH64;
                int c = idx % GD_SDPA_DH64;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * GD_SDPA_DH64;
                qsh[ii * GD_SDPA_DH64 + c] = (float)q[qb2 + c];
                gsh[ii * GD_SDPA_DH64 + c] = (float)go[qb2 + c];
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

                for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
                    int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                    ss += qsh[ii * GD_SDPA_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_DH64 + c] * vreg[x];
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
                    for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                        dvacc[x] += pj * gsh[ii * GD_SDPA_DH64 + c];
                        dkacc[x] += p.scale * ds * qsh[ii * GD_SDPA_DH64 + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int base = (((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits + s) *
                   2 * GD_SDPA_DH64;
        for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
            part[base + c] = dkacc[x];
            part[base + GD_SDPA_DH64 + c] = dvacc[x];
        }
    }
    (void)bias;
}

/* VLM hot path: fp16 prefix-causal + suffix sliding-window backward with
 * head_dim=64. Prefix/image keys remain globally visible; windowing only prunes
 * suffix/suffix pairs. Keep this separate from the pure GPT window kernels so
 * prefix branches do not regress prefix_len==0 workloads. */
static inline bool gd_sdpa_prefix_window_allowed(int i, int j, int Tq, int Tk,
                                                 int window, int prefix_len)
{
    int qpos = i + (Tk - Tq);

    if (prefix_len > 0) {
        if (qpos < prefix_len) {
            return j < prefix_len;
        }
        if (j < prefix_len) {
            return true;
        }
    }
    return j <= qpos && (qpos - j) < window;
}

static inline int gd_sdpa_prefix_window_kb_start(int q0, int Tq, int Tk,
                                                 int window, int prefix_len)
{
    int qpos = q0 + (Tk - Tq);
    int first = 0;

    if (window <= 0) {
        return 0;
    }
    if (prefix_len > 0) {
        if (qpos < prefix_len) {
            qpos = prefix_len;
        }
        first = qpos - window + 1;
        if (first < prefix_len) {
            first = prefix_len;
        }
        if (first > Tk) {
            return Tk;
        }
        return first;
    }
    first = qpos - window + 1;
    if (first <= 0) {
        return 0;
    }
    if (first > Tk) {
        return Tk;
    }
    return first;
}

static inline int gd_sdpa_prefix_window_kb_prefix_end(int Tk, int window,
                                                      int prefix_len)
{
    if (window <= 0 || prefix_len <= 0) {
        return 0;
    }
    return prefix_len < Tk ? prefix_len : Tk;
}

static inline int gd_sdpa_prefix_window_kb_end(int q0, int bq, int Tq, int Tk,
                                               int prefix_len)
{
    int qmax = q0 + bq - 1;
    int qmaxpos = 0;
    int lim = 0;

    if (qmax > Tq - 1) {
        qmax = Tq - 1;
    }
    qmaxpos = qmax + (Tk - Tq);
    lim = qmaxpos + 1;
    if (prefix_len > 0 && qmaxpos < prefix_len) {
        lim = prefix_len;
    }
    if (lim < 0) {
        return 0;
    }
    return lim > Tk ? Tk : lim;
}

static inline int gd_sdpa_prefix_window_qb_start(int k0, int qblk, int Tk,
                                                 int Tq, int prefix_len)
{
    int qmin = 0;

    if (prefix_len > 0 && k0 < prefix_len) {
        return 0;
    }
    qmin = k0 - (Tk - Tq);
    if (qmin <= 0) {
        return 0;
    }
    if (qmin > Tq) {
        return Tq;
    }
    return (qmin / qblk) * qblk;
}

static inline int gd_sdpa_prefix_window_qb_end(int k0, int key_count, int Tk,
                                               int Tq, int window,
                                               int prefix_len)
{
    int kend = 0;
    int lim = 0;

    if (window <= 0) {
        return Tq;
    }
    if (prefix_len > 0 && k0 < prefix_len) {
        return Tq;
    }
    kend = k0 + key_count;
    if (kend > Tk) {
        kend = Tk;
    }
    lim = kend + window - 1 - (Tk - Tq);
    if (lim < 0) {
        return 0;
    }
    return lim > Tq ? Tq : lim;
}

kernel void gd_sdpa_bwd_stats_dq_split_prefix_window_lane8_dh64_f16(
    device const half *go            [[buffer(0)]],
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
    threadgroup float ksh[GD_SDPA_BK * GD_SDPA_DH64];
    threadgroup float vsh[GD_SDPA_BK * GD_SDPA_DH64];

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
    int qbase = active ? (((b * p.Tq + i) * p.Hq + hq) * GD_SDPA_DH64) : 0;

    float qreg[GD_SDPA_DH64_CHANS];
    float goreg[GD_SDPA_DH64_CHANS];
    float acc[GD_SDPA_DH64_CHANS];
    float ksum[GD_SDPA_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
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
    int kb_start = gd_sdpa_prefix_window_kb_start(q0, p.Tq, p.Tk, p.window,
                                                  p.prefix_len);
    int kb_prefix_end = gd_sdpa_prefix_window_kb_prefix_end(p.Tk, p.window,
                                                            p.prefix_len);
    int kb_end = gd_sdpa_prefix_window_kb_end(q0, GD_SDPA_CAUSAL_QROWS,
                                              p.Tq, p.Tk, p.prefix_len);
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
        for (int idx = (int)tid; idx < tile * GD_SDPA_DH64;
             idx += GD_SDPA_CAUSAL_THREADS) {
            int jj = idx / GD_SDPA_DH64;
            int c = idx % GD_SDPA_DH64;
            int kbase = ((b * p.Tk + (kb + jj)) * p.Hkv + hkv) * GD_SDPA_DH64;
            ksh[jj * GD_SDPA_DH64 + c] = (float)k[kbase + c];
            vsh[jj * GD_SDPA_DH64 + c] = (float)v[kbase + c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_prefix_window_allowed(i, j, p.Tq, p.Tk,
                                                  p.window, p.prefix_len)) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        ss += qreg[x] * ksh[jj * GD_SDPA_DH64 + c];
                        dp += goreg[x] * vsh[jj * GD_SDPA_DH64 + c];
                    }
                    ss = gd_sdpa_sum_lanes8(ss) * p.scale;
                    dp = gd_sdpa_sum_lanes8(dp);
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    raw = raw * corr + e * dp;
                    for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_LANES;
                        float kc = ksh[jj * GD_SDPA_DH64 + c];
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
        int vbase = (((b * p.Hq + hq) * p.Tq + i) * p.n_splits + sp) *
                    2 * GD_SDPA_DH64;
        if (lane == 0) {
            stats_part[sbase + 0] = m;
            stats_part[sbase + 1] = l;
            stats_part[sbase + 2] = raw;
        }
        for (int x = 0; x < GD_SDPA_DH64_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_LANES;
            dq_part[vbase + c] = acc[x];
            dq_part[vbase + GD_SDPA_DH64 + c] = ksum[x];
        }
    }
    (void)bias;
}

kernel void gd_sdpa_bwd_dkv_split_prefix_window_k16_dh64_f16(
    device const half *go             [[buffer(0)]],
    device const half *q              [[buffer(1)]],
    device const half *k              [[buffer(2)]],
    device const half *v              [[buffer(3)]],
    device const half *bias           [[buffer(4)]],
    device float *part                [[buffer(5)]],
    constant gd_metal_sdpa_params &p  [[buffer(6)]],
    device const float *stats         [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_SDPA_BK * GD_SDPA_DH64];
    threadgroup float gsh[GD_SDPA_BK * GD_SDPA_DH64];
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
    int kbase = active ? (((b * p.Tk + j) * p.Hkv + hkv) * GD_SDPA_DH64) : 0;

    float kreg[GD_SDPA_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
        kreg[x] = active ? (float)k[kbase + c] : 0.0f;
        vreg[x] = active ? (float)v[kbase + c] : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
    }

    int k0 = kblk * GD_SDPA_DKV_WIDE_KEYS;
    int qb_start = gd_sdpa_prefix_window_qb_start(k0, GD_SDPA_BK, p.Tk,
                                                  p.Tq, p.prefix_len);
    int q_cap = gd_sdpa_prefix_window_qb_end(k0, GD_SDPA_DKV_WIDE_KEYS,
                                             p.Tk, p.Tq, p.window,
                                             p.prefix_len);
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
            for (int idx = (int)tid; idx < tile * GD_SDPA_DH64;
                 idx += GD_SDPA_DKV_WIDE_THREADS) {
                int ii = idx / GD_SDPA_DH64;
                int c = idx % GD_SDPA_DH64;
                int qb2 = ((b * p.Tq + (qb + ii)) * p.Hq + hq) * GD_SDPA_DH64;
                qsh[ii * GD_SDPA_DH64 + c] = (float)q[qb2 + c];
                gsh[ii * GD_SDPA_DH64 + c] = (float)go[qb2 + c];
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

                for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
                    int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                    ss += qsh[ii * GD_SDPA_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_DH64 + c] * vreg[x];
                }
                ss = gd_sdpa_sum_lanes8(ss);
                dp = gd_sdpa_sum_lanes8(dp);
                if (active && gd_sdpa_prefix_window_allowed(i, j, p.Tq, p.Tk,
                                                            p.window,
                                                            p.prefix_len)) {
                    float l = lsh[ii];
                    if (l > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / l;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
                        dvacc[x] += pj * gsh[ii * GD_SDPA_DH64 + c];
                        dkacc[x] += p.scale * ds * qsh[ii * GD_SDPA_DH64 + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int base = (((b * p.Hkv + hkv) * p.Tk + j) * p.n_splits + s) *
                   2 * GD_SDPA_DH64;
        for (int x = 0; x < GD_SDPA_DH64_WIDE_CHANS; ++x) {
            int c = lane + x * GD_SDPA_DKV_WIDE_LANES;
            part[base + c] = dkacc[x];
            part[base + GD_SDPA_DH64 + c] = dvacc[x];
        }
    }
    (void)bias;
}
