#include <metal_stdlib>
#include "metal_sdpa_varlen_types.h"

using namespace metal;

static inline float gd_sdpa_varlen_load(device const uchar *buf,
                                        ulong byte_offset,
                                        uint dtype,
                                        ulong idx)
{
    if (dtype == uint(GD_METAL_SDPA_DTYPE_F16)) {
        return float(reinterpret_cast<device const half *>(buf + byte_offset)[idx]);
    }
    return reinterpret_cast<device const float *>(buf + byte_offset)[idx];
}

static inline void gd_sdpa_varlen_store(device uchar *buf,
                                        ulong byte_offset,
                                        uint dtype,
                                        ulong idx,
                                        float value)
{
    if (dtype == uint(GD_METAL_SDPA_DTYPE_F16)) {
        reinterpret_cast<device half *>(buf + byte_offset)[idx] = half(value);
    } else {
        reinterpret_cast<device float *>(buf + byte_offset)[idx] = value;
    }
}

static inline bool gd_sdpa_varlen_allowed(int i, int j, int causal, int window, int prefix_len)
{
    if (causal != 0) {
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

static inline int gd_sdpa_varlen_kb_end(int q0, int bq, int T, int causal, int prefix_len)
{
    if (causal == 0) {
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

static inline int gd_sdpa_varlen_kb_start(int q0, int T, int window, int prefix_len)
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

static inline int gd_sdpa_varlen_qb_start(int k0, int qblk, int causal, int prefix_len)
{
    if (causal == 0) {
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

static inline int gd_sdpa_varlen_qb_end(int k0, int key_count, int T, int window, int prefix_len)
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

kernel void gd_sdpa_varlen_kernel(device const uchar *qbuf [[buffer(0)]],
                                  device const uchar *kbuf [[buffer(1)]],
                                  device const uchar *vbuf [[buffer(2)]],
                                  device const uchar *cubuf [[buffer(3)]],
                                  device uchar *outbuf [[buffer(4)]],
                                  constant gd_metal_sdpa_varlen_args &p [[buffer(5)]],
                                  uint tgid [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    threadgroup float vsh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);

    const int bq = int(GD_METAL_SDPA_BQ);
    const int bk = int(GD_METAL_SDPA_BK);
    int qb = int(tgid) % int(p.n_qb_max);
    int r = int(tgid) / int(p.n_qb_max);
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    if (b >= int(p.batch) || p.dh > uint(GD_METAL_SDPA_DHT)) {
        return;
    }
    int start = cu[b];
    int end = cu[b + 1];
    int T = end - start;
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * bq + int(tid);
    bool active = i < T;
    int qg = start + i;
    ulong qbase = active ? ulong((qg * int(p.hq) + hq) * int(p.dh)) : 0ul;

    float qreg[GD_METAL_SDPA_DHT];
    float acc[GD_METAL_SDPA_DHT];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_sdpa_varlen_load(qbuf, p.q_offset, p.dtype, qbase + ulong(c)) : 0.0f;
        acc[c] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * bq;
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, int(p.window), int(p.prefix_len));
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, int(p.window), int(p.prefix_len));
    int kb_end = gd_sdpa_varlen_kb_end(q0, bq, T, int(p.causal), int(p.prefix_len));
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
            int kg = start + kb0 + jj;
            ulong kbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
            ksh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c));
            vsh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(vbuf, p.v_offset, p.dtype, kbase + ulong(c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb0 + jj;
                if (gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
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
            gd_sdpa_varlen_store(outbuf, p.out_offset, p.dtype, qbase + ulong(c), acc[c] * inv_l);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_stats_kernel(device const uchar *gobuf [[buffer(0)]],
                                            device const uchar *qbuf [[buffer(1)]],
                                            device const uchar *kbuf [[buffer(2)]],
                                            device const uchar *vbuf [[buffer(3)]],
                                            device const uchar *cubuf [[buffer(4)]],
                                            device uchar *statsbuf [[buffer(5)]],
                                            constant gd_metal_sdpa_varlen_args &p [[buffer(6)]],
                                            uint tgid [[threadgroup_position_in_grid]],
                                            uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    threadgroup float vsh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device float *stats = reinterpret_cast<device float *>(statsbuf + p.stats_offset);

    const int bq = int(GD_METAL_SDPA_BQ);
    const int bk = int(GD_METAL_SDPA_BK);
    int qb = int(tgid) % int(p.n_qb_max);
    int r = int(tgid) / int(p.n_qb_max);
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    if (b >= int(p.batch) || p.dh > uint(GD_METAL_SDPA_DHT)) {
        return;
    }
    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * bq + int(tid);
    bool active = i < T;
    int qg = start + i;
    ulong qbase = active ? ulong((qg * int(p.hq) + hq) * int(p.dh)) : 0ul;

    float qreg[GD_METAL_SDPA_DHT];
    float goreg[GD_METAL_SDPA_DHT];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_sdpa_varlen_load(qbuf, p.q_offset, p.dtype, qbase + ulong(c)) : 0.0f;
        goreg[c] = active ? gd_sdpa_varlen_load(gobuf, p.grad_out_offset, p.dtype, qbase + ulong(c)) : 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int q0 = qb * bq;
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, int(p.window), int(p.prefix_len));
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, int(p.window), int(p.prefix_len));
    int kb_end = gd_sdpa_varlen_kb_end(q0, bq, T, int(p.causal), int(p.prefix_len));
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
            int kg = start + kb0 + jj;
            ulong kbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
            ksh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c));
            vsh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(vbuf, p.v_offset, p.dtype, kbase + ulong(c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb0 + jj;
                if (gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < int(p.dh); ++c) {
                        s += qreg[c] * ksh[jj * int(p.dh) + c];
                        dp += goreg[c] * vsh[jj * int(p.dh) + c];
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
        ulong sbase = ulong(qg * int(p.hq) + hq) * 3ul;
        stats[sbase + 0ul] = m;
        stats[sbase + 1ul] = l;
        stats[sbase + 2ul] = l > 0.0f ? raw / l : 0.0f;
    }
}

kernel void gd_sdpa_varlen_bwd_kernel(device const uchar *gobuf [[buffer(0)]],
                                      device const uchar *qbuf [[buffer(1)]],
                                      device const uchar *kbuf [[buffer(2)]],
                                      device const uchar *vbuf [[buffer(3)]],
                                      device const uchar *cubuf [[buffer(4)]],
                                      device uchar *dqbuf [[buffer(5)]],
                                      device const uchar *statsbuf [[buffer(6)]],
                                      constant gd_metal_sdpa_varlen_args &p [[buffer(7)]],
                                      uint tgid [[threadgroup_position_in_grid]],
                                      uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    threadgroup float vsh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const float *stats = reinterpret_cast<device const float *>(statsbuf + p.stats_offset);

    const int bq = int(GD_METAL_SDPA_BQ);
    const int bk = int(GD_METAL_SDPA_BK);
    int qb = int(tgid) % int(p.n_qb_max);
    int r = int(tgid) / int(p.n_qb_max);
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    if (b >= int(p.batch) || p.dh > uint(GD_METAL_SDPA_DHT)) {
        return;
    }
    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * bq + int(tid);
    bool active = i < T;
    int qg = start + i;
    ulong qbase = active ? ulong((qg * int(p.hq) + hq) * int(p.dh)) : 0ul;
    ulong sbase = active ? ulong(qg * int(p.hq) + hq) * 3ul : 0ul;
    float m = active ? stats[sbase + 0ul] : 0.0f;
    float l = active ? stats[sbase + 1ul] : 0.0f;
    float D = active ? stats[sbase + 2ul] : 0.0f;
    bool run = active && l > 0.0f;

    float qreg[GD_METAL_SDPA_DHT];
    float goreg[GD_METAL_SDPA_DHT];
    float dqacc[GD_METAL_SDPA_DHT];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_sdpa_varlen_load(qbuf, p.q_offset, p.dtype, qbase + ulong(c)) : 0.0f;
        goreg[c] = active ? gd_sdpa_varlen_load(gobuf, p.grad_out_offset, p.dtype, qbase + ulong(c)) : 0.0f;
        dqacc[c] = 0.0f;
    }

    int q0 = qb * bq;
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, int(p.window), int(p.prefix_len));
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, int(p.window), int(p.prefix_len));
    int kb_end = gd_sdpa_varlen_kb_end(q0, bq, T, int(p.causal), int(p.prefix_len));
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
            int kg = start + kb0 + jj;
            ulong kbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
            ksh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c));
            vsh[jj * int(p.dh) + c] = gd_sdpa_varlen_load(vbuf, p.v_offset, p.dtype, kbase + ulong(c));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (run) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb0 + jj;
                if (gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < int(p.dh); ++c) {
                        s += qreg[c] * ksh[jj * int(p.dh) + c];
                        dp += goreg[c] * vsh[jj * int(p.dh) + c];
                    }
                    s *= p.scale;
                    float pj = exp(s - m) / l;
                    float ds = pj * (dp - D);
                    for (int c = 0; c < int(p.dh); ++c) {
                        dqacc[c] += p.scale * ds * ksh[jj * int(p.dh) + c];
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        for (int c = 0; c < int(p.dh); ++c) {
            gd_sdpa_varlen_store(dqbuf, p.grad_q_offset, p.dtype, qbase + ulong(c), dqacc[c]);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv_kernel(device const uchar *gobuf [[buffer(0)]],
                                          device const uchar *qbuf [[buffer(1)]],
                                          device const uchar *kbuf [[buffer(2)]],
                                          device const uchar *vbuf [[buffer(3)]],
                                          device const uchar *cubuf [[buffer(4)]],
                                          device uchar *dkbuf [[buffer(5)]],
                                          device uchar *dvbuf [[buffer(6)]],
                                          device const uchar *statsbuf [[buffer(7)]],
                                          constant gd_metal_sdpa_varlen_args &p [[buffer(8)]],
                                          uint tgid [[threadgroup_position_in_grid]],
                                          uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    threadgroup float gsh[GD_METAL_SDPA_BK * GD_METAL_SDPA_DHT];
    threadgroup float msh[GD_METAL_SDPA_BK];
    threadgroup float lsh[GD_METAL_SDPA_BK];
    threadgroup float dsh[GD_METAL_SDPA_BK];
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const float *stats = reinterpret_cast<device const float *>(statsbuf + p.stats_offset);

    const int bq = int(GD_METAL_SDPA_BQ);
    const int bk = int(GD_METAL_SDPA_BK);
    int kb = int(tgid) % int(p.n_qb_max);
    int r = int(tgid) / int(p.n_qb_max);
    int hkv = r % int(p.hkv);
    int b = r / int(p.hkv);
    if (b >= int(p.batch) || p.dh > uint(GD_METAL_SDPA_DHT)) {
        return;
    }
    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int j = kb * bq + int(tid);
    bool active = j < T;
    int kg = start + j;
    ulong kbase = active ? ulong((kg * int(p.hkv) + hkv) * int(p.dh)) : 0ul;

    float kreg[GD_METAL_SDPA_DHT];
    float vreg[GD_METAL_SDPA_DHT];
    float dkacc[GD_METAL_SDPA_DHT];
    float dvacc[GD_METAL_SDPA_DHT];
    for (int c = 0; c < int(p.dh); ++c) {
        kreg[c] = active ? gd_sdpa_varlen_load(kbuf, p.k_offset, p.dtype, kbase + ulong(c)) : 0.0f;
        vreg[c] = active ? gd_sdpa_varlen_load(vbuf, p.v_offset, p.dtype, kbase + ulong(c)) : 0.0f;
        dkacc[c] = 0.0f;
        dvacc[c] = 0.0f;
    }

    int k0 = kb * bq;
    int qb_start = gd_sdpa_varlen_qb_start(k0, bk, int(p.causal), int(p.prefix_len));
    int qb_end = gd_sdpa_varlen_qb_end(k0, bq, T, int(p.window), int(p.prefix_len));
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = qb_start; qb < qb_end; qb += bk) {
            int tile = qb_end - qb;
            if (tile > bk) {
                tile = bk;
            }
            for (int idx = int(tid); idx < tile * int(p.dh); idx += bq) {
                int ii = idx / int(p.dh);
                int c = idx % int(p.dh);
                int qg = start + qb + ii;
                ulong qbase = ulong((qg * int(p.hq) + hq) * int(p.dh));
                qsh[ii * int(p.dh) + c] = gd_sdpa_varlen_load(qbuf, p.q_offset, p.dtype, qbase + ulong(c));
                gsh[ii * int(p.dh) + c] = gd_sdpa_varlen_load(gobuf, p.grad_out_offset, p.dtype, qbase + ulong(c));
            }
            for (int ii = int(tid); ii < tile; ii += bq) {
                int qg = start + qb + ii;
                ulong sb = ulong(qg * int(p.hq) + hq) * 3ul;
                msh[ii] = stats[sb + 0ul];
                lsh[ii] = stats[sb + 1ul];
                dsh[ii] = stats[sb + 2ul];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    int i = qb + ii;
                    if (!gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                        continue;
                    }
                    float l = lsh[ii];
                    if (l <= 0.0f) {
                        continue;
                    }
                    float s = 0.0f;
                    float dp = 0.0f;
                    for (int c = 0; c < int(p.dh); ++c) {
                        s += qsh[ii * int(p.dh) + c] * kreg[c];
                        dp += gsh[ii * int(p.dh) + c] * vreg[c];
                    }
                    s *= p.scale;
                    float pj = exp(s - msh[ii]) / l;
                    float ds = pj * (dp - dsh[ii]);
                    for (int c = 0; c < int(p.dh); ++c) {
                        dvacc[c] += pj * gsh[ii * int(p.dh) + c];
                        dkacc[c] += p.scale * ds * qsh[ii * int(p.dh) + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        for (int c = 0; c < int(p.dh); ++c) {
            gd_sdpa_varlen_store(dkbuf, p.grad_k_offset, p.dtype, kbase + ulong(c), dkacc[c]);
            gd_sdpa_varlen_store(dvbuf, p.grad_v_offset, p.dtype, kbase + ulong(c), dvacc[c]);
        }
    }
}

#define GD_SDPA_VARLEN_DH64 64
#define GD_SDPA_VARLEN_DH64_CHANS (GD_SDPA_VARLEN_DH64 / int(GD_METAL_SDPA_DKV_LANES))
#define GD_SDPA_VARLEN_DH64_WIDE_CHANS (GD_SDPA_VARLEN_DH64 / int(GD_METAL_SDPA_DKV_WIDE_LANES))

static inline float gd_sdpa_varlen_sum_lanes8(float x)
{
    x += simd_shuffle_xor(x, 1);
    x += simd_shuffle_xor(x, 2);
    x += simd_shuffle_xor(x, 4);
    return x;
}

kernel void gd_sdpa_varlen_prefix_window_lane8_dh64_f16_kernel(
    device const half *q [[buffer(0)]],
    device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]],
    device const int *cu [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant gd_metal_sdpa_varlen_args &p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float vsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];

    int n_qb = int(p.n_qb_max);
    int qb = int(tgid) % n_qb;
    int r = int(tgid) / n_qb;
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    int local_q = int(tid) / int(GD_METAL_SDPA_DKV_LANES);
    int lane = int(tid) - local_q * int(GD_METAL_SDPA_DKV_LANES);
    if (b >= int(p.batch)) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * int(GD_METAL_SDPA_CAUSAL_QROWS) + local_q;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * int(p.hq) + hq) * GD_SDPA_VARLEN_DH64) : 0;

    float qreg[GD_SDPA_VARLEN_DH64_CHANS];
    float acc[GD_SDPA_VARLEN_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
        qreg[x] = active ? float(q[qbase + c]) : 0.0f;
        acc[x] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    int q0 = qb * int(GD_METAL_SDPA_CAUSAL_QROWS);
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, int(p.window), int(p.prefix_len));
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, int(p.window), int(p.prefix_len));
    int kb_end = gd_sdpa_varlen_kb_end(q0,
                                       int(GD_METAL_SDPA_CAUSAL_QROWS),
                                       T,
                                       int(p.causal),
                                       int(p.prefix_len));
    for (int kb = 0; kb < kb_end; kb += int(GD_METAL_SDPA_BK)) {
        if (kb < kb_start && (kb_prefix_end == 0 || kb >= kb_prefix_end)) {
            kb = kb_start - int(GD_METAL_SDPA_BK);
            continue;
        }
        int tile = kb_end - kb;
        if (tile > int(GD_METAL_SDPA_BK)) {
            tile = int(GD_METAL_SDPA_BK);
        }
        for (int idx = int(tid); idx < tile * GD_SDPA_VARLEN_DH64;
             idx += int(GD_METAL_SDPA_CAUSAL_THREADS)) {
            int jj = idx / GD_SDPA_VARLEN_DH64;
            int c = idx % GD_SDPA_VARLEN_DH64;
            int kg = start + kb + jj;
            int kbase = ((kg * int(p.hkv) + hkv) * GD_SDPA_VARLEN_DH64);
            ksh[jj * GD_SDPA_VARLEN_DH64 + c] = float(k[kbase + c]);
            vsh[jj * GD_SDPA_VARLEN_DH64 + c] = float(v[kbase + c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float ss = 0.0f;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
                        ss += qreg[x] * ksh[jj * GD_SDPA_VARLEN_DH64 + c];
                    }
                    ss = gd_sdpa_varlen_sum_lanes8(ss) * p.scale;
                    float mnew = (ss > m) ? ss : m;
                    float corr = exp(m - mnew);
                    float e = exp(ss - mnew);
                    l = l * corr + e;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
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
            int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
            out[qbase + c] = half(acc[x] * inv_l);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_stats_dq_prefix_window_lane8_dh64_f16_kernel(
    device const half *go [[buffer(0)]],
    device const half *q [[buffer(1)]],
    device const half *k [[buffer(2)]],
    device const half *v [[buffer(3)]],
    device const int *cu [[buffer(4)]],
    device half *dq [[buffer(5)]],
    constant gd_metal_sdpa_varlen_args &p [[buffer(6)]],
    device float *stats [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float vsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];

    int n_qb = int(p.n_qb_max);
    int qb = int(tgid) % n_qb;
    int r = int(tgid) / n_qb;
    int hq = r % int(p.hq);
    int b = r / int(p.hq);
    int local_q = int(tid) / int(GD_METAL_SDPA_DKV_LANES);
    int lane = int(tid) - local_q * int(GD_METAL_SDPA_DKV_LANES);
    if (b >= int(p.batch)) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int hkv = hq / group;
    int i = qb * int(GD_METAL_SDPA_CAUSAL_QROWS) + local_q;
    bool active = i < T;
    int qg = start + i;
    int qbase = active ? ((qg * int(p.hq) + hq) * GD_SDPA_VARLEN_DH64) : 0;

    float qreg[GD_SDPA_VARLEN_DH64_CHANS];
    float goreg[GD_SDPA_VARLEN_DH64_CHANS];
    float acc[GD_SDPA_VARLEN_DH64_CHANS];
    float ksum[GD_SDPA_VARLEN_DH64_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
        qreg[x] = active ? float(q[qbase + c]) : 0.0f;
        goreg[x] = active ? float(go[qbase + c]) : 0.0f;
        acc[x] = 0.0f;
        ksum[x] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;

    int q0 = qb * int(GD_METAL_SDPA_CAUSAL_QROWS);
    int kb_start = gd_sdpa_varlen_kb_start(q0, T, int(p.window), int(p.prefix_len));
    int kb_prefix_end = gd_sdpa_varlen_kb_prefix_end(T, int(p.window), int(p.prefix_len));
    int kb_end = gd_sdpa_varlen_kb_end(q0,
                                       int(GD_METAL_SDPA_CAUSAL_QROWS),
                                       T,
                                       int(p.causal),
                                       int(p.prefix_len));
    for (int kb = 0; kb < kb_end; kb += int(GD_METAL_SDPA_BK)) {
        if (kb < kb_start && (kb_prefix_end == 0 || kb >= kb_prefix_end)) {
            kb = kb_start - int(GD_METAL_SDPA_BK);
            continue;
        }
        int tile = kb_end - kb;
        if (tile > int(GD_METAL_SDPA_BK)) {
            tile = int(GD_METAL_SDPA_BK);
        }
        for (int idx = int(tid); idx < tile * GD_SDPA_VARLEN_DH64;
             idx += int(GD_METAL_SDPA_CAUSAL_THREADS)) {
            int jj = idx / GD_SDPA_VARLEN_DH64;
            int c = idx % GD_SDPA_VARLEN_DH64;
            int kg = start + kb + jj;
            int kbase = ((kg * int(p.hkv) + hkv) * GD_SDPA_VARLEN_DH64);
            ksh[jj * GD_SDPA_VARLEN_DH64 + c] = float(k[kbase + c]);
            vsh[jj * GD_SDPA_VARLEN_DH64 + c] = float(v[kbase + c]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (int jj = 0; jj < tile; ++jj) {
                int j = kb + jj;
                if (gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float ss = 0.0f;
                    float dp = 0.0f;
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
                        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
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
                        int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
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
        int sbase = (qg * int(p.hq) + hq) * 3;
        if (lane == 0) {
            stats[sbase + 0] = m;
            stats[sbase + 1] = l;
            stats[sbase + 2] = D;
        }
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_CHANS; ++x) {
            int c = lane + x * int(GD_METAL_SDPA_DKV_LANES);
            dq[qbase + c] = half(p.scale * (acc[x] - D * ksum[x]) * inv_l);
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv_prefix_window_k16_dh64_f16_kernel(
    device const half *go [[buffer(0)]],
    device const half *q [[buffer(1)]],
    device const half *k [[buffer(2)]],
    device const half *v [[buffer(3)]],
    device const int *cu [[buffer(4)]],
    device half *dk [[buffer(5)]],
    device half *dv [[buffer(6)]],
    constant gd_metal_sdpa_varlen_args &p [[buffer(7)]],
    device const float *stats [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float gsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float msh[GD_METAL_SDPA_BK];
    threadgroup float lsh[GD_METAL_SDPA_BK];
    threadgroup float dsh[GD_METAL_SDPA_BK];

    int n_kb = (int(p.max_seqlen) + int(GD_METAL_SDPA_DKV_WIDE_KEYS) - 1) /
               int(GD_METAL_SDPA_DKV_WIDE_KEYS);
    int kblk = int(tgid) % n_kb;
    int r = int(tgid) / n_kb;
    int hkv = r % int(p.hkv);
    int b = r / int(p.hkv);
    int local_key = int(tid) / int(GD_METAL_SDPA_DKV_WIDE_LANES);
    int lane = int(tid) - local_key * int(GD_METAL_SDPA_DKV_WIDE_LANES);
    if (b >= int(p.batch)) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int j = kblk * int(GD_METAL_SDPA_DKV_WIDE_KEYS) + local_key;
    bool active = j < T;
    int kg = start + j;
    int kbase = active ? ((kg * int(p.hkv) + hkv) * GD_SDPA_VARLEN_DH64) : 0;

    float kreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
        kreg[x] = active ? float(k[kbase + c]) : 0.0f;
        vreg[x] = active ? float(v[kbase + c]) : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
    }

    int k0 = kblk * int(GD_METAL_SDPA_DKV_WIDE_KEYS);
    int qb_start = gd_sdpa_varlen_qb_start(k0, int(GD_METAL_SDPA_BK), int(p.causal), int(p.prefix_len));
    int qb_end = gd_sdpa_varlen_qb_end(k0,
                                       int(GD_METAL_SDPA_DKV_WIDE_KEYS),
                                       T,
                                       int(p.window),
                                       int(p.prefix_len));
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = qb_start; qb < qb_end; qb += int(GD_METAL_SDPA_BK)) {
            int tile = qb_end - qb;
            if (tile > int(GD_METAL_SDPA_BK)) {
                tile = int(GD_METAL_SDPA_BK);
            }
            for (int idx = int(tid); idx < tile * GD_SDPA_VARLEN_DH64;
                 idx += int(GD_METAL_SDPA_DKV_WIDE_THREADS)) {
                int ii = idx / GD_SDPA_VARLEN_DH64;
                int c = idx % GD_SDPA_VARLEN_DH64;
                int qg = start + qb + ii;
                int qbase = (qg * int(p.hq) + hq) * GD_SDPA_VARLEN_DH64;
                qsh[ii * GD_SDPA_VARLEN_DH64 + c] = float(q[qbase + c]);
                gsh[ii * GD_SDPA_VARLEN_DH64 + c] = float(go[qbase + c]);
            }
            for (int ii = int(tid); ii < tile; ii += int(GD_METAL_SDPA_DKV_WIDE_THREADS)) {
                int qg = start + qb + ii;
                int sb = (qg * int(p.hq) + hq) * 3;
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
                    int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
                    ss += qsh[ii * GD_SDPA_VARLEN_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_VARLEN_DH64 + c] * vreg[x];
                }
                ss = gd_sdpa_varlen_sum_lanes8(ss);
                dp = gd_sdpa_varlen_sum_lanes8(dp);
                if (active && gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float ll = lsh[ii];
                    if (ll > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / ll;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
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
            int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
            dk[kbase + c] = half(dkacc[x]);
            dv[kbase + c] = half(dvacc[x]);
        }
    }
}

static inline int gd_sdpa_varlen_split_len(int T, int n)
{
    int len = (T + n - 1) / n;
    len = ((len + int(GD_METAL_SDPA_BK) - 1) / int(GD_METAL_SDPA_BK)) * int(GD_METAL_SDPA_BK);
    return (len < int(GD_METAL_SDPA_BK)) ? int(GD_METAL_SDPA_BK) : len;
}

kernel void gd_sdpa_varlen_bwd_dkv_split_prefix_window_k16_dh64_f16_kernel(
    device const half *go [[buffer(0)]],
    device const half *q [[buffer(1)]],
    device const half *k [[buffer(2)]],
    device const half *v [[buffer(3)]],
    device const int *cu [[buffer(4)]],
    device float *part [[buffer(5)]],
    constant gd_metal_sdpa_varlen_args &p [[buffer(6)]],
    device const float *stats [[buffer(7)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float gsh[GD_METAL_SDPA_BK * GD_SDPA_VARLEN_DH64];
    threadgroup float msh[GD_METAL_SDPA_BK];
    threadgroup float lsh[GD_METAL_SDPA_BK];
    threadgroup float dsh[GD_METAL_SDPA_BK];

    int n_kb = (int(p.max_seqlen) + int(GD_METAL_SDPA_DKV_WIDE_KEYS) - 1) /
               int(GD_METAL_SDPA_DKV_WIDE_KEYS);
    int split = int(tgid) % int(p.n_splits);
    int t2 = int(tgid) / int(p.n_splits);
    int kblk = t2 % n_kb;
    int r = t2 / n_kb;
    int hkv = r % int(p.hkv);
    int b = r / int(p.hkv);
    int local_key = int(tid) / int(GD_METAL_SDPA_DKV_WIDE_LANES);
    int lane = int(tid) - local_key * int(GD_METAL_SDPA_DKV_WIDE_LANES);
    if (b >= int(p.batch)) {
        return;
    }

    int start = cu[b];
    int T = cu[b + 1] - start;
    int group = int(p.hq) / int(p.hkv);
    int j = kblk * int(GD_METAL_SDPA_DKV_WIDE_KEYS) + local_key;
    bool active = j < T;
    int kg = start + j;
    int kbase = active ? ((kg * int(p.hkv) + hkv) * GD_SDPA_VARLEN_DH64) : 0;

    float kreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float vreg[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dkacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    float dvacc[GD_SDPA_VARLEN_DH64_WIDE_CHANS];
    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
        int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
        kreg[x] = active ? float(k[kbase + c]) : 0.0f;
        vreg[x] = active ? float(v[kbase + c]) : 0.0f;
        dkacc[x] = 0.0f;
        dvacc[x] = 0.0f;
    }

    int k0 = kblk * int(GD_METAL_SDPA_DKV_WIDE_KEYS);
    int qb_start = gd_sdpa_varlen_qb_start(k0, int(GD_METAL_SDPA_BK), int(p.causal), int(p.prefix_len));
    int qb_end = gd_sdpa_varlen_qb_end(k0,
                                       int(GD_METAL_SDPA_DKV_WIDE_KEYS),
                                       T,
                                       int(p.window),
                                       int(p.prefix_len));
    int slen = gd_sdpa_varlen_split_len(T, int(p.n_splits));
    int q_lo = split * slen;
    int q_hi = q_lo + slen;
    if (q_lo < qb_start) {
        q_lo = qb_start;
    }
    if (q_hi > qb_end) {
        q_hi = qb_end;
    }
    for (int g = 0; g < group; ++g) {
        int hq = hkv * group + g;
        for (int qb = q_lo; qb < q_hi; qb += int(GD_METAL_SDPA_BK)) {
            int tile = q_hi - qb;
            if (tile > int(GD_METAL_SDPA_BK)) {
                tile = int(GD_METAL_SDPA_BK);
            }
            for (int idx = int(tid); idx < tile * GD_SDPA_VARLEN_DH64;
                 idx += int(GD_METAL_SDPA_DKV_WIDE_THREADS)) {
                int ii = idx / GD_SDPA_VARLEN_DH64;
                int c = idx % GD_SDPA_VARLEN_DH64;
                int qg = start + qb + ii;
                int qbase = (qg * int(p.hq) + hq) * GD_SDPA_VARLEN_DH64;
                qsh[ii * GD_SDPA_VARLEN_DH64 + c] = float(q[qbase + c]);
                gsh[ii * GD_SDPA_VARLEN_DH64 + c] = float(go[qbase + c]);
            }
            for (int ii = int(tid); ii < tile; ii += int(GD_METAL_SDPA_DKV_WIDE_THREADS)) {
                int qg = start + qb + ii;
                int sb = (qg * int(p.hq) + hq) * 3;
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
                    int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
                    ss += qsh[ii * GD_SDPA_VARLEN_DH64 + c] * kreg[x];
                    dp += gsh[ii * GD_SDPA_VARLEN_DH64 + c] * vreg[x];
                }
                ss = gd_sdpa_varlen_sum_lanes8(ss);
                dp = gd_sdpa_varlen_sum_lanes8(dp);
                if (active && gd_sdpa_varlen_allowed(i, j, int(p.causal), int(p.window), int(p.prefix_len))) {
                    float ll = lsh[ii];
                    if (ll > 0.0f) {
                        ss *= p.scale;
                        pj = exp(ss - msh[ii]) / ll;
                        ds = pj * (dp - dsh[ii]);
                    }
                }
                if (active) {
                    for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
                        int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
                        dvacc[x] += pj * gsh[ii * GD_SDPA_VARLEN_DH64 + c];
                        dkacc[x] += p.scale * ds * qsh[ii * GD_SDPA_VARLEN_DH64 + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (active) {
        int pbase = ((kg * int(p.hkv) + hkv) * int(p.n_splits) + split) *
                    2 * GD_SDPA_VARLEN_DH64;
        for (int x = 0; x < GD_SDPA_VARLEN_DH64_WIDE_CHANS; ++x) {
            int c = lane + x * int(GD_METAL_SDPA_DKV_WIDE_LANES);
            part[pbase + c] = dkacc[x];
            part[pbase + GD_SDPA_VARLEN_DH64 + c] = dvacc[x];
        }
    }
}

kernel void gd_sdpa_varlen_bwd_dkv_reduce_f16_kernel(
    device const float *part [[buffer(0)]],
    device half *dk [[buffer(1)]],
    device half *dv [[buffer(2)]],
    constant gd_metal_sdpa_varlen_args &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    int total = int(p.total_tokens) * int(p.hkv) * GD_SDPA_VARLEN_DH64;
    if (int(gid) >= total) {
        return;
    }
    int c = int(gid) % GD_SDPA_VARLEN_DH64;
    int row = int(gid) / GD_SDPA_VARLEN_DH64;
    int base = row * GD_SDPA_VARLEN_DH64 + c;
    int pbase = row * int(p.n_splits) * 2 * GD_SDPA_VARLEN_DH64;
    float adk = 0.0f;
    float adv = 0.0f;
    for (int s = 0; s < int(p.n_splits); ++s) {
        int off = pbase + s * 2 * GD_SDPA_VARLEN_DH64;
        adk += part[off + c];
        adv += part[off + GD_SDPA_VARLEN_DH64 + c];
    }
    dk[base] = half(adk);
    dv[base] = half(adv);
}
