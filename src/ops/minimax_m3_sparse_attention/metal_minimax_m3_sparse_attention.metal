#include <metal_stdlib>
#include "metal_minimax_m3_sparse_attention_types.h"

using namespace metal;

static inline float gd_m3s_load_f16(device const uchar *buf, ulong byte_offset, ulong idx)
{
    return float(reinterpret_cast<device const half *>(buf + byte_offset)[idx]);
}

static inline void gd_m3s_store_f16(device uchar *buf, ulong byte_offset, ulong idx, float value)
{
    reinterpret_cast<device half *>(buf + byte_offset)[idx] = half(value);
}

static inline int gd_m3s_min_i(int a, int b)
{
    return a < b ? a : b;
}

static inline bool gd_m3s_topk_contains(device const int *topk,
                                        constant gd_metal_minimax_m3_sparse_args &p,
                                        int hkv,
                                        int qg,
                                        int block_id)
{
    const ulong base = (ulong(hkv) * p.total_tokens + ulong(qg)) * ulong(p.topk);
    for (int t = 0; t < int(p.topk); ++t) {
        if (topk[base + ulong(t)] == block_id) {
            return true;
        }
    }
    return false;
}

kernel void gd_minimax_m3_sparse_attention_kernel(
    device const uchar *qbuf [[buffer(0)]],
    device const uchar *kbuf [[buffer(1)]],
    device const uchar *vbuf [[buffer(2)]],
    device const uchar *cubuf [[buffer(3)]],
    device const uchar *topkbuf [[buffer(4)]],
    device uchar *outbuf [[buffer(5)]],
    constant gd_metal_minimax_m3_sparse_args &p [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const int *topk = reinterpret_cast<device const int *>(topkbuf + p.topk_offset);

    const int q0 = int(gid.x) * int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS);
    const int hq = int(gid.y);
    const int b = int(gid.z);
    if (b >= int(p.batch) || hq >= int(p.hq) || p.dh > uint(GD_METAL_MINIMAX_M3_MAX_HEAD_DIM) ||
        p.block_size == 0U || p.block_size > uint(GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE) ||
        p.topk == 0U || p.topk > uint(GD_METAL_MINIMAX_M3_MAX_TOPK)) {
        return;
    }
    const int start = cu[b];
    const int T = cu[b + 1] - start;
    if (q0 >= T) {
        return;
    }

    const int i = q0 + int(tid);
    const bool active = i < T;
    const int group = int(p.hq) / int(p.hkv);
    const int hkv = hq / group;
    const int qg = start + i;
    const ulong qbase = active ? ulong((qg * int(p.hq) + hq) * int(p.dh)) : 0ul;

    float qreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float acc[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_m3s_load_f16(qbuf, p.q_offset, qbase + ulong(c)) : 0.0f;
        acc[c] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    if (active) {
        const ulong topk_base = (ulong(hkv) * p.total_tokens + ulong(qg)) * ulong(p.topk);
        for (int t = 0; t < int(p.topk); ++t) {
            const int blk = topk[topk_base + ulong(t)];
            if (blk < 0) {
                continue;
            }
            const int k_begin = blk * int(p.block_size);
            const int k_end = gd_m3s_min_i(k_begin + int(p.block_size), T);
            for (int j = k_begin; j < k_end && j <= i; ++j) {
                const int kg = start + j;
                const ulong kvbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
                float dot = 0.0f;
                for (int c = 0; c < int(p.dh); ++c) {
                    dot += qreg[c] * gd_m3s_load_f16(kbuf, p.k_offset, kvbase + ulong(c));
                }
                const float s = dot * p.scale;
                const float mnew = s > m ? s : m;
                const float corr = exp(m - mnew);
                const float e = exp(s - mnew);
                l = l * corr + e;
                for (int c = 0; c < int(p.dh); ++c) {
                    const float v_c = gd_m3s_load_f16(vbuf, p.v_offset, kvbase + ulong(c));
                    acc[c] = acc[c] * corr + e * v_c;
                }
                m = mnew;
            }
        }
        const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < int(p.dh); ++c) {
            gd_m3s_store_f16(outbuf, p.out_offset, qbase + ulong(c), acc[c] * inv_l);
        }
    }
}

kernel void gd_minimax_m3_sparse_attention_q4_kernel(
    device const uchar *qbuf [[buffer(0)]],
    device const uchar *kbuf [[buffer(1)]],
    device const uchar *vbuf [[buffer(2)]],
    device const uchar *cubuf [[buffer(3)]],
    device const uchar *topkbuf [[buffer(4)]],
    device uchar *outbuf [[buffer(5)]],
    constant gd_metal_minimax_m3_sparse_args &p [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float scratch[GD_METAL_MINIMAX_M3_QTILE_THREADS];
    threadgroup float sh_e[GD_METAL_MINIMAX_M3_QTILE];
    threadgroup float sh_corr[GD_METAL_MINIMAX_M3_QTILE];
    threadgroup float sh_inv_l[GD_METAL_MINIMAX_M3_QTILE];
    threadgroup float sh_k[GD_METAL_MINIMAX_M3_ATTENTION_THREADS];
    threadgroup float sh_v[GD_METAL_MINIMAX_M3_ATTENTION_THREADS];
    threadgroup int sh_selected[GD_METAL_MINIMAX_M3_QTILE];
    threadgroup int sh_blk;
    threadgroup int sh_process;

    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const int *topk = reinterpret_cast<device const int *>(topkbuf + p.topk_offset);

    const int q_tile = int(gid.x) * int(GD_METAL_MINIMAX_M3_QTILE);
    const int hq = int(gid.y);
    const int b = int(gid.z);
    const int row = int(tid / uint(GD_METAL_MINIMAX_M3_ATTENTION_THREADS));
    const int lane = int(tid - uint(row) * uint(GD_METAL_MINIMAX_M3_ATTENTION_THREADS));
    if (b >= int(p.batch) || hq >= int(p.hq) || p.dh > uint(GD_METAL_MINIMAX_M3_MAX_HEAD_DIM) ||
        p.block_size == 0U || p.block_size > uint(GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE) ||
        p.topk == 0U || p.topk > uint(GD_METAL_MINIMAX_M3_MAX_TOPK)) {
        return;
    }
    const int start = cu[b];
    const int T = cu[b + 1] - start;
    if (q_tile >= T) {
        return;
    }
    const int group = int(p.hq) / int(p.hkv);
    const int hkv = hq / group;
    const int i = q_tile + row;
    const bool active = row < int(GD_METAL_MINIMAX_M3_QTILE) && i < T;
    const int qg = start + i;
    const ulong qbase = active ? ulong((qg * int(p.hq) + hq) * int(p.dh)) : 0ul;
    const float q_c = (active && lane < int(p.dh)) ? gd_m3s_load_f16(qbuf, p.q_offset, qbase + ulong(lane)) : 0.0f;
    float acc_c = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int outer_row = 0; outer_row < int(GD_METAL_MINIMAX_M3_QTILE); ++outer_row) {
        for (int outer_t = 0; outer_t < int(p.topk); ++outer_t) {
            if (tid == 0U) {
                int candidate = -1;
                int process = 0;
                const int cand_i = q_tile + outer_row;
                if (cand_i < T) {
                    const int cand_qg = start + cand_i;
                    const ulong cand_base = (ulong(hkv) * p.total_tokens + ulong(cand_qg)) * ulong(p.topk);
                    candidate = topk[cand_base + ulong(outer_t)];
                    if (candidate >= 0) {
                        bool duplicate = false;
                        for (int prev_row = 0; prev_row <= outer_row; ++prev_row) {
                            const int prev_i = q_tile + prev_row;
                            if (prev_i >= T) {
                                continue;
                            }
                            const int prev_limit = prev_row == outer_row ? outer_t : int(p.topk);
                            const int prev_qg = start + prev_i;
                            const ulong prev_base = (ulong(hkv) * p.total_tokens + ulong(prev_qg)) * ulong(p.topk);
                            for (int prev_t = 0; prev_t < prev_limit; ++prev_t) {
                                if (topk[prev_base + ulong(prev_t)] == candidate) {
                                    duplicate = true;
                                }
                            }
                        }
                        process = duplicate ? 0 : 1;
                    }
                }
                sh_blk = candidate;
                sh_process = process;
                for (int rr = 0; rr < int(GD_METAL_MINIMAX_M3_QTILE); ++rr) {
                    const int rr_i = q_tile + rr;
                    const int rr_qg = start + rr_i;
                    sh_selected[rr] = (process != 0 && rr_i < T &&
                                       gd_m3s_topk_contains(topk, p, hkv, rr_qg, candidate)) ? 1 : 0;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sh_process == 0) {
                continue;
            }

            const int blk = sh_blk;
            const int k_begin = blk * int(p.block_size);
            const int k_end = gd_m3s_min_i(k_begin + int(p.block_size), T);
            for (int j = k_begin; j < k_end; ++j) {
                const int kg = start + j;
                if (tid < uint(GD_METAL_MINIMAX_M3_ATTENTION_THREADS)) {
                    const int c = int(tid);
                    const ulong kvbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
                    sh_k[c] = c < int(p.dh) ? gd_m3s_load_f16(kbuf, p.k_offset, kvbase + ulong(c)) : 0.0f;
                    sh_v[c] = c < int(p.dh) ? gd_m3s_load_f16(vbuf, p.v_offset, kvbase + ulong(c)) : 0.0f;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const bool visible = active && sh_selected[row] != 0 && j <= i;
                const int sbase = row * int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS);
                scratch[tid] = (visible && lane < int(p.dh)) ? q_c * sh_k[lane] : 0.0f;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = uint(GD_METAL_MINIMAX_M3_ATTENTION_THREADS) >> 1; stride > 0U; stride >>= 1) {
                    if (uint(lane) < stride) {
                        scratch[uint(sbase + lane)] += scratch[uint(sbase + lane) + stride];
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (lane == 0) {
                    if (visible) {
                        const float s = scratch[uint(sbase)] * p.scale;
                        const float mnew = s > m ? s : m;
                        const float corr = exp(m - mnew);
                        const float e = exp(s - mnew);
                        l = l * corr + e;
                        m = mnew;
                        sh_corr[row] = corr;
                        sh_e[row] = e;
                    } else {
                        sh_corr[row] = 1.0f;
                        sh_e[row] = 0.0f;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (active && lane < int(p.dh)) {
                    acc_c = acc_c * sh_corr[row] + sh_e[row] * sh_v[lane];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    if (lane == 0) {
        sh_inv_l[row] = l > 0.0f ? 1.0f / l : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active && lane < int(p.dh)) {
        gd_m3s_store_f16(outbuf, p.out_offset, qbase + ulong(lane), acc_c * sh_inv_l[row]);
    }
}

kernel void gd_minimax_m3_sparse_attention_bwd_dq_kernel(
    device const uchar *gobuf [[buffer(0)]],
    device const uchar *qbuf [[buffer(1)]],
    device const uchar *kbuf [[buffer(2)]],
    device const uchar *vbuf [[buffer(3)]],
    device const uchar *cubuf [[buffer(4)]],
    device const uchar *topkbuf [[buffer(5)]],
    device uchar *dqbuf [[buffer(6)]],
    device uchar *statsbuf [[buffer(7)]],
    constant gd_metal_minimax_m3_sparse_args &p [[buffer(8)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const int *topk = reinterpret_cast<device const int *>(topkbuf + p.topk_offset);
    device float *stats = reinterpret_cast<device float *>(statsbuf + p.stats_offset);

    const int q0 = int(gid.x) * int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS);
    const int hq = int(gid.y);
    const int b = int(gid.z);
    if (b >= int(p.batch) || hq >= int(p.hq) || p.dh > uint(GD_METAL_MINIMAX_M3_MAX_HEAD_DIM) ||
        p.block_size == 0U || p.block_size > uint(GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE) ||
        p.topk == 0U || p.topk > uint(GD_METAL_MINIMAX_M3_MAX_TOPK)) {
        return;
    }
    const int start = cu[b];
    const int T = cu[b + 1] - start;
    if (q0 >= T) {
        return;
    }

    const int i = q0 + int(tid);
    if (i >= T) {
        return;
    }
    const int group = int(p.hq) / int(p.hkv);
    const int hkv = hq / group;
    const int qg = start + i;
    const ulong qbase = ulong((qg * int(p.hq) + hq) * int(p.dh));

    float qreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float goreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float acc[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float ksum[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = gd_m3s_load_f16(qbuf, p.q_offset, qbase + ulong(c));
        goreg[c] = gd_m3s_load_f16(gobuf, p.grad_out_offset, qbase + ulong(c));
        acc[c] = 0.0f;
        ksum[c] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;
    float raw = 0.0f;
    const ulong topk_base = (ulong(hkv) * p.total_tokens + ulong(qg)) * ulong(p.topk);
    for (int t = 0; t < int(p.topk); ++t) {
        const int blk = topk[topk_base + ulong(t)];
        if (blk < 0) {
            continue;
        }
        const int k_begin = blk * int(p.block_size);
        const int k_end = gd_m3s_min_i(k_begin + int(p.block_size), T);
        for (int j = k_begin; j < k_end && j <= i; ++j) {
            const int kg = start + j;
            const ulong kvbase = ulong((kg * int(p.hkv) + hkv) * int(p.dh));
            float dot = 0.0f;
            float dp = 0.0f;
            for (int c = 0; c < int(p.dh); ++c) {
                const float k_c = gd_m3s_load_f16(kbuf, p.k_offset, kvbase + ulong(c));
                const float v_c = gd_m3s_load_f16(vbuf, p.v_offset, kvbase + ulong(c));
                dot += qreg[c] * k_c;
                dp += goreg[c] * v_c;
            }
            const float s = dot * p.scale;
            const float mnew = s > m ? s : m;
            const float corr = exp(m - mnew);
            const float e = exp(s - mnew);
            l = l * corr + e;
            raw = raw * corr + e * dp;
            for (int c = 0; c < int(p.dh); ++c) {
                const float k_c = gd_m3s_load_f16(kbuf, p.k_offset, kvbase + ulong(c));
                acc[c] = acc[c] * corr + e * dp * k_c;
                ksum[c] = ksum[c] * corr + e * k_c;
            }
            m = mnew;
        }
    }

    const float D = l > 0.0f ? raw / l : 0.0f;
    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    const ulong sbase = (ulong(qg) * ulong(p.hq) + ulong(hq)) * 3ul;
    stats[sbase + 0ul] = m;
    stats[sbase + 1ul] = l;
    stats[sbase + 2ul] = D;
    for (int c = 0; c < int(p.dh); ++c) {
        gd_m3s_store_f16(dqbuf,
                         p.grad_q_offset,
                         qbase + ulong(c),
                         p.scale * (acc[c] - D * ksum[c]) * inv_l);
    }
}

kernel void gd_minimax_m3_sparse_attention_bwd_dkv_kernel(
    device const uchar *gobuf [[buffer(0)]],
    device const uchar *qbuf [[buffer(1)]],
    device const uchar *kbuf [[buffer(2)]],
    device const uchar *vbuf [[buffer(3)]],
    device const uchar *cubuf [[buffer(4)]],
    device const uchar *topkbuf [[buffer(5)]],
    device uchar *dkbuf [[buffer(6)]],
    device uchar *dvbuf [[buffer(7)]],
    device const uchar *statsbuf [[buffer(8)]],
    constant gd_metal_minimax_m3_sparse_args &p [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float qsh[GD_METAL_MINIMAX_M3_DKV_QTILE * GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    threadgroup float gsh[GD_METAL_MINIMAX_M3_DKV_QTILE * GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    threadgroup float msh[GD_METAL_MINIMAX_M3_DKV_QTILE];
    threadgroup float lsh[GD_METAL_MINIMAX_M3_DKV_QTILE];
    threadgroup float dsh[GD_METAL_MINIMAX_M3_DKV_QTILE];

    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device const int *topk = reinterpret_cast<device const int *>(topkbuf + p.topk_offset);
    device const float *stats = reinterpret_cast<device const float *>(statsbuf + p.stats_offset);

    const int k0 = int(gid.x) * int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS);
    const int hkv = int(gid.y);
    const int b = int(gid.z);
    if (b >= int(p.batch) || hkv >= int(p.hkv) || p.dh > uint(GD_METAL_MINIMAX_M3_MAX_HEAD_DIM) ||
        p.block_size == 0U || p.block_size > uint(GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE) ||
        p.topk == 0U || p.topk > uint(GD_METAL_MINIMAX_M3_MAX_TOPK)) {
        return;
    }
    const int start = cu[b];
    const int T = cu[b + 1] - start;
    if (k0 >= T) {
        return;
    }

    const int j = k0 + int(tid);
    const bool active = j < T;
    const int kg = start + j;
    const int block_id = j / int(p.block_size);
    const int group = int(p.hq) / int(p.hkv);
    const ulong kvbase = active ? ulong((kg * int(p.hkv) + hkv) * int(p.dh)) : 0ul;

    float kreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float vreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float dkacc[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    float dvacc[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    for (int c = 0; c < int(p.dh); ++c) {
        kreg[c] = active ? gd_m3s_load_f16(kbuf, p.k_offset, kvbase + ulong(c)) : 0.0f;
        vreg[c] = active ? gd_m3s_load_f16(vbuf, p.v_offset, kvbase + ulong(c)) : 0.0f;
        dkacc[c] = 0.0f;
        dvacc[c] = 0.0f;
    }

    const int q_start = (k0 / int(GD_METAL_MINIMAX_M3_DKV_QTILE)) * int(GD_METAL_MINIMAX_M3_DKV_QTILE);
    for (int g = 0; g < group; ++g) {
        const int hq = hkv * group + g;
        for (int qb = q_start; qb < T; qb += int(GD_METAL_MINIMAX_M3_DKV_QTILE)) {
            int tile = T - qb;
            if (tile > int(GD_METAL_MINIMAX_M3_DKV_QTILE)) {
                tile = int(GD_METAL_MINIMAX_M3_DKV_QTILE);
            }
            for (int idx = int(tid); idx < tile * int(p.dh); idx += int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS)) {
                const int ii = idx / int(p.dh);
                const int c = idx - ii * int(p.dh);
                const int qg = start + qb + ii;
                const ulong qbase = ulong((qg * int(p.hq) + hq) * int(p.dh));
                qsh[ii * int(p.dh) + c] = gd_m3s_load_f16(qbuf, p.q_offset, qbase + ulong(c));
                gsh[ii * int(p.dh) + c] = gd_m3s_load_f16(gobuf, p.grad_out_offset, qbase + ulong(c));
            }
            for (int ii = int(tid); ii < tile; ii += int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS)) {
                const int qg = start + qb + ii;
                const ulong sbase = (ulong(qg) * ulong(p.hq) + ulong(hq)) * 3ul;
                msh[ii] = stats[sbase + 0ul];
                lsh[ii] = stats[sbase + 1ul];
                dsh[ii] = stats[sbase + 2ul];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (active) {
                for (int ii = 0; ii < tile; ++ii) {
                    const int i = qb + ii;
                    if (i < j || lsh[ii] <= 0.0f) {
                        continue;
                    }
                    const int qg = start + i;
                    if (!gd_m3s_topk_contains(topk, p, hkv, qg, block_id)) {
                        continue;
                    }
                    float dot = 0.0f;
                    float dp = 0.0f;
                    const int qoff = ii * int(p.dh);
                    for (int c = 0; c < int(p.dh); ++c) {
                        dot += qsh[qoff + c] * kreg[c];
                        dp += gsh[qoff + c] * vreg[c];
                    }
                    const float pj = exp(dot * p.scale - msh[ii]) / lsh[ii];
                    const float ds = pj * (dp - dsh[ii]);
                    for (int c = 0; c < int(p.dh); ++c) {
                        dvacc[c] += pj * gsh[qoff + c];
                        dkacc[c] += p.scale * ds * qsh[qoff + c];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (active) {
        for (int c = 0; c < int(p.dh); ++c) {
            gd_m3s_store_f16(dkbuf, p.grad_k_offset, kvbase + ulong(c), dkacc[c]);
            gd_m3s_store_f16(dvbuf, p.grad_v_offset, kvbase + ulong(c), dvacc[c]);
        }
    }
}
