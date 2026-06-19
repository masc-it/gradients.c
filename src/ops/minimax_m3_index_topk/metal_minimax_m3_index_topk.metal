#include <metal_stdlib>
#include "../minimax_m3_sparse_attention/metal_minimax_m3_sparse_attention_types.h"

using namespace metal;

static inline float gd_m3_load_f16(device const uchar *buf, ulong byte_offset, ulong idx)
{
    return float(reinterpret_cast<device const half *>(buf + byte_offset)[idx]);
}

static inline int gd_m3_max_i(int a, int b)
{
    return a > b ? a : b;
}

static inline int gd_m3_min_i(int a, int b)
{
    return a < b ? a : b;
}

kernel void gd_minimax_m3_index_topk_kernel(
    device const uchar *qbuf [[buffer(0)]],
    device const uchar *kbuf [[buffer(1)]],
    device const uchar *cubuf [[buffer(2)]],
    device uchar *topkbuf [[buffer(3)]],
    constant gd_metal_minimax_m3_sparse_args &p [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    threadgroup float ksh[GD_METAL_MINIMAX_M3_DKV_QTILE * GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];

    device const int *cu = reinterpret_cast<device const int *>(cubuf + p.cu_offset);
    device int *topk = reinterpret_cast<device int *>(topkbuf + p.topk_offset);

    const int q0 = int(gid.x) * int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS);
    const int h = int(gid.y);
    const int b = int(gid.z);
    if (b >= int(p.batch) || h >= int(p.hkv) || p.dh > uint(GD_METAL_MINIMAX_M3_MAX_HEAD_DIM) ||
        p.block_size == 0U || p.block_size > uint(GD_METAL_MINIMAX_M3_MAX_BLOCK_SIZE) ||
        p.topk == 0U || p.topk > uint(GD_METAL_MINIMAX_M3_MAX_TOPK)) {
        return;
    }

    const int start = cu[b];
    const int T = cu[b + 1] - start;
    if (q0 >= T) {
        return;
    }

    const int row = int(tid);
    const int i = q0 + row;
    const bool active = i < T;
    const int qg = start + i;
    const int block_size = int(p.block_size);
    const int valid_blocks = active ? (i + block_size) / block_size : 0;
    const int max_i = gd_m3_min_i(q0 + int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS) - 1, T - 1);
    const int max_valid_blocks = (max_i + block_size) / block_size;
    const int local_start = gd_m3_max_i(0, valid_blocks - int(p.local_blocks));
    const ulong qbase = active ? ulong((qg * int(p.hkv) + h) * int(p.dh)) : 0ul;

    float qreg[GD_METAL_MINIMAX_M3_MAX_HEAD_DIM];
    for (int c = 0; c < int(p.dh); ++c) {
        qreg[c] = active ? gd_m3_load_f16(qbuf, p.q_offset, qbase + ulong(c)) : 0.0f;
    }

    int selected_idx[GD_METAL_MINIMAX_M3_MAX_TOPK];
    float selected_score[GD_METAL_MINIMAX_M3_MAX_TOPK];
    for (int t = 0; t < int(GD_METAL_MINIMAX_M3_MAX_TOPK); ++t) {
        selected_idx[t] = -1;
        selected_score[t] = -INFINITY;
    }

    for (int blk = 0; blk < max_valid_blocks; ++blk) {
        float block_score = -INFINITY;
        const int k_begin = blk * block_size;
        const int k_end = gd_m3_min_i(k_begin + block_size, T);
        for (int kb = k_begin; kb < k_end; kb += int(GD_METAL_MINIMAX_M3_DKV_QTILE)) {
            int tile = k_end - kb;
            if (tile > int(GD_METAL_MINIMAX_M3_DKV_QTILE)) {
                tile = int(GD_METAL_MINIMAX_M3_DKV_QTILE);
            }
            for (int idx = int(tid); idx < tile * int(p.dh); idx += int(GD_METAL_MINIMAX_M3_ATTENTION_THREADS)) {
                const int jj = idx / int(p.dh);
                const int c = idx - jj * int(p.dh);
                const int kg = start + kb + jj;
                const ulong kbase = ulong((kg * int(p.hkv) + h) * int(p.dh));
                ksh[jj * int(p.dh) + c] = gd_m3_load_f16(kbuf, p.k_offset, kbase + ulong(c));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (active && blk < valid_blocks) {
                for (int jj = 0; jj < tile; ++jj) {
                    const int j = kb + jj;
                    if (j <= i) {
                        const int koff = jj * int(p.dh);
                        float s = 0.0f;
                        for (int c = 0; c < int(p.dh); ++c) {
                            s += qreg[c] * ksh[koff + c];
                        }
                        block_score = s > block_score ? s : block_score;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (active && blk < valid_blocks) {
            if (!(block_score == block_score)) {
                block_score = -INFINITY;
            }
            /* Match the M3 top-k priority scheme: force init blocks highest,
             * local blocks next, then use index scores for the rest. */
            if (blk < int(p.init_blocks)) {
                block_score = 1.0e30f;
            }
            if (blk >= local_start && blk < valid_blocks && p.local_blocks > 0U) {
                block_score = 1.0e29f;
            }
            for (int slot = 0; slot < int(p.topk); ++slot) {
                if (block_score > selected_score[slot]) {
                    for (int move = int(p.topk) - 1; move > slot; --move) {
                        selected_score[move] = selected_score[move - 1];
                        selected_idx[move] = selected_idx[move - 1];
                    }
                    selected_score[slot] = block_score;
                    selected_idx[slot] = blk;
                    break;
                }
            }
        }
    }

    if (active) {
        const ulong out_base = (ulong(h) * p.total_tokens + ulong(qg)) * ulong(p.topk);
        for (int t = 0; t < int(p.topk); ++t) {
            topk[out_base + ulong(t)] = t < valid_blocks ? selected_idx[t] : -1;
        }
    }
}
