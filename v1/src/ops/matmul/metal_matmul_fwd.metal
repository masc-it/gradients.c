#include "metal_common.metal"

kernel void gd_matmul(device const float *a              [[buffer(0)]],
                      device const float *b              [[buffer(1)]],
                      device float *out                  [[buffer(2)]],
                      constant gd_metal_matmul_params &p  [[buffer(3)]],
                      uint2 gid                          [[thread_position_in_grid]])
{
    int batch_total = 1;
    for (int i = 0; i < p.batch_ndim; ++i) {
        batch_total *= p.out_batch_sizes[i];
    }
    int col = (int)gid.x;
    int grow = (int)gid.y; /* batch_lin * m + row */
    if (col >= p.n || grow >= batch_total * p.m) {
        return;
    }
    int batch_lin = grow / p.m;
    int row = grow % p.m;

    int bidx[GD_METAL_MAX_DIMS];
    int tmp = batch_lin;
    for (int i = p.batch_ndim - 1; i >= 0; --i) {
        bidx[i] = tmp % p.out_batch_sizes[i];
        tmp /= p.out_batch_sizes[i];
    }
    int a_base = 0, a_bstride = 1;
    for (int i = p.a_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.a_batch_ndim - i);
        int coord = (p.a_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        a_base += coord * a_bstride * p.a_mat;
        a_bstride *= p.a_batch_sizes[i];
    }
    int b_base = 0, b_bstride = 1;
    for (int i = p.b_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.b_batch_ndim - i);
        int coord = (p.b_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        b_base += coord * b_bstride * p.b_mat;
        b_bstride *= p.b_batch_sizes[i];
    }

    float acc = 0.0f;
    for (int kk = 0; kk < p.k; ++kk) {
        int a_off = p.trans_a ? (kk * p.a_cols + row) : (row * p.a_cols + kk);
        int b_off = p.trans_b ? (col * p.b_cols + kk) : (kk * p.b_cols + col);
        acc += a[a_base + a_off] * b[b_base + b_off];
    }
    out[batch_lin * p.out_mat + row * p.n + col] = acc;
}
kernel void gd_matmul_tiled(device const float *a              [[buffer(0)]],
                            device const float *b              [[buffer(1)]],
                            device float *out                  [[buffer(2)]],
                            constant gd_metal_matmul_params &p  [[buffer(3)]],
                            uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
{
    threadgroup float As[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup float Bs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    const int nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN)
                       * (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);

    int tx = (int)lid.x; /* 0 .. BN/TN */
    int ty = (int)lid.y; /* 0 .. BM/TM */
    int tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    int m0 = (int)tgpos.y * GD_METAL_GEMM_BM; /* first output row of block */
    int n0 = (int)tgpos.x * GD_METAL_GEMM_BN; /* first output col of block */
    int row0 = m0 + ty * GD_METAL_GEMM_TM;
    int col0 = n0 + tx * GD_METAL_GEMM_TN;
    int batch_lin = (int)tgpos.z;

    int bidx[GD_METAL_MAX_DIMS];
    int tmp = batch_lin;
    for (int i = p.batch_ndim - 1; i >= 0; --i) {
        bidx[i] = tmp % p.out_batch_sizes[i];
        tmp /= p.out_batch_sizes[i];
    }
    int a_base = 0, a_bstride = 1;
    for (int i = p.a_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.a_batch_ndim - i);
        int coord = (p.a_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        a_base += coord * a_bstride * p.a_mat;
        a_bstride *= p.a_batch_sizes[i];
    }
    int b_base = 0, b_bstride = 1;
    for (int i = p.b_batch_ndim - 1; i >= 0; --i) {
        int out_pos = p.batch_ndim - (p.b_batch_ndim - i);
        int coord = (p.b_batch_sizes[i] == 1) ? 0 : bidx[out_pos];
        b_base += coord * b_bstride * p.b_mat;
        b_bstride *= p.b_batch_sizes[i];
    }

    float4 acc[GD_METAL_GEMM_TM];
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }
    int n_tiles = (p.k + GD_METAL_GEMM_BK - 1) / GD_METAL_GEMM_BK;
    for (int t = 0; t < n_tiles; ++t) {
        int kbase = t * GD_METAL_GEMM_BK;
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BM;
            int mr = e % GD_METAL_GEMM_BM;
            int gr = m0 + mr;
            int gk = kbase + kr;
            float val = 0.0f;
            if (gr < p.m && gk < p.k) {
                int a_off = p.trans_a ? (gk * p.a_cols + gr) : (gr * p.a_cols + gk);
                val = a[a_base + a_off];
            }
            As[kr][mr] = val;
        }
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BN;
            int nc = e % GD_METAL_GEMM_BN;
            int gk = kbase + kr;
            int gc = n0 + nc;
            float val = 0.0f;
            if (gc < p.n && gk < p.k) {
                int b_off = p.trans_b ? (gc * p.b_cols + gk) : (gk * p.b_cols + gc);
                val = b[b_base + b_off];
            }
            Bs[kr][nc] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            float4 breg = *(threadgroup float4 *)(&Bs[kk][tx * GD_METAL_GEMM_TN]);
            float4 areg = *(threadgroup float4 *)(&As[kk][ty * GD_METAL_GEMM_TM]);
            acc[0] += areg.x * breg;
            acc[1] += areg.y * breg;
            acc[2] += areg.z * breg;
            acc[3] += areg.w * breg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        int gr = row0 + tm;
        if (gr >= p.m) {
            continue;
        }
        float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (int tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            int gc = col0 + tn;
            if (gc < p.n) {
                out[batch_lin * p.out_mat + gr * p.n + gc] = vals[tn];
            }
        }
    }
}
