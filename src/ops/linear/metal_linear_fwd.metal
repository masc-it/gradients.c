#include "metal_common.metal"

kernel void gd_linear(device const float *x               [[buffer(0)]],
                      device const float *w               [[buffer(1)]],
                      device const float *bias            [[buffer(2)]],
                      device float *out                   [[buffer(3)]],
                      constant gd_metal_linear_params &p   [[buffer(4)]],
                      uint2 gid                           [[thread_position_in_grid]])
{
    int o = (int)gid.x;
    int r = (int)gid.y;
    if (o >= p.out_features || r >= p.rows) {
        return;
    }
    float acc = p.has_bias ? bias[o] : 0.0f;
    for (int kk = 0; kk < p.in_features; ++kk) {
        int w_off = p.trans_w ? (o * p.in_features + kk) : (kk * p.out_features + o);
        acc += x[r * p.in_features + kk] * w[w_off];
    }
    out[r * p.out_features + o] = acc;
}
kernel void gd_linear_tiled(device const float *x               [[buffer(0)]],
                            device const float *w               [[buffer(1)]],
                            device const float *bias            [[buffer(2)]],
                            device float *out                   [[buffer(3)]],
                            constant gd_metal_linear_params &p   [[buffer(4)]],
                            uint3 tgpos [[threadgroup_position_in_grid]],
                            uint3 lid   [[thread_position_in_threadgroup]])
{
    threadgroup float Xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup float Ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    const int nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN)
                       * (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);

    int tx = (int)lid.x;
    int ty = (int)lid.y;
    int tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    int r0 = (int)tgpos.y * GD_METAL_GEMM_BM; /* first row of block */
    int o0 = (int)tgpos.x * GD_METAL_GEMM_BN; /* first out-feature of block */
    int row0 = r0 + ty * GD_METAL_GEMM_TM;
    int col0 = o0 + tx * GD_METAL_GEMM_TN;

    float4 acc[GD_METAL_GEMM_TM];
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }
    int n_tiles = (p.in_features + GD_METAL_GEMM_BK - 1) / GD_METAL_GEMM_BK;
    for (int t = 0; t < n_tiles; ++t) {
        int kbase = t * GD_METAL_GEMM_BK;
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BM;
            int mr = e % GD_METAL_GEMM_BM;
            int gr = r0 + mr;
            int gk = kbase + kr;
            Xs[kr][mr] = (gr < p.rows && gk < p.in_features)
                             ? x[gr * p.in_features + gk] : 0.0f;
        }
        for (int e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            int kr = e / GD_METAL_GEMM_BN;
            int nc = e % GD_METAL_GEMM_BN;
            int gk = kbase + kr;
            int gc = o0 + nc;
            float val = 0.0f;
            if (gc < p.out_features && gk < p.in_features) {
                int w_off = p.trans_w ? (gc * p.in_features + gk) : (gk * p.out_features + gc);
                val = w[w_off];
            }
            Ws[kr][nc] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            float4 wreg = *(threadgroup float4 *)(&Ws[kk][tx * GD_METAL_GEMM_TN]);
            float4 xreg = *(threadgroup float4 *)(&Xs[kk][ty * GD_METAL_GEMM_TM]);
            acc[0] += xreg.x * wreg;
            acc[1] += xreg.y * wreg;
            acc[2] += xreg.z * wreg;
            acc[3] += xreg.w * wreg;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        int gr = row0 + tm;
        if (gr >= p.rows) {
            continue;
        }
        float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (int tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            int gc = col0 + tn;
            if (gc < p.out_features) {
                out[gr * p.out_features + gc] = (p.has_bias ? bias[gc] : 0.0f) + vals[tn];
            }
        }
    }
}
