#include <metal_stdlib>
using namespace metal;

/* Square GEMM perf probe: M,N,K all multiples of 64, no transpose/bias/batch,
 * so every block is interior (no bounds checks). C = A*B, A[M,K] B[K,N]. */

/* ---- fp32 register-blocked (our shipping kernel shape: 64x64 block, 16x16
 * threads, 4x4 float4 micro-tile, BK=8) ---- */
kernel void gemm_reg_f32(device const float *A   [[buffer(0)]],
                         device const float *B   [[buffer(1)]],
                         device float *C         [[buffer(2)]],
                         constant int3 &d        [[buffer(3)]],
                         uint3 tg [[threadgroup_position_in_grid]],
                         uint3 lid [[thread_position_in_threadgroup]])
{
    int Md = d.x, Nd = d.y, Kd = d.z;
    threadgroup float As[8][64];
    threadgroup float Bs[8][64];
    int tx = (int)lid.x, ty = (int)lid.y;
    int tid = ty * 16 + tx;
    int m0 = (int)tg.y * 64, n0 = (int)tg.x * 64;
    int row0 = m0 + ty * 4, col0 = n0 + tx * 4;
    float4 acc[4] = {float4(0), float4(0), float4(0), float4(0)};
    for (int t = 0; t < Kd; t += 8) {
        for (int e = tid; e < 8 * 64; e += 256) { int kr = e / 64, mr = e % 64; As[kr][mr] = A[(m0 + mr) * Kd + (t + kr)]; }
        for (int e = tid; e < 8 * 64; e += 256) { int kr = e / 64, nc = e % 64; Bs[kr][nc] = B[(t + kr) * Nd + (n0 + nc)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 8; ++kk) {
            float4 b = *(threadgroup float4 *)(&Bs[kk][tx * 4]);
            float4 a = *(threadgroup float4 *)(&As[kk][ty * 4]);
            acc[0] += a.x * b; acc[1] += a.y * b; acc[2] += a.z * b; acc[3] += a.w * b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4; ++i) {
        float4 r = acc[i];
        int gr = row0 + i;
        C[gr * Nd + col0 + 0] = r.x; C[gr * Nd + col0 + 1] = r.y;
        C[gr * Nd + col0 + 2] = r.z; C[gr * Nd + col0 + 3] = r.w;
    }
}

/* ---- fp32 simdgroup_matrix (64x64 block, 4 simdgroups 2x2, BK=8) ---- */
kernel void gemm_simd_f32(device const float *A [[buffer(0)]],
                          device const float *B [[buffer(1)]],
                          device float *C       [[buffer(2)]],
                          constant int3 &d      [[buffer(3)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint sg  [[simdgroup_index_in_threadgroup]])
{
    int Md = d.x, Nd = d.y, Kd = d.z;
    threadgroup float As[64 * 32];
    threadgroup float Bs[32 * 64];
    int m0 = (int)tg.y * 64, n0 = (int)tg.x * 64, sr = (int)sg / 2, sc = (int)sg % 2;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int t = 0; t < Kd; t += 32) {
        for (int e = (int)tid; e < 64 * 32; e += 128) { int mr = e / 32, kr = e % 32; As[mr * 32 + kr] = A[(m0 + mr) * Kd + (t + kr)]; }
        for (int e = (int)tid; e < 32 * 64; e += 128) { int kr = e / 64, nc = e % 64; Bs[kr * 64 + nc] = B[(t + kr) * Nd + (n0 + nc)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int ksub = 0; ksub < 32; ksub += 8) {
            simdgroup_float8x8 af[4], bf[4];
            for (int i = 0; i < 4; ++i) simdgroup_load(af[i], As + (sr * 32 + i * 8) * 32 + ksub, 32);
            for (int j = 0; j < 4; ++j) simdgroup_load(bf[j], Bs + ksub * 64 + (sc * 32 + j * 8), 64);
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *o = C + m0 * Nd + n0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        simdgroup_store(acc[i][j], o + (sr * 32 + i * 8) * Nd + (sc * 32 + j * 8), Nd);
}

/* ---- bf16 simdgroup_matrix (bf16 inputs, fp32 accumulate) ---- */
kernel void gemm_simd_bf16(device const bfloat *A [[buffer(0)]],
                           device const bfloat *B [[buffer(1)]],
                           device float *C        [[buffer(2)]],
                           constant int3 &d       [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint sg  [[simdgroup_index_in_threadgroup]])
{
    int Md = d.x, Nd = d.y, Kd = d.z;
    threadgroup bfloat As[64 * 32];
    threadgroup bfloat Bs[32 * 64];
    int m0 = (int)tg.y * 64, n0 = (int)tg.x * 64, sr = (int)sg / 2, sc = (int)sg % 2;
    simdgroup_float8x8 acc[4][4];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int t = 0; t < Kd; t += 32) {
        for (int e = (int)tid; e < 64 * 32; e += 128) { int mr = e / 32, kr = e % 32; As[mr * 32 + kr] = A[(m0 + mr) * Kd + (t + kr)]; }
        for (int e = (int)tid; e < 32 * 64; e += 128) { int kr = e / 64, nc = e % 64; Bs[kr * 64 + nc] = B[(t + kr) * Nd + (n0 + nc)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int ksub = 0; ksub < 32; ksub += 8) {
            simdgroup_bfloat8x8 af[4], bf[4];
            for (int i = 0; i < 4; ++i) simdgroup_load(af[i], As + (sr * 32 + i * 8) * 32 + ksub, 32);
            for (int j = 0; j < 4; ++j) simdgroup_load(bf[j], Bs + ksub * 64 + (sc * 32 + j * 8), 64);
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *o = C + m0 * Nd + n0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        simdgroup_store(acc[i][j], o + (sr * 32 + i * 8) * Nd + (sc * 32 + j * 8), Nd);
}
