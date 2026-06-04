#include <metal_stdlib>
#include "metal_matmul_types.h"

using namespace metal;

#include "../_shared/gemm/metal_gemm_common.h"

kernel void gd_matmul_f16_reg(device const uchar *xbuf [[buffer(0)]],
                               device const uchar *wbuf [[buffer(1)]],
                               device const uchar *bbuf [[buffer(2)]],
                               device uchar *ybuf [[buffer(3)]],
                               constant gd_metal_gemm_args &p [[buffer(4)]],
                               uint3 tgpos [[threadgroup_position_in_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    gd_gemm_f16_reg_tile(xbuf, wbuf, bbuf, ybuf, p, tgpos, simd_lane, simdgroup_id);
}

kernel void gd_matmul_f16_nt_reg(device const uchar *xbuf [[buffer(0)]],
                                 device const uchar *wbuf [[buffer(1)]],
                                 device const uchar *bbuf [[buffer(2)]],
                                 device uchar *ybuf [[buffer(3)]],
                                 constant gd_metal_gemm_args &p [[buffer(4)]],
                                 uint3 tgpos [[threadgroup_position_in_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    (void)bbuf;
    gd_gemm_f16_reg_nt_tile(xbuf, wbuf, ybuf, p, tgpos, simd_lane, simdgroup_id);
}

kernel void gd_matmul_f16_tn_reg(device const uchar *xbuf [[buffer(0)]],
                                 device const uchar *wbuf [[buffer(1)]],
                                 device const uchar *bbuf [[buffer(2)]],
                                 device uchar *ybuf [[buffer(3)]],
                                 constant gd_metal_gemm_args &p [[buffer(4)]],
                                 uint3 tgpos [[threadgroup_position_in_grid]],
                                 uint simd_lane [[thread_index_in_simdgroup]],
                                 uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    (void)bbuf;
    gd_gemm_f16_reg_tn_tile(xbuf, wbuf, ybuf, p, tgpos, simd_lane, simdgroup_id);
}

kernel void gd_matmul_f16_tiled(device const uchar *xbuf [[buffer(0)]],
                                device const uchar *wbuf [[buffer(1)]],
                                device const uchar *bbuf [[buffer(2)]],
                                device uchar *ybuf [[buffer(3)]],
                                constant gd_metal_gemm_args &p [[buffer(4)]],
                                uint3 tgpos [[threadgroup_position_in_grid]],
                                uint3 lid [[thread_position_in_threadgroup]])
{
    threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    gd_gemm_f16_tile(xbuf, wbuf, bbuf, ybuf, p, xs, ws, tgpos, lid);
}

kernel void gd_matmul_f16_nt_tiled(device const uchar *xbuf [[buffer(0)]],
                                   device const uchar *wbuf [[buffer(1)]],
                                   device const uchar *bbuf [[buffer(2)]],
                                   device uchar *ybuf [[buffer(3)]],
                                   constant gd_metal_gemm_args &p [[buffer(4)]],
                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                   uint3 lid [[thread_position_in_threadgroup]])
{
    (void)bbuf;
    threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    gd_gemm_f16_nt_tile(xbuf, wbuf, ybuf, p, xs, ws, tgpos, lid);
}

kernel void gd_matmul_f16_tn_tiled(device const uchar *xbuf [[buffer(0)]],
                                   device const uchar *wbuf [[buffer(1)]],
                                   device const uchar *bbuf [[buffer(2)]],
                                   device uchar *ybuf [[buffer(3)]],
                                   constant gd_metal_gemm_args &p [[buffer(4)]],
                                   uint3 tgpos [[threadgroup_position_in_grid]],
                                   uint3 lid [[thread_position_in_threadgroup]])
{
    (void)bbuf;
    threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM];
    threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN];
    gd_gemm_f16_tn_tile(xbuf, wbuf, ybuf, p, xs, ws, tgpos, lid);
}
