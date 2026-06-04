#include <metal_stdlib>
#include "metal_linear_types.h"

using namespace metal;

#include "../_shared/gemm/metal_gemm_common.h"

kernel void gd_linear_f16_reg(device const uchar *xbuf [[buffer(0)]],
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

kernel void gd_linear_f16_tiled(device const uchar *xbuf [[buffer(0)]],
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

kernel void gd_reduce_rows_f16(device const uchar *xbuf [[buffer(0)]],
                               device uchar *ybuf [[buffer(1)]],
                               constant gd_metal_reduce_rows_args &p [[buffer(2)]],
                               uint3 tgpos [[threadgroup_position_in_grid]],
                               uint simd_lane [[thread_index_in_simdgroup]],
                               uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const uint col = tgpos.x * GD_METAL_REDUCE_ROWS_SIMDGROUPS + simdgroup_id;
    if (col >= p.cols) {
        return;
    }
    float sum = 0.0f;
    for (uint row = simd_lane; row < p.rows; row += 32u) {
        sum += float(gd_load_f16(xbuf, p.x_offset + ulong(row) * p.x_row_bytes + ulong(col) * 2ul));
    }
    sum = simd_sum(sum);
    if (simd_lane == 0u) {
        gd_store_f16(ybuf, p.y_offset + ulong(col) * 2ul, half(sum));
    }
}
