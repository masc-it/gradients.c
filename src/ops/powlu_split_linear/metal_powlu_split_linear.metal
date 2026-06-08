#include <metal_stdlib>
#include "../_shared/gemm/metal_gemm_types.h"
#include "metal_powlu_split_linear_types.h"

using namespace metal;

#define GD_METAL_UNROLL _Pragma("clang loop unroll(full)")

static inline float gd_powlu_exp(const float x)
{
    return fast::exp(x);
}

static inline float gd_powlu_log(const float x)
{
    return fast::log(x);
}

static inline float gd_powlu_sqrt(const float x)
{
    return fast::sqrt(x);
}

static inline float gd_powlu_sigmoid_stable(const float x)
{
    if (x >= 0.0f) {
        const float e = gd_powlu_exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = gd_powlu_exp(x);
    return e / (1.0f + e);
}

static inline float2 gd_powlu_gate_and_grad(const float z, const float m)
{
    const float s = gd_powlu_sigmoid_stable(z);
    if (z <= 0.0f) {
        return float2(z * s, s * (1.0f + z * (1.0f - s)));
    }
    const float r = gd_powlu_sqrt(z);
    const float rp1 = r + 1.0f;
    const float a = m / rp1;
    const float lz = gd_powlu_log(max(z, 0x1p-126f));
    const float g = gd_powlu_exp(a * lz);
    const float da = -m / (2.0f * r * rp1 * rp1);
    const float gate = g * s;
    return float2(gate, gate * (a / z + da * lz + (1.0f - s)));
}

static inline uint2 gd_simdgroup_matrix_thread_coords(uint simd_lane)
{
    const uint qid = simd_lane / 4u;
    const uint sm = (qid & 4u) + ((simd_lane / 2u) & 3u);
    const uint sn = ((qid & 2u) << 1u) + ((simd_lane & 1u) << 1u);
    return uint2(sm, sn);
}

static inline void gd_simdgroup_load_f16_transpose_b(thread simdgroup_matrix<half, 8, 8> &frag,
                                                     device const half *src,
                                                     ulong src_stride,
                                                     uint row0,
                                                     uint col0,
                                                     uint2 elem)
{
    thread auto &vals = frag.thread_elements();
    vals[0] = src[ulong(col0 + elem.y) * src_stride + ulong(row0 + elem.x)];
    vals[1] = src[ulong(col0 + elem.y + 1u) * src_stride + ulong(row0 + elem.x)];
}

kernel void gd_powlu_split_linear_backward_x12_f16_reg_kernel(
    device const uchar *x12buf [[buffer(0)]],
    device const uchar *wbuf [[buffer(1)]],
    device const uchar *gradbuf [[buffer(2)]],
    device uchar *dx12buf [[buffer(3)]],
    constant gd_metal_powlu_split_linear_bwd_x12_args &p [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]])
{
    const uint row0 = tgpos.y * GD_METAL_GEMM_REG_TILE;
    const uint col0 = (tgpos.x * GD_METAL_GEMM_REG_SIMDGROUPS + simdgroup_id) *
                      GD_METAL_GEMM_REG_TILE;
    if (col0 >= p.hidden) {
        return;
    }

    const ulong grad_stride = p.grad_row_bytes / 2ul;
    const ulong w_stride = p.w_row_bytes / 2ul;
    const ulong x12_stride = p.x12_row_bytes / 2ul;
    const ulong dx12_stride = p.dx12_row_bytes / 2ul;
    device const half *grad = reinterpret_cast<device const half *>(gradbuf + p.grad_offset);
    device const half *w = reinterpret_cast<device const half *>(wbuf + p.w_offset);
    device const half *x12 = reinterpret_cast<device const half *>(x12buf + p.x12_offset);
    device half *dx12 = reinterpret_cast<device half *>(dx12buf + p.dx12_offset);

    simdgroup_matrix<half, 8, 8> grad_frag[GD_METAL_GEMM_REG_NBLK];
    simdgroup_matrix<half, 8, 8> w_frag[GD_METAL_GEMM_REG_NBLK];
    simdgroup_matrix<float, 8, 8> acc[GD_METAL_GEMM_REG_NBLK][GD_METAL_GEMM_REG_NBLK];

    GD_METAL_UNROLL
    for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint2 elem = gd_simdgroup_matrix_thread_coords(simd_lane);
    for (uint k0 = 0; k0 < p.out_cols; k0 += 8u) {
        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            simdgroup_load(grad_frag[mi],
                           grad + ulong(row0 + mi * 8u) * grad_stride + ulong(k0),
                           grad_stride);
        }
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            gd_simdgroup_load_f16_transpose_b(w_frag[nj],
                                              w,
                                              w_stride,
                                              k0,
                                              col0 + nj * 8u,
                                              elem);
        }

        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            GD_METAL_UNROLL
            for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
                const uint ns = (mi & 1u) != 0u ? (GD_METAL_GEMM_REG_NBLK - 1u - nj) : nj;
                simdgroup_multiply_accumulate(acc[mi][ns], grad_frag[mi], w_frag[ns], acc[mi][ns]);
            }
        }
    }

    GD_METAL_UNROLL
    for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
        const uint gr = row0 + mi * 8u + elem.x;
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            const uint gc = col0 + nj * 8u + elem.y;
            thread const auto &vals = acc[mi][nj].thread_elements();
            const float d0 = vals[0];
            const float d1 = vals[1];
            const ulong xrow = ulong(gr) * x12_stride;
            const ulong dxrow = ulong(gr) * dx12_stride;
            const float x10 = float(x12[xrow + ulong(gc)]);
            const float x20 = float(x12[xrow + ulong(p.hidden + gc)]);
            const float2 gg0 = gd_powlu_gate_and_grad(x20, p.m);
            dx12[dxrow + ulong(gc)] = half(d0 * gg0.x);
            dx12[dxrow + ulong(p.hidden + gc)] = half(d0 * x10 * gg0.y);
            {
                const uint gc1 = gc + 1u;
                const float x11 = float(x12[xrow + ulong(gc1)]);
                const float x21 = float(x12[xrow + ulong(p.hidden + gc1)]);
                const float2 gg1 = gd_powlu_gate_and_grad(x21, p.m);
                dx12[dxrow + ulong(gc1)] = half(d1 * gg1.x);
                dx12[dxrow + ulong(p.hidden + gc1)] = half(d1 * x11 * gg1.y);
            }
        }
    }
}
