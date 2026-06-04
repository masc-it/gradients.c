#include <metal_stdlib>
#include "metal_kernel_types.h"

using namespace metal;

#define GD_METAL_UNROLL _Pragma("clang loop unroll(full)")

static inline void gd_write_pattern(device uchar *dst, ulong byte, uint elem_size, uint pattern)
{
    for (uint i = 0; i < elem_size; ++i) {
        dst[byte + i] = uchar((pattern >> (8u * i)) & 255u);
    }
}

static inline ulong gd_splitmix64(ulong x)
{
    x += 0x9E3779B97F4A7C15ul;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
    return x ^ (x >> 31);
}

static inline ushort gd_bf16_bits(float v)
{
    uint bits = as_type<uint>(v);
    uint lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return ushort(bits >> 16);
}

static inline void gd_write_float_dtype(device uchar *dst, ulong byte, uint dtype, float v)
{
    if (dtype == 1u) {
        ushort bits = as_type<ushort>(half(v));
        gd_write_pattern(dst, byte, 2u, uint(bits));
    } else if (dtype == 2u) {
        gd_write_pattern(dst, byte, 2u, uint(gd_bf16_bits(v)));
    } else {
        gd_write_pattern(dst, byte, 4u, as_type<uint>(v));
    }
}

kernel void gd_fill_kernel(device uchar *dst [[buffer(0)]],
                           constant gd_metal_fill_args &args [[buffer(1)]],
                           uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    gd_write_pattern(dst, args.byte_offset + i * ulong(args.elem_size), args.elem_size, args.pattern);
}

kernel void gd_rand_uniform_kernel(device uchar *dst [[buffer(0)]],
                                   constant gd_metal_rand_uniform_args &args [[buffer(1)]],
                                   uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    ulong r = gd_splitmix64(args.seed + i);
    uint mant = uint((r >> 40) & 0xfffffful);
    float u = float(mant) * (1.0f / 16777216.0f);
    float v = args.low + (args.high - args.low) * u;
    ulong elem_size = args.dtype == 3u ? 4ul : 2ul;
    gd_write_float_dtype(dst, args.byte_offset + i * elem_size, args.dtype, v);
}

static inline half gd_load_f16(device const uchar *buf, ulong byte)
{
    return *(reinterpret_cast<device const half *>(buf + byte));
}

static inline void gd_store_f16(device uchar *buf, ulong byte, half v)
{
    *(reinterpret_cast<device half *>(buf + byte)) = v;
}

static inline uint2 gd_simdgroup_matrix_thread_coords(uint simd_lane)
{
    const uint qid = simd_lane / 4u;
    const uint sm = (qid & 4u) + ((simd_lane / 2u) & 3u);
    const uint sn = ((qid & 2u) << 1u) + ((simd_lane & 1u) << 1u);
    return uint2(sm, sn);
}

static inline void gd_gemm_f16_reg_tile(device const uchar *xbuf,
                                        device const uchar *wbuf,
                                        device const uchar *bbuf,
                                        device uchar *ybuf,
                                        constant gd_metal_gemm_args &p,
                                        uint3 tgpos,
                                        uint simd_lane,
                                        uint simdgroup_id)
{
    const uint row0 = tgpos.y * GD_METAL_GEMM_REG_TILE;
    const uint col0 = (tgpos.x * GD_METAL_GEMM_REG_SIMDGROUPS + simdgroup_id) *
                      GD_METAL_GEMM_REG_TILE;
    if (col0 >= p.cols) {
        return;
    }

    const ulong x_stride = p.x_row_bytes / 2ul;
    const ulong w_stride = p.w_row_bytes / 2ul;
    const ulong y_stride = p.y_row_bytes / 2ul;
    device const half *x = reinterpret_cast<device const half *>(xbuf + p.x_offset);
    device const half *w = reinterpret_cast<device const half *>(wbuf + p.w_offset);
    device const half *bias = reinterpret_cast<device const half *>(bbuf + p.bias_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + p.y_offset);

    simdgroup_matrix<half, 8, 8> x_frag[GD_METAL_GEMM_REG_NBLK];
    simdgroup_matrix<half, 8, 8> w_frag[GD_METAL_GEMM_REG_NBLK];
    simdgroup_matrix<float, 8, 8> acc[GD_METAL_GEMM_REG_NBLK][GD_METAL_GEMM_REG_NBLK];

    GD_METAL_UNROLL
    for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            acc[mi][nj] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    for (uint k0 = 0; k0 < p.inner; k0 += 8u) {
        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            simdgroup_load(x_frag[mi],
                           x + ulong(row0 + mi * 8u) * x_stride + ulong(k0),
                           x_stride);
        }
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            simdgroup_load(w_frag[nj],
                           w + ulong(k0) * w_stride + ulong(col0 + nj * 8u),
                           w_stride);
        }

        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            GD_METAL_UNROLL
            for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
                const uint ns = (mi & 1u) != 0u ? (GD_METAL_GEMM_REG_NBLK - 1u - nj) : nj;
                simdgroup_multiply_accumulate(acc[mi][ns], x_frag[mi], w_frag[ns], acc[mi][ns]);
            }
        }
    }

    const uint2 elem = gd_simdgroup_matrix_thread_coords(simd_lane);
    GD_METAL_UNROLL
    for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
        const uint gr = row0 + mi * 8u + elem.x;
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            const uint gc = col0 + nj * 8u + elem.y;
            thread const auto &vals = acc[mi][nj].thread_elements();
            float v0 = vals[0];
            float v1 = vals[1];
            if (p.has_bias != 0u) {
                v0 += float(bias[gc]);
                v1 += float(bias[gc + 1u]);
            }
            y[ulong(gr) * y_stride + ulong(gc)] = half(v0);
            y[ulong(gr) * y_stride + ulong(gc + 1u)] = half(v1);
        }
    }
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

static inline void gd_simdgroup_load_f16_transpose_a(thread simdgroup_matrix<half, 8, 8> &frag,
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

static inline void gd_gemm_f16_reg_nt_tile(device const uchar *xbuf,
                                           device const uchar *wbuf,
                                           device uchar *ybuf,
                                           constant gd_metal_gemm_args &p,
                                           uint3 tgpos,
                                           uint simd_lane,
                                           uint simdgroup_id)
{
    const uint row0 = tgpos.y * GD_METAL_GEMM_REG_TILE;
    const uint col0 = (tgpos.x * GD_METAL_GEMM_REG_SIMDGROUPS + simdgroup_id) *
                      GD_METAL_GEMM_REG_TILE;
    if (col0 >= p.cols) {
        return;
    }

    const ulong x_stride = p.x_row_bytes / 2ul;
    const ulong w_stride = p.w_row_bytes / 2ul;
    const ulong y_stride = p.y_row_bytes / 2ul;
    device const half *x = reinterpret_cast<device const half *>(xbuf + p.x_offset);
    device const half *w = reinterpret_cast<device const half *>(wbuf + p.w_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + p.y_offset);

    simdgroup_matrix<half, 8, 8> x_frag[GD_METAL_GEMM_REG_NBLK];
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
    for (uint k0 = 0; k0 < p.inner; k0 += 8u) {
        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            simdgroup_load(x_frag[mi],
                           x + ulong(row0 + mi * 8u) * x_stride + ulong(k0),
                           x_stride);
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
                simdgroup_multiply_accumulate(acc[mi][ns], x_frag[mi], w_frag[ns], acc[mi][ns]);
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
            y[ulong(gr) * y_stride + ulong(gc)] = half(vals[0]);
            y[ulong(gr) * y_stride + ulong(gc + 1u)] = half(vals[1]);
        }
    }
}

static inline void gd_gemm_f16_reg_tn_tile(device const uchar *xbuf,
                                           device const uchar *wbuf,
                                           device uchar *ybuf,
                                           constant gd_metal_gemm_args &p,
                                           uint3 tgpos,
                                           uint simd_lane,
                                           uint simdgroup_id)
{
    const uint row0 = tgpos.y * GD_METAL_GEMM_REG_TILE;
    const uint col0 = (tgpos.x * GD_METAL_GEMM_REG_SIMDGROUPS + simdgroup_id) *
                      GD_METAL_GEMM_REG_TILE;
    if (col0 >= p.cols) {
        return;
    }

    const ulong x_stride = p.x_row_bytes / 2ul;
    const ulong w_stride = p.w_row_bytes / 2ul;
    const ulong y_stride = p.y_row_bytes / 2ul;
    device const half *x = reinterpret_cast<device const half *>(xbuf + p.x_offset);
    device const half *w = reinterpret_cast<device const half *>(wbuf + p.w_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + p.y_offset);

    simdgroup_matrix<half, 8, 8> x_frag[GD_METAL_GEMM_REG_NBLK];
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
    for (uint k0 = 0; k0 < p.inner; k0 += 8u) {
        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            gd_simdgroup_load_f16_transpose_a(x_frag[mi],
                                              x,
                                              x_stride,
                                              row0 + mi * 8u,
                                              k0,
                                              elem);
        }
        GD_METAL_UNROLL
        for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
            simdgroup_load(w_frag[nj],
                           w + ulong(k0) * w_stride + ulong(col0 + nj * 8u),
                           w_stride);
        }

        GD_METAL_UNROLL
        for (uint mi = 0; mi < GD_METAL_GEMM_REG_NBLK; ++mi) {
            GD_METAL_UNROLL
            for (uint nj = 0; nj < GD_METAL_GEMM_REG_NBLK; ++nj) {
                const uint ns = (mi & 1u) != 0u ? (GD_METAL_GEMM_REG_NBLK - 1u - nj) : nj;
                simdgroup_multiply_accumulate(acc[mi][ns], x_frag[mi], w_frag[ns], acc[mi][ns]);
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
            y[ulong(gr) * y_stride + ulong(gc)] = half(vals[0]);
            y[ulong(gr) * y_stride + ulong(gc + 1u)] = half(vals[1]);
        }
    }
}

static inline void gd_gemm_f16_tile(device const uchar *xbuf,
                                    device const uchar *wbuf,
                                    device const uchar *bbuf,
                                    device uchar *ybuf,
                                    constant gd_metal_gemm_args &p,
                                    threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM],
                                    threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN],
                                    uint3 tgpos,
                                    uint3 lid)
{
    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) *
                          (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);
    const uint tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    const uint m0 = tgpos.y * GD_METAL_GEMM_BM;
    const uint n0 = tgpos.x * GD_METAL_GEMM_BN;
    const uint row0 = m0 + ty * GD_METAL_GEMM_TM;
    const uint col0 = n0 + tx * GD_METAL_GEMM_TN;

    float4 acc[GD_METAL_GEMM_TM];
    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }

    const uint n_tiles = p.inner / GD_METAL_GEMM_BK + ((p.inner % GD_METAL_GEMM_BK) != 0u ? 1u : 0u);
    for (uint t = 0; t < n_tiles; ++t) {
        const uint kbase = t * GD_METAL_GEMM_BK;

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BM;
            const uint mr = e - kr * GD_METAL_GEMM_BM;
            const uint gr = m0 + mr;
            const uint gk = kbase + kr;
            half v = half(0.0);
            if (gr < p.rows && gk < p.inner) {
                v = gd_load_f16(xbuf, p.x_offset + ulong(gr) * p.x_row_bytes + ulong(gk) * 2ul);
            }
            xs[kr][mr] = v;
        }

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BN;
            const uint nc = e - kr * GD_METAL_GEMM_BN;
            const uint gk = kbase + kr;
            const uint gc = n0 + nc;
            half v = half(0.0);
            if (gc < p.cols && gk < p.inner) {
                v = gd_load_f16(wbuf, p.w_offset + ulong(gk) * p.w_row_bytes + ulong(gc) * 2ul);
            }
            ws[kr][nc] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            const float4 wreg = float4(*(reinterpret_cast<threadgroup half4 *>(&ws[kk][tx * GD_METAL_GEMM_TN])));
            const float4 xreg = float4(*(reinterpret_cast<threadgroup half4 *>(&xs[kk][ty * GD_METAL_GEMM_TM])));
            acc[0] += xreg.x * wreg;
            acc[1] += xreg.y * wreg;
            acc[2] += xreg.z * wreg;
            acc[3] += xreg.w * wreg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        const uint gr = row0 + tm;
        if (gr >= p.rows) {
            continue;
        }
        const float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (uint tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            const uint gc = col0 + tn;
            if (gc < p.cols) {
                float v = vals[tn];
                if (p.has_bias != 0u) {
                    v += float(gd_load_f16(bbuf, p.bias_offset + ulong(gc) * 2ul));
                }
                gd_store_f16(ybuf, p.y_offset + ulong(gr) * p.y_row_bytes + ulong(gc) * 2ul, half(v));
            }
        }
    }
}

static inline void gd_gemm_f16_nt_tile(device const uchar *xbuf,
                                       device const uchar *wbuf,
                                       device uchar *ybuf,
                                       constant gd_metal_gemm_args &p,
                                       threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM],
                                       threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN],
                                       uint3 tgpos,
                                       uint3 lid)
{
    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) *
                          (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);
    const uint tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    const uint m0 = tgpos.y * GD_METAL_GEMM_BM;
    const uint n0 = tgpos.x * GD_METAL_GEMM_BN;
    const uint row0 = m0 + ty * GD_METAL_GEMM_TM;
    const uint col0 = n0 + tx * GD_METAL_GEMM_TN;

    float4 acc[GD_METAL_GEMM_TM];
    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }

    const uint n_tiles = p.inner / GD_METAL_GEMM_BK + ((p.inner % GD_METAL_GEMM_BK) != 0u ? 1u : 0u);
    for (uint t = 0; t < n_tiles; ++t) {
        const uint kbase = t * GD_METAL_GEMM_BK;

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BM;
            const uint mr = e - kr * GD_METAL_GEMM_BM;
            const uint gr = m0 + mr;
            const uint gk = kbase + kr;
            half v = half(0.0);
            if (gr < p.rows && gk < p.inner) {
                v = gd_load_f16(xbuf, p.x_offset + ulong(gr) * p.x_row_bytes + ulong(gk) * 2ul);
            }
            xs[kr][mr] = v;
        }

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BN;
            const uint nc = e - kr * GD_METAL_GEMM_BN;
            const uint gk = kbase + kr;
            const uint gc = n0 + nc;
            half v = half(0.0);
            if (gc < p.cols && gk < p.inner) {
                v = gd_load_f16(wbuf, p.w_offset + ulong(gc) * p.w_row_bytes + ulong(gk) * 2ul);
            }
            ws[kr][nc] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            const float4 wreg = float4(*(reinterpret_cast<threadgroup half4 *>(&ws[kk][tx * GD_METAL_GEMM_TN])));
            const float4 xreg = float4(*(reinterpret_cast<threadgroup half4 *>(&xs[kk][ty * GD_METAL_GEMM_TM])));
            acc[0] += xreg.x * wreg;
            acc[1] += xreg.y * wreg;
            acc[2] += xreg.z * wreg;
            acc[3] += xreg.w * wreg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        const uint gr = row0 + tm;
        if (gr >= p.rows) {
            continue;
        }
        const float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (uint tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            const uint gc = col0 + tn;
            if (gc < p.cols) {
                gd_store_f16(ybuf, p.y_offset + ulong(gr) * p.y_row_bytes + ulong(gc) * 2ul, half(vals[tn]));
            }
        }
    }
}

static inline void gd_gemm_f16_tn_tile(device const uchar *xbuf,
                                       device const uchar *wbuf,
                                       device uchar *ybuf,
                                       constant gd_metal_gemm_args &p,
                                       threadgroup half xs[GD_METAL_GEMM_BK][GD_METAL_GEMM_BM],
                                       threadgroup half ws[GD_METAL_GEMM_BK][GD_METAL_GEMM_BN],
                                       uint3 tgpos,
                                       uint3 lid)
{
    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint nthreads = (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) *
                          (GD_METAL_GEMM_BM / GD_METAL_GEMM_TM);
    const uint tid = ty * (GD_METAL_GEMM_BN / GD_METAL_GEMM_TN) + tx;
    const uint m0 = tgpos.y * GD_METAL_GEMM_BM;
    const uint n0 = tgpos.x * GD_METAL_GEMM_BN;
    const uint row0 = m0 + ty * GD_METAL_GEMM_TM;
    const uint col0 = n0 + tx * GD_METAL_GEMM_TN;

    float4 acc[GD_METAL_GEMM_TM];
    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        acc[tm] = float4(0.0f);
    }

    const uint n_tiles = p.inner / GD_METAL_GEMM_BK + ((p.inner % GD_METAL_GEMM_BK) != 0u ? 1u : 0u);
    for (uint t = 0; t < n_tiles; ++t) {
        const uint kbase = t * GD_METAL_GEMM_BK;

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BM; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BM;
            const uint mr = e - kr * GD_METAL_GEMM_BM;
            const uint gr = m0 + mr;
            const uint gk = kbase + kr;
            half v = half(0.0);
            if (gr < p.rows && gk < p.inner) {
                v = gd_load_f16(xbuf, p.x_offset + ulong(gk) * p.x_row_bytes + ulong(gr) * 2ul);
            }
            xs[kr][mr] = v;
        }

        for (uint e = tid; e < GD_METAL_GEMM_BK * GD_METAL_GEMM_BN; e += nthreads) {
            const uint kr = e / GD_METAL_GEMM_BN;
            const uint nc = e - kr * GD_METAL_GEMM_BN;
            const uint gk = kbase + kr;
            const uint gc = n0 + nc;
            half v = half(0.0);
            if (gc < p.cols && gk < p.inner) {
                v = gd_load_f16(wbuf, p.w_offset + ulong(gk) * p.w_row_bytes + ulong(gc) * 2ul);
            }
            ws[kr][nc] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < GD_METAL_GEMM_BK; ++kk) {
            const float4 wreg = float4(*(reinterpret_cast<threadgroup half4 *>(&ws[kk][tx * GD_METAL_GEMM_TN])));
            const float4 xreg = float4(*(reinterpret_cast<threadgroup half4 *>(&xs[kk][ty * GD_METAL_GEMM_TM])));
            acc[0] += xreg.x * wreg;
            acc[1] += xreg.y * wreg;
            acc[2] += xreg.z * wreg;
            acc[3] += xreg.w * wreg;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint tm = 0; tm < GD_METAL_GEMM_TM; ++tm) {
        const uint gr = row0 + tm;
        if (gr >= p.rows) {
            continue;
        }
        const float vals[4] = {acc[tm].x, acc[tm].y, acc[tm].z, acc[tm].w};
        for (uint tn = 0; tn < GD_METAL_GEMM_TN; ++tn) {
            const uint gc = col0 + tn;
            if (gc < p.cols) {
                gd_store_f16(ybuf, p.y_offset + ulong(gr) * p.y_row_bytes + ulong(gc) * 2ul, half(vals[tn]));
            }
        }
    }
}

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
