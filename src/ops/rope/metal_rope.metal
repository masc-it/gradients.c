#include <metal_stdlib>
#include "metal_rope_types.h"

using namespace metal;

static inline float gd_rope_pair_angle(constant gd_metal_rope_args &args,
                                       uint pair,
                                       int position)
{
    return float(position) * exp2(float(pair) * args.freq_scale);
}

static inline void gd_rope_rotate_f16(device const half *x,
                                      device half *out,
                                      device const int *pos,
                                      constant gd_metal_rope_args &args,
                                      uint gid)
{
    const uint lane = gid % args.lanes_per_row;
    const uint row = gid / args.lanes_per_row;
    const uint half_dims = args.n_dims >> 1U;
    const uint tail = args.head_dim - args.n_dims;
    const ulong base = ulong(row) * ulong(args.head_dim);
    const int position = pos[row / args.heads];
    if (lane < half_dims) {
        const uint a = args.interleaved != 0U ? (lane << 1U) : lane;
        const uint b = args.interleaved != 0U ? ((lane << 1U) + 1U) : (lane + half_dims);
        const float angle = gd_rope_pair_angle(args, lane, position);
        const float c = cos(angle);
        const float s = sin(angle) * args.sin_sign;
        const float x0 = float(x[base + ulong(a)]);
        const float x1 = float(x[base + ulong(b)]);
        out[base + ulong(a)] = half(x0 * c - x1 * s);
        out[base + ulong(b)] = half(x0 * s + x1 * c);
    }
    if (lane < tail) {
        const ulong d = ulong(args.n_dims + lane);
        out[base + d] = x[base + d];
    }
}

static inline void gd_rope_rotate_f32(device const float *x,
                                      device float *out,
                                      device const int *pos,
                                      constant gd_metal_rope_args &args,
                                      uint gid)
{
    const uint lane = gid % args.lanes_per_row;
    const uint row = gid / args.lanes_per_row;
    const uint half_dims = args.n_dims >> 1U;
    const uint tail = args.head_dim - args.n_dims;
    const ulong base = ulong(row) * ulong(args.head_dim);
    const int position = pos[row / args.heads];
    if (lane < half_dims) {
        const uint a = args.interleaved != 0U ? (lane << 1U) : lane;
        const uint b = args.interleaved != 0U ? ((lane << 1U) + 1U) : (lane + half_dims);
        const float angle = gd_rope_pair_angle(args, lane, position);
        const float c = cos(angle);
        const float s = sin(angle) * args.sin_sign;
        const float x0 = x[base + ulong(a)];
        const float x1 = x[base + ulong(b)];
        out[base + ulong(a)] = x0 * c - x1 * s;
        out[base + ulong(b)] = x0 * s + x1 * c;
    }
    if (lane < tail) {
        const ulong d = ulong(args.n_dims + lane);
        out[base + d] = x[base + d];
    }
}

static inline void gd_rope_rotate_full_f16(device const half *x,
                                           device half *out,
                                           device const int *pos,
                                           constant gd_metal_rope_args &args,
                                           uint gid)
{
    const uint half_dims = args.n_dims >> 1U;
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    const uint a = args.interleaved != 0U ? (pair << 1U) : pair;
    const uint b = args.interleaved != 0U ? ((pair << 1U) + 1U) : (pair + half_dims);
    const int position = pos[token];
    const float angle = gd_rope_pair_angle(args, pair, position);
    const float c = cos(angle);
    const float s = sin(angle) * args.sin_sign;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong base = (ulong(token) * ulong(args.heads) + ulong(h)) * ulong(args.head_dim);
        const float x0 = float(x[base + ulong(a)]);
        const float x1 = float(x[base + ulong(b)]);
        out[base + ulong(a)] = half(x0 * c - x1 * s);
        out[base + ulong(b)] = half(x0 * s + x1 * c);
    }
}

static inline void gd_rope_rotate_full_f32(device const float *x,
                                           device float *out,
                                           device const int *pos,
                                           constant gd_metal_rope_args &args,
                                           uint gid)
{
    const uint half_dims = args.n_dims >> 1U;
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    const uint a = args.interleaved != 0U ? (pair << 1U) : pair;
    const uint b = args.interleaved != 0U ? ((pair << 1U) + 1U) : (pair + half_dims);
    const int position = pos[token];
    const float angle = gd_rope_pair_angle(args, pair, position);
    const float c = cos(angle);
    const float s = sin(angle) * args.sin_sign;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong base = (ulong(token) * ulong(args.heads) + ulong(h)) * ulong(args.head_dim);
        const float x0 = x[base + ulong(a)];
        const float x1 = x[base + ulong(b)];
        out[base + ulong(a)] = x0 * c - x1 * s;
        out[base + ulong(b)] = x0 * s + x1 * c;
    }
}

kernel void gd_rope_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                               device const uchar *posbuf [[buffer(1)]],
                               device uchar *outbuf [[buffer(2)]],
                               constant gd_metal_rope_args &args [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= args.rows * args.lanes_per_row) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    gd_rope_rotate_f16(x, out, pos, args, gid);
}

kernel void gd_rope_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                               device const uchar *posbuf [[buffer(1)]],
                               device uchar *outbuf [[buffer(2)]],
                               constant gd_metal_rope_args &args [[buffer(3)]],
                               uint gid [[thread_position_in_grid]])
{
    if (gid >= args.rows * args.lanes_per_row) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    gd_rope_rotate_f32(x, out, pos, args, gid);
}

kernel void gd_rope_full_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *posbuf [[buffer(1)]],
                                    device uchar *outbuf [[buffer(2)]],
                                    constant gd_metal_rope_args &args [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]])
{
    const uint leading_count = args.rows / args.heads;
    const uint half_dims = args.n_dims >> 1U;
    if (gid >= leading_count * half_dims) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    gd_rope_rotate_full_f16(x, out, pos, args, gid);
}

kernel void gd_rope_full_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *posbuf [[buffer(1)]],
                                    device uchar *outbuf [[buffer(2)]],
                                    constant gd_metal_rope_args &args [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]])
{
    const uint leading_count = args.rows / args.heads;
    const uint half_dims = args.n_dims >> 1U;
    if (gid >= leading_count * half_dims) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device float *out = reinterpret_cast<device float *>(outbuf + args.out_offset);
    gd_rope_rotate_full_f32(x, out, pos, args, gid);
}

kernel void gd_rope_backward_f16_kernel(device const uchar *gbuf [[buffer(0)]],
                                        device const uchar *posbuf [[buffer(1)]],
                                        device uchar *dxbuf [[buffer(2)]],
                                        constant gd_metal_rope_args &args [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]])
{
    if (gid >= args.rows * args.lanes_per_row) {
        return;
    }
    device const half *g = reinterpret_cast<device const half *>(gbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.out_offset);
    gd_rope_rotate_f16(g, dx, pos, args, gid);
}

kernel void gd_rope_backward_f32_kernel(device const uchar *gbuf [[buffer(0)]],
                                        device const uchar *posbuf [[buffer(1)]],
                                        device uchar *dxbuf [[buffer(2)]],
                                        constant gd_metal_rope_args &args [[buffer(3)]],
                                        uint gid [[thread_position_in_grid]])
{
    if (gid >= args.rows * args.lanes_per_row) {
        return;
    }
    device const float *g = reinterpret_cast<device const float *>(gbuf + args.x_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.out_offset);
    gd_rope_rotate_f32(g, dx, pos, args, gid);
}
