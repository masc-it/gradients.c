#include <metal_stdlib>
#include "metal_qkv_split_rope_types.h"

using namespace metal;

static inline float gd_qkv_split_rope_pair_angle(constant gd_metal_qkv_split_rope_args &args,
                                                 uint pair,
                                                 int position)
{
    return float(position) * exp2(float(pair) * args.freq_scale);
}

static inline void gd_qkv_split_rope_pair_indices(constant gd_metal_qkv_split_rope_args &args,
                                                  uint pair,
                                                  thread uint &a,
                                                  thread uint &b)
{
    if (args.interleaved != 0U) {
        a = pair << 1U;
        b = a + 1U;
    } else {
        a = pair;
        b = pair + (args.n_dims >> 1U);
    }
}

static inline void gd_qkv_split_rope_dynamic_cos_sin(constant gd_metal_qkv_split_rope_args &args,
                                                     uint pair,
                                                     int position,
                                                     thread float &c,
                                                     thread float &s)
{
    const float angle = gd_qkv_split_rope_pair_angle(args, pair, position);
    c = cos(angle);
    s = sin(angle) * args.sin_sign;
}

static inline void gd_qkv_split_rope_table_cos_sin(constant gd_metal_qkv_split_rope_args &args,
                                                   constant const float *table,
                                                   uint pair,
                                                   int position,
                                                   thread float &c,
                                                   thread float &s)
{
    if (position >= 0 && uint(position) < GD_METAL_QKV_SPLIT_ROPE_TABLE_POSITIONS &&
        pair < GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS) {
        const ulong idx = (ulong(uint(position)) * ulong(GD_METAL_QKV_SPLIT_ROPE_TABLE_PAIRS) +
                           ulong(pair)) *
                          2ul;
        c = table[idx];
        s = table[idx + 1ul] * args.sin_sign;
        return;
    }
    gd_qkv_split_rope_dynamic_cos_sin(args, pair, position, c, s);
}

kernel void gd_qkv_split_rope_forward_f16_kernel(device const uchar *qkvbuf [[buffer(0)]],
                                                 device const uchar *posbuf [[buffer(1)]],
                                                 device uchar *qbuf [[buffer(2)]],
                                                 device uchar *kbuf [[buffer(3)]],
                                                 device uchar *vbuf [[buffer(4)]],
                                                 constant gd_metal_qkv_split_rope_args &args [[buffer(5)]],
                                                 uint gid [[thread_position_in_grid]])
{
    const uint half_dims = args.n_dims >> 1U;
    const uint total = uint(args.tokens) * half_dims;
    if (gid >= total) {
        return;
    }
    device const half *qkv = reinterpret_cast<device const half *>(qkvbuf + args.qkv_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *q = reinterpret_cast<device half *>(qbuf + args.q_offset);
    device half *k = reinterpret_cast<device half *>(kbuf + args.k_offset);
    device half *v = reinterpret_cast<device half *>(vbuf + args.v_offset);
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    uint a;
    uint b;
    float c;
    float s;
    gd_qkv_split_rope_pair_indices(args, pair, a, b);
    gd_qkv_split_rope_dynamic_cos_sin(args, pair, pos[token], c, s);
    const ulong head_dim = ulong(args.head_dim);
    const ulong head_area = ulong(args.heads) * head_dim;
    const ulong qkv_token_base = ulong(token) * (head_area * 3ul);
    const ulong out_token_base = ulong(token) * head_area;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong head_base = ulong(h) * head_dim;
        const ulong src_q = qkv_token_base + head_base;
        const ulong src_k = qkv_token_base + head_area + head_base;
        const ulong src_v = qkv_token_base + head_area * 2ul + head_base;
        const ulong dst = out_token_base + head_base;
        const float q0 = float(qkv[src_q + ulong(a)]);
        const float q1 = float(qkv[src_q + ulong(b)]);
        const float k0 = float(qkv[src_k + ulong(a)]);
        const float k1 = float(qkv[src_k + ulong(b)]);
        q[dst + ulong(a)] = half(q0 * c - q1 * s);
        q[dst + ulong(b)] = half(q0 * s + q1 * c);
        k[dst + ulong(a)] = half(k0 * c - k1 * s);
        k[dst + ulong(b)] = half(k0 * s + k1 * c);
        v[dst + ulong(a)] = qkv[src_v + ulong(a)];
        v[dst + ulong(b)] = qkv[src_v + ulong(b)];
    }
}

kernel void gd_qkv_split_rope_backward_f16_kernel(device const uchar *gqbuf [[buffer(0)]],
                                                  device const uchar *gkbuf [[buffer(1)]],
                                                  device const uchar *gvbuf [[buffer(2)]],
                                                  device const uchar *posbuf [[buffer(3)]],
                                                  device uchar *dqkvbuf [[buffer(4)]],
                                                  constant gd_metal_qkv_split_rope_args &args [[buffer(5)]],
                                                  uint gid [[thread_position_in_grid]])
{
    const uint half_dims = args.n_dims >> 1U;
    const uint total = uint(args.tokens) * half_dims;
    if (gid >= total) {
        return;
    }
    device const half *gq = reinterpret_cast<device const half *>(gqbuf + args.q_offset);
    device const half *gk = reinterpret_cast<device const half *>(gkbuf + args.k_offset);
    device const half *gv = reinterpret_cast<device const half *>(gvbuf + args.v_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *dqkv = reinterpret_cast<device half *>(dqkvbuf + args.qkv_offset);
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    uint a;
    uint b;
    float c;
    float s;
    gd_qkv_split_rope_pair_indices(args, pair, a, b);
    gd_qkv_split_rope_dynamic_cos_sin(args, pair, pos[token], c, s);
    const ulong head_dim = ulong(args.head_dim);
    const ulong head_area = ulong(args.heads) * head_dim;
    const ulong qkv_token_base = ulong(token) * (head_area * 3ul);
    const ulong grad_token_base = ulong(token) * head_area;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong head_base = ulong(h) * head_dim;
        const ulong src = grad_token_base + head_base;
        const ulong dst_q = qkv_token_base + head_base;
        const ulong dst_k = qkv_token_base + head_area + head_base;
        const ulong dst_v = qkv_token_base + head_area * 2ul + head_base;
        const float q0 = float(gq[src + ulong(a)]);
        const float q1 = float(gq[src + ulong(b)]);
        const float k0 = float(gk[src + ulong(a)]);
        const float k1 = float(gk[src + ulong(b)]);
        dqkv[dst_q + ulong(a)] = half(q0 * c - q1 * s);
        dqkv[dst_q + ulong(b)] = half(q0 * s + q1 * c);
        dqkv[dst_k + ulong(a)] = half(k0 * c - k1 * s);
        dqkv[dst_k + ulong(b)] = half(k0 * s + k1 * c);
        dqkv[dst_v + ulong(a)] = gv[src + ulong(a)];
        dqkv[dst_v + ulong(b)] = gv[src + ulong(b)];
    }
}

kernel void gd_qkv_split_rope_forward_table_f16_kernel(device const uchar *qkvbuf [[buffer(0)]],
                                                       device const uchar *posbuf [[buffer(1)]],
                                                       device uchar *qbuf [[buffer(2)]],
                                                       device uchar *kbuf [[buffer(3)]],
                                                       device uchar *vbuf [[buffer(4)]],
                                                       constant gd_metal_qkv_split_rope_args &args [[buffer(5)]],
                                                       constant const float *table [[buffer(6)]],
                                                       uint gid [[thread_position_in_grid]])
{
    const uint half_dims = args.n_dims >> 1U;
    const uint total = uint(args.tokens) * half_dims;
    if (gid >= total) {
        return;
    }
    device const half *qkv = reinterpret_cast<device const half *>(qkvbuf + args.qkv_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *q = reinterpret_cast<device half *>(qbuf + args.q_offset);
    device half *k = reinterpret_cast<device half *>(kbuf + args.k_offset);
    device half *v = reinterpret_cast<device half *>(vbuf + args.v_offset);
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    uint a;
    uint b;
    float c;
    float s;
    gd_qkv_split_rope_pair_indices(args, pair, a, b);
    gd_qkv_split_rope_table_cos_sin(args, table, pair, pos[token], c, s);
    const ulong head_dim = ulong(args.head_dim);
    const ulong head_area = ulong(args.heads) * head_dim;
    const ulong qkv_token_base = ulong(token) * (head_area * 3ul);
    const ulong out_token_base = ulong(token) * head_area;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong head_base = ulong(h) * head_dim;
        const ulong src_q = qkv_token_base + head_base;
        const ulong src_k = qkv_token_base + head_area + head_base;
        const ulong src_v = qkv_token_base + head_area * 2ul + head_base;
        const ulong dst = out_token_base + head_base;
        const float q0 = float(qkv[src_q + ulong(a)]);
        const float q1 = float(qkv[src_q + ulong(b)]);
        const float k0 = float(qkv[src_k + ulong(a)]);
        const float k1 = float(qkv[src_k + ulong(b)]);
        q[dst + ulong(a)] = half(q0 * c - q1 * s);
        q[dst + ulong(b)] = half(q0 * s + q1 * c);
        k[dst + ulong(a)] = half(k0 * c - k1 * s);
        k[dst + ulong(b)] = half(k0 * s + k1 * c);
        v[dst + ulong(a)] = qkv[src_v + ulong(a)];
        v[dst + ulong(b)] = qkv[src_v + ulong(b)];
    }
}

kernel void gd_qkv_split_rope_backward_table_f16_kernel(device const uchar *gqbuf [[buffer(0)]],
                                                        device const uchar *gkbuf [[buffer(1)]],
                                                        device const uchar *gvbuf [[buffer(2)]],
                                                        device const uchar *posbuf [[buffer(3)]],
                                                        device uchar *dqkvbuf [[buffer(4)]],
                                                        constant gd_metal_qkv_split_rope_args &args [[buffer(5)]],
                                                        constant const float *table [[buffer(6)]],
                                                        uint gid [[thread_position_in_grid]])
{
    const uint half_dims = args.n_dims >> 1U;
    const uint total = uint(args.tokens) * half_dims;
    if (gid >= total) {
        return;
    }
    device const half *gq = reinterpret_cast<device const half *>(gqbuf + args.q_offset);
    device const half *gk = reinterpret_cast<device const half *>(gkbuf + args.k_offset);
    device const half *gv = reinterpret_cast<device const half *>(gvbuf + args.v_offset);
    device const int *pos = reinterpret_cast<device const int *>(posbuf + args.pos_offset);
    device half *dqkv = reinterpret_cast<device half *>(dqkvbuf + args.qkv_offset);
    const uint pair = gid % half_dims;
    const uint token = gid / half_dims;
    uint a;
    uint b;
    float c;
    float s;
    gd_qkv_split_rope_pair_indices(args, pair, a, b);
    gd_qkv_split_rope_table_cos_sin(args, table, pair, pos[token], c, s);
    const ulong head_dim = ulong(args.head_dim);
    const ulong head_area = ulong(args.heads) * head_dim;
    const ulong qkv_token_base = ulong(token) * (head_area * 3ul);
    const ulong grad_token_base = ulong(token) * head_area;
    for (uint h = 0U; h < args.heads; ++h) {
        const ulong head_base = ulong(h) * head_dim;
        const ulong src = grad_token_base + head_base;
        const ulong dst_q = qkv_token_base + head_base;
        const ulong dst_k = qkv_token_base + head_area + head_base;
        const ulong dst_v = qkv_token_base + head_area * 2ul + head_base;
        const float q0 = float(gq[src + ulong(a)]);
        const float q1 = float(gq[src + ulong(b)]);
        const float k0 = float(gk[src + ulong(a)]);
        const float k1 = float(gk[src + ulong(b)]);
        dqkv[dst_q + ulong(a)] = half(q0 * c - q1 * s);
        dqkv[dst_q + ulong(b)] = half(q0 * s + q1 * c);
        dqkv[dst_k + ulong(a)] = half(k0 * c - k1 * s);
        dqkv[dst_k + ulong(b)] = half(k0 * s + k1 * c);
        dqkv[dst_v + ulong(a)] = gv[src + ulong(a)];
        dqkv[dst_v + ulong(b)] = gv[src + ulong(b)];
    }
}
