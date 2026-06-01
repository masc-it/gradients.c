#ifndef GD_METAL_COMMON_METAL
#define GD_METAL_COMMON_METAL

#include <metal_stdlib>
#include "metal_kernel_types.h"

using namespace metal;

#define GD_CLIP_NORM_TG 256
#define GD_CLIP_NORM_ITEMS 4
#define GD_RMS_TG 256
#define GD_CE_TG 256
#define GD_SDPA_BQ GD_METAL_SDPA_BQ
#define GD_SDPA_BK GD_METAL_SDPA_BK
#define GD_SDPA_DHT GD_METAL_SDPA_DHT
#define GD_SDPA_DKV_KEYS GD_METAL_SDPA_DKV_KEYS
#define GD_SDPA_DKV_LANES (GD_SDPA_BQ / GD_SDPA_DKV_KEYS)
#define GD_SDPA_DKV_CMAX ((GD_SDPA_DHT + GD_SDPA_DKV_LANES - 1) / GD_SDPA_DKV_LANES)
#define GD_SDPA_DKV_WIDE_KEYS GD_METAL_SDPA_DKV_WIDE_KEYS
#define GD_SDPA_DKV_WIDE_LANES GD_METAL_SDPA_DKV_WIDE_LANES
#define GD_SDPA_DKV_WIDE_THREADS GD_METAL_SDPA_DKV_WIDE_THREADS
#define GD_SDPA_CAUSAL_QROWS GD_METAL_SDPA_CAUSAL_QROWS
#define GD_SDPA_CAUSAL_THREADS GD_METAL_SDPA_CAUSAL_THREADS

static __attribute__((unused)) int gd_broadcast_offset(thread const int *out_index,
                                                       int out_ndim,
                                                       constant int *in_sizes,
                                                       int in_ndim)
{
    int stride = 1;
    int offset = 0;
    for (int i = in_ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_ndim - i);
        int coord = (in_sizes[i] == 1) ? 0 : out_index[out_pos];
        offset += coord * stride;
        stride *= in_sizes[i];
    }
    return offset;
}

static inline __attribute__((unused)) float gd_load_float(device const uchar *x,
                                                           int dtype,
                                                           uint idx)
{
    if (dtype == GD_METAL_DT_F16) {
        return (float)((device const half *)x)[idx];
    }
    return ((device const float *)x)[idx];
}

static inline __attribute__((unused)) void gd_store_float(device uchar *out,
                                                          int dtype,
                                                          uint idx,
                                                          float value)
{
    if (dtype == GD_METAL_DT_F16) {
        ((device half *)out)[idx] = (half)value;
    } else {
        ((device float *)out)[idx] = value;
    }
}

static inline __attribute__((unused)) void gd_binary(device const uchar *a,
                                                     device const uchar *b,
                                                     device uchar *out,
                                                     constant gd_metal_ew_params &p,
                                                     uint gid,
                                                     int op)
{
    if ((int)gid >= p.numel) {
        return;
    }
    if (p.same_shape) {
        float av = gd_load_float(a, p.dtype, gid);
        float bv = gd_load_float(b, p.dtype, gid);
        gd_store_float(out, p.dtype, gid, (op == 0) ? (av + bv) : (av * bv));
        return;
    }
    int index[GD_METAL_MAX_DIMS];
    int lin = (int)gid;
    for (int i = p.ndim - 1; i >= 0; --i) {
        index[i] = lin % p.out_sizes[i];
        lin /= p.out_sizes[i];
    }
    int ao = gd_broadcast_offset(index, p.ndim, p.a_sizes, p.a_ndim);
    int bo = gd_broadcast_offset(index, p.ndim, p.b_sizes, p.b_ndim);
    float av = gd_load_float(a, p.dtype, (uint)ao);
    float bv = gd_load_float(b, p.dtype, (uint)bo);
    gd_store_float(out, p.dtype, gid, (op == 0) ? (av + bv) : (av * bv));
}

static inline __attribute__((unused)) float gd_sigmoid_stable(float x)
{
    if (x >= 0.0f) {
        float e = exp(-x);
        return 1.0f / (1.0f + e);
    }
    float e = exp(x);
    return e / (1.0f + e);
}

static inline __attribute__((unused)) float gd_erff(float x)
{
    float s = x < 0.0f ? -1.0f : 1.0f;
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                       - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

#endif /* GD_METAL_COMMON_METAL */
