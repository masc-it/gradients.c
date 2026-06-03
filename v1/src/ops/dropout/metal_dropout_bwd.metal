#include "metal_common.metal"

static inline uint gd_dropout_bwd_hash(uint seed_lo,
                                       uint seed_hi,
                                       uint run_lo,
                                       uint run_hi,
                                       uint index)
{
    uint h = seed_lo ^ (seed_hi * 0x9e3779b9u);

    h ^= run_lo * 0x85ebca6bu;
    h ^= run_hi * 0xc2b2ae35u;
    h ^= index * 0x27d4eb2du;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

static inline bool gd_dropout_bwd_keep(constant gd_metal_dropout_params &p, uint index)
{
    uint r = gd_dropout_bwd_hash(p.seed_lo, p.seed_hi, p.run_lo, p.run_hi, index);
    float u = ((float)(r >> 8) + 0.5f) * (1.0f / 16777216.0f);

    return u >= p.p;
}

kernel void gd_dropout_bwd(device const uchar *go                [[buffer(0)]],
                           device uchar *dx                      [[buffer(1)]],
                           constant gd_metal_dropout_params &p   [[buffer(2)]],
                           uint gid                              [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }
    float keep_scale = 1.0f / (1.0f - p.p);
    float v = gd_load_float(go, p.dtype, gid);
    float y = gd_dropout_bwd_keep(p, gid) ? v * keep_scale : 0.0f;
    gd_store_float(dx, p.dtype, gid, y);
}
