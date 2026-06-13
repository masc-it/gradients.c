#include <metal_stdlib>
#include "metal_dropout_types.h"

using namespace metal;

static inline ulong gd_dropout_splitmix64(ulong x)
{
    x += 0x9E3779B97F4A7C15ul;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
    return x ^ (x >> 31);
}

static inline bool gd_dropout_keep(ulong seed, ulong index, uint threshold)
{
    const ulong r = gd_dropout_splitmix64(seed + index);
    const uint mant = uint((r >> 40) & 0xfffffful);
    return mant >= threshold;
}

static inline void gd_dropout_forward_f16_one(device const half *x,
                                              device half *y,
                                              device uchar *mask,
                                              constant gd_metal_dropout_args &args,
                                              ulong i)
{
    const bool keep = gd_dropout_keep(args.seed, i, args.threshold);
    mask[i] = keep ? uchar(1) : uchar(0);
    y[i] = keep ? half(float(x[i]) * args.scale) : half(0.0h);
}

static inline void gd_dropout_forward_f32_one(device const float *x,
                                              device float *y,
                                              device uchar *mask,
                                              constant gd_metal_dropout_args &args,
                                              ulong i)
{
    const bool keep = gd_dropout_keep(args.seed, i, args.threshold);
    mask[i] = keep ? uchar(1) : uchar(0);
    y[i] = keep ? x[i] * args.scale : 0.0f;
}

static inline void gd_dropout_backward_recompute_f16_one(device const half *g,
                                                         device half *dx,
                                                         constant gd_metal_dropout_args &args,
                                                         ulong i)
{
    const bool keep = gd_dropout_keep(args.seed, i, args.threshold);
    dx[i] = keep ? half(float(g[i]) * args.scale) : half(0.0h);
}

static inline void gd_dropout_backward_recompute_f32_one(device const float *g,
                                                         device float *dx,
                                                         constant gd_metal_dropout_args &args,
                                                         ulong i)
{
    const bool keep = gd_dropout_keep(args.seed, i, args.threshold);
    dx[i] = keep ? g[i] * args.scale : 0.0f;
}

static inline void gd_dropout_backward_mask_f16_one(device const uchar *mask,
                                                    device const half *g,
                                                    device half *dx,
                                                    constant gd_metal_dropout_args &args,
                                                    ulong i)
{
    dx[i] = mask[i] != uchar(0) ? half(float(g[i]) * args.scale) : half(0.0h);
}

static inline void gd_dropout_backward_mask_f32_one(device const uchar *mask,
                                                    device const float *g,
                                                    device float *dx,
                                                    constant gd_metal_dropout_args &args,
                                                    ulong i)
{
    dx[i] = mask[i] != uchar(0) ? g[i] * args.scale : 0.0f;
}

kernel void gd_dropout_forward_f16_kernel(device const uchar *xbuf [[buffer(0)]],
                                          device uchar *ybuf [[buffer(1)]],
                                          device uchar *maskbuf [[buffer(2)]],
                                          constant gd_metal_dropout_args &args [[buffer(3)]],
                                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + args.y_offset);
    device uchar *mask = maskbuf + args.mask_offset;
    if (base + 3ul < args.count) {
        gd_dropout_forward_f16_one(x, y, mask, args, base + 0ul);
        gd_dropout_forward_f16_one(x, y, mask, args, base + 1ul);
        gd_dropout_forward_f16_one(x, y, mask, args, base + 2ul);
        gd_dropout_forward_f16_one(x, y, mask, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_forward_f16_one(x, y, mask, args, i);
    }
}

kernel void gd_dropout_forward_f32_kernel(device const uchar *xbuf [[buffer(0)]],
                                          device uchar *ybuf [[buffer(1)]],
                                          device uchar *maskbuf [[buffer(2)]],
                                          constant gd_metal_dropout_args &args [[buffer(3)]],
                                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const float *x = reinterpret_cast<device const float *>(xbuf + args.x_offset);
    device float *y = reinterpret_cast<device float *>(ybuf + args.y_offset);
    device uchar *mask = maskbuf + args.mask_offset;
    if (base + 3ul < args.count) {
        gd_dropout_forward_f32_one(x, y, mask, args, base + 0ul);
        gd_dropout_forward_f32_one(x, y, mask, args, base + 1ul);
        gd_dropout_forward_f32_one(x, y, mask, args, base + 2ul);
        gd_dropout_forward_f32_one(x, y, mask, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_forward_f32_one(x, y, mask, args, i);
    }
}

static inline void gd_dropout_add_forward_f16_one(device const half *residual,
                                                  device const half *x,
                                                  device half *y,
                                                  device uchar *mask,
                                                  constant gd_metal_dropout_add_args &args,
                                                  ulong i)
{
    const bool keep = gd_dropout_keep(args.seed, i, args.threshold);
    const half dropped = keep ? half(float(x[i]) * args.scale) : half(0.0h);
    mask[i] = keep ? uchar(1) : uchar(0);
    y[i] = residual[i] + dropped;
}

kernel void gd_dropout_add_forward_f16_kernel(device const uchar *residualbuf [[buffer(0)]],
                                              device const uchar *xbuf [[buffer(1)]],
                                              device uchar *ybuf [[buffer(2)]],
                                              device uchar *maskbuf [[buffer(3)]],
                                              constant gd_metal_dropout_add_args &args [[buffer(4)]],
                                              uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *residual = reinterpret_cast<device const half *>(residualbuf + args.residual_offset);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device half *y = reinterpret_cast<device half *>(ybuf + args.y_offset);
    device uchar *mask = maskbuf + args.mask_offset;
    if (base + 3ul < args.count) {
        gd_dropout_add_forward_f16_one(residual, x, y, mask, args, base + 0ul);
        gd_dropout_add_forward_f16_one(residual, x, y, mask, args, base + 1ul);
        gd_dropout_add_forward_f16_one(residual, x, y, mask, args, base + 2ul);
        gd_dropout_add_forward_f16_one(residual, x, y, mask, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_add_forward_f16_one(residual, x, y, mask, args, i);
    }
}

kernel void gd_dropout_backward_recompute_f16_kernel(device const uchar *gbuf [[buffer(0)]],
                                                     device uchar *dxbuf [[buffer(1)]],
                                                     constant gd_metal_dropout_args &args [[buffer(2)]],
                                                     uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const half *g = reinterpret_cast<device const half *>(gbuf + args.x_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        gd_dropout_backward_recompute_f16_one(g, dx, args, base + 0ul);
        gd_dropout_backward_recompute_f16_one(g, dx, args, base + 1ul);
        gd_dropout_backward_recompute_f16_one(g, dx, args, base + 2ul);
        gd_dropout_backward_recompute_f16_one(g, dx, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_backward_recompute_f16_one(g, dx, args, i);
    }
}

kernel void gd_dropout_backward_recompute_f32_kernel(device const uchar *gbuf [[buffer(0)]],
                                                     device uchar *dxbuf [[buffer(1)]],
                                                     constant gd_metal_dropout_args &args [[buffer(2)]],
                                                     uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const float *g = reinterpret_cast<device const float *>(gbuf + args.x_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        gd_dropout_backward_recompute_f32_one(g, dx, args, base + 0ul);
        gd_dropout_backward_recompute_f32_one(g, dx, args, base + 1ul);
        gd_dropout_backward_recompute_f32_one(g, dx, args, base + 2ul);
        gd_dropout_backward_recompute_f32_one(g, dx, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_backward_recompute_f32_one(g, dx, args, i);
    }
}

kernel void gd_dropout_backward_mask_f16_kernel(device const uchar *maskbuf [[buffer(0)]],
                                                device const uchar *gbuf [[buffer(1)]],
                                                device uchar *dxbuf [[buffer(2)]],
                                                constant gd_metal_dropout_args &args [[buffer(3)]],
                                                uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const uchar *mask = maskbuf + args.mask_offset;
    device const half *g = reinterpret_cast<device const half *>(gbuf + args.x_offset);
    device half *dx = reinterpret_cast<device half *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        gd_dropout_backward_mask_f16_one(mask, g, dx, args, base + 0ul);
        gd_dropout_backward_mask_f16_one(mask, g, dx, args, base + 1ul);
        gd_dropout_backward_mask_f16_one(mask, g, dx, args, base + 2ul);
        gd_dropout_backward_mask_f16_one(mask, g, dx, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_backward_mask_f16_one(mask, g, dx, args, i);
    }
}

kernel void gd_dropout_backward_mask_f32_kernel(device const uchar *maskbuf [[buffer(0)]],
                                                device const uchar *gbuf [[buffer(1)]],
                                                device uchar *dxbuf [[buffer(2)]],
                                                constant gd_metal_dropout_args &args [[buffer(3)]],
                                                uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * ulong(GD_METAL_DROPOUT_ELEMENTS_PER_THREAD);
    if (base >= args.count) {
        return;
    }
    device const uchar *mask = maskbuf + args.mask_offset;
    device const float *g = reinterpret_cast<device const float *>(gbuf + args.x_offset);
    device float *dx = reinterpret_cast<device float *>(dxbuf + args.y_offset);
    if (base + 3ul < args.count) {
        gd_dropout_backward_mask_f32_one(mask, g, dx, args, base + 0ul);
        gd_dropout_backward_mask_f32_one(mask, g, dx, args, base + 1ul);
        gd_dropout_backward_mask_f32_one(mask, g, dx, args, base + 2ul);
        gd_dropout_backward_mask_f32_one(mask, g, dx, args, base + 3ul);
        return;
    }
    for (ulong i = base; i < args.count; ++i) {
        gd_dropout_backward_mask_f32_one(mask, g, dx, args, i);
    }
}
