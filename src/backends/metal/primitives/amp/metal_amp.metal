#include <metal_stdlib>
#include "metal_amp_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

#define GD_AMP_SCALE_INDEX 0u
#define GD_AMP_INV_SCALE_INDEX 1u
#define GD_AMP_GROWTH_TRACKER_INDEX 0u
#define GD_AMP_FOUND_INF_INDEX 1u
#define GD_AMP_LAST_FOUND_INF_INDEX 2u

static inline void gd_amp_mark_found_inf(device uchar *found_inf_buf, ulong found_inf_offset)
{
    device atomic_uint *flag = reinterpret_cast<device atomic_uint *>(found_inf_buf + found_inf_offset);
    atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

static inline float gd_amp_load_scale(device const uchar *scale_buf, ulong scale_offset)
{
    return reinterpret_cast<device const float *>(scale_buf + scale_offset)[GD_AMP_SCALE_INDEX];
}

static inline float gd_amp_load_inv_scale(device const uchar *scale_buf, ulong inv_scale_offset)
{
    return *(reinterpret_cast<device const float *>(scale_buf + inv_scale_offset));
}

static inline void gd_amp_store_dtype(device uchar *dst,
                                      ulong byte_offset,
                                      uint dtype,
                                      float value)
{
    if (dtype == 1u) {
        *(reinterpret_cast<device half *>(dst + byte_offset)) = half(value);
        return;
    }
    if (dtype == 3u) {
        *(reinterpret_cast<device float *>(dst + byte_offset)) = value;
        return;
    }
    gd_write_float_dtype(dst, byte_offset, dtype, value);
}

static inline float gd_amp_load_dtype(device const uchar *src,
                                      ulong byte_offset,
                                      uint dtype)
{
    if (dtype == 1u) {
        return float(*(reinterpret_cast<device const half *>(src + byte_offset)));
    }
    if (dtype == 3u) {
        return *(reinterpret_cast<device const float *>(src + byte_offset));
    }
    return gd_load_float_dtype(src, byte_offset, dtype);
}

kernel void gd_amp_begin_step_kernel(device uchar *scale_buf [[buffer(0)]],
                                     device uchar *flags_buf [[buffer(1)]],
                                     constant gd_metal_amp_state_args &args [[buffer(2)]])
{
    device float *scale = reinterpret_cast<device float *>(scale_buf + args.scale_offset);
    device uint *flags = reinterpret_cast<device uint *>(flags_buf + args.flags_offset);
    const float s = clamp(scale[GD_AMP_SCALE_INDEX], args.min_scale, args.max_scale);
    scale[GD_AMP_SCALE_INDEX] = s;
    scale[GD_AMP_INV_SCALE_INDEX] = 1.0f / s;
    flags[GD_AMP_FOUND_INF_INDEX] = 0u;
}

kernel void gd_amp_finish_step_kernel(device uchar *scale_buf [[buffer(0)]],
                                      device uchar *flags_buf [[buffer(1)]],
                                      constant gd_metal_amp_state_args &args [[buffer(2)]])
{
    device float *scale = reinterpret_cast<device float *>(scale_buf + args.scale_offset);
    device uint *flags = reinterpret_cast<device uint *>(flags_buf + args.flags_offset);
    const bool found_inf = flags[GD_AMP_FOUND_INF_INDEX] != 0u;
    float s = scale[GD_AMP_SCALE_INDEX];
    uint tracker = flags[GD_AMP_GROWTH_TRACKER_INDEX];

    flags[GD_AMP_LAST_FOUND_INF_INDEX] = found_inf ? 1u : 0u;
    if (found_inf) {
        s = max(s * args.backoff_factor, args.min_scale);
        tracker = 0u;
    } else {
        tracker += 1u;
        if (tracker >= args.growth_interval) {
            s = min(s * args.growth_factor, args.max_scale);
            tracker = 0u;
        }
    }
    scale[GD_AMP_SCALE_INDEX] = s;
    scale[GD_AMP_INV_SCALE_INDEX] = 1.0f / s;
    flags[GD_AMP_GROWTH_TRACKER_INDEX] = tracker;
}

kernel void gd_amp_fill_scale_kernel(device uchar *dst [[buffer(0)]],
                                     device const uchar *scale_buf [[buffer(1)]],
                                     constant gd_metal_amp_tensor_args &args [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    const ulong byte = args.dst_offset + i * gd_dtype_elem_size(args.dtype);
    gd_amp_store_dtype(dst, byte, args.dtype, gd_amp_load_scale(scale_buf, args.scale_offset));
}

kernel void gd_amp_scale_kernel(device uchar *dst [[buffer(0)]],
                                device const uchar *src [[buffer(1)]],
                                device const uchar *scale_buf [[buffer(2)]],
                                constant gd_metal_amp_tensor_args &args [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }
    const ulong elem_size = gd_dtype_elem_size(args.dtype);
    const ulong dst_byte = args.dst_offset + i * elem_size;
    const ulong src_byte = args.src_offset + i * elem_size;
    const float value = gd_amp_load_dtype(src, src_byte, args.dtype) *
                        gd_amp_load_scale(scale_buf, args.scale_offset);
    gd_amp_store_dtype(dst, dst_byte, args.dtype, value);
}

kernel void gd_amp_unscale_kernel(device uchar *grad [[buffer(0)]],
                                  device const uchar *scale_buf [[buffer(1)]],
                                  device uchar *found_inf_buf [[buffer(2)]],
                                  constant gd_metal_amp_unscale_args &args [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }

    const float inv_scale = gd_amp_load_inv_scale(scale_buf, args.inv_scale_offset);
    if (args.grad_dtype == 1u) {
        device half *grad_h = reinterpret_cast<device half *>(grad + args.grad_offset);
        const float g = float(grad_h[i]) * inv_scale;
        if (!isfinite(g)) {
            gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
        }
        grad_h[i] = half(g);
        return;
    }

    if (args.grad_dtype == 3u) {
        device float *grad_f = reinterpret_cast<device float *>(grad + args.grad_offset);
        const float g = grad_f[i] * inv_scale;
        if (!isfinite(g)) {
            gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
        }
        grad_f[i] = g;
        return;
    }

    const ulong byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    float g = gd_load_float_dtype(grad, byte, args.grad_dtype) * inv_scale;
    if (!isfinite(g)) {
        gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
    }
    gd_write_float_dtype(grad, byte, args.grad_dtype, g);
}
