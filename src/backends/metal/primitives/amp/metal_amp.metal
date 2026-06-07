#include <metal_stdlib>
#include "metal_amp_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

static inline void gd_amp_mark_found_inf(device uchar *found_inf_buf, ulong found_inf_offset)
{
    device atomic_uint *flag = reinterpret_cast<device atomic_uint *>(found_inf_buf + found_inf_offset);
    atomic_store_explicit(flag, 1u, memory_order_relaxed);
}

kernel void gd_amp_unscale_kernel(device uchar *grad [[buffer(0)]],
                                  device uchar *found_inf_buf [[buffer(1)]],
                                  constant gd_metal_amp_unscale_args &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }

    if (args.grad_dtype == 1u) {
        device half *grad_h = reinterpret_cast<device half *>(grad + args.grad_offset);
        const float g = float(grad_h[i]) * args.inv_scale;
        if (!isfinite(g)) {
            gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
        }
        grad_h[i] = half(g);
        return;
    }

    if (args.grad_dtype == 3u) {
        device float *grad_f = reinterpret_cast<device float *>(grad + args.grad_offset);
        const float g = grad_f[i] * args.inv_scale;
        if (!isfinite(g)) {
            gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
        }
        grad_f[i] = g;
        return;
    }

    const ulong byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    float g = gd_load_float_dtype(grad, byte, args.grad_dtype) * args.inv_scale;
    if (!isfinite(g)) {
        gd_amp_mark_found_inf(found_inf_buf, args.found_inf_offset);
    }
    gd_write_float_dtype(grad, byte, args.grad_dtype, g);
}
