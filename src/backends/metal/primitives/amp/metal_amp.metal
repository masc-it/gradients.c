#include <metal_stdlib>
#include "metal_amp_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

kernel void gd_amp_unscale_kernel(device uchar *grad [[buffer(0)]],
                                  device uchar *found_inf_buf [[buffer(1)]],
                                  constant gd_metal_amp_unscale_args &args [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]])
{
    const ulong i = ulong(gid);
    if (i >= args.count) {
        return;
    }

    const ulong byte = args.grad_offset + i * gd_dtype_elem_size(args.grad_dtype);
    float g = gd_load_float_dtype(grad, byte, args.grad_dtype) * args.inv_scale;
    if (!isfinite(g)) {
        device atomic_uint *flag = reinterpret_cast<device atomic_uint *>(found_inf_buf + args.found_inf_offset);
        atomic_store_explicit(flag, 1u, memory_order_relaxed);
    }
    gd_write_float_dtype(grad, byte, args.grad_dtype, g);
}
