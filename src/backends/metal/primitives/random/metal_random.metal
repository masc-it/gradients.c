#include <metal_stdlib>
#include "metal_random_types.h"

using namespace metal;

#include "backends/metal/metal_common.h"

static inline ulong gd_splitmix64(ulong x)
{
    x += 0x9E3779B97F4A7C15ul;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;
    return x ^ (x >> 31);
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
    ulong elem_size = gd_dtype_elem_size(args.dtype);
    gd_write_float_dtype(dst, args.byte_offset + i * elem_size, args.dtype, v);
}
