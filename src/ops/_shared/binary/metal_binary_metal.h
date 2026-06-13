#ifndef GD_OPS_SHARED_BINARY_METAL_H
#define GD_OPS_SHARED_BINARY_METAL_H

#include "metal_binary_types.h"

using namespace metal;

static inline ulong gd_binary_row_bcast_offset(ulong row,
                                               ulong col,
                                               constant gd_metal_u64 *strides)
{
    return row * ulong(strides[0]) + col * ulong(strides[1]);
}

static inline ulong gd_binary_bcast_offset(ulong linear,
                                           constant gd_metal_binary_bcast_args &args,
                                           constant gd_metal_u64 *strides)
{
    ulong rem = linear;
    ulong offset = 0ul;
    for (int dim = int(args.rank) - 1; dim >= 0; --dim) {
        const ulong size = ulong(args.out_shape[dim]);
        const ulong coord = size > 1ul ? rem % size : 0ul;
        if (size > 1ul) {
            rem /= size;
        }
        offset += coord * ulong(strides[dim]);
    }
    return offset;
}

#endif /* GD_OPS_SHARED_BINARY_METAL_H */
