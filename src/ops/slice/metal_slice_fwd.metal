#include "metal_common.metal"

static inline void gd_slice_copy_elem(device const uchar *src,
                                      device uchar *dst,
                                      int elem_size)
{
    if (elem_size == 4) {
        *((device uint *)dst) = *((device const uint *)src);
    } else if (elem_size == 2) {
        *((device ushort *)dst) = *((device const ushort *)src);
    } else if (elem_size == 8) {
        *((device ulong *)dst) = *((device const ulong *)src);
    } else {
        for (int b = 0; b < elem_size; ++b) {
            dst[b] = src[b];
        }
    }
}

kernel void gd_slice(device const uchar *x                         [[buffer(0)]],
                     device uchar *out                            [[buffer(1)]],
                     constant gd_metal_slice_params &p             [[buffer(2)]],
                     uint gid                                      [[thread_position_in_grid]])
{
    int out_lin = (int)gid;
    int tmp = out_lin;
    int in_off = 0;

    if (out_lin >= p.numel) {
        return;
    }
    for (int axis = p.ndim - 1; axis >= 0; --axis) {
        int coord = tmp % p.slice_sizes[axis];
        tmp /= p.slice_sizes[axis];
        if (axis == p.dim) {
            coord += p.start;
        }
        in_off += coord * p.in_strides[axis];
    }
    gd_slice_copy_elem(x + in_off * p.elem_size,
                       out + out_lin * p.elem_size,
                       p.elem_size);
}
