#include "metal_common.metal"

static inline void gd_slice_bwd_copy_or_zero(device const uchar *src,
                                             device uchar *dst,
                                             int elem_size,
                                             bool copy_src)
{
    if (elem_size == 4) {
        *((device uint *)dst) = copy_src ? *((device const uint *)src) : 0u;
    } else if (elem_size == 2) {
        *((device ushort *)dst) = copy_src ? *((device const ushort *)src) : (ushort)0;
    } else if (elem_size == 8) {
        *((device ulong *)dst) = copy_src ? *((device const ulong *)src) : (ulong)0;
    } else {
        for (int b = 0; b < elem_size; ++b) {
            dst[b] = copy_src ? src[b] : (uchar)0;
        }
    }
}

kernel void gd_slice_bwd(device const uchar *go                    [[buffer(0)]],
                         device uchar *dx                         [[buffer(1)]],
                         constant gd_metal_slice_params &p         [[buffer(2)]],
                         uint gid                                  [[thread_position_in_grid]])
{
    int full_lin = (int)gid;
    int tmp = full_lin;
    int slice_off = 0;
    int stride = 1;
    bool inside = true;

    if (full_lin >= p.numel) {
        return;
    }
    for (int axis = p.ndim - 1; axis >= 0; --axis) {
        int coord = tmp % p.full_sizes[axis];
        int slice_coord = coord;
        tmp /= p.full_sizes[axis];
        if (axis == p.dim) {
            if (coord < p.start || coord >= p.start + p.len) {
                inside = false;
            }
            slice_coord = coord - p.start;
        }
        if (inside) {
            slice_off += slice_coord * stride;
        }
        stride *= p.slice_sizes[axis];
    }
    gd_slice_bwd_copy_or_zero(go + slice_off * p.elem_size,
                              dx + full_lin * p.elem_size,
                              p.elem_size,
                              inside);
}
