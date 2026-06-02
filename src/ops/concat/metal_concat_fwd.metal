#include "metal_common.metal"

static inline void gd_concat_copy_elem(device const uchar *src,
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
        for (int i = 0; i < elem_size; ++i) {
            dst[i] = src[i];
        }
    }
}

kernel void gd_concat(device const uchar *x                         [[buffer(0)]],
                      device uchar *out                            [[buffer(1)]],
                      constant gd_metal_concat_params &p            [[buffer(2)]],
                      uint gid                                      [[thread_position_in_grid]])
{
    if ((int)gid >= p.numel) {
        return;
    }

    int inner_idx = (int)gid % p.inner;
    int tmp = (int)gid / p.inner;
    int dim_idx = tmp % p.input_dim;
    int outer_idx = tmp / p.input_dim;
    int out_lin = (outer_idx * p.output_dim + dim_idx + p.dst_start) * p.inner + inner_idx;

    gd_concat_copy_elem(x + (int)gid * p.elem_size,
                        out + out_lin * p.elem_size,
                        p.elem_size);
}
