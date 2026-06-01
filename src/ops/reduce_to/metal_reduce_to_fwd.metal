#include "metal_common.metal"

kernel void gd_reduce_to(device const uchar *go               [[buffer(0)]],
                         device uchar *out                    [[buffer(1)]],
                         constant gd_metal_reduce_to_params &p [[buffer(2)]],
                         uint gid                             [[thread_position_in_grid]])
{
    int t = (int)gid;
    if (t >= p.target_numel) {
        return;
    }

    /* Fast path for the dominant GPT case: reduce only extra leading dims,
     * e.g. [B, M, N] -> [M, N]. Both tensors are contiguous, so each output
     * element is a short strided sum over leading slices. Avoids the generic
     * coordinate/divmod path for millions of elements with reduce_count=4/8. */
    if (p.go_ndim == p.target_ndim + 1) {
        bool leading_only = true;
        for (int i = 0; i < p.target_ndim; ++i) {
            if (p.go_sizes[i + 1] != p.target_sizes[i]) {
                leading_only = false;
            }
        }
        if (leading_only) {
            float acc = 0.0f;
            for (int r = 0; r < p.go_sizes[0]; ++r) {
                acc += gd_load_float(go, p.dtype, (uint)(r * p.target_numel + t));
            }
            gd_store_float(out, p.dtype, (uint)t, acc);
            return;
        }
    }

    /* Target coordinates (contiguous target). */
    int tcoord[GD_METAL_MAX_DIMS];
    int lin = t;
    for (int k = p.target_ndim - 1; k >= 0; --k) {
        tcoord[k] = (p.target_sizes[k] > 0) ? (lin % p.target_sizes[k]) : 0;
        if (p.target_sizes[k] > 0) {
            lin /= p.target_sizes[k];
        }
    }

    /* Contiguous strides for go. */
    int go_stride[GD_METAL_MAX_DIMS];
    int s = 1;
    for (int i = p.go_ndim - 1; i >= 0; --i) {
        go_stride[i] = s;
        s *= p.go_sizes[i];
    }

    /* Partition go dims into fixed (matched to a target coord) and reduced
     * (extra leading dim, or target dim of size 1 that go broadcasts over). */
    int reduce_dims[GD_METAL_MAX_DIMS];
    int n_reduce = 0;
    int reduce_count = 1;
    int base = 0;
    for (int i = 0; i < p.go_ndim; ++i) {
        int out_pos = p.target_ndim - (p.go_ndim - i);
        bool reduced = (out_pos < 0) ||
                       (p.target_sizes[out_pos] == 1 && p.go_sizes[i] > 1);
        if (reduced) {
            reduce_dims[n_reduce++] = i;
            reduce_count *= p.go_sizes[i];
        } else {
            int coord = (out_pos < 0 || p.go_sizes[i] == 1) ? 0 : tcoord[out_pos];
            base += coord * go_stride[i];
        }
    }

    float acc = 0.0f;
    for (int r = 0; r < reduce_count; ++r) {
        int rem = r;
        int goff = base;
        for (int j = n_reduce - 1; j >= 0; --j) {
            int dim = reduce_dims[j];
            goff += (rem % p.go_sizes[dim]) * go_stride[dim];
            rem /= p.go_sizes[dim];
        }
        acc += gd_load_float(go, p.dtype, (uint)goff);
    }
    gd_store_float(out, p.dtype, (uint)t, acc);
}
