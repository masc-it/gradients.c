#include <metal_stdlib>
#include "metal_add_types.h"
#include "../_shared/binary/metal_binary_metal.h"

using namespace metal;

#define GD_ADD_VEC ulong(GD_METAL_ADD_F16_VECTOR_WIDTH)

static inline bool gd_add_half4_aligned(ulong byte_offset, ulong elem_index)
{
    return ((byte_offset + elem_index * 2ul) & 7ul) == 0ul;
}

static inline half4 gd_add_load4(device const half *ptr, ulong base, ulong stride, ulong byte_offset)
{
    if (stride == 0ul) {
        return half4(ptr[base]);
    }
    if (stride == 1ul && gd_add_half4_aligned(byte_offset, base)) {
        return *reinterpret_cast<device const half4 *>(ptr + base);
    }
    return half4(ptr[base], ptr[base + stride], ptr[base + stride * 2ul], ptr[base + stride * 3ul]);
}

static inline void gd_add_store4(device half *ptr, ulong base, ulong byte_offset, half4 value)
{
    if (gd_add_half4_aligned(byte_offset, base)) {
        *reinterpret_cast<device half4 *>(ptr + base) = value;
    } else {
        ptr[base + 0ul] = value[0];
        ptr[base + 1ul] = value[1];
        ptr[base + 2ul] = value[2];
        ptr[base + 3ul] = value[3];
    }
}

kernel void gd_add_kernel(device const uchar *xbuf [[buffer(0)]],
                          device const uchar *ybuf [[buffer(1)]],
                          device uchar *outbuf [[buffer(2)]],
                          constant gd_metal_binary_args &args [[buffer(3)]],
                          uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_ADD_VEC;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (base >= args.count) {
        return;
    }
    if (base + 3ul < args.count) {
        const half4 xv = gd_add_load4(x, base, 1ul, ulong(args.x_offset));
        const half4 yv = gd_add_load4(y, base, 1ul, ulong(args.y_offset));
        gd_add_store4(out, base, ulong(args.out_offset), xv + yv);
    } else {
        for (ulong lane = 0ul; lane < GD_ADD_VEC && base + lane < args.count; ++lane) {
            out[base + lane] = x[base + lane] + y[base + lane];
        }
    }
}

kernel void gd_add_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                device const uchar *ybuf [[buffer(1)]],
                                device uchar *outbuf [[buffer(2)]],
                                constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                uint gid [[thread_position_in_grid]])
{
    const ulong base = ulong(gid) * GD_ADD_VEC;
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (base >= args.count) {
        return;
    }
    const uint last_dim = args.rank - 1u;
    const ulong last_size = ulong(args.out_shape[last_dim]);
    const ulong inner = base % last_size;
    const ulong x_last_stride = ulong(args.x_strides[last_dim]);
    const ulong y_last_stride = ulong(args.y_strides[last_dim]);
    if (base + 3ul < args.count && inner + 3ul < last_size &&
        (x_last_stride == 0ul || x_last_stride == 1ul) &&
        (y_last_stride == 0ul || y_last_stride == 1ul)) {
        const ulong xi = gd_binary_bcast_offset(base, args, args.x_strides);
        const ulong yi = gd_binary_bcast_offset(base, args, args.y_strides);
        const half4 xv = gd_add_load4(x, xi, x_last_stride, ulong(args.x_offset));
        const half4 yv = gd_add_load4(y, yi, y_last_stride, ulong(args.y_offset));
        gd_add_store4(out, base, ulong(args.out_offset), xv + yv);
    } else {
        for (ulong lane = 0ul; lane < GD_ADD_VEC && base + lane < args.count; ++lane) {
            const ulong i = base + lane;
            const ulong xi = gd_binary_bcast_offset(i, args, args.x_strides);
            const ulong yi = gd_binary_bcast_offset(i, args, args.y_strides);
            out[i] = x[xi] + y[yi];
        }
    }
}

kernel void gd_add_row_bcast_kernel(device const uchar *xbuf [[buffer(0)]],
                                    device const uchar *ybuf [[buffer(1)]],
                                    device uchar *outbuf [[buffer(2)]],
                                    constant gd_metal_binary_bcast_args &args [[buffer(3)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    const ulong base_col = ulong(gid.x) * GD_ADD_VEC;
    const ulong row = ulong(gid.y);
    const ulong rows = ulong(args.out_shape[0]);
    const ulong cols = ulong(args.out_shape[1]);
    device const half *x = reinterpret_cast<device const half *>(xbuf + args.x_offset);
    device const half *y = reinterpret_cast<device const half *>(ybuf + args.y_offset);
    device half *out = reinterpret_cast<device half *>(outbuf + args.out_offset);
    if (row >= rows || base_col >= cols) {
        return;
    }
    const ulong out_i = row * cols + base_col;
    const ulong x_col_stride = ulong(args.x_strides[1]);
    const ulong y_col_stride = ulong(args.y_strides[1]);
    const ulong x_base = row * ulong(args.x_strides[0]) + base_col * x_col_stride;
    const ulong y_base = row * ulong(args.y_strides[0]) + base_col * y_col_stride;
    if (base_col + 3ul < cols &&
        (x_col_stride == 0ul || x_col_stride == 1ul) &&
        (y_col_stride == 0ul || y_col_stride == 1ul)) {
        const half4 xv = gd_add_load4(x, x_base, x_col_stride, ulong(args.x_offset));
        const half4 yv = gd_add_load4(y, y_base, y_col_stride, ulong(args.y_offset));
        gd_add_store4(out, out_i, ulong(args.out_offset), xv + yv);
    } else {
        for (ulong lane = 0ul; lane < GD_ADD_VEC && base_col + lane < cols; ++lane) {
            const ulong col = base_col + lane;
            const ulong xi = gd_binary_row_bcast_offset(row, col, args.x_strides);
            const ulong yi = gd_binary_row_bcast_offset(row, col, args.y_strides);
            out[row * cols + col] = x[xi] + y[yi];
        }
    }
}
