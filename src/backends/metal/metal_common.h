#ifndef GD_METAL_COMMON_H
#define GD_METAL_COMMON_H

/* Metal-only helpers shared by backend primitives and optimizer kernels. */

#ifndef __METAL_VERSION__
#error "metal_common.h is for Metal shader translation units only"
#endif

static inline void gd_write_pattern(device uchar *dst, ulong byte, uint elem_size, uint pattern)
{
    for (uint i = 0; i < elem_size; ++i) {
        dst[byte + i] = uchar((pattern >> (8u * i)) & 255u);
    }
}

static inline ushort gd_bf16_bits(float v)
{
    uint bits = as_type<uint>(v);
    uint lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return ushort(bits >> 16);
}

static inline void gd_write_float_dtype(device uchar *dst, ulong byte, uint dtype, float v)
{
    if (dtype == 1u) {
        ushort bits = as_type<ushort>(half(v));
        gd_write_pattern(dst, byte, 2u, uint(bits));
    } else if (dtype == 2u) {
        gd_write_pattern(dst, byte, 2u, uint(gd_bf16_bits(v)));
    } else {
        gd_write_pattern(dst, byte, 4u, as_type<uint>(v));
    }
}

static inline float gd_load_float_dtype(device const uchar *src, ulong byte, uint dtype)
{
    if (dtype == 1u) {
        return float(*(reinterpret_cast<device const half *>(src + byte)));
    }
    if (dtype == 3u) {
        return *(reinterpret_cast<device const float *>(src + byte));
    }
    return 0.0f;
}

static inline ulong gd_dtype_elem_size(uint dtype)
{
    return dtype == 3u ? 4ul : 2ul;
}

#endif /* GD_METAL_COMMON_H */
