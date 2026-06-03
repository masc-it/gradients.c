#include "gradients/dtype.h"

#include <stdint.h>
#include <string.h>

#include "internal.h"

gd_compute_policy gd_compute_policy_default(void)
{
    return (gd_compute_policy){GD_DTYPE_F32, GD_DTYPE_F32};
}

uint16_t _gd_f32_to_f16_bits(float value)
{
    uint32_t bits = 0U;
    uint32_t sign = 0U;
    uint32_t exp = 0U;
    uint32_t mant = 0U;
    int32_t half_exp = 0;

    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16U) & 0x8000U;
    exp = (bits >> 23U) & 0xffU;
    mant = bits & 0x7fffffU;

    if (exp == 0xffU) {
        if (mant == 0U) {
            return (uint16_t)(sign | 0x7c00U);
        }
        mant >>= 13U;
        if (mant == 0U) {
            mant = 1U;
        }
        return (uint16_t)(sign | 0x7c00U | mant | 0x0200U);
    }

    half_exp = (int32_t)exp - 127 + 15;
    if (half_exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    if (half_exp <= 0) {
        uint32_t shift = 0U;
        uint32_t half_mant = 0U;
        uint32_t round_bit = 0U;
        uint32_t rest = 0U;

        if (half_exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000U;
        shift = (uint32_t)(14 - half_exp);
        half_mant = mant >> shift;
        round_bit = 1U << (shift - 1U);
        rest = mant & (round_bit - 1U);
        if ((mant & round_bit) != 0U && (rest != 0U || (half_mant & 1U) != 0U)) {
            half_mant++;
        }
        return (uint16_t)(sign | half_mant);
    }

    {
        uint32_t half_mant = mant >> 13U;
        uint32_t rem = mant & 0x1fffU;

        if (rem > 0x1000U || (rem == 0x1000U && (half_mant & 1U) != 0U)) {
            half_mant++;
            if (half_mant == 0x400U) {
                half_mant = 0U;
                half_exp++;
                if (half_exp >= 31) {
                    return (uint16_t)(sign | 0x7c00U);
                }
            }
        }
        return (uint16_t)(sign | ((uint32_t)half_exp << 10U) | half_mant);
    }
}

float _gd_f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16U;
    uint32_t exp = ((uint32_t)bits >> 10U) & 0x1fU;
    uint32_t mant = (uint32_t)bits & 0x03ffU;
    uint32_t out = 0U;
    float value = 0.0F;

    if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else {
            int32_t e = -14;
            while ((mant & 0x0400U) == 0U) {
                mant <<= 1U;
                e--;
            }
            mant &= 0x03ffU;
            out = sign | ((uint32_t)(e + 127) << 23U) | (mant << 13U);
        }
    } else if (exp == 0x1fU) {
        out = sign | 0x7f800000U | (mant << 13U);
    } else {
        out = sign | ((exp + (127U - 15U)) << 23U) | (mant << 13U);
    }
    memcpy(&value, &out, sizeof(value));
    return value;
}

size_t gd_dtype_sizeof(gd_dtype dtype)
{
    switch (dtype) {
    case GD_DTYPE_BOOL:
    case GD_DTYPE_I8:
    case GD_DTYPE_U8:
    case GD_DTYPE_FP8_E4M3:
    case GD_DTYPE_FP8_E5M2:
        return 1U;
    case GD_DTYPE_I16:
    case GD_DTYPE_U16:
    case GD_DTYPE_F16:
    case GD_DTYPE_BF16:
        return 2U;
    case GD_DTYPE_I32:
    case GD_DTYPE_U32:
    case GD_DTYPE_F32:
        return 4U;
    case GD_DTYPE_I64:
    case GD_DTYPE_U64:
        return 8U;
    case GD_DTYPE_INVALID:
    case GD_DTYPE_QUANTIZED:
        return 0U;
    }
    return 0U;
}

const char *gd_dtype_name(gd_dtype dtype)
{
    switch (dtype) {
    case GD_DTYPE_INVALID:
        return "invalid";
    case GD_DTYPE_F32:
        return "f32";
    case GD_DTYPE_F16:
        return "f16";
    case GD_DTYPE_BF16:
        return "bf16";
    case GD_DTYPE_FP8_E4M3:
        return "fp8_e4m3";
    case GD_DTYPE_FP8_E5M2:
        return "fp8_e5m2";
    case GD_DTYPE_I8:
        return "i8";
    case GD_DTYPE_U8:
        return "u8";
    case GD_DTYPE_I16:
        return "i16";
    case GD_DTYPE_U16:
        return "u16";
    case GD_DTYPE_I32:
        return "i32";
    case GD_DTYPE_U32:
        return "u32";
    case GD_DTYPE_I64:
        return "i64";
    case GD_DTYPE_U64:
        return "u64";
    case GD_DTYPE_BOOL:
        return "bool";
    case GD_DTYPE_QUANTIZED:
        return "quantized";
    }
    return "unknown";
}
