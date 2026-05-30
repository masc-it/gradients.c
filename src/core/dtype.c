#include "gradients/dtype.h"

gd_compute_policy gd_compute_policy_default(void)
{
    return (gd_compute_policy){GD_DTYPE_F32, GD_DTYPE_F32};
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
