#ifndef GRADIENTS_DTYPE_H
#define GRADIENTS_DTYPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum gd_dtype {
    GD_DTYPE_INVALID = 0,
    GD_DTYPE_F32,
    GD_DTYPE_F16,
    GD_DTYPE_BF16,
    GD_DTYPE_FP8_E4M3,
    GD_DTYPE_FP8_E5M2,
    GD_DTYPE_I8,
    GD_DTYPE_U8,
    GD_DTYPE_I16,
    GD_DTYPE_U16,
    GD_DTYPE_I32,
    GD_DTYPE_U32,
    GD_DTYPE_I64,
    GD_DTYPE_U64,
    GD_DTYPE_BOOL,
    GD_DTYPE_QUANTIZED
} gd_dtype;

typedef struct gd_compute_policy {
    gd_dtype compute_dtype;
    gd_dtype accum_dtype;
} gd_compute_policy;

gd_compute_policy gd_compute_policy_default(void);

size_t gd_dtype_sizeof(gd_dtype dtype);
const char *gd_dtype_name(gd_dtype dtype);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_DTYPE_H */
