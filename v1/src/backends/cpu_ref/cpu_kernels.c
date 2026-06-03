#include "cpu_backend.h"

#include <stdint.h>

#include "../../core/internal.h"

int64_t _gd_cpu_desc_numel(const gd_tensor_desc *desc)
{
    int64_t numel = 1;
    int i = 0;

    for (i = 0; i < desc->ndim; ++i) {
        numel *= desc->sizes[i];
    }
    return numel;
}

void _gd_cpu_unravel(int64_t linear, const gd_tensor_desc *desc, int64_t *index)
{
    int i = 0;

    for (i = desc->ndim - 1; i >= 0; --i) {
        index[i] = linear % desc->sizes[i];
        linear /= desc->sizes[i];
    }
}

/* Linear offset into a contiguous input, broadcasting against an output index. */
int64_t _gd_cpu_broadcast_offset(const int64_t *out_index,
                                 int out_ndim,
                                 const gd_tensor_desc *in_desc)
{
    int64_t stride = 1;
    int64_t offset = 0;
    int i = 0;

    for (i = in_desc->ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_desc->ndim - i);
        int64_t coord = in_desc->sizes[i] == 1 ? 0 : out_index[out_pos];

        offset += coord * stride;
        stride *= in_desc->sizes[i];
    }
    return offset;
}

gd_status _gd_cpu_load_float(const gd_tensor_desc *desc, const void *data,
                             int64_t i, float *out)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        *out = ((const float *)data)[i];
        return GD_OK;
    case GD_DTYPE_F16:
        *out = _gd_f16_bits_to_f32(((const uint16_t *)data)[i]);
        return GD_OK;
    case GD_DTYPE_I32:
        *out = (float)((const int32_t *)data)[i];
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF float load dtype is not implemented");
    }
}

gd_status _gd_cpu_store_float(const gd_tensor_desc *desc, void *data,
                              int64_t i, float value)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        ((float *)data)[i] = value;
        return GD_OK;
    case GD_DTYPE_F16:
        ((uint16_t *)data)[i] = _gd_f32_to_f16_bits(value);
        return GD_OK;
    case GD_DTYPE_I32:
        ((int32_t *)data)[i] = (int32_t)value;
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF float store dtype is not implemented");
    }
}
