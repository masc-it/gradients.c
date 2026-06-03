#include "gradients/quant.h"

#include "internal.h"

gd_status gd_quant_register_format(gd_context *ctx, const gd_quant_format *format)
{
    if (ctx == NULL || format == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_quant_register_format argument is NULL");
    }
    return _gd_error(GD_ERR_UNSUPPORTED, "quant registry is not implemented yet");
}

const gd_quant_format *gd_quant_find_format(gd_context *ctx, const char *name)
{
    if (ctx == NULL || name == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_quant_find_format argument is NULL");
        return NULL;
    }
    _gd_set_last_error(GD_ERR_UNSUPPORTED, "quant registry is not implemented yet");
    return NULL;
}

gd_status gd_quant_desc_create(gd_context *ctx,
                               const gd_quant_format *format,
                               int group_size,
                               int axis,
                               gd_dtype scale_dtype,
                               gd_tensor *scales,
                               gd_tensor *zeros,
                               const void *extra,
                               size_t extra_size,
                               gd_quant_desc **out)
{
    if (ctx == NULL || format == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_quant_desc_create argument is NULL");
    }
    (void)group_size;
    (void)axis;
    (void)scale_dtype;
    (void)scales;
    (void)zeros;
    (void)extra;
    (void)extra_size;
    *out = NULL;
    return _gd_error(GD_ERR_UNSUPPORTED, "quant descriptors are not implemented yet");
}

gd_status gd_quant_desc_retain(gd_quant_desc *desc)
{
    if (desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_quant_desc_retain desc is NULL");
    }
    return _gd_error(GD_ERR_UNSUPPORTED, "quant descriptors are not implemented yet");
}

void gd_quant_desc_release(gd_quant_desc *desc)
{
    (void)desc;
    _gd_set_last_error(GD_OK, NULL);
}

gd_status gd_quantize(gd_context *ctx,
                      gd_tensor *src,
                      const gd_quant_desc *qdesc,
                      gd_tensor **packed_out)
{
    if (ctx == NULL || src == NULL || qdesc == NULL || packed_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_quantize argument is NULL");
    }
    *packed_out = NULL;
    return _gd_error(GD_ERR_UNSUPPORTED, "quantize op is not implemented yet");
}

gd_status gd_dequantize(gd_context *ctx,
                        gd_tensor *packed,
                        gd_dtype out_dtype,
                        gd_tensor **out)
{
    if (ctx == NULL || packed == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_dequantize argument is NULL");
    }
    (void)out_dtype;
    *out = NULL;
    return _gd_error(GD_ERR_UNSUPPORTED, "dequantize op is not implemented yet");
}
