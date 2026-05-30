#ifndef GRADIENTS_QUANT_H
#define GRADIENTS_QUANT_H

#include <stddef.h>

#include "gradients/dtype.h"
#include "gradients/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_context gd_context;
typedef struct gd_tensor gd_tensor;
typedef struct gd_quant_format gd_quant_format;
typedef struct gd_quant_desc gd_quant_desc;

typedef gd_status (*gd_quant_pack_ref_fn)(const gd_tensor *src,
                                          const gd_quant_desc *qdesc,
                                          gd_tensor *dst_packed);
typedef gd_status (*gd_quant_unpack_ref_fn)(const gd_tensor *src_packed,
                                            gd_dtype out_dtype,
                                            gd_tensor *dst);

struct gd_quant_format {
    const char *name;
    int bits_num;
    int bits_den;
    int values_per_block;
    int bytes_per_block;
    int preferred_layout;
    gd_quant_pack_ref_fn pack_ref;
    gd_quant_unpack_ref_fn unpack_ref;
};

gd_status gd_quant_register_format(gd_context *ctx,
                                   const gd_quant_format *format);
const gd_quant_format *gd_quant_find_format(gd_context *ctx,
                                            const char *name);

gd_status gd_quant_desc_create(gd_context *ctx,
                               const gd_quant_format *format,
                               int group_size,
                               int axis,
                               gd_dtype scale_dtype,
                               gd_tensor *scales,
                               gd_tensor *zeros,
                               const void *extra,
                               size_t extra_size,
                               gd_quant_desc **out);
gd_status gd_quant_desc_retain(gd_quant_desc *desc);
void gd_quant_desc_release(gd_quant_desc *desc);

gd_status gd_quantize(gd_context *ctx,
                      gd_tensor *src,
                      const gd_quant_desc *qdesc,
                      gd_tensor **packed_out);
gd_status gd_dequantize(gd_context *ctx,
                        gd_tensor *packed,
                        gd_dtype out_dtype,
                        gd_tensor **out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_QUANT_H */
