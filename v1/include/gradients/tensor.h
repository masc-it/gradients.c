#ifndef GRADIENTS_TENSOR_H
#define GRADIENTS_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "gradients/device.h"
#include "gradients/dtype.h"
#include "gradients/quant.h"
#include "gradients/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GD_MAX_DIMS 8

typedef struct gd_context gd_context;
typedef struct gd_storage gd_storage;
typedef struct gd_tensor gd_tensor;

typedef enum gd_layout {
    GD_LAYOUT_STRIDED = 0,
    GD_LAYOUT_CONTIGUOUS,
    GD_LAYOUT_CHANNELS_LAST,
    GD_LAYOUT_PACKED_QUANT,
    GD_LAYOUT_BLOCKED,
    GD_LAYOUT_BACKEND_OPAQUE
} gd_layout;

typedef enum gd_memory_kind {
    GD_MEM_HOST = 0,
    GD_MEM_DEVICE,
    GD_MEM_PINNED_HOST,
    GD_MEM_UNIFIED
} gd_memory_kind;

typedef struct gd_storage_desc {
    gd_device device;
    gd_memory_kind memory;
    size_t nbytes;
    size_t alignment;
} gd_storage_desc;

typedef struct gd_tensor_desc {
    gd_dtype dtype;
    gd_device device;
    gd_layout layout;
    int ndim;
    int64_t sizes[GD_MAX_DIMS];
    int64_t strides[GD_MAX_DIMS];
    int64_t storage_offset_bytes;
    const gd_quant_desc *quant;
} gd_tensor_desc;

gd_status gd_tensor_desc_contiguous(gd_dtype dtype,
                                    gd_device device,
                                    int ndim,
                                    const int64_t *sizes,
                                    gd_tensor_desc *out);

gd_status gd_tensor_desc_nbytes(const gd_tensor_desc *desc,
                                size_t *nbytes_out,
                                size_t *alignment_out);

gd_status gd_storage_create(gd_context *ctx,
                            const gd_storage_desc *desc,
                            gd_storage **out);
gd_status gd_storage_retain(gd_storage *storage);
void gd_storage_release(gd_storage *storage);
gd_status gd_storage_data_cpu(gd_storage *storage, void **out);
gd_status gd_storage_copy_from_cpu(gd_context *ctx,
                                   gd_storage *dst,
                                   size_t dst_offset,
                                   const void *src,
                                   size_t nbytes);
gd_status gd_storage_copy_to_cpu(gd_context *ctx,
                                 gd_storage *src,
                                 size_t src_offset,
                                 void *dst,
                                 size_t nbytes);
size_t gd_storage_nbytes(const gd_storage *storage);
gd_device gd_storage_device(const gd_storage *storage);

gd_status gd_tensor_empty(gd_context *ctx,
                          const gd_tensor_desc *desc,
                          gd_tensor **out);
gd_status gd_tensor_from_storage(gd_context *ctx,
                                 gd_storage *storage,
                                 const gd_tensor_desc *desc,
                                 gd_tensor **out);
gd_status gd_tensor_retain(gd_tensor *tensor);
void gd_tensor_release(gd_tensor *tensor);

/* Raw byte transfer helpers. Buffers must already use tensor dtype/layout bytes;
 * no numeric dtype conversion is performed. Use gd_cast for typed conversion. */
gd_status gd_tensor_copy_from_cpu(gd_context *ctx,
                                  gd_tensor *dst,
                                  const void *src,
                                  size_t nbytes);
gd_status gd_tensor_copy_to_cpu(gd_context *ctx,
                                gd_tensor *src,
                                void *dst,
                                size_t nbytes);

int gd_tensor_ndim(const gd_tensor *tensor);
int64_t gd_tensor_size(const gd_tensor *tensor, int dim);
int64_t gd_tensor_stride(const gd_tensor *tensor, int dim);
gd_dtype gd_tensor_dtype(const gd_tensor *tensor);
gd_device gd_tensor_device(const gd_tensor *tensor);
gd_layout gd_tensor_layout(const gd_tensor *tensor);
gd_storage *gd_tensor_storage(const gd_tensor *tensor);
const gd_quant_desc *gd_tensor_quant(const gd_tensor *tensor);

gd_status gd_tensor_view(gd_tensor *base,
                         const gd_tensor_desc *view_desc,
                         gd_tensor **out);
gd_status gd_tensor_reshape(gd_tensor *tensor,
                            int ndim,
                            const int64_t *sizes,
                            gd_tensor **out);
gd_status gd_tensor_transpose(gd_tensor *tensor,
                              int d0,
                              int d1,
                              gd_tensor **out);
gd_status gd_tensor_slice(gd_tensor *tensor,
                          int dim,
                          int64_t start,
                          int64_t len,
                          gd_tensor **out);
gd_status gd_tensor_contiguous(gd_context *ctx,
                               gd_tensor *tensor,
                               gd_tensor **out);

gd_status gd_tensor_set_requires_grad(gd_tensor *tensor, bool requires_grad);
bool gd_tensor_requires_grad(const gd_tensor *tensor);
gd_status gd_tensor_grad(gd_tensor *tensor, gd_tensor **grad_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TENSOR_H */
