#ifndef GRADIENTS_TRANSFER_H
#define GRADIENTS_TRANSFER_H

#include <stddef.h>

#include <gradients/memory.h>
#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

gd_status gd_synchronize(gd_context *ctx);

gd_status gd_span_upload(gd_context *ctx,
                         const gd_span *dst,
                         size_t dst_offset,
                         const void *src,
                         size_t nbytes);

gd_status gd_span_download(gd_context *ctx,
                           const gd_span *src,
                           size_t src_offset,
                           void *dst,
                           size_t nbytes);

gd_status gd_upload(gd_context *ctx, const void *src, size_t nbytes, gd_tensor *dst);
gd_status gd_download(gd_context *ctx, const gd_tensor *src, void *dst, size_t nbytes);

gd_status gd_tensor_write(gd_context *ctx, gd_tensor *dst, const void *src, size_t nbytes);
gd_status gd_tensor_read(gd_context *ctx, const gd_tensor *src, void *dst, size_t nbytes);
gd_status gd_tensor_write_f32(gd_context *ctx, gd_tensor *dst, const float *src, size_t count);
gd_status gd_tensor_read_f32(gd_context *ctx, const gd_tensor *src, float *dst, size_t count);
gd_status gd_tensor_from_f32(gd_context *ctx,
                             gd_arena_kind arena,
                             gd_dtype dtype,
                             gd_shape shape,
                             const float *src,
                             size_t count,
                             bool requires_grad,
                             gd_tensor *out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_TRANSFER_H */
