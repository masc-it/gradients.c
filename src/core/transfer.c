#include <gradients/transfer.h>

#include "backend.h"
#include "dtype_internal.h"
#include "memory_internal.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static bool gd_range_inside(size_t capacity, size_t offset, size_t nbytes)
{
    return offset <= capacity && nbytes <= capacity - offset;
}

gd_status gd_synchronize(gd_context *ctx)
{
    return gd_context_synchronize(ctx);
}

gd_status gd_span_upload(gd_context *ctx,
                         const gd_span *dst,
                         size_t dst_offset,
                         const void *src,
                         size_t nbytes)
{
    gd_status st;
    gd_backend *backend;
    if (ctx == NULL || dst == NULL || src == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_context_wait_for_span(ctx, dst);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_range_inside(dst->nbytes, dst_offset, nbytes) ||
        dst->offset > SIZE_MAX - dst_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "span upload range out of bounds");
    }
    st = gd_context_flush_backend(ctx);
    if (st != GD_OK) {
        return st;
    }
    backend = gd_context_backend(ctx);
    if (backend == NULL || dst->buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "span upload missing backend buffer");
    }
    st = gd_backend_upload(backend,
                           (gd_backend_buffer *)dst->buffer,
                           dst->offset + dst_offset,
                           src,
                           nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend upload failed");
    }
    return GD_OK;
}

gd_status gd_span_download(gd_context *ctx,
                           const gd_span *src,
                           size_t src_offset,
                           void *dst,
                           size_t nbytes)
{
    gd_status st;
    gd_backend *backend;
    if (ctx == NULL || src == NULL || dst == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_context_wait_for_span(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_range_inside(src->nbytes, src_offset, nbytes) ||
        src->offset > SIZE_MAX - src_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "span download range out of bounds");
    }
    st = gd_context_flush_backend(ctx);
    if (st != GD_OK) {
        return st;
    }
    backend = gd_context_backend(ctx);
    if (backend == NULL || src->buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "span download missing backend buffer");
    }
    st = gd_backend_download(backend,
                             (gd_backend_buffer *)src->buffer,
                             src->offset + src_offset,
                             dst,
                             nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend download failed");
    }
    return GD_OK;
}

gd_status gd_upload(gd_context *ctx, const void *src, size_t nbytes, gd_tensor *dst)
{
    gd_status st;
    size_t logical_nbytes;
    if (ctx == NULL || src == NULL || dst == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, dst);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(dst)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor upload requires contiguous tensor");
    }
    st = gd_tensor_logical_nbytes(dst, &logical_nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "tensor upload invalid shape");
    }
    if (nbytes != logical_nbytes) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor upload size mismatch");
    }
    st = gd_span_upload(ctx, &dst->storage, dst->view_offset, src, nbytes);
    if (st != GD_OK) {
        return st;
    }
    dst->version += 1U;
    if (dst->version == 0U) {
        dst->version = 1U;
    }
    return GD_OK;
}

gd_status gd_download(gd_context *ctx, const gd_tensor *src, void *dst, size_t nbytes)
{
    gd_status st;
    size_t logical_nbytes;
    if (ctx == NULL || src == NULL || dst == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(src)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor download requires contiguous tensor");
    }
    st = gd_tensor_logical_nbytes(src, &logical_nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "tensor download invalid shape");
    }
    if (nbytes != logical_nbytes) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor download size mismatch");
    }
    return gd_span_download(ctx, &src->storage, src->view_offset, dst, nbytes);
}

gd_status gd_tensor_write(gd_context *ctx, gd_tensor *dst, const void *src, size_t nbytes)
{
    return gd_upload(ctx, src, nbytes, dst);
}

gd_status gd_tensor_read(gd_context *ctx, const gd_tensor *src, void *dst, size_t nbytes)
{
    return gd_download(ctx, src, dst, nbytes);
}

static gd_status gd_tensor_f32_transfer_validate(gd_context *ctx,
                                                 const gd_tensor *tensor,
                                                 size_t count,
                                                 const char *message)
{
    gd_status st;
    int64_t numel;
    if (ctx == NULL || tensor == NULL || count == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (tensor->dtype != GD_DTYPE_F16 && tensor->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "f32 tensor transfer supports f16 and f32 tensors");
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "f32 tensor transfer requires contiguous tensor");
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, message);
    }
    if (numel < 0 || (uint64_t)numel != (uint64_t)count) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "f32 tensor transfer count mismatch");
    }
    return GD_OK;
}

gd_status gd_tensor_write_f32(gd_context *ctx, gd_tensor *dst, const float *src, size_t count)
{
    gd_status st;
    size_t i;
    size_t nbytes;
    uint16_t *packed;
    if (ctx == NULL || dst == NULL || src == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_f32_transfer_validate(ctx, dst, count, "f32 tensor write invalid shape");
    if (st != GD_OK) {
        return st;
    }
    if (dst->dtype == GD_DTYPE_F32) {
        if (count > SIZE_MAX / sizeof(src[0])) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor write size overflow");
        }
        return gd_tensor_write(ctx, dst, src, count * sizeof(src[0]));
    }
    if (count > SIZE_MAX / sizeof(packed[0])) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor write size overflow");
    }
    nbytes = count * sizeof(packed[0]);
    packed = (uint16_t *)malloc(nbytes);
    if (packed == NULL) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor write pack allocation failed");
    }
    for (i = 0U; i < count; ++i) {
        packed[i] = gd_f32_to_f16_bits(src[i]);
    }
    st = gd_tensor_write(ctx, dst, packed, nbytes);
    free(packed);
    return st;
}

gd_status gd_tensor_read_f32(gd_context *ctx, const gd_tensor *src, float *dst, size_t count)
{
    gd_status st;
    size_t i;
    size_t nbytes;
    uint16_t *packed;
    if (ctx == NULL || src == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_f32_transfer_validate(ctx, src, count, "f32 tensor read invalid shape");
    if (st != GD_OK) {
        return st;
    }
    if (src->dtype == GD_DTYPE_F32) {
        if (count > SIZE_MAX / sizeof(dst[0])) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor read size overflow");
        }
        return gd_tensor_read(ctx, src, dst, count * sizeof(dst[0]));
    }
    if (count > SIZE_MAX / sizeof(packed[0])) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor read size overflow");
    }
    nbytes = count * sizeof(packed[0]);
    packed = (uint16_t *)malloc(nbytes);
    if (packed == NULL) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "f32 tensor read pack allocation failed");
    }
    st = gd_tensor_read(ctx, src, packed, nbytes);
    if (st == GD_OK) {
        for (i = 0U; i < count; ++i) {
            dst[i] = gd_f16_bits_to_f32(packed[i]);
        }
    }
    free(packed);
    return st;
}
