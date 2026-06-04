#include <gradients/transfer.h>

#include "backend.h"
#include "memory_internal.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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
