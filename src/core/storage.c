#include "gradients/tensor.h"

#include <stdint.h>
#include <stdlib.h>

#include "internal.h"
#include "refcount.h"
#include "storage_internal.h"
#include "../backends/backend.h"

struct gd_storage {
    gd_refcount refcount;
    gd_storage_desc desc;
    _gd_backend *backend;
    void *handle;
};

static void *storage_host_ptr(const gd_storage *storage)
{
    void *ptr = NULL;

    if (storage == NULL || storage->backend == NULL ||
        storage->backend->vt->storage_host_ptr == NULL) {
        return NULL;
    }
    if (storage->backend->vt->storage_host_ptr(storage->backend, storage->handle, &ptr) != GD_OK) {
        return NULL;
    }
    return ptr;
}

gd_status gd_storage_create(gd_context *ctx,
                            const gd_storage_desc *desc,
                            gd_storage **out)
{
    gd_status status = GD_OK;
    gd_storage *storage = NULL;
    _gd_backend *backend = NULL;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_create out is NULL");
    }
    *out = NULL;
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_create ctx is NULL");
    }
    if (desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage desc is NULL");
    }
    if (desc->nbytes == 0U) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage nbytes must be nonzero");
    }

    backend = _gd_context_backend(ctx, desc->device);
    if (backend == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for storage device");
    }
    if (backend->vt->storage_alloc == NULL || backend->vt->storage_free == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not implement storage allocation");
    }

    storage = calloc(1U, sizeof(*storage));
    if (storage == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate storage header");
    }

    /* Backend validates device/memory-kind/alignment legality and owns the bytes. */
    status = backend->vt->storage_alloc(backend, desc, &storage->handle);
    if (status != GD_OK) {
        free(storage);
        return status;
    }
    _gd_profile_record_alloc(ctx, backend, desc->nbytes);

    _gd_refcount_init(&storage->refcount);
    storage->desc = *desc;
    storage->backend = backend;

    *out = storage;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_storage_retain(gd_storage *storage)
{
    gd_status status = GD_OK;

    if (storage == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_retain storage is NULL");
    }
    status = _gd_refcount_retain(&storage->refcount);
    if (status != GD_OK) {
        return _gd_error(status, "cannot retain released storage");
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void gd_storage_release(gd_storage *storage)
{
    if (storage == NULL) {
        return;
    }
    if (_gd_refcount_release(&storage->refcount) != 0) {
        _gd_profile_record_free(storage->backend->ctx, storage->backend, storage->desc.nbytes);
        storage->backend->vt->storage_free(storage->backend, storage->handle);
        free(storage);
    }
    _gd_set_last_error(GD_OK, NULL);
}

static int storage_is_host_accessible(const gd_storage *storage)
{
    if (!storage->backend->caps.host_visible) {
        return 0;
    }
    return storage->desc.memory == GD_MEM_HOST || storage->desc.memory == GD_MEM_UNIFIED ||
           storage->desc.memory == GD_MEM_PINNED_HOST;
}

gd_status gd_storage_data_cpu(gd_storage *storage, void **out)
{
    void *ptr = NULL;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_data_cpu out is NULL");
    }
    *out = NULL;
    if (storage == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_data_cpu storage is NULL");
    }
    if (!storage_is_host_accessible(storage)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "storage is not CPU accessible");
    }
    ptr = storage_host_ptr(storage);
    if (ptr == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "storage has no host pointer");
    }
    *out = ptr;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_storage_copy_from_cpu(gd_context *ctx,
                                   gd_storage *dst,
                                   size_t dst_offset,
                                   const void *src,
                                   size_t nbytes)
{
    if (ctx == NULL || dst == NULL || src == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage_copy_from_cpu argument is NULL");
    }
    if (dst_offset > dst->desc.nbytes || nbytes > dst->desc.nbytes - dst_offset) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy range exceeds destination storage");
    }
    if (dst->backend->vt->upload == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not implement host upload");
    }
    {
        uint64_t start = _gd_profile_enabled(ctx) ? _gd_profile_now_ns() : 0U;
        gd_status status = dst->backend->vt->upload(dst->backend, dst->handle, dst_offset,
                                                    src, nbytes);
        if (start != 0U) {
            _gd_profile_record_upload(ctx, dst->backend, _gd_profile_now_ns() - start, nbytes);
        }
        return status;
    }
}

gd_status gd_storage_copy_to_cpu(gd_context *ctx,
                                 gd_storage *src,
                                 size_t src_offset,
                                 void *dst,
                                 size_t nbytes)
{
    if (ctx == NULL || src == NULL || dst == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage_copy_to_cpu argument is NULL");
    }
    if (src_offset > src->desc.nbytes || nbytes > src->desc.nbytes - src_offset) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy range exceeds source storage");
    }
    if (src->backend->vt->download == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "backend does not implement host download");
    }
    {
        uint64_t start = _gd_profile_enabled(ctx) ? _gd_profile_now_ns() : 0U;
        gd_status status = src->backend->vt->download(src->backend, src->handle, src_offset,
                                                      dst, nbytes);
        if (start != 0U) {
            _gd_profile_record_download(ctx, src->backend, _gd_profile_now_ns() - start,
                                        nbytes, true);
        }
        return status;
    }
}

size_t gd_storage_nbytes(const gd_storage *storage)
{
    if (storage == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_nbytes storage is NULL");
        return 0U;
    }
    _gd_set_last_error(GD_OK, NULL);
    return storage->desc.nbytes;
}

gd_device gd_storage_device(const gd_storage *storage)
{
    if (storage == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_storage_device storage is NULL");
        return (gd_device){GD_DEVICE_CPU, 0};
    }
    _gd_set_last_error(GD_OK, NULL);
    return storage->desc.device;
}

void *_gd_storage_data_mut(gd_storage *storage)
{
    return storage_host_ptr(storage);
}

const void *_gd_storage_data(const gd_storage *storage)
{
    return storage_host_ptr(storage);
}

size_t _gd_storage_nbytes(const gd_storage *storage)
{
    return storage == NULL ? 0U : storage->desc.nbytes;
}

const gd_storage_desc *_gd_storage_desc(const gd_storage *storage)
{
    return storage == NULL ? NULL : &storage->desc;
}

void *_gd_storage_handle(const gd_storage *storage)
{
    return storage == NULL ? NULL : storage->handle;
}
