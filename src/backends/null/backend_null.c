#include "../../core/backend.h"

struct gd_backend {
    int unused;
};

struct gd_backend_buffer {
    int unused;
};

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    if (out_backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_backend = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_destroy(gd_backend *backend)
{
    (void)backend;
}

gd_backend_kind gd_backend_kind_query(const gd_backend *backend)
{
    (void)backend;
    return 0;
}

const char *gd_backend_name(const gd_backend *backend)
{
    (void)backend;
    return "none";
}

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer)
{
    (void)backend;
    (void)nbytes;
    if (out_buffer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_buffer_destroy(gd_backend_buffer *buffer)
{
    (void)buffer;
}

size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer)
{
    (void)buffer;
    return 0U;
}

void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer)
{
    (void)buffer;
    return 0;
}

bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer)
{
    (void)buffer;
    return false;
}

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)src;
    (void)nbytes;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes)
{
    (void)backend;
    (void)buffer;
    (void)offset;
    (void)dst;
    (void)nbytes;
    return GD_ERR_UNSUPPORTED;
}

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence)
{
    (void)backend;
    if (out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = 0;
    return GD_ERR_UNSUPPORTED;
}

void gd_backend_fence_destroy(gd_backend_fence *fence)
{
    if (fence != NULL) {
        fence->handle = 0;
    }
}

bool gd_backend_fence_is_complete(gd_backend_fence *fence)
{
    (void)fence;
    return true;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    (void)fence;
    return GD_OK;
}
