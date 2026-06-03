#ifndef GD_CORE_BACKEND_H
#define GD_CORE_BACKEND_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>

typedef struct gd_backend gd_backend;
typedef struct gd_backend_buffer gd_backend_buffer;

typedef struct gd_backend_fence {
    void *handle;
} gd_backend_fence;

typedef enum gd_backend_kind {
    GD_BACKEND_METAL = 1,
} gd_backend_kind;

gd_status gd_backend_create_default(gd_backend **out_backend);
void gd_backend_destroy(gd_backend *backend);

gd_backend_kind gd_backend_kind_query(const gd_backend *backend);
const char *gd_backend_name(const gd_backend *backend);

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer);
void gd_backend_buffer_destroy(gd_backend_buffer *buffer);
size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer);
void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer);
bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer);

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes);
gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes);

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence);
void gd_backend_fence_destroy(gd_backend_fence *fence);
bool gd_backend_fence_is_complete(gd_backend_fence *fence);
gd_status gd_backend_fence_wait(gd_backend_fence *fence);

#endif /* GD_CORE_BACKEND_H */
