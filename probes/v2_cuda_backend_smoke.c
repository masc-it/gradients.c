#include "src/core/backend.h"

#include <stdint.h>
#include <stdio.h>

int main(void)
{
    gd_backend *backend = NULL;
    gd_backend_buffer *buffer = NULL;
    gd_backend_fence fence = {0};
    const uint32_t src[4] = {1U, 2U, 3U, 4U};
    uint32_t dst[4] = {0U, 0U, 0U, 0U};
    gd_status st;

    st = gd_backend_create_default(&backend);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_create_default failed: %d\n", (int)st);
        return 1;
    }
    if (gd_backend_kind_query(backend) != GD_BACKEND_CUDA) {
        fprintf(stderr, "expected CUDA backend, got %s\n", gd_backend_name(backend));
        gd_backend_destroy(backend);
        return 2;
    }

    st = gd_backend_buffer_create(backend, sizeof(src), &buffer);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_buffer_create failed: %d\n", (int)st);
        gd_backend_destroy(backend);
        return 3;
    }
    if (!gd_backend_buffer_is_host_visible(buffer) ||
        gd_backend_buffer_host_ptr(buffer) == NULL ||
        gd_backend_buffer_nbytes(buffer) != sizeof(src)) {
        fprintf(stderr, "bad CUDA buffer metadata\n");
        gd_backend_buffer_destroy(buffer);
        gd_backend_destroy(backend);
        return 4;
    }

    st = gd_backend_upload(backend, buffer, 0U, src, sizeof(src));
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_upload failed: %d\n", (int)st);
        gd_backend_buffer_destroy(buffer);
        gd_backend_destroy(backend);
        return 5;
    }
    st = gd_backend_fill(backend, buffer, sizeof(uint32_t), 2U, sizeof(uint32_t), 0xa5a5a5a5U);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_fill failed: %d\n", (int)st);
        gd_backend_buffer_destroy(buffer);
        gd_backend_destroy(backend);
        return 6;
    }
    st = gd_backend_record_fence(backend, &fence);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_record_fence failed: %d\n", (int)st);
        gd_backend_buffer_destroy(buffer);
        gd_backend_destroy(backend);
        return 7;
    }
    st = gd_backend_fence_wait(&fence);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_fence_wait failed: %d\n", (int)st);
        gd_backend_fence_destroy(&fence);
        gd_backend_buffer_destroy(buffer);
        gd_backend_destroy(backend);
        return 8;
    }
    gd_backend_fence_destroy(&fence);

    st = gd_backend_download(backend, buffer, 0U, dst, sizeof(dst));
    gd_backend_buffer_destroy(buffer);
    gd_backend_destroy(backend);
    if (st != GD_OK) {
        fprintf(stderr, "gd_backend_download failed: %d\n", (int)st);
        return 9;
    }
    if (dst[0] != 1U || dst[1] != 0xa5a5a5a5U || dst[2] != 0xa5a5a5a5U || dst[3] != 4U) {
        fprintf(stderr,
                "bad CUDA smoke data: %08x %08x %08x %08x\n",
                dst[0],
                dst[1],
                dst[2],
                dst[3]);
        return 10;
    }

    printf("[cuda-probe] backend=cuda ok\n");
    return 0;
}
