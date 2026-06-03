#include "../../core/backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

struct gd_backend {
    void *device;
    void *queue;
};

struct gd_backend_buffer {
    void *buffer;
    size_t nbytes;
};

static id<MTLDevice> gd_metal_device(gd_backend *backend)
{
    return (__bridge id<MTLDevice>)backend->device;
}

static id<MTLCommandQueue> gd_metal_queue(gd_backend *backend)
{
    return (__bridge id<MTLCommandQueue>)backend->queue;
}

static id<MTLBuffer> gd_metal_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static id<MTLCommandBuffer> gd_metal_command_buffer(gd_backend_fence *fence)
{
    return (__bridge id<MTLCommandBuffer>)fence->handle;
}

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    gd_backend *backend;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    if (out_backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_backend = NULL;
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    queue = [device newCommandQueue];
    if (queue == nil) {
        return GD_ERR_INTERNAL;
    }
    backend = (gd_backend *)calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    backend->device = (void *)CFBridgingRetain(device);
    backend->queue = (void *)CFBridgingRetain(queue);
    *out_backend = backend;
    return GD_OK;
}

void gd_backend_destroy(gd_backend *backend)
{
    if (backend == NULL) {
        return;
    }
    if (backend->queue != NULL) {
        CFRelease(backend->queue);
    }
    if (backend->device != NULL) {
        CFRelease(backend->device);
    }
    free(backend);
}

gd_backend_kind gd_backend_kind_query(const gd_backend *backend)
{
    if (backend == NULL) {
        return 0;
    }
    return GD_BACKEND_METAL;
}

const char *gd_backend_name(const gd_backend *backend)
{
    if (backend == NULL) {
        return "none";
    }
    return "metal";
}

gd_status gd_backend_buffer_create(gd_backend *backend,
                                   size_t nbytes,
                                   gd_backend_buffer **out_buffer)
{
    gd_backend_buffer *buffer;
    id<MTLBuffer> metal_buffer;
    if (backend == NULL || out_buffer == NULL || nbytes == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = NULL;
    metal_buffer = [gd_metal_device(backend) newBufferWithLength:nbytes
                                                         options:MTLResourceStorageModeShared];
    if (metal_buffer == nil) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    buffer = (gd_backend_buffer *)calloc(1U, sizeof(*buffer));
    if (buffer == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    buffer->buffer = (void *)CFBridgingRetain(metal_buffer);
    buffer->nbytes = nbytes;
    *out_buffer = buffer;
    return GD_OK;
}

void gd_backend_buffer_destroy(gd_backend_buffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    if (buffer->buffer != NULL) {
        CFRelease(buffer->buffer);
    }
    free(buffer);
}

size_t gd_backend_buffer_nbytes(const gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->nbytes : 0U;
}

void *gd_backend_buffer_host_ptr(gd_backend_buffer *buffer)
{
    if (buffer == NULL) {
        return NULL;
    }
    return [gd_metal_buffer(buffer) contents];
}

bool gd_backend_buffer_is_host_visible(const gd_backend_buffer *buffer)
{
    return buffer != NULL;
}

gd_status gd_backend_upload(gd_backend *backend,
                            gd_backend_buffer *buffer,
                            size_t offset,
                            const void *src,
                            size_t nbytes)
{
    unsigned char *dst;
    (void)backend;
    if (buffer == NULL || src == NULL || nbytes == 0U ||
        offset > buffer->nbytes || nbytes > buffer->nbytes - offset) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst = (unsigned char *)[gd_metal_buffer(buffer) contents];
    if (dst == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    memcpy(dst + offset, src, nbytes);
    return GD_OK;
}

gd_status gd_backend_download(gd_backend *backend,
                              gd_backend_buffer *buffer,
                              size_t offset,
                              void *dst,
                              size_t nbytes)
{
    unsigned char *src;
    (void)backend;
    if (buffer == NULL || dst == NULL || nbytes == 0U ||
        offset > buffer->nbytes || nbytes > buffer->nbytes - offset) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    src = (unsigned char *)[gd_metal_buffer(buffer) contents];
    if (src == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    memcpy(dst, src + offset, nbytes);
    return GD_OK;
}

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL || out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = NULL;
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    [command_buffer commit];
    out_fence->handle = (void *)CFBridgingRetain(command_buffer);
    return GD_OK;
}

void gd_backend_fence_destroy(gd_backend_fence *fence)
{
    if (fence == NULL) {
        return;
    }
    if (fence->handle != NULL) {
        CFRelease(fence->handle);
        fence->handle = NULL;
    }
}

bool gd_backend_fence_is_complete(gd_backend_fence *fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (fence == NULL) {
        return true;
    }
    command_buffer = gd_metal_command_buffer(fence);
    return command_buffer.status == MTLCommandBufferStatusCompleted;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (fence == NULL) {
        return GD_OK;
    }
    command_buffer = gd_metal_command_buffer(fence);
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        return GD_ERR_INTERNAL;
    }
    return GD_OK;
}
