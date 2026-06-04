#include "metal_backend_internal.h"

#import <Foundation/Foundation.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define GD_METAL_MAX_THREADS_PER_GROUP 256U

typedef struct gd_metal_fill_args {
    uint64_t byte_offset;
    uint64_t count;
    uint32_t elem_size;
    uint32_t pattern;
} gd_metal_fill_args;

typedef struct gd_metal_rand_uniform_args {
    uint64_t byte_offset;
    uint64_t count;
    uint32_t dtype;
    uint32_t pad0;
    uint64_t seed;
    float low;
    float high;
} gd_metal_rand_uniform_args;

static id<MTLDevice> gd_metal_device(gd_backend *backend)
{
    return (__bridge id<MTLDevice>)backend->device;
}

static id<MTLCommandQueue> gd_metal_queue(gd_backend *backend)
{
    return (__bridge id<MTLCommandQueue>)backend->queue;
}

static id<MTLComputePipelineState> gd_metal_fill_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->fill_pso;
}

static id<MTLComputePipelineState> gd_metal_rand_uniform_pso(gd_backend *backend)
{
    return (__bridge id<MTLComputePipelineState>)backend->rand_uniform_pso;
}

static id<MTLCommandBuffer> gd_metal_active_command_buffer(gd_backend *backend)
{
    return (__bridge id<MTLCommandBuffer>)backend->active_command_buffer;
}

static id<MTLBuffer> gd_metal_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static id<MTLCommandBuffer> gd_metal_command_buffer(gd_backend_fence *fence)
{
    return (__bridge id<MTLCommandBuffer>)fence->handle;
}

static const char *gd_metal_kernel_source(void)
{
    return "#include <metal_stdlib>\n"
           "using namespace metal;\n"
           "struct gd_fill_args { ulong byte_offset; ulong count; uint elem_size; uint pattern; };\n"
           "struct gd_rand_uniform_args { ulong byte_offset; ulong count; uint dtype; uint pad0; ulong seed; float low; float high; };\n"
           "static inline void gd_write_pattern(device uchar *dst, ulong byte, uint elem_size, uint pattern) {\n"
           "  for (uint i = 0; i < elem_size; ++i) { dst[byte + i] = uchar((pattern >> (8u * i)) & 255u); }\n"
           "}\n"
           "static inline ulong gd_splitmix64(ulong x) {\n"
           "  x += 0x9E3779B97F4A7C15ul;\n"
           "  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ul;\n"
           "  x = (x ^ (x >> 27)) * 0x94D049BB133111EBul;\n"
           "  return x ^ (x >> 31);\n"
           "}\n"
           "static inline ushort gd_bf16_bits(float v) {\n"
           "  uint bits = as_type<uint>(v);\n"
           "  uint lsb = (bits >> 16) & 1u;\n"
           "  bits += 0x7fffu + lsb;\n"
           "  return ushort(bits >> 16);\n"
           "}\n"
           "static inline void gd_write_float_dtype(device uchar *dst, ulong byte, uint dtype, float v) {\n"
           "  if (dtype == 1u) { ushort bits = as_type<ushort>(half(v)); gd_write_pattern(dst, byte, 2u, uint(bits)); }\n"
           "  else if (dtype == 2u) { gd_write_pattern(dst, byte, 2u, uint(gd_bf16_bits(v))); }\n"
           "  else { gd_write_pattern(dst, byte, 4u, as_type<uint>(v)); }\n"
           "}\n"
           "kernel void gd_fill_kernel(device uchar *dst [[buffer(0)]], constant gd_fill_args &args [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
           "  ulong i = ulong(gid);\n"
           "  if (i >= args.count) { return; }\n"
           "  gd_write_pattern(dst, args.byte_offset + i * ulong(args.elem_size), args.elem_size, args.pattern);\n"
           "}\n"
           "kernel void gd_rand_uniform_kernel(device uchar *dst [[buffer(0)]], constant gd_rand_uniform_args &args [[buffer(1)]], uint gid [[thread_position_in_grid]]) {\n"
           "  ulong i = ulong(gid);\n"
           "  if (i >= args.count) { return; }\n"
           "  ulong r = gd_splitmix64(args.seed + i);\n"
           "  uint mant = uint((r >> 40) & 0xfffffful);\n"
           "  float u = float(mant) * (1.0f / 16777216.0f);\n"
           "  float v = args.low + (args.high - args.low) * u;\n"
           "  ulong elem_size = args.dtype == 3u ? 4ul : 2ul;\n"
           "  gd_write_float_dtype(dst, args.byte_offset + i * elem_size, args.dtype, v);\n"
           "}\n";
}

static gd_status gd_metal_make_pipelines(gd_backend *backend)
{
    NSError *error = nil;
    NSString *source;
    id<MTLLibrary> library;
    id<MTLFunction> fill_function;
    id<MTLFunction> rand_uniform_function;
    id<MTLComputePipelineState> fill_pso;
    id<MTLComputePipelineState> rand_uniform_pso;
    source = [NSString stringWithUTF8String:gd_metal_kernel_source()];
    if (source == nil) {
        return GD_ERR_INTERNAL;
    }
    library = [gd_metal_device(backend) newLibraryWithSource:source options:nil error:&error];
    if (library == nil) {
        return GD_ERR_INTERNAL;
    }
    fill_function = [library newFunctionWithName:@"gd_fill_kernel"];
    rand_uniform_function = [library newFunctionWithName:@"gd_rand_uniform_kernel"];
    if (fill_function == nil || rand_uniform_function == nil) {
        return GD_ERR_INTERNAL;
    }
    fill_pso = [gd_metal_device(backend) newComputePipelineStateWithFunction:fill_function error:&error];
    rand_uniform_pso = [gd_metal_device(backend) newComputePipelineStateWithFunction:rand_uniform_function error:&error];
    if (fill_pso == nil || rand_uniform_pso == nil) {
        return GD_ERR_INTERNAL;
    }
    backend->fill_pso = (void *)CFBridgingRetain(fill_pso);
    backend->rand_uniform_pso = (void *)CFBridgingRetain(rand_uniform_pso);
    return GD_OK;
}

static bool gd_metal_byte_range_valid(const gd_backend_buffer *buffer, size_t offset, size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_metal_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > SIZE_MAX / elem_size) {
        return false;
    }
    *out_nbytes = count * elem_size;
    return true;
}

gd_status gd_metal_command_for_op(gd_backend *backend,
                                  id<MTLCommandBuffer> *out_command_buffer,
                                  bool *out_immediate)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL || out_command_buffer == NULL || out_immediate == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    command_buffer = gd_metal_active_command_buffer(backend);
    if (command_buffer != nil) {
        *out_command_buffer = command_buffer;
        *out_immediate = false;
        return GD_OK;
    }
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    if (backend->scope_active) {
        backend->active_command_buffer = (void *)CFBridgingRetain(command_buffer);
        *out_command_buffer = gd_metal_active_command_buffer(backend);
        *out_immediate = false;
        return GD_OK;
    }
    *out_command_buffer = command_buffer;
    *out_immediate = true;
    return GD_OK;
}

gd_status gd_metal_finish_immediate(id<MTLCommandBuffer> command_buffer, bool immediate)
{
    if (!immediate) {
        return GD_OK;
    }
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        return GD_ERR_INTERNAL;
    }
    return GD_OK;
}

gd_status gd_backend_create_default(gd_backend **out_backend)
{
    gd_backend *backend;
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    gd_status st;
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
    st = gd_metal_make_pipelines(backend);
    if (st != GD_OK) {
        gd_backend_destroy(backend);
        return st;
    }
    *out_backend = backend;
    return GD_OK;
}

void gd_backend_destroy(gd_backend *backend)
{
    if (backend == NULL) {
        return;
    }
    if (backend->active_command_buffer != NULL) {
        CFRelease(backend->active_command_buffer);
    }
    if (backend->linear_bias_pso != NULL) {
        CFRelease(backend->linear_bias_pso);
    }
    if (backend->rand_uniform_pso != NULL) {
        CFRelease(backend->rand_uniform_pso);
    }
    if (backend->fill_pso != NULL) {
        CFRelease(backend->fill_pso);
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

gd_status gd_backend_scope_begin(gd_backend *backend)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->scope_active || backend->active_command_buffer != NULL) {
        return GD_ERR_BAD_STATE;
    }
    command_buffer = [gd_metal_queue(backend) commandBuffer];
    if (command_buffer == nil) {
        return GD_ERR_INTERNAL;
    }
    backend->active_command_buffer = (void *)CFBridgingRetain(command_buffer);
    backend->scope_active = true;
    return GD_OK;
}

gd_status gd_backend_flush(gd_backend *backend)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (backend->active_command_buffer == NULL) {
        return GD_OK;
    }
    command_buffer = gd_metal_active_command_buffer(backend);
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        CFRelease(backend->active_command_buffer);
        backend->active_command_buffer = NULL;
        return GD_ERR_INTERNAL;
    }
    CFRelease(backend->active_command_buffer);
    backend->active_command_buffer = NULL;
    return GD_OK;
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
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
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
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    src = (unsigned char *)[gd_metal_buffer(buffer) contents];
    if (src == NULL) {
        return GD_ERR_UNSUPPORTED;
    }
    memcpy(dst, src + offset, nbytes);
    return GD_OK;
}

gd_status gd_backend_fill(gd_backend *backend,
                          gd_backend_buffer *buffer,
                          size_t offset,
                          size_t count,
                          size_t elem_size,
                          uint32_t pattern)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_fill_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t nbytes;
    gd_status st;
    if (backend == NULL || buffer == NULL ||
        (elem_size != 1U && elem_size != 2U && elem_size != 4U) ||
        count > UINT32_MAX || !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.byte_offset = offset;
    args.count = count;
    args.elem_size = (uint32_t)elem_size;
    args.pattern = pattern;
    [encoder setComputePipelineState:gd_metal_fill_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(buffer) offset:0U atIndex:0U];
    [encoder setBytes:&args length:sizeof(args) atIndex:1U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_rand_uniform(gd_backend *backend,
                                  gd_backend_buffer *buffer,
                                  size_t offset,
                                  size_t count,
                                  uint32_t dtype,
                                  uint64_t seed,
                                  float low,
                                  float high)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    gd_metal_rand_uniform_args args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    size_t elem_size;
    size_t nbytes;
    gd_status st;
    if (dtype == 1U || dtype == 2U) {
        elem_size = 2U;
    } else if (dtype == 3U) {
        elem_size = 4U;
    } else {
        return GD_ERR_UNSUPPORTED;
    }
    if (backend == NULL || buffer == NULL || count > UINT32_MAX || !(low <= high) ||
        !gd_metal_count_bytes(count, elem_size, &nbytes) ||
        !gd_metal_byte_range_valid(buffer, offset, nbytes)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    args.byte_offset = offset;
    args.count = count;
    args.dtype = dtype;
    args.pad0 = 0U;
    args.seed = seed;
    args.low = low;
    args.high = high;
    [encoder setComputePipelineState:gd_metal_rand_uniform_pso(backend)];
    [encoder setBuffer:gd_metal_buffer(buffer) offset:0U atIndex:0U];
    [encoder setBytes:&args length:sizeof(args) atIndex:1U];
    grid = MTLSizeMake((NSUInteger)count, 1U, 1U);
    threads = MTLSizeMake((NSUInteger)(count < GD_METAL_MAX_THREADS_PER_GROUP ? count : GD_METAL_MAX_THREADS_PER_GROUP),
                          1U,
                          1U);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}

gd_status gd_backend_record_fence(gd_backend *backend, gd_backend_fence *out_fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (backend == NULL || out_fence == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    out_fence->handle = NULL;
    if (backend->active_command_buffer != NULL) {
        command_buffer = gd_metal_active_command_buffer(backend);
        [command_buffer commit];
        out_fence->handle = backend->active_command_buffer;
        backend->active_command_buffer = NULL;
        backend->scope_active = false;
        return GD_OK;
    }
    backend->scope_active = false;
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
    if (fence == NULL || fence->handle == NULL) {
        return true;
    }
    command_buffer = gd_metal_command_buffer(fence);
    return command_buffer.status == MTLCommandBufferStatusCompleted;
}

gd_status gd_backend_fence_wait(gd_backend_fence *fence)
{
    id<MTLCommandBuffer> command_buffer;
    if (fence == NULL || fence->handle == NULL) {
        return GD_OK;
    }
    command_buffer = gd_metal_command_buffer(fence);
    [command_buffer waitUntilCompleted];
    if (command_buffer.status == MTLCommandBufferStatusError) {
        return GD_ERR_INTERNAL;
    }
    return GD_OK;
}
