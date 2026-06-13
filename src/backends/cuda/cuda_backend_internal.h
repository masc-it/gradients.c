#ifndef GD_CUDA_BACKEND_INTERNAL_H
#define GD_CUDA_BACKEND_INTERNAL_H

#include "../../core/backend.h"

#include <cuda_runtime.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GD_CUDA_DEFAULT_THREADS_PER_BLOCK 256U

struct gd_backend {
    cudaStream_t stream;
    bool scope_active;
};

struct gd_backend_buffer {
    void *ptr;
    size_t nbytes;
};

gd_status gd_cuda_status(cudaError_t err);
bool gd_cuda_byte_range_valid(const gd_backend_buffer *buffer, size_t offset, size_t nbytes);
bool gd_cuda_count_bytes(size_t count, size_t elem_size, size_t *out_nbytes);
gd_status gd_cuda_finish_if_immediate(gd_backend *backend);

static inline cudaStream_t gd_cuda_stream(gd_backend *backend)
{
    return backend != NULL ? backend->stream : (cudaStream_t)0;
}

static inline void *gd_cuda_buffer_ptr(gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->ptr : NULL;
}

static inline const void *gd_cuda_buffer_const_ptr(const gd_backend_buffer *buffer)
{
    return buffer != NULL ? buffer->ptr : NULL;
}

static inline unsigned int gd_cuda_blocks_for_count(size_t count, unsigned int threads_per_block)
{
    if (threads_per_block == 0U) {
        return 0U;
    }
    return (unsigned int)((count + (size_t)threads_per_block - 1U) / (size_t)threads_per_block);
}

#endif /* GD_CUDA_BACKEND_INTERNAL_H */
