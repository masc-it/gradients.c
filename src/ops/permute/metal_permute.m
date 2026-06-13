#include "../../backends/metal/metal_backend_internal.h"
#include "metal_permute_types.h"

#include <gradients/tensor.h>

#include <limits.h>
#include <stdint.h>
#include <string.h>

typedef enum gd_permute_kernel_kind {
    GD_PERMUTE_KERNEL_SCALAR = 0,
    GD_PERMUTE_KERNEL_BLOCK = 1,
    GD_PERMUTE_KERNEL_SUFFIX16 = 2,
    GD_PERMUTE_KERNEL_HWC_TO_CHW = 3,
    GD_PERMUTE_KERNEL_CHW_TO_HWC = 4,
    GD_PERMUTE_KERNEL_TRANSPOSE = 5,
} gd_permute_kernel_kind;

static id<MTLBuffer> gd_permute_buffer(gd_backend_buffer *buffer)
{
    return (__bridge id<MTLBuffer>)buffer->buffer;
}

static size_t gd_permute_dtype_size(uint32_t dtype)
{
    if (dtype == (uint32_t)GD_DTYPE_U8) {
        return 1U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16 || dtype == (uint32_t)GD_DTYPE_BF16) {
        return 2U;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32 || dtype == (uint32_t)GD_DTYPE_I32) {
        return 4U;
    }
    return 0U;
}

static id<MTLComputePipelineState> gd_permute_typed_pso(gd_backend *backend,
                                                        size_t elem_size,
                                                        gd_permute_kernel_kind kind)
{
    if (backend == NULL) {
        return nil;
    }
    if (kind == GD_PERMUTE_KERNEL_SUFFIX16) {
        return (__bridge id<MTLComputePipelineState>)backend->permute_suffix16_pso;
    }
    if (elem_size == 1U) {
        if (kind == GD_PERMUTE_KERNEL_BLOCK) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_block_u8_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_HWC_TO_CHW) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_hwc_to_chw_u8_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_CHW_TO_HWC) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_chw_to_hwc_u8_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_TRANSPOSE) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_transpose_u8_pso;
        }
        return (__bridge id<MTLComputePipelineState>)backend->permute_u8_pso;
    }
    if (elem_size == 2U) {
        if (kind == GD_PERMUTE_KERNEL_BLOCK) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_block_u16_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_HWC_TO_CHW) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_hwc_to_chw_u16_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_CHW_TO_HWC) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_chw_to_hwc_u16_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_TRANSPOSE) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_transpose_u16_pso;
        }
        return (__bridge id<MTLComputePipelineState>)backend->permute_u16_pso;
    }
    if (elem_size == 4U) {
        if (kind == GD_PERMUTE_KERNEL_BLOCK) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_block_u32_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_HWC_TO_CHW) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_hwc_to_chw_u32_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_CHW_TO_HWC) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_chw_to_hwc_u32_pso;
        }
        if (kind == GD_PERMUTE_KERNEL_TRANSPOSE) {
            return (__bridge id<MTLComputePipelineState>)backend->permute_transpose_u32_pso;
        }
        return (__bridge id<MTLComputePipelineState>)backend->permute_u32_pso;
    }
    return nil;
}

static bool gd_permute_byte_range_valid(const gd_backend_buffer *buffer,
                                        size_t offset,
                                        size_t nbytes)
{
    return buffer != NULL && offset <= buffer->nbytes && nbytes <= buffer->nbytes - offset;
}

static bool gd_permute_count_bytes(uint64_t count, size_t elem_size, size_t *out_nbytes)
{
    if (out_nbytes == NULL || count == 0U || elem_size == 0U || count > (uint64_t)(SIZE_MAX / elem_size)) {
        return false;
    }
    *out_nbytes = (size_t)count * elem_size;
    return true;
}

static bool gd_permute_contiguous_view(const gd_backend_tensor_view *view)
{
    int64_t expected = 1;
    int32_t d;
    if (view == NULL || view->rank > GD_MAX_DIMS || view->count == 0U) {
        return false;
    }
    if (view->rank == 0U) {
        return view->count == 1U;
    }
    for (d = (int32_t)view->rank - 1; d >= 0; --d) {
        if (view->shape[d] <= 0 || view->strides[d] != expected ||
            expected > INT64_MAX / view->shape[d]) {
            return false;
        }
        expected *= view->shape[d];
    }
    return view->count == (size_t)expected;
}

static bool gd_permute_view_range_valid(const gd_backend_tensor_view *view)
{
    size_t nbytes;
    size_t elem_size;
    if (view == NULL || view->buffer == NULL || view->count == 0U) {
        return false;
    }
    elem_size = gd_permute_dtype_size(view->dtype);
    if (!gd_permute_count_bytes((uint64_t)view->count, elem_size, &nbytes)) {
        return false;
    }
    return elem_size != 0U && view->offset % elem_size == 0U &&
           gd_permute_byte_range_valid(view->buffer, view->offset, nbytes);
}

static bool gd_permute_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
{
    if (out == NULL) {
        return false;
    }
    if (a != 0U && b > UINT64_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static bool gd_permute_axes_are_permutation(const gd_backend_permute_args *args)
{
    uint32_t d;
    uint32_t seen = 0U;
    if (args == NULL || args->rank > GD_MAX_DIMS) {
        return false;
    }
    for (d = 0U; d < args->rank; ++d) {
        uint32_t bit;
        if (args->axes[d] >= args->rank) {
            return false;
        }
        bit = 1U << args->axes[d];
        if ((seen & bit) != 0U) {
            return false;
        }
        seen |= bit;
    }
    return true;
}

static gd_status gd_permute_validate_args(const gd_backend_tensor_view *src,
                                          const gd_backend_tensor_view *dst,
                                          const gd_backend_permute_args *args,
                                          size_t *out_elem_size)
{
    size_t elem_size;
    uint32_t d;
    uint64_t inner = 1U;
    if (src == NULL || dst == NULL || args == NULL || out_elem_size == NULL ||
        !gd_permute_view_range_valid(src) || !gd_permute_view_range_valid(dst) ||
        !gd_permute_contiguous_view(src) || !gd_permute_contiguous_view(dst) ||
        src->dtype != dst->dtype || src->rank != dst->rank || src->rank != args->rank ||
        args->rank > GD_MAX_DIMS || args->active_rank > args->rank || args->count == 0U ||
        args->count > UINT32_MAX || src->count != (size_t)args->count ||
        dst->count != (size_t)args->count || args->inner == 0U ||
        !gd_permute_axes_are_permutation(args)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_permute_dtype_size(src->dtype);
    if (elem_size == 0U) {
        return GD_ERR_UNSUPPORTED;
    }
    for (d = 0U; d < args->rank; ++d) {
        if (src->shape[d] <= 0 || dst->shape[d] <= 0 ||
            dst->shape[d] != src->shape[args->axes[d]]) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    for (d = args->active_rank; d < args->rank; ++d) {
        if (args->axes[d] != d || dst->shape[d] <= 0 ||
            !gd_permute_u64_mul(inner, (uint64_t)dst->shape[d], &inner)) {
            return GD_ERR_INVALID_ARGUMENT;
        }
    }
    if (inner != args->inner) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    *out_elem_size = elem_size;
    return GD_OK;
}

static bool gd_permute_is_matrix_transpose(const gd_backend_tensor_view *src,
                                           const gd_backend_tensor_view *dst,
                                           const gd_backend_permute_args *args,
                                           uint64_t *out_rows,
                                           uint64_t *out_cols,
                                           uint64_t *out_batch)
{
    if (src == NULL || dst == NULL || args == NULL || out_rows == NULL ||
        out_cols == NULL || out_batch == NULL) {
        return false;
    }
    if (args->rank == 2U) {
        if (args->axes[0] != 1U || args->axes[1] != 0U ||
            src->shape[0] <= 0 || src->shape[1] <= 0) {
            return false;
        }
        *out_rows = (uint64_t)src->shape[0];
        *out_cols = (uint64_t)src->shape[1];
        *out_batch = 1U;
        return dst->shape[0] == src->shape[1] && dst->shape[1] == src->shape[0];
    }
    if (args->rank == 3U) {
        if (args->axes[0] != 0U || args->axes[1] != 2U || args->axes[2] != 1U ||
            src->shape[0] <= 0 || src->shape[1] <= 0 || src->shape[2] <= 0) {
            return false;
        }
        *out_rows = (uint64_t)src->shape[1];
        *out_cols = (uint64_t)src->shape[2];
        *out_batch = (uint64_t)src->shape[0];
        return dst->shape[0] == src->shape[0] && dst->shape[1] == src->shape[2] &&
               dst->shape[2] == src->shape[1];
    }
    return false;
}

static bool gd_permute_is_hwc_to_chw(const gd_backend_tensor_view *src,
                                     const gd_backend_tensor_view *dst,
                                     const gd_backend_permute_args *args,
                                     uint64_t *out_pixels)
{
    uint64_t hw;
    uint64_t pixels;
    uint64_t n;
    uint64_t c;
    if (src == NULL || dst == NULL || args == NULL || out_pixels == NULL) {
        return false;
    }
    if (args->rank == 3U) {
        if (args->axes[0] != 2U || args->axes[1] != 0U || args->axes[2] != 1U ||
            src->shape[2] <= 0 || src->shape[2] > 16 || dst->shape[0] != src->shape[2]) {
            return false;
        }
        if (!gd_permute_u64_mul((uint64_t)src->shape[0], (uint64_t)src->shape[1], &pixels)) {
            return false;
        }
        *out_pixels = pixels;
        return pixels > 0U;
    }
    if (args->rank == 4U) {
        if (args->axes[0] != 0U || args->axes[1] != 3U || args->axes[2] != 1U ||
            args->axes[3] != 2U || src->shape[3] <= 0 || src->shape[3] > 16 ||
            dst->shape[1] != src->shape[3]) {
            return false;
        }
        n = (uint64_t)src->shape[0];
        c = (uint64_t)src->shape[3];
        if (!gd_permute_u64_mul((uint64_t)src->shape[1], (uint64_t)src->shape[2], &hw) ||
            !gd_permute_u64_mul(n, hw, &pixels) || pixels == 0U ||
            !gd_permute_u64_mul(pixels, c, &hw) || hw != args->count) {
            return false;
        }
        *out_pixels = pixels;
        return true;
    }
    return false;
}

static bool gd_permute_is_chw_to_hwc(const gd_backend_tensor_view *src,
                                     const gd_backend_tensor_view *dst,
                                     const gd_backend_permute_args *args,
                                     uint64_t *out_pixels)
{
    uint64_t hw;
    uint64_t pixels;
    uint64_t n;
    uint64_t c;
    if (src == NULL || dst == NULL || args == NULL || out_pixels == NULL) {
        return false;
    }
    if (args->rank == 3U) {
        if (args->axes[0] != 1U || args->axes[1] != 2U || args->axes[2] != 0U ||
            src->shape[0] <= 0 || src->shape[0] > 16 || dst->shape[2] != src->shape[0]) {
            return false;
        }
        if (!gd_permute_u64_mul((uint64_t)src->shape[1], (uint64_t)src->shape[2], &pixels)) {
            return false;
        }
        *out_pixels = pixels;
        return pixels > 0U;
    }
    if (args->rank == 4U) {
        if (args->axes[0] != 0U || args->axes[1] != 2U || args->axes[2] != 3U ||
            args->axes[3] != 1U || src->shape[1] <= 0 || src->shape[1] > 16 ||
            dst->shape[3] != src->shape[1]) {
            return false;
        }
        n = (uint64_t)src->shape[0];
        c = (uint64_t)src->shape[1];
        if (!gd_permute_u64_mul((uint64_t)src->shape[2], (uint64_t)src->shape[3], &hw) ||
            !gd_permute_u64_mul(n, hw, &pixels) || pixels == 0U ||
            !gd_permute_u64_mul(pixels, c, &hw) || hw != args->count) {
            return false;
        }
        *out_pixels = pixels;
        return true;
    }
    return false;
}

static bool gd_permute_can_use_suffix16(const gd_backend_tensor_view *src,
                                        const gd_backend_tensor_view *dst,
                                        const gd_backend_permute_args *args,
                                        size_t elem_size,
                                        uint64_t *out_vecs_per_block,
                                        uint64_t *out_blocks)
{
    uint64_t inner_bytes;
    uint64_t blocks;
    if (src == NULL || dst == NULL || args == NULL || out_vecs_per_block == NULL ||
        out_blocks == NULL || elem_size == 0U || args->inner == 0U ||
        args->count % args->inner != 0U || src->offset % 16U != 0U || dst->offset % 16U != 0U) {
        return false;
    }
    if (!gd_permute_u64_mul(args->inner, (uint64_t)elem_size, &inner_bytes) ||
        inner_bytes < 16U || inner_bytes % 16U != 0U) {
        return false;
    }
    blocks = args->count / args->inner;
    if (blocks == 0U) {
        return false;
    }
    *out_vecs_per_block = inner_bytes / 16U;
    *out_blocks = blocks;
    return true;
}

static uint32_t gd_permute_threads_for_count(uint64_t count)
{
    if (count <= 64U) {
        return 64U;
    }
    if (count <= 128U) {
        return 128U;
    }
    return GD_METAL_PERMUTE_THREADS;
}

static MTLSize gd_permute_suffix16_threads(uint64_t vecs_per_block, uint64_t blocks)
{
    uint64_t tx = vecs_per_block < 32U ? vecs_per_block : 32U;
    uint64_t ty;
    if (tx == 0U) {
        tx = 1U;
    }
    ty = GD_METAL_PERMUTE_THREADS / tx;
    if (ty == 0U) {
        ty = 1U;
    }
    if (blocks < ty) {
        ty = blocks == 0U ? 1U : blocks;
    }
    return MTLSizeMake((NSUInteger)tx, (NSUInteger)ty, 1U);
}

gd_status gd_backend_permute(gd_backend *backend,
                             const gd_backend_tensor_view *src,
                             const gd_backend_tensor_view *dst,
                             const gd_backend_permute_args *args)
{
    id<MTLCommandBuffer> command_buffer;
    id<MTLComputeCommandEncoder> encoder;
    id<MTLComputePipelineState> pso;
    gd_metal_permute_args metal_args;
    MTLSize grid;
    MTLSize threads;
    bool immediate;
    gd_permute_kernel_kind kind = GD_PERMUTE_KERNEL_SCALAR;
    size_t elem_size;
    uint64_t grid_count = 0U;
    uint64_t vecs_per_block = 0U;
    uint64_t blocks = 0U;
    uint64_t transpose_rows = 0U;
    uint64_t transpose_cols = 0U;
    uint64_t transpose_batch = 0U;
    gd_status st;
    uint32_t d;
    if (backend == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_permute_validate_args(src, dst, args, &elem_size);
    if (st != GD_OK) {
        return st;
    }
    if (gd_permute_is_hwc_to_chw(src, dst, args, &grid_count)) {
        kind = GD_PERMUTE_KERNEL_HWC_TO_CHW;
    } else if (gd_permute_is_chw_to_hwc(src, dst, args, &grid_count)) {
        kind = GD_PERMUTE_KERNEL_CHW_TO_HWC;
    } else if (gd_permute_is_matrix_transpose(src,
                                             dst,
                                             args,
                                             &transpose_rows,
                                             &transpose_cols,
                                             &transpose_batch)) {
        kind = GD_PERMUTE_KERNEL_TRANSPOSE;
    } else if (gd_permute_can_use_suffix16(src, dst, args, elem_size, &vecs_per_block, &blocks)) {
        kind = GD_PERMUTE_KERNEL_SUFFIX16;
    } else if (args->active_rank > 0U && args->inner >= 4U && args->inner <= 1024U &&
               args->count % args->inner == 0U) {
        kind = GD_PERMUTE_KERNEL_BLOCK;
        grid_count = args->count / args->inner;
    } else {
        grid_count = args->count;
    }
    pso = gd_permute_typed_pso(backend, elem_size, kind);
    if (pso == nil) {
        return GD_ERR_UNSUPPORTED;
    }
    st = gd_metal_command_for_op(backend, &command_buffer, &immediate);
    if (st != GD_OK) {
        return st;
    }
    encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        return GD_ERR_INTERNAL;
    }
    memset(&metal_args, 0, sizeof(metal_args));
    metal_args.src_offset = (uint64_t)src->offset;
    metal_args.dst_offset = (uint64_t)dst->offset;
    metal_args.count = args->count;
    metal_args.inner = args->inner;
    metal_args.rank = args->rank;
    metal_args.active_rank = args->active_rank;
    metal_args.reserved0 = (uint32_t)elem_size;
    for (d = 0U; d < args->rank; ++d) {
        metal_args.dst_shape[d] = (uint64_t)dst->shape[d];
        metal_args.src_strides[d] = (uint64_t)src->strides[d];
        metal_args.axes[d] = args->axes[d];
    }
    [encoder setComputePipelineState:pso];
    [encoder setBuffer:gd_permute_buffer(src->buffer) offset:0U atIndex:0U];
    [encoder setBuffer:gd_permute_buffer(dst->buffer) offset:0U atIndex:1U];
    [encoder setBytes:&metal_args length:sizeof(metal_args) atIndex:2U];
    if (kind == GD_PERMUTE_KERNEL_SUFFIX16) {
        grid = MTLSizeMake((NSUInteger)vecs_per_block, (NSUInteger)blocks, 1U);
        threads = gd_permute_suffix16_threads(vecs_per_block, blocks);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    } else if (kind == GD_PERMUTE_KERNEL_TRANSPOSE) {
        grid = MTLSizeMake((NSUInteger)((transpose_cols + 15U) / 16U),
                           (NSUInteger)((transpose_rows + 15U) / 16U),
                           (NSUInteger)transpose_batch);
        threads = MTLSizeMake(16U, 16U, 1U);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threads];
    } else {
        grid = MTLSizeMake((NSUInteger)grid_count, 1U, 1U);
        threads = MTLSizeMake((NSUInteger)gd_permute_threads_for_count(grid_count), 1U, 1U);
        [encoder dispatchThreads:grid threadsPerThreadgroup:threads];
    }
    [encoder endEncoding];
    return gd_metal_finish_immediate(command_buffer, immediate);
}
