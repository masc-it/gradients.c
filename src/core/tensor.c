#include <gradients/tensor.h>

#include "backend.h"
#include "memory_internal.h"

#include <stdint.h>
#include <string.h>

size_t gd_dtype_size(gd_dtype dtype)
{
    switch (dtype) {
    case GD_DTYPE_F16:
    case GD_DTYPE_BF16:
        return 2U;
    case GD_DTYPE_F32:
    case GD_DTYPE_I32:
        return 4U;
    case GD_DTYPE_U8:
        return 1U;
    case GD_DTYPE_INVALID:
    default:
        return 0U;
    }
}

const char *gd_dtype_name(gd_dtype dtype)
{
    switch (dtype) {
    case GD_DTYPE_F16:
        return "f16";
    case GD_DTYPE_BF16:
        return "bf16";
    case GD_DTYPE_F32:
        return "f32";
    case GD_DTYPE_I32:
        return "i32";
    case GD_DTYPE_U8:
        return "u8";
    case GD_DTYPE_INVALID:
    default:
        return "invalid";
    }
}

static bool gd_i64_mul_overflow(int64_t a, int64_t b, int64_t *out)
{
    if (a < 0 || b < 0 || out == NULL) {
        return true;
    }
    if (a != 0 && b > INT64_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static bool gd_i64_add_overflow(int64_t a, int64_t b, int64_t *out)
{
    if (a < 0 || b < 0 || out == NULL) {
        return true;
    }
    if (b > INT64_MAX - a) {
        return true;
    }
    *out = a + b;
    return false;
}

static gd_status gd_shape_numel(uint32_t rank, const int64_t *shape, int64_t *out_numel)
{
    uint32_t i;
    int64_t n = 1;
    if (rank > GD_MAX_DIMS || out_numel == NULL || (rank > 0U && shape == NULL)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < rank; ++i) {
        if (shape[i] <= 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (gd_i64_mul_overflow(n, shape[i], &n)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    *out_numel = n;
    return GD_OK;
}

static float gd_f16_bits_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    int32_t exp = (int32_t)(((uint32_t)bits >> 10) & 0x1fU);
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1;
        }
        mant &= 0x3ffU;
        exp += 1;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

static gd_status gd_shape_storage_nbytes(gd_dtype dtype,
                                         uint32_t rank,
                                         const int64_t *shape,
                                         size_t *out_nbytes)
{
    int64_t numel;
    size_t elem_size;
    gd_status st;
    if (out_nbytes == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_dtype_size(dtype);
    if (elem_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_shape_numel(rank, shape, &numel);
    if (st != GD_OK) {
        return st;
    }
    if ((uint64_t)numel > (uint64_t)(SIZE_MAX / elem_size)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out_nbytes = (size_t)numel * elem_size;
    return GD_OK;
}

static void gd_tensor_set_contiguous_strides(gd_tensor *tensor)
{
    uint32_t i;
    int64_t stride = 1;
    if (tensor == NULL) {
        return;
    }
    for (i = tensor->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        tensor->strides[dim] = stride;
        stride *= tensor->shape[dim];
    }
}

static gd_status gd_tensor_logical_span_from_allocation(const gd_tensor *tensor,
                                                        size_t *out_nbytes)
{
    uint32_t i;
    int64_t max_elem_offset = 0;
    size_t elem_size;
    if (tensor == NULL || out_nbytes == NULL || tensor->rank > GD_MAX_DIMS) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < tensor->rank; ++i) {
        int64_t term;
        if (tensor->shape[i] <= 0 || tensor->strides[i] < 0) {
            return GD_ERR_INVALID_ARGUMENT;
        }
        if (gd_i64_mul_overflow(tensor->shape[i] - 1, tensor->strides[i], &term) ||
            gd_i64_add_overflow(max_elem_offset, term, &max_elem_offset)) {
            return GD_ERR_OUT_OF_MEMORY;
        }
    }
    if ((uint64_t)max_elem_offset > (uint64_t)(SIZE_MAX / elem_size)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    {
        size_t bytes = (size_t)max_elem_offset * elem_size;
        if (bytes > SIZE_MAX - elem_size || tensor->view_offset > SIZE_MAX - bytes - elem_size) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        *out_nbytes = tensor->view_offset + bytes + elem_size;
    }
    return GD_OK;
}

bool gd_tensor_is_contiguous(const gd_tensor *tensor)
{
    uint32_t i;
    int64_t stride = 1;
    if (tensor == NULL || tensor->rank > GD_MAX_DIMS) {
        return false;
    }
    for (i = tensor->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        if (tensor->shape[dim] <= 0 || tensor->strides[dim] != stride) {
            return false;
        }
        if (tensor->shape[dim] != 0 && stride > INT64_MAX / tensor->shape[dim]) {
            return false;
        }
        stride *= tensor->shape[dim];
    }
    return true;
}

size_t gd_tensor_storage_offset(const gd_tensor *tensor)
{
    if (tensor == NULL || tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return 0U;
    }
    return tensor->storage.offset + tensor->view_offset;
}

gd_status gd_tensor_numel(const gd_tensor *tensor, int64_t *out_numel)
{
    if (tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_shape_numel(tensor->rank, tensor->shape, out_numel);
}

gd_status gd_tensor_logical_nbytes(const gd_tensor *tensor, size_t *out_nbytes)
{
    int64_t numel;
    size_t elem_size;
    gd_status st;
    if (tensor == NULL || out_nbytes == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK) {
        return st;
    }
    if ((uint64_t)numel > (uint64_t)(SIZE_MAX / elem_size)) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out_nbytes = (size_t)numel * elem_size;
    return GD_OK;
}

gd_status gd_tensor_item(gd_context *ctx, const gd_tensor *src, float *out)
{
    gd_backend *backend;
    gd_status st;
    int64_t numel;
    size_t offset;
    if (out != NULL) {
        *out = 0.0f;
    }
    if (ctx == NULL || src == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_numel(src, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "tensor item invalid shape");
    }
    if (numel != 1) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor item requires exactly one element");
    }
    if (src->dtype != GD_DTYPE_F16 && src->dtype != GD_DTYPE_F32) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "tensor item supports f16 and f32 tensors");
    }
    if (src->storage.offset > SIZE_MAX - src->view_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor item invalid descriptor");
    }
    st = gd_context_wait_for_span(ctx, &src->storage);
    if (st != GD_OK) {
        return st;
    }
    st = gd_context_flush_backend(ctx);
    if (st != GD_OK) {
        return st;
    }
    backend = gd_context_backend(ctx);
    if (backend == NULL || src->storage.buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "tensor item missing backend buffer");
    }
    offset = src->storage.offset + src->view_offset;
    if (src->dtype == GD_DTYPE_F16) {
        uint16_t bits = 0U;
        st = gd_backend_download(backend,
                                 (gd_backend_buffer *)src->storage.buffer,
                                 offset,
                                 &bits,
                                 sizeof(bits));
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend tensor item download failed");
        }
        *out = gd_f16_bits_to_f32(bits);
        return GD_OK;
    }
    {
        float value = 0.0f;
        st = gd_backend_download(backend,
                                 (gd_backend_buffer *)src->storage.buffer,
                                 offset,
                                 &value,
                                 sizeof(value));
        if (st != GD_OK) {
            return gd_context_set_error(ctx, st, "backend tensor item download failed");
        }
        *out = value;
    }
    return GD_OK;
}

gd_status gd_tensor_validate(gd_context *ctx, const gd_tensor *tensor)
{
    size_t logical_span;
    gd_status st;
    if (ctx == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (tensor->device != GD_DEVICE_GPU || tensor->layout != GD_LAYOUT_STRIDED ||
        tensor->rank > GD_MAX_DIMS || gd_dtype_size(tensor->dtype) == 0U) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid tensor descriptor");
    }
    st = gd_context_validate_span(ctx, &tensor->storage, "tensor storage is stale");
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_logical_span_from_allocation(tensor, &logical_span);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "invalid tensor shape or strides");
    }
    if (logical_span > tensor->storage.nbytes) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "tensor logical span exceeds allocation");
    }
    return GD_OK;
}

static gd_status gd_tensor_count(const gd_tensor *tensor, size_t *out_count)
{
    int64_t numel;
    gd_status st;
    if (tensor == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK) {
        return st;
    }
    if ((uint64_t)numel > (uint64_t)SIZE_MAX) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static bool gd_dtype_one_pattern(gd_dtype dtype, uint32_t *out_pattern)
{
    if (out_pattern == NULL) {
        return false;
    }
    switch (dtype) {
    case GD_DTYPE_F16:
        *out_pattern = 0x00003c00U;
        return true;
    case GD_DTYPE_BF16:
        *out_pattern = 0x00003f80U;
        return true;
    case GD_DTYPE_F32:
        *out_pattern = 0x3f800000U;
        return true;
    case GD_DTYPE_I32:
    case GD_DTYPE_U8:
        *out_pattern = 1U;
        return true;
    case GD_DTYPE_INVALID:
    default:
        return false;
    }
}

static bool gd_dtype_rand_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_BF16 || dtype == GD_DTYPE_F32;
}

static gd_status gd_tensor_fill_pattern(gd_context *ctx, gd_tensor *tensor, uint32_t pattern)
{
    gd_backend *backend;
    gd_status st;
    size_t count;
    size_t elem_size;
    size_t offset;
    if (ctx == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor fill requires contiguous tensor");
    }
    st = gd_context_wait_for_span(ctx, &tensor->storage);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_count(tensor, &count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "tensor fill invalid shape");
    }
    elem_size = gd_dtype_size(tensor->dtype);
    if (elem_size == 0U || tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor fill invalid descriptor");
    }
    offset = tensor->storage.offset + tensor->view_offset;
    backend = gd_context_backend(ctx);
    if (backend == NULL || tensor->storage.buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "tensor fill missing backend buffer");
    }
    st = gd_backend_fill(backend,
                         (gd_backend_buffer *)tensor->storage.buffer,
                         offset,
                         count,
                         elem_size,
                         pattern);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend tensor fill failed");
    }
    tensor->version += 1U;
    if (tensor->version == 0U) {
        tensor->version = 1U;
    }
    return GD_OK;
}

static gd_status gd_tensor_alloc_then_zero_or_one(gd_context *ctx,
                                                  gd_arena_kind arena,
                                                  gd_dtype dtype,
                                                  uint32_t rank,
                                                  const int64_t *shape,
                                                  size_t alignment,
                                                  bool one,
                                                  gd_tensor *out)
{
    gd_tensor tensor;
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_empty(ctx, arena, dtype, rank, shape, alignment, &tensor);
    if (st != GD_OK) {
        return st;
    }
    st = one ? gd_tensor_one_(ctx, &tensor) : gd_tensor_zero_(ctx, &tensor);
    if (st != GD_OK) {
        return st;
    }
    *out = tensor;
    return GD_OK;
}

gd_status gd_tensor_empty(gd_context *ctx,
                          gd_arena_kind arena,
                          gd_dtype dtype,
                          uint32_t rank,
                          const int64_t *shape,
                          size_t alignment,
                          gd_tensor *out)
{
    gd_tensor tensor;
    size_t nbytes;
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_shape_storage_nbytes(dtype, rank, shape, &nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "invalid tensor allocation shape");
    }
    memset(&tensor, 0, sizeof(tensor));
    st = gd_context_next_tensor_id(ctx, &tensor.id);
    if (st != GD_OK) {
        return st;
    }
    tensor.version = 0U;
    tensor.dtype = dtype;
    tensor.device = GD_DEVICE_GPU;
    tensor.layout = GD_LAYOUT_STRIDED;
    tensor.rank = rank;
    if (rank > 0U) {
        memcpy(tensor.shape, shape, (size_t)rank * sizeof(tensor.shape[0]));
    }
    gd_tensor_set_contiguous_strides(&tensor);
    st = gd_context_alloc_span(ctx, arena, nbytes, alignment, &tensor.storage);
    if (st != GD_OK) {
        return st;
    }
    tensor.view_offset = 0U;
    tensor.is_view = false;
    tensor.requires_grad = false;
    tensor.is_leaf = true;
    st = gd_tensor_validate(ctx, &tensor);
    if (st != GD_OK) {
        return st;
    }
    *out = tensor;
    return GD_OK;
}

gd_status gd_tensor_zeros(gd_context *ctx,
                          gd_arena_kind arena,
                          gd_dtype dtype,
                          uint32_t rank,
                          const int64_t *shape,
                          size_t alignment,
                          gd_tensor *out)
{
    return gd_tensor_alloc_then_zero_or_one(ctx, arena, dtype, rank, shape, alignment, false, out);
}

gd_status gd_tensor_ones(gd_context *ctx,
                         gd_arena_kind arena,
                         gd_dtype dtype,
                         uint32_t rank,
                         const int64_t *shape,
                         size_t alignment,
                         gd_tensor *out)
{
    return gd_tensor_alloc_then_zero_or_one(ctx, arena, dtype, rank, shape, alignment, true, out);
}

gd_status gd_tensor_rand(gd_context *ctx,
                         gd_arena_kind arena,
                         gd_dtype dtype,
                         uint32_t rank,
                         const int64_t *shape,
                         size_t alignment,
                         uint64_t seed,
                         gd_tensor *out)
{
    return gd_tensor_rand_uniform(ctx, arena, dtype, rank, shape, alignment,
                                  seed, 0.0f, 1.0f, out);
}

gd_status gd_tensor_rand_uniform(gd_context *ctx,
                                 gd_arena_kind arena,
                                 gd_dtype dtype,
                                 uint32_t rank,
                                 const int64_t *shape,
                                 size_t alignment,
                                 uint64_t seed,
                                 float low,
                                 float high,
                                 gd_tensor *out)
{
    gd_tensor tensor;
    gd_status st;
    size_t unused_nbytes;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dtype_rand_supported(dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor rand requires floating dtype");
    }
    if (!(low <= high)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor rand invalid range");
    }
    st = gd_shape_storage_nbytes(dtype, rank, shape, &unused_nbytes);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "invalid tensor rand shape");
    }
    st = gd_tensor_empty(ctx, arena, dtype, rank, shape, alignment, &tensor);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_rand_uniform_(ctx, &tensor, seed, low, high);
    if (st != GD_OK) {
        return st;
    }
    *out = tensor;
    return GD_OK;
}

gd_status gd_tensor_slice(gd_context *ctx,
                          const gd_tensor *base,
                          uint32_t dim,
                          int64_t start,
                          int64_t length,
                          gd_tensor *out)
{
    gd_tensor view;
    gd_status st;
    size_t elem_size;
    int64_t elem_delta;
    size_t byte_delta;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || base == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, base);
    if (st != GD_OK) {
        return st;
    }
    if (dim >= base->rank || start < 0 || length <= 0 ||
        start > base->shape[dim] || length > base->shape[dim] - start) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid tensor slice range");
    }
    elem_size = gd_dtype_size(base->dtype);
    if (gd_i64_mul_overflow(start, base->strides[dim], &elem_delta) ||
        (uint64_t)elem_delta > (uint64_t)(SIZE_MAX / elem_size)) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "tensor slice offset overflow");
    }
    byte_delta = (size_t)elem_delta * elem_size;
    if (base->view_offset > SIZE_MAX - byte_delta) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "tensor slice offset overflow");
    }
    view = *base;
    st = gd_context_next_tensor_id(ctx, &view.id);
    if (st != GD_OK) {
        return st;
    }
    view.version = 0U;
    view.shape[dim] = length;
    view.view_offset = base->view_offset + byte_delta;
    view.is_view = true;
    view.is_leaf = false;
    st = gd_tensor_validate(ctx, &view);
    if (st != GD_OK) {
        return st;
    }
    *out = view;
    return GD_OK;
}

gd_status gd_tensor_contiguous(gd_context *ctx,
                               gd_arena_kind arena,
                               const gd_tensor *src,
                               size_t alignment,
                               gd_tensor *out)
{
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || src == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_empty(ctx, arena, src->dtype, src->rank, src->shape, alignment, out);
    if (st != GD_OK) {
        return st;
    }
    out->requires_grad = src->requires_grad;
    out->is_leaf = false;
    return GD_OK;
}

gd_status gd_tensor_zero_(gd_context *ctx, gd_tensor *tensor)
{
    return gd_tensor_fill_pattern(ctx, tensor, 0U);
}

gd_status gd_tensor_one_(gd_context *ctx, gd_tensor *tensor)
{
    uint32_t pattern;
    if (tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dtype_one_pattern(tensor->dtype, &pattern)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor one invalid dtype");
    }
    return gd_tensor_fill_pattern(ctx, tensor, pattern);
}

gd_status gd_tensor_rand_(gd_context *ctx, gd_tensor *tensor, uint64_t seed)
{
    return gd_tensor_rand_uniform_(ctx, tensor, seed, 0.0f, 1.0f);
}

gd_status gd_tensor_rand_uniform_(gd_context *ctx,
                                  gd_tensor *tensor,
                                  uint64_t seed,
                                  float low,
                                  float high)
{
    gd_backend *backend;
    gd_status st;
    size_t count;
    size_t offset;
    if (ctx == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_dtype_rand_supported(tensor->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor rand requires floating dtype");
    }
    if (!(low <= high)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor rand invalid range");
    }
    st = gd_tensor_validate(ctx, tensor);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(tensor)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "tensor rand requires contiguous tensor");
    }
    st = gd_context_wait_for_span(ctx, &tensor->storage);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_count(tensor, &count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "tensor rand invalid shape");
    }
    if (tensor->storage.offset > SIZE_MAX - tensor->view_offset) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "tensor rand invalid descriptor");
    }
    offset = tensor->storage.offset + tensor->view_offset;
    backend = gd_context_backend(ctx);
    if (backend == NULL || tensor->storage.buffer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "tensor rand missing backend buffer");
    }
    st = gd_backend_rand_uniform(backend,
                                 (gd_backend_buffer *)tensor->storage.buffer,
                                 offset,
                                 count,
                                 (uint32_t)tensor->dtype,
                                 seed,
                                 low,
                                 high);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend tensor rand failed");
    }
    tensor->version += 1U;
    if (tensor->version == 0U) {
        tensor->version = 1U;
    }
    return GD_OK;
}
