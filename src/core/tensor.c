#include <gradients/tensor.h>

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
    st = gd_tensor_validate(ctx, &tensor);
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
    view.shape[dim] = length;
    view.view_offset = base->view_offset + byte_delta;
    view.is_view = true;
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
    return GD_OK;
}
