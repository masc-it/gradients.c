#include "gradients/graph.h"
#include "gradients/tensor.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "refcount.h"
#include "storage_internal.h"
#include "tensor_internal.h"
#include "../backends/backend.h"
#include "../graph/graph_internal.h"

struct gd_tensor {
    gd_refcount refcount;
    gd_tensor_desc desc;
    gd_storage *storage;
    gd_graph *graph;
    int value_id;
    bool requires_grad;
    gd_tensor *grad;
    char *name;
};

static gd_status checked_mul_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "checked_mul_size out is NULL");
    }
    if (a != 0U && b > SIZE_MAX / a) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "size overflow");
    }
    *out = a * b;
    return GD_OK;
}

static gd_status checked_add_size(size_t a, size_t b, size_t *out)
{
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "checked_add_size out is NULL");
    }
    if (b > SIZE_MAX - a) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "size overflow");
    }
    *out = a + b;
    return GD_OK;
}

static gd_status checked_mul_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "checked_mul_i64 out is NULL");
    }
    if (a < 0 || b < 0 || (a != 0 && b > INT64_MAX / a)) {
        return _gd_error(GD_ERR_SHAPE, "int64 multiplication overflow");
    }
    *out = a * b;
    return GD_OK;
}

static gd_status checked_add_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "checked_add_i64 out is NULL");
    }
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
        return _gd_error(GD_ERR_SHAPE, "int64 addition overflow");
    }
    *out = a + b;
    return GD_OK;
}

static gd_status tensor_numel_desc(const gd_tensor_desc *desc, size_t *out)
{
    gd_status status = GD_OK;
    size_t numel = 1U;
    int i = 0;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tensor_numel_desc out is NULL");
    }
    for (i = 0; i < desc->ndim; ++i) {
        status = checked_mul_size(numel, (size_t)desc->sizes[i], &numel);
        if (status != GD_OK) {
            return status;
        }
    }
    *out = numel;
    return GD_OK;
}

static gd_status validate_tensor_desc(const gd_tensor_desc *desc)
{
    int i = 0;

    if (desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tensor desc is NULL");
    }
    if (desc->ndim < 0 || desc->ndim > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "tensor ndim is out of range");
    }
    if (desc->dtype == GD_DTYPE_INVALID) {
        return _gd_error(GD_ERR_DTYPE, "tensor dtype is invalid");
    }
    if (desc->dtype == GD_DTYPE_QUANTIZED && desc->quant == NULL) {
        return _gd_error(GD_ERR_DTYPE, "quantized tensor requires quant desc");
    }
    if (desc->dtype != GD_DTYPE_QUANTIZED && desc->quant != NULL) {
        return _gd_error(GD_ERR_DTYPE, "non-quantized tensor cannot have quant desc");
    }
    if (desc->layout != GD_LAYOUT_CONTIGUOUS && desc->layout != GD_LAYOUT_STRIDED &&
        desc->layout != GD_LAYOUT_CHANNELS_LAST && desc->layout != GD_LAYOUT_PACKED_QUANT &&
        desc->layout != GD_LAYOUT_BLOCKED && desc->layout != GD_LAYOUT_BACKEND_OPAQUE) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown tensor layout");
    }
    if (desc->storage_offset_bytes < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "negative storage offset is not supported");
    }
    for (i = 0; i < desc->ndim; ++i) {
        if (desc->sizes[i] <= 0) {
            return _gd_error(GD_ERR_SHAPE, "tensor dimensions must be positive");
        }
        if (desc->layout != GD_LAYOUT_PACKED_QUANT && desc->strides[i] < 0) {
            return _gd_error(GD_ERR_SHAPE, "negative strides are not supported");
        }
    }
    return GD_OK;
}

static bool desc_is_contiguous_strides(const gd_tensor_desc *desc)
{
    int i = 0;
    int64_t expected = 1;

    if (desc == NULL || desc->layout == GD_LAYOUT_PACKED_QUANT ||
        desc->layout == GD_LAYOUT_BACKEND_OPAQUE || desc->layout == GD_LAYOUT_BLOCKED ||
        desc->layout == GD_LAYOUT_CHANNELS_LAST) {
        return false;
    }

    for (i = desc->ndim - 1; i >= 0; --i) {
        if (desc->sizes[i] != 1 && desc->strides[i] != expected) {
            return false;
        }
        if (desc->sizes[i] > 1) {
            if (expected > INT64_MAX / desc->sizes[i]) {
                return false;
            }
            expected *= desc->sizes[i];
        }
    }
    return true;
}

static gd_status make_tensor_from_storage(gd_storage *storage,
                                          const gd_tensor_desc *desc,
                                          gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_tensor *tensor = NULL;
    size_t required = 0U;
    size_t alignment = 0U;
    const gd_storage_desc *storage_desc = NULL;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "tensor out is NULL");
    }
    *out = NULL;
    if (storage == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage is NULL");
    }

    status = gd_tensor_desc_nbytes(desc, &required, &alignment);
    if (status != GD_OK) {
        return status;
    }
    (void)alignment;

    storage_desc = _gd_storage_desc(storage);
    if (storage_desc == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage desc is NULL");
    }
    if (!gd_device_equal(storage_desc->device, desc->device)) {
        return _gd_error(GD_ERR_DEVICE, "tensor device must match storage device");
    }
    if (required > _gd_storage_nbytes(storage)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "storage is too small for tensor desc");
    }

    tensor = calloc(1U, sizeof(*tensor));
    if (tensor == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate tensor");
    }

    status = gd_storage_retain(storage);
    if (status != GD_OK) {
        free(tensor);
        return status;
    }

    _gd_refcount_init(&tensor->refcount);
    tensor->desc = *desc;
    tensor->storage = storage;
    tensor->graph = NULL;
    tensor->value_id = -1;

    *out = tensor;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_tensor_desc_contiguous(gd_dtype dtype,
                                    gd_device device,
                                    int ndim,
                                    const int64_t *sizes,
                                    gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int i = 0;
    int64_t stride = 1;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_desc_contiguous out is NULL");
    }
    if (ndim < 0 || ndim > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "ndim is out of range");
    }
    if (ndim > 0 && sizes == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sizes is NULL");
    }

    memset(out, 0, sizeof(*out));
    out->dtype = dtype;
    out->device = device;
    out->layout = GD_LAYOUT_CONTIGUOUS;
    out->ndim = ndim;
    out->storage_offset_bytes = 0;
    out->quant = NULL;

    for (i = 0; i < ndim; ++i) {
        if (sizes[i] <= 0) {
            return _gd_error(GD_ERR_SHAPE, "sizes must be positive");
        }
        out->sizes[i] = sizes[i];
    }

    for (i = ndim - 1; i >= 0; --i) {
        out->strides[i] = stride;
        if (i > 0) {
            status = checked_mul_i64(stride, out->sizes[i], &stride);
            if (status != GD_OK) {
                return status;
            }
        }
    }

    return validate_tensor_desc(out);
}

gd_status gd_tensor_desc_nbytes(const gd_tensor_desc *desc,
                                size_t *nbytes_out,
                                size_t *alignment_out)
{
    gd_status status = GD_OK;
    size_t elem_size = 0U;
    size_t max_elem_offset = 0U;
    size_t total = 0U;
    int i = 0;

    if (nbytes_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "nbytes_out is NULL");
    }
    *nbytes_out = 0U;
    if (alignment_out != NULL) {
        *alignment_out = 0U;
    }

    status = validate_tensor_desc(desc);
    if (status != GD_OK) {
        return status;
    }
    if (desc->layout == GD_LAYOUT_PACKED_QUANT || desc->dtype == GD_DTYPE_QUANTIZED) {
        return _gd_error(GD_ERR_UNSUPPORTED, "packed quant sizing is not implemented yet");
    }
    if (desc->layout == GD_LAYOUT_CHANNELS_LAST || desc->layout == GD_LAYOUT_BLOCKED ||
        desc->layout == GD_LAYOUT_BACKEND_OPAQUE) {
        return _gd_error(GD_ERR_UNSUPPORTED, "layout sizing is not implemented for this layout");
    }

    elem_size = gd_dtype_sizeof(desc->dtype);
    if (elem_size == 0U) {
        return _gd_error(GD_ERR_DTYPE, "dtype has no fixed element size");
    }

    if (desc->layout == GD_LAYOUT_CONTIGUOUS) {
        status = tensor_numel_desc(desc, &total);
        if (status != GD_OK) {
            return status;
        }
        status = checked_mul_size(total, elem_size, &total);
        if (status != GD_OK) {
            return status;
        }
    } else {
        for (i = 0; i < desc->ndim; ++i) {
            size_t dim_extent = 0U;
            size_t stride = 0U;
            size_t offset = 0U;

            dim_extent = (size_t)(desc->sizes[i] - 1);
            stride = (size_t)desc->strides[i];
            status = checked_mul_size(dim_extent, stride, &offset);
            if (status != GD_OK) {
                return status;
            }
            status = checked_add_size(max_elem_offset, offset, &max_elem_offset);
            if (status != GD_OK) {
                return status;
            }
        }
        status = checked_add_size(max_elem_offset, 1U, &total);
        if (status != GD_OK) {
            return status;
        }
        status = checked_mul_size(total, elem_size, &total);
        if (status != GD_OK) {
            return status;
        }
    }

    status = checked_add_size((size_t)desc->storage_offset_bytes, total, &total);
    if (status != GD_OK) {
        return status;
    }

    *nbytes_out = total;
    if (alignment_out != NULL) {
        *alignment_out = elem_size;
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_tensor_empty(gd_context *ctx,
                          const gd_tensor_desc *desc,
                          gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_storage_desc storage_desc;
    gd_storage *storage = NULL;
    size_t nbytes = 0U;
    size_t alignment = 0U;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_empty ctx is NULL");
    }
    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_empty out is NULL");
    }
    *out = NULL;

    status = gd_tensor_desc_nbytes(desc, &nbytes, &alignment);
    if (status != GD_OK) {
        return status;
    }

    {
        _gd_backend *backend = _gd_context_backend(ctx, desc->device);
        gd_memory_kind memory = GD_MEM_HOST;
        if (backend == NULL) {
            return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for tensor device");
        }
        memory = backend->caps.default_memory;
        storage_desc = (gd_storage_desc){desc->device, memory, nbytes, alignment};
    }
    status = gd_storage_create(ctx, &storage_desc, &storage);
    if (status != GD_OK) {
        return status;
    }

    status = make_tensor_from_storage(storage, desc, out);
    gd_storage_release(storage);
    return status;
}

gd_status gd_tensor_from_storage(gd_context *ctx,
                                 gd_storage *storage,
                                 const gd_tensor_desc *desc,
                                 gd_tensor **out)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "ctx is NULL");
    }
    return make_tensor_from_storage(storage, desc, out);
}

gd_status gd_tensor_retain(gd_tensor *tensor)
{
    gd_status status = GD_OK;

    if (tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_retain tensor is NULL");
    }
    status = _gd_refcount_retain(&tensor->refcount);
    if (status != GD_OK) {
        return _gd_error(status, "cannot retain released tensor");
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void gd_tensor_release(gd_tensor *tensor)
{
    if (tensor == NULL) {
        return;
    }

    if (_gd_refcount_release(&tensor->refcount) != 0) {
        gd_tensor_release(tensor->grad);
        gd_storage_release(tensor->storage);
        if (tensor->graph != NULL) {
            _gd_graph_note_virtual_tensor_release(tensor->graph, tensor);
        }
        free(tensor->name);
        free(tensor);
    }

    _gd_set_last_error(GD_OK, NULL);
}

gd_status gd_tensor_copy_from_cpu(gd_context *ctx,
                                  gd_tensor *dst,
                                  const void *src,
                                  size_t nbytes)
{
    size_t offset = 0U;

    if (ctx == NULL || dst == NULL || src == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy_from_cpu argument is NULL");
    }
    if (dst->storage == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "destination tensor is virtual");
    }
    if (!desc_is_contiguous_strides(&dst->desc)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "copy_from_cpu requires contiguous tensor");
    }
    offset = (size_t)dst->desc.storage_offset_bytes;
    if (offset > _gd_storage_nbytes(dst->storage) ||
        nbytes > _gd_storage_nbytes(dst->storage) - offset) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy_from_cpu byte count exceeds storage");
    }

    return gd_storage_copy_from_cpu(ctx, dst->storage, offset, src, nbytes);
}

gd_status gd_tensor_copy_to_cpu(gd_context *ctx,
                                gd_tensor *src,
                                void *dst,
                                size_t nbytes)
{
    size_t offset = 0U;

    if (ctx == NULL || src == NULL || dst == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy_to_cpu argument is NULL");
    }
    if (src->storage == NULL) {
        gd_status status = GD_OK;
        const gd_tensor_desc *vdesc = NULL;
        gd_storage *vstorage = NULL;
        size_t voffset = 0U;
        size_t need = 0U;
        size_t align = 0U;

        if (src->graph == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "tensor has no storage or graph");
        }
        status = _gd_graph_value_storage(src->graph, src->value_id, true, &vstorage, &voffset,
                                         &vdesc);
        if (status != GD_OK) {
            return status;
        }
        status = gd_tensor_desc_nbytes(vdesc, &need, &align);
        if (status != GD_OK) {
            return status;
        }
        if (nbytes > need) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy_to_cpu byte count exceeds value");
        }
        /* Backend-routed download (blocking); no host-pointer assumption. */
        return gd_storage_copy_to_cpu(ctx, vstorage, voffset, dst, nbytes);
    }
    if (!desc_is_contiguous_strides(&src->desc)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "copy_to_cpu requires contiguous tensor");
    }
    offset = (size_t)src->desc.storage_offset_bytes;
    if (offset > _gd_storage_nbytes(src->storage) ||
        nbytes > _gd_storage_nbytes(src->storage) - offset) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "copy_to_cpu byte count exceeds storage");
    }

    return gd_storage_copy_to_cpu(ctx, src->storage, offset, dst, nbytes);
}

int gd_tensor_ndim(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_ndim tensor is NULL");
        return -1;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.ndim;
}

int64_t gd_tensor_size(const gd_tensor *tensor, int dim)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_size tensor is NULL");
        return -1;
    }
    if (dim < 0 || dim >= tensor->desc.ndim) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "tensor dim is out of range");
        return -1;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.sizes[dim];
}

int64_t gd_tensor_stride(const gd_tensor *tensor, int dim)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_stride tensor is NULL");
        return -1;
    }
    if (dim < 0 || dim >= tensor->desc.ndim) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "tensor dim is out of range");
        return -1;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.strides[dim];
}

gd_dtype gd_tensor_dtype(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_dtype tensor is NULL");
        return GD_DTYPE_INVALID;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.dtype;
}

gd_device gd_tensor_device(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_device tensor is NULL");
        return (gd_device){GD_DEVICE_CPU, 0};
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.device;
}

gd_layout gd_tensor_layout(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_layout tensor is NULL");
        return GD_LAYOUT_STRIDED;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.layout;
}

gd_storage *gd_tensor_storage(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_storage tensor is NULL");
        return NULL;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->storage;
}

const gd_quant_desc *gd_tensor_quant(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_quant tensor is NULL");
        return NULL;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->desc.quant;
}

gd_status gd_tensor_view(gd_tensor *base,
                         const gd_tensor_desc *view_desc,
                         gd_tensor **out)
{
    if (base == NULL || view_desc == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_view argument is NULL");
    }
    *out = NULL;
    if (base->storage == NULL) {
        /* Virtual (graph) tensor: record a functional reshape into the new
         * contiguous shape. Values are immutable, so this is observationally a
         * zero-copy view. Only element-order-preserving (contiguous, equal
         * numel) views are representable in v1. */
        size_t base_numel = 0U;
        size_t view_numel = 0U;

        if (base->graph == NULL) {
            return _gd_error(GD_ERR_INVALID_STATE, "virtual tensor has no graph");
        }
        if (view_desc->dtype != base->desc.dtype) {
            return _gd_error(GD_ERR_DTYPE, "view dtype must match base dtype");
        }
        if (!gd_device_equal(view_desc->device, base->desc.device)) {
            return _gd_error(GD_ERR_DEVICE, "view device must match base device");
        }
        if (view_desc->quant != base->desc.quant) {
            return _gd_error(GD_ERR_DTYPE, "view quant descriptor must match base");
        }
        if (!desc_is_contiguous_strides(view_desc)) {
            return _gd_error(GD_ERR_UNSUPPORTED,
                             "only contiguous views of virtual tensors are supported");
        }
        if (tensor_numel_desc(&base->desc, &base_numel) != GD_OK ||
            tensor_numel_desc(view_desc, &view_numel) != GD_OK) {
            return _gd_error(GD_ERR_SHAPE, "failed to compute view element count");
        }
        if (base_numel != view_numel) {
            return _gd_error(GD_ERR_SHAPE, "view must preserve element count");
        }
        return _gd_graph_emit(base->graph, _GD_OP_COPY, &base, 1, NULL, view_desc, out);
    }
    if (view_desc->dtype != base->desc.dtype) {
        return _gd_error(GD_ERR_DTYPE, "view dtype must match base dtype");
    }
    if (!gd_device_equal(view_desc->device, base->desc.device)) {
        return _gd_error(GD_ERR_DEVICE, "view device must match base device");
    }
    if (view_desc->quant != base->desc.quant) {
        return _gd_error(GD_ERR_DTYPE, "view quant descriptor must match base");
    }
    return make_tensor_from_storage(base->storage, view_desc, out);
}

gd_status gd_tensor_reshape(gd_tensor *tensor,
                            int ndim,
                            const int64_t *sizes,
                            gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_tensor_desc desc;
    size_t old_numel = 0U;
    size_t new_numel = 0U;

    if (tensor == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_reshape argument is NULL");
    }
    *out = NULL;
    if (!desc_is_contiguous_strides(&tensor->desc)) {
        return _gd_error(GD_ERR_UNSUPPORTED, "reshape requires contiguous-compatible tensor");
    }

    status = tensor_numel_desc(&tensor->desc, &old_numel);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_desc_contiguous(tensor->desc.dtype,
                                       tensor->desc.device,
                                       ndim,
                                       sizes,
                                       &desc);
    if (status != GD_OK) {
        return status;
    }
    desc.storage_offset_bytes = tensor->desc.storage_offset_bytes;
    desc.quant = tensor->desc.quant;
    status = tensor_numel_desc(&desc, &new_numel);
    if (status != GD_OK) {
        return status;
    }
    if (old_numel != new_numel) {
        return _gd_error(GD_ERR_SHAPE, "reshape must preserve element count");
    }

    return gd_tensor_view(tensor, &desc, out);
}

gd_status gd_tensor_transpose(gd_tensor *tensor,
                              int d0,
                              int d1,
                              gd_tensor **out)
{
    gd_tensor_desc desc;
    int64_t tmp = 0;

    if (tensor == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_transpose argument is NULL");
    }
    *out = NULL;
    if (d0 < 0) {
        d0 += tensor->desc.ndim;
    }
    if (d1 < 0) {
        d1 += tensor->desc.ndim;
    }
    if (d0 < 0 || d0 >= tensor->desc.ndim || d1 < 0 || d1 >= tensor->desc.ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose dims are out of range");
    }

    desc = tensor->desc;
    tmp = desc.sizes[d0];
    desc.sizes[d0] = desc.sizes[d1];
    desc.sizes[d1] = tmp;
    tmp = desc.strides[d0];
    desc.strides[d0] = desc.strides[d1];
    desc.strides[d1] = tmp;
    desc.layout = desc_is_contiguous_strides(&desc) ? GD_LAYOUT_CONTIGUOUS : GD_LAYOUT_STRIDED;

    return gd_tensor_view(tensor, &desc, out);
}

gd_status gd_tensor_slice(gd_tensor *tensor,
                          int dim,
                          int64_t start,
                          int64_t len,
                          gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_tensor_desc desc;
    int64_t offset_elems = 0;
    int64_t offset_bytes = 0;
    size_t elem_size = 0U;

    if (tensor == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_slice argument is NULL");
    }
    *out = NULL;
    if (dim < 0) {
        dim += tensor->desc.ndim;
    }
    if (dim < 0 || dim >= tensor->desc.ndim) {
        return _gd_error(GD_ERR_SHAPE, "slice dim is out of range");
    }
    if (start < 0 || len <= 0 || start > tensor->desc.sizes[dim] ||
        len > tensor->desc.sizes[dim] - start) {
        return _gd_error(GD_ERR_SHAPE, "slice range is invalid");
    }

    elem_size = gd_dtype_sizeof(tensor->desc.dtype);
    if (elem_size == 0U) {
        return _gd_error(GD_ERR_DTYPE, "slice requires fixed-size dtype");
    }

    desc = tensor->desc;
    desc.sizes[dim] = len;
    status = checked_mul_i64(start, tensor->desc.strides[dim], &offset_elems);
    if (status != GD_OK) {
        return status;
    }
    status = checked_mul_i64(offset_elems, (int64_t)elem_size, &offset_bytes);
    if (status != GD_OK) {
        return status;
    }
    status = checked_add_i64(desc.storage_offset_bytes, offset_bytes, &desc.storage_offset_bytes);
    if (status != GD_OK) {
        return status;
    }
    desc.layout = desc_is_contiguous_strides(&desc) ? GD_LAYOUT_CONTIGUOUS : GD_LAYOUT_STRIDED;

    return gd_tensor_view(tensor, &desc, out);
}

gd_status gd_tensor_contiguous(gd_context *ctx,
                               gd_tensor *tensor,
                               gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_tensor_desc desc;
    gd_tensor *result = NULL;
    const unsigned char *src_base = NULL;
    unsigned char *dst_base = NULL;
    size_t elem = 0U;
    int64_t numel = 1;
    int64_t i = 0;
    int d = 0;

    if (ctx == NULL || tensor == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_contiguous argument is NULL");
    }
    *out = NULL;

    if (desc_is_contiguous_strides(&tensor->desc)) {
        if (gd_tensor_retain(tensor) != GD_OK) {
            return _gd_error(GD_ERR_INVALID_STATE, "failed to retain contiguous tensor");
        }
        *out = tensor;
        _gd_set_last_error(GD_OK, NULL);
        return GD_OK;
    }

    /* Non-contiguous: eagerly gather a fresh contiguous copy of materialized data.
     * Virtual (graph-produced) values are always contiguous, so they never reach
     * here; only strided materialized views (transpose/slice) do. */
    if (tensor->storage == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED,
                         "cannot materialize a non-contiguous virtual tensor");
    }
    if (tensor->desc.dtype == GD_DTYPE_QUANTIZED) {
        return _gd_error(GD_ERR_UNSUPPORTED, "contiguous copy of packed quant is not supported");
    }
    elem = gd_dtype_sizeof(tensor->desc.dtype);
    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "contiguous copy requires a fixed-size dtype");
    }

    status = gd_tensor_desc_contiguous(tensor->desc.dtype, tensor->desc.device,
                                      tensor->desc.ndim, tensor->desc.sizes, &desc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, &result);
    if (status != GD_OK) {
        return status;
    }

    {
        void *src_host = NULL;
        void *dst_host = NULL;

        /* Strided gather indexes host memory directly, so both sides must be
         * host-visible. Device-backed strided copies belong to a graph op. */
        status = gd_storage_data_cpu(tensor->storage, &src_host);
        if (status != GD_OK) {
            gd_tensor_release(result);
            return status;
        }
        status = gd_storage_data_cpu(result->storage, &dst_host);
        if (status != GD_OK) {
            gd_tensor_release(result);
            return status;
        }
        src_base = (const unsigned char *)src_host +
                   (size_t)tensor->desc.storage_offset_bytes;
        dst_base = dst_host;
    }

    for (d = 0; d < tensor->desc.ndim; ++d) {
        numel *= tensor->desc.sizes[d];
    }
    for (i = 0; i < numel; ++i) {
        int64_t rem = i;
        int64_t src_off = 0;
        for (d = tensor->desc.ndim - 1; d >= 0; --d) {
            int64_t coord = rem % tensor->desc.sizes[d];
            rem /= tensor->desc.sizes[d];
            src_off += coord * tensor->desc.strides[d];
        }
        memcpy(dst_base + (size_t)i * elem, src_base + (size_t)src_off * elem, elem);
    }

    *out = result;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_tensor_create_virtual(gd_graph *graph,
                                    int value_id,
                                    const gd_tensor_desc *desc,
                                    gd_tensor **out)
{
    gd_status status = GD_OK;
    gd_tensor *tensor = NULL;

    if (graph == NULL || desc == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "_gd_tensor_create_virtual argument is NULL");
    }
    if (value_id < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "value id must be nonnegative");
    }
    *out = NULL;

    status = validate_tensor_desc(desc);
    if (status != GD_OK) {
        return status;
    }

    tensor = calloc(1U, sizeof(*tensor));
    if (tensor == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate virtual tensor");
    }

    _gd_refcount_init(&tensor->refcount);
    tensor->desc = *desc;
    tensor->storage = NULL;
    tensor->graph = graph;
    tensor->value_id = value_id;

    status = _gd_graph_note_virtual_tensor_create(graph, tensor);
    if (status != GD_OK) {
        free(tensor);
        return status;
    }

    *out = tensor;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

bool _gd_tensor_is_virtual(const gd_tensor *tensor)
{
    return tensor != NULL && tensor->storage == NULL && tensor->graph != NULL;
}

int _gd_tensor_value_id(const gd_tensor *tensor)
{
    return tensor == NULL ? -1 : tensor->value_id;
}

gd_graph *_gd_tensor_graph(const gd_tensor *tensor)
{
    return tensor == NULL ? NULL : tensor->graph;
}

gd_status _gd_tensor_materialize_from_graph(gd_context *ctx, gd_tensor *tensor)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *vdesc = NULL;
    gd_storage *vstorage = NULL;
    size_t voffset = 0U;
    gd_storage_desc storage_desc;
    gd_storage *storage = NULL;
    void *storage_data = NULL;
    size_t nbytes = 0U;
    size_t alignment = 0U;
    gd_graph *graph = NULL;

    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "_gd_tensor_materialize_from_graph argument is NULL");
    }
    if (tensor->storage != NULL) {
        return GD_OK;
    }
    if (tensor->graph == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "virtual tensor has no graph");
    }

    status = _gd_graph_value_storage(tensor->graph, tensor->value_id, true, &vstorage, &voffset,
                                     &vdesc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_desc_nbytes(vdesc, &nbytes, &alignment);
    if (status != GD_OK) {
        return status;
    }

    storage_desc = (gd_storage_desc){vdesc->device, GD_MEM_HOST, nbytes, alignment};
    status = gd_storage_create(ctx, &storage_desc, &storage);
    if (status != GD_OK) {
        return status;
    }
    /* Fill the new host storage by downloading the value through the backend. */
    status = gd_storage_data_cpu(storage, &storage_data);
    if (status != GD_OK) {
        gd_storage_release(storage);
        return status;
    }
    status = gd_storage_copy_to_cpu(ctx, vstorage, voffset, storage_data, nbytes);
    if (status != GD_OK) {
        gd_storage_release(storage);
        return status;
    }

    graph = tensor->graph;
    tensor->desc = *vdesc;
    tensor->storage = storage;
    tensor->graph = NULL;
    tensor->value_id = -1;
    _gd_graph_note_virtual_tensor_release(graph, tensor);

    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

bool _gd_tensor_is_contiguous(const gd_tensor *tensor)
{
    return tensor != NULL && desc_is_contiguous_strides(&tensor->desc);
}

gd_status _gd_tensor_ensure_grad(gd_context *ctx, gd_tensor *tensor, gd_tensor **grad_out)
{
    gd_status status = GD_OK;
    gd_tensor_desc desc;
    gd_tensor *grad = NULL;

    if (ctx == NULL || tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "ensure_grad argument is NULL");
    }
    if (!tensor->requires_grad) {
        return _gd_error(GD_ERR_INVALID_STATE, "tensor does not require grad");
    }
    if (tensor->storage == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "grad slot requires a materialized leaf");
    }
    if (tensor->grad != NULL) {
        if (grad_out != NULL) {
            *grad_out = tensor->grad;
        }
        return GD_OK;
    }

    status = gd_tensor_desc_contiguous(GD_DTYPE_F32, tensor->desc.device, tensor->desc.ndim,
                                      tensor->desc.sizes, &desc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, &grad);
    if (status != GD_OK) {
        return status;
    }
    tensor->grad = grad;
    if (grad_out != NULL) {
        *grad_out = grad;
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_tensor_zero(gd_tensor *tensor)
{
    void *data = NULL;

    if (tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "zero tensor is NULL");
    }
    if (tensor->storage == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "cannot zero a virtual tensor");
    }
    data = _gd_storage_data_mut(tensor->storage);
    if (data == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "tensor storage has no data");
    }
    memset((unsigned char *)data + (size_t)tensor->desc.storage_offset_bytes, 0,
           _gd_storage_nbytes(tensor->storage) - (size_t)tensor->desc.storage_offset_bytes);
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

const gd_tensor_desc *_gd_tensor_desc_ptr(const gd_tensor *tensor)
{
    return tensor == NULL ? NULL : &tensor->desc;
}

gd_status gd_tensor_set_requires_grad(gd_tensor *tensor, bool requires_grad)
{
    if (tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_set_requires_grad tensor is NULL");
    }
    if (requires_grad && tensor->desc.dtype != GD_DTYPE_F32 &&
        tensor->desc.dtype != GD_DTYPE_F16 && tensor->desc.dtype != GD_DTYPE_BF16) {
        return _gd_error(GD_ERR_DTYPE, "only floating tensors can require gradients");
    }
    tensor->requires_grad = requires_grad;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

bool gd_tensor_requires_grad(const gd_tensor *tensor)
{
    if (tensor == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_requires_grad tensor is NULL");
        return false;
    }
    _gd_set_last_error(GD_OK, NULL);
    return tensor->requires_grad;
}

gd_status gd_tensor_grad(gd_tensor *tensor, gd_tensor **grad_out)
{
    if (grad_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_grad grad_out is NULL");
    }
    *grad_out = NULL;
    if (tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_grad tensor is NULL");
    }
    *grad_out = tensor->grad;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_tensor_set_name(gd_tensor *tensor, const char *name)
{
    char *copy = NULL;
    size_t len = 0U;

    if (tensor == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_tensor_set_name tensor is NULL");
    }
    if (name != NULL) {
        len = strlen(name);
        copy = malloc(len + 1U);
        if (copy == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate tensor name");
        }
        memcpy(copy, name, len + 1U);
    }

    free(tensor->name);
    tensor->name = copy;
    if (tensor->storage == NULL && tensor->graph != NULL) {
        gd_status status = _gd_graph_set_value_name(tensor->graph, tensor->value_id, name);
        if (status != GD_OK) {
            return status;
        }
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
