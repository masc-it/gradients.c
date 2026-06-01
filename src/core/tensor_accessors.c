#include "gradients/graph.h"
#include "gradients/tensor.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "tensor_internal.h"
#include "../graph/graph_internal.h"

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
