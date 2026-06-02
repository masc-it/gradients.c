#include "../../backends/cpu_ref/cpu_op.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int64_t slice_prod_sizes(const gd_tensor_desc *desc, int begin, int end)
{
    int64_t prod = 1;
    int i = 0;

    for (i = begin; i < end; ++i) {
        prod *= desc->sizes[i];
    }
    return prod;
}

static gd_status checked_elem_bytes(int64_t elems, size_t elem_size, size_t *out)
{
    if (elems < 0 || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice byte size argument is invalid");
    }
    if (elem_size == 0U || (uint64_t)elems > (uint64_t)(SIZE_MAX / elem_size)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "slice byte count overflows");
    }
    *out = (size_t)elems * elem_size;
    return GD_OK;
}

static gd_status slice_copy_contiguous(const gd_tensor_desc *out_desc,
                                       void *out,
                                       const gd_tensor_desc *x_desc,
                                       const void *x,
                                       int dim,
                                       int64_t start,
                                       int64_t len)
{
    gd_status status = GD_OK;
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t outer = slice_prod_sizes(x_desc, 0, dim);
    int64_t inner = slice_prod_sizes(x_desc, dim + 1, x_desc->ndim);
    size_t block_bytes = 0U;
    int64_t o = 0;

    status = checked_elem_bytes(len * inner, elem, &block_bytes);
    if (status != GD_OK) {
        return status;
    }
    for (o = 0; o < outer; ++o) {
        int64_t src_elems = (o * x_desc->sizes[dim] + start) * inner;
        int64_t dst_elems = o * len * inner;
        memcpy((unsigned char *)out + (size_t)dst_elems * elem,
               (const unsigned char *)x + (size_t)src_elems * elem,
               block_bytes);
    }
    return GD_OK;
}

static gd_status slice_copy_generic(const gd_tensor_desc *out_desc,
                                    void *out,
                                    const gd_tensor_desc *x_desc,
                                    const void *x,
                                    int dim,
                                    int64_t start)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = _gd_cpu_desc_numel(out_desc);
    int64_t lin = 0;

    for (lin = 0; lin < total; ++lin) {
        int64_t tmp = lin;
        int64_t in_off = 0;
        int axis = 0;

        for (axis = out_desc->ndim - 1; axis >= 0; --axis) {
            int64_t coord = tmp % out_desc->sizes[axis];
            tmp /= out_desc->sizes[axis];
            if (axis == dim) {
                coord += start;
            }
            in_off += coord * x_desc->strides[axis];
        }
        memcpy((unsigned char *)out + (size_t)lin * elem,
               (const unsigned char *)x + (size_t)in_off * elem,
               elem);
    }
    return GD_OK;
}

static gd_status slice_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    void *x_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *x_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;
    gd_status status = GD_OK;
    int dim = 0;

    status = _gd_cpu_exec_input(exec, node, 0, &x_data, &x_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    dim = node->attrs.dim;
    if (gd_dtype_sizeof(out_desc->dtype) == 0U || out_desc->dtype != x_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "slice CPU requires matching fixed-size dtypes");
    }
    if (x_desc->layout == GD_LAYOUT_CONTIGUOUS) {
        return slice_copy_contiguous(out_desc, out_data, x_desc, x_data, dim,
                                     node->attrs.slice_start, node->attrs.slice_len);
    }
    return slice_copy_generic(out_desc, out_data, x_desc, x_data, dim,
                              node->attrs.slice_start);
}

const _gd_cpu_op _gd_cpu_op_slice = {
    .kind = _GD_OP_SLICE,
    .name = "slice",
    .support = _gd_cpu_support_default,
    .run = slice_run,
};
