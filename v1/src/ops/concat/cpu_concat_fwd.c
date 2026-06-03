#include "../../backends/cpu_ref/cpu_op.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static gd_status concat_prod_sizes(const gd_tensor_desc *desc,
                                   int begin,
                                   int end,
                                   int64_t *out)
{
    int64_t prod = 1;
    int i = 0;

    if (desc == NULL || out == NULL || begin < 0 || end < begin || end > desc->ndim) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat product arguments are invalid");
    }
    for (i = begin; i < end; ++i) {
        if (desc->sizes[i] <= 0 || prod > INT64_MAX / desc->sizes[i]) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "concat shape product overflows");
        }
        prod *= desc->sizes[i];
    }
    *out = prod;
    return GD_OK;
}

static gd_status checked_elem_bytes(int64_t elems, size_t elem_size, size_t *out)
{
    if (elems < 0 || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat byte size argument is invalid");
    }
    if (elem_size == 0U || (uint64_t)elems > (uint64_t)(SIZE_MAX / elem_size)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "concat byte count overflows");
    }
    *out = (size_t)elems * elem_size;
    return GD_OK;
}

static gd_status concat_copy_contiguous(const gd_tensor_desc *out_desc,
                                        void *out,
                                        const gd_tensor_desc *in_desc,
                                        const void *in,
                                        int dim,
                                        int64_t dst_start)
{
    gd_status status = GD_OK;
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t outer = 0;
    int64_t inner = 0;
    int64_t block_elems = 0;
    size_t block_bytes = 0U;
    int64_t o = 0;

    status = concat_prod_sizes(out_desc, 0, dim, &outer);
    if (status != GD_OK) {
        return status;
    }
    status = concat_prod_sizes(out_desc, dim + 1, out_desc->ndim, &inner);
    if (status != GD_OK) {
        return status;
    }
    if (in_desc->sizes[dim] > INT64_MAX / inner) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "concat copy block size overflows");
    }
    block_elems = in_desc->sizes[dim] * inner;
    status = checked_elem_bytes(block_elems, elem, &block_bytes);
    if (status != GD_OK) {
        return status;
    }
    for (o = 0; o < outer; ++o) {
        int64_t src_elems = o * block_elems;
        int64_t dst_elems = (o * out_desc->sizes[dim] + dst_start) * inner;

        memcpy((unsigned char *)out + (size_t)dst_elems * elem,
               (const unsigned char *)in + (size_t)src_elems * elem,
               block_bytes);
    }
    return GD_OK;
}

static gd_status concat_copy_generic(const gd_tensor_desc *out_desc,
                                     void *out,
                                     const gd_tensor_desc *in_desc,
                                     const void *in,
                                     int dim,
                                     int64_t dst_start)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = _gd_cpu_desc_numel(in_desc);
    int64_t lin = 0;

    for (lin = 0; lin < total; ++lin) {
        int64_t tmp = lin;
        int64_t in_off = 0;
        int64_t out_off = 0;
        int axis = 0;

        for (axis = in_desc->ndim - 1; axis >= 0; --axis) {
            int64_t coord = tmp % in_desc->sizes[axis];
            int64_t out_coord = coord;

            tmp /= in_desc->sizes[axis];
            if (axis == dim) {
                out_coord += dst_start;
            }
            in_off += coord * in_desc->strides[axis];
            out_off += out_coord * out_desc->strides[axis];
        }
        memcpy((unsigned char *)out + (size_t)out_off * elem,
               (const unsigned char *)in + (size_t)in_off * elem,
               elem);
    }
    return GD_OK;
}

static gd_status concat_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    void *out_data = NULL;
    const gd_tensor_desc *out_desc = NULL;
    gd_status status = GD_OK;
    int dim = 0;
    int64_t dst_start = 0;
    int i = 0;

    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) {
        return status;
    }
    dim = node->attrs.dim;
    if (gd_dtype_sizeof(out_desc->dtype) == 0U || out_desc->quant != NULL) {
        return _gd_error(GD_ERR_DTYPE, "concat CPU requires a fixed-size output dtype");
    }
    for (i = 0; i < node->n_inputs; ++i) {
        void *in_data = NULL;
        const gd_tensor_desc *in_desc = NULL;

        status = _gd_cpu_exec_input(exec, node, i, &in_data, &in_desc);
        if (status != GD_OK) {
            return status;
        }
        if (in_desc->dtype != out_desc->dtype || gd_dtype_sizeof(in_desc->dtype) == 0U ||
            in_desc->quant != NULL) {
            return _gd_error(GD_ERR_DTYPE, "concat CPU requires matching fixed-size dtypes");
        }
        if (in_desc->layout == GD_LAYOUT_CONTIGUOUS && out_desc->layout == GD_LAYOUT_CONTIGUOUS) {
            status = concat_copy_contiguous(out_desc, out_data, in_desc, in_data, dim, dst_start);
        } else {
            status = concat_copy_generic(out_desc, out_data, in_desc, in_data, dim, dst_start);
        }
        if (status != GD_OK) {
            return status;
        }
        dst_start += in_desc->sizes[dim];
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_concat = {
    .kind = _GD_OP_CONCAT,
    .name = "concat",
    .support = _gd_cpu_support_default,
    .run = concat_run,
};
