#include "../../backends/cpu_ref/cpu_op.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static int64_t slice_bwd_prod_sizes(const gd_tensor_desc *desc, int begin, int end)
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
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice_bwd byte size argument is invalid");
    }
    if (elem_size == 0U || (uint64_t)elems > (uint64_t)(SIZE_MAX / elem_size)) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "slice_bwd byte count overflows");
    }
    *out = (size_t)elems * elem_size;
    return GD_OK;
}

static gd_status slice_bwd_contiguous(const gd_tensor_desc *dx_desc,
                                      void *dx,
                                      const gd_tensor_desc *go_desc,
                                      const void *go,
                                      int dim,
                                      int64_t start,
                                      int64_t len)
{
    gd_status status = GD_OK;
    size_t elem = gd_dtype_sizeof(dx_desc->dtype);
    int64_t outer = slice_bwd_prod_sizes(dx_desc, 0, dim);
    int64_t inner = slice_bwd_prod_sizes(dx_desc, dim + 1, dx_desc->ndim);
    int64_t total = _gd_cpu_desc_numel(dx_desc);
    size_t total_bytes = 0U;
    size_t block_bytes = 0U;
    int64_t o = 0;

    (void)go_desc;
    status = checked_elem_bytes(total, elem, &total_bytes);
    if (status != GD_OK) {
        return status;
    }
    status = checked_elem_bytes(len * inner, elem, &block_bytes);
    if (status != GD_OK) {
        return status;
    }
    memset(dx, 0, total_bytes);
    for (o = 0; o < outer; ++o) {
        int64_t dst_elems = (o * dx_desc->sizes[dim] + start) * inner;
        int64_t src_elems = o * len * inner;
        memcpy((unsigned char *)dx + (size_t)dst_elems * elem,
               (const unsigned char *)go + (size_t)src_elems * elem,
               block_bytes);
    }
    return GD_OK;
}

static gd_status slice_bwd_generic(const gd_tensor_desc *dx_desc,
                                   void *dx,
                                   const gd_tensor_desc *go_desc,
                                   const void *go,
                                   int dim,
                                   int64_t start)
{
    gd_status status = GD_OK;
    size_t elem = gd_dtype_sizeof(dx_desc->dtype);
    int64_t total_dx = _gd_cpu_desc_numel(dx_desc);
    int64_t total_go = _gd_cpu_desc_numel(go_desc);
    size_t total_bytes = 0U;
    int64_t lin = 0;

    status = checked_elem_bytes(total_dx, elem, &total_bytes);
    if (status != GD_OK) {
        return status;
    }
    memset(dx, 0, total_bytes);
    for (lin = 0; lin < total_go; ++lin) {
        int64_t tmp = lin;
        int64_t src_off = 0;
        int64_t dst_off = 0;
        int axis = 0;

        for (axis = go_desc->ndim - 1; axis >= 0; --axis) {
            int64_t coord = tmp % go_desc->sizes[axis];
            int64_t dst_coord = coord;
            tmp /= go_desc->sizes[axis];
            if (axis == dim) {
                dst_coord += start;
            }
            src_off += coord * go_desc->strides[axis];
            dst_off += dst_coord * dx_desc->strides[axis];
        }
        memcpy((unsigned char *)dx + (size_t)dst_off * elem,
               (const unsigned char *)go + (size_t)src_off * elem,
               elem);
    }
    return GD_OK;
}

static gd_status slice_bwd_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    void *go_data = NULL;
    void *x_data = NULL;
    void *dx_data = NULL;
    const gd_tensor_desc *go_desc = NULL;
    const gd_tensor_desc *x_desc = NULL;
    const gd_tensor_desc *dx_desc = NULL;
    gd_status status = GD_OK;
    int dim = 0;

    status = _gd_cpu_exec_input(exec, node, 0, &go_data, &go_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_input(exec, node, 1, &x_data, &x_desc);
    if (status != GD_OK) {
        return status;
    }
    status = _gd_cpu_exec_output(exec, node, 0, &dx_data, &dx_desc);
    if (status != GD_OK) {
        return status;
    }
    (void)x_data;
    (void)x_desc;
    dim = node->attrs.dim;
    if (gd_dtype_sizeof(dx_desc->dtype) == 0U || dx_desc->dtype != go_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "slice_bwd CPU requires matching fixed-size dtypes");
    }
    if (dx_desc->layout == GD_LAYOUT_CONTIGUOUS && go_desc->layout == GD_LAYOUT_CONTIGUOUS) {
        return slice_bwd_contiguous(dx_desc, dx_data, go_desc, go_data, dim,
                                    node->attrs.slice_start, node->attrs.slice_len);
    }
    return slice_bwd_generic(dx_desc, dx_data, go_desc, go_data, dim,
                             node->attrs.slice_start);
}

const _gd_cpu_op _gd_cpu_op_slice_bwd = {
    .kind = _GD_OP_SLICE_BWD,
    .name = "slice_bwd",
    .support = _gd_cpu_support_default,
    .run = slice_bwd_run,
};
