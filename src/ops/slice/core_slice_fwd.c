#include "../op_impl.h"
#include "../meta_common.h"
#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#include <stdint.h>

#define GD_SLICE_N_INPUTS 1
#define GD_SLICE_N_OUTPUTS 1

static gd_status meta_slice_desc(const gd_tensor_desc *x,
                                 int dim,
                                 int64_t start,
                                 int64_t len,
                                 int *norm_dim_out,
                                 gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int norm_dim = 0;
    int i = 0;

    if (x == NULL || norm_dim_out == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice meta argument is NULL");
    }
    if (gd_dtype_sizeof(x->dtype) == 0U || x->quant != NULL) {
        return _gd_error(GD_ERR_DTYPE, "slice requires an unquantized fixed-size dtype");
    }
    status = _gd_meta_normalize_dim(dim, x->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    if (start < 0 || len <= 0 || start > x->sizes[norm_dim] ||
        len > x->sizes[norm_dim] - start) {
        return _gd_error(GD_ERR_SHAPE, "slice range is invalid");
    }
    for (i = 0; i < x->ndim; ++i) {
        sizes[i] = x->sizes[i];
    }
    sizes[norm_dim] = len;
    status = gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, sizes, out);
    if (status != GD_OK) {
        return status;
    }
    *norm_dim_out = norm_dim;
    return GD_OK;
}

static gd_status slice_meta(const gd_tensor_desc *const *inputs,
                            int n_inputs,
                            _gd_op_attrs *attrs,
                            gd_tensor_desc *outputs,
                            int *n_outputs)
{
    gd_status status = GD_OK;
    int norm_dim = 0;

    if (inputs == NULL || outputs == NULL || n_outputs == NULL || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice meta arguments are NULL");
    }
    if (n_inputs != GD_SLICE_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice input count mismatch");
    }
    if (inputs[0] == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice input desc is NULL");
    }
    status = _gd_meta_set_output_count(GD_SLICE_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    status = meta_slice_desc(inputs[0], attrs->dim, attrs->slice_start,
                             attrs->slice_len, &norm_dim, &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    attrs->dim = norm_dim;
    return GD_OK;
}

const _gd_op_def _gd_opdef_slice = {
    .kind = _GD_OP_SLICE,
    .name = "slice",
    .min_inputs = GD_SLICE_N_INPUTS,
    .max_inputs = GD_SLICE_N_INPUTS,
    .n_outputs = GD_SLICE_N_OUTPUTS,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = slice_meta,
};

gd_status gd_slice(gd_context *ctx,
                   gd_tensor *x,
                   int dim,
                   int64_t start,
                   int64_t len,
                   gd_tensor **out)
{
    gd_tensor *inputs[GD_SLICE_N_INPUTS] = {x};
    gd_tensor *outputs[GD_SLICE_N_OUTPUTS] = {0};
    _gd_op_attrs attrs = {0};
    gd_status status = GD_OK;

    if (ctx == NULL || x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_slice argument is NULL");
    }
    *out = NULL;
    attrs.dim = dim;
    attrs.slice_start = start;
    attrs.slice_len = len;
    status = _gd_emit_checked(ctx, _GD_OP_SLICE, inputs, GD_SLICE_N_INPUTS,
                              &attrs, outputs, GD_SLICE_N_OUTPUTS);
    if (status != GD_OK) {
        return status;
    }
    *out = outputs[0];
    return GD_OK;
}
