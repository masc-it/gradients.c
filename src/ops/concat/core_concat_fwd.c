#include "../op_impl.h"
#include "../meta_common.h"
#include "gradients/ops.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#include <stdint.h>

#define GD_CONCAT_MIN_INPUTS 1
#define GD_CONCAT_N_OUTPUTS 1

static gd_status meta_concat_desc(const gd_tensor_desc *const *inputs,
                                  int n_inputs,
                                  int dim,
                                  int *norm_dim_out,
                                  gd_tensor_desc *out)
{
    const gd_tensor_desc *base = NULL;
    int64_t sizes[GD_MAX_DIMS];
    int64_t concat_size = 0;
    gd_status status = GD_OK;
    int norm_dim = 0;
    int i = 0;
    int axis = 0;

    if (inputs == NULL || norm_dim_out == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat meta argument is NULL");
    }
    if (n_inputs < GD_CONCAT_MIN_INPUTS || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat input count is out of range");
    }
    base = inputs[0];
    if (base == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat input desc is NULL");
    }
    if (gd_dtype_sizeof(base->dtype) == 0U || base->quant != NULL) {
        return _gd_error(GD_ERR_DTYPE, "concat requires an unquantized fixed-size dtype");
    }
    status = _gd_meta_normalize_dim(dim, base->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    for (axis = 0; axis < base->ndim; ++axis) {
        sizes[axis] = base->sizes[axis];
    }

    for (i = 0; i < n_inputs; ++i) {
        const gd_tensor_desc *x = inputs[i];

        if (x == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat input desc is NULL");
        }
        if (x->dtype != base->dtype) {
            return _gd_error(GD_ERR_DTYPE, "concat inputs must share dtype");
        }
        if (gd_dtype_sizeof(x->dtype) == 0U || x->quant != NULL) {
            return _gd_error(GD_ERR_DTYPE, "concat requires unquantized fixed-size dtypes");
        }
        status = _gd_meta_require_same_device(base, x, "concat inputs must share a device");
        if (status != GD_OK) {
            return status;
        }
        if (x->ndim != base->ndim) {
            return _gd_error(GD_ERR_SHAPE, "concat inputs must have equal rank");
        }
        for (axis = 0; axis < base->ndim; ++axis) {
            if (axis != norm_dim && x->sizes[axis] != base->sizes[axis]) {
                return _gd_error(GD_ERR_SHAPE,
                                 "concat input shapes must match except along dim");
            }
        }
        if (x->sizes[norm_dim] > INT64_MAX - concat_size) {
            return _gd_error(GD_ERR_SHAPE, "concat dimension overflows int64");
        }
        concat_size += x->sizes[norm_dim];
    }

    sizes[norm_dim] = concat_size;
    status = gd_tensor_desc_contiguous(base->dtype, base->device, base->ndim, sizes, out);
    if (status != GD_OK) {
        return status;
    }
    *norm_dim_out = norm_dim;
    return GD_OK;
}

static gd_status concat_meta(const gd_tensor_desc *const *inputs,
                             int n_inputs,
                             _gd_op_attrs *attrs,
                             gd_tensor_desc *outputs,
                             int *n_outputs)
{
    gd_status status = GD_OK;
    int norm_dim = 0;

    if (inputs == NULL || outputs == NULL || n_outputs == NULL || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "concat meta arguments are NULL");
    }
    status = _gd_meta_set_output_count(GD_CONCAT_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    status = meta_concat_desc(inputs, n_inputs, attrs->dim, &norm_dim, &outputs[0]);
    if (status != GD_OK) {
        return status;
    }
    attrs->dim = norm_dim;
    return GD_OK;
}

const _gd_op_def _gd_opdef_concat = {
    .kind = _GD_OP_CONCAT,
    .name = "concat",
    .min_inputs = GD_CONCAT_MIN_INPUTS,
    .max_inputs = _GD_OP_MAX_INPUTS,
    .n_outputs = GD_CONCAT_N_OUTPUTS,
    .flags = GD_OPF_PUBLIC | GD_OPF_DIFF,
    .meta = concat_meta,
};

gd_status gd_concat(gd_context *ctx,
                    gd_tensor *const *inputs,
                    int n_inputs,
                    int dim,
                    gd_tensor **out)
{
    gd_tensor *local_inputs[_GD_OP_MAX_INPUTS];
    gd_tensor *outputs[GD_CONCAT_N_OUTPUTS] = {0};
    _gd_op_attrs attrs = {0};
    gd_status status = GD_OK;
    int i = 0;

    if (ctx == NULL || inputs == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_concat argument is NULL");
    }
    *out = NULL;
    if (n_inputs < GD_CONCAT_MIN_INPUTS || n_inputs > _GD_OP_MAX_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_concat input count is out of range");
    }
    for (i = 0; i < n_inputs; ++i) {
        if (inputs[i] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_concat input tensor is NULL");
        }
        local_inputs[i] = inputs[i];
    }
    attrs.dim = dim;
    status = _gd_emit_checked(ctx, _GD_OP_CONCAT, local_inputs, n_inputs,
                              &attrs, outputs, GD_CONCAT_N_OUTPUTS);
    if (status != GD_OK) {
        return status;
    }
    *out = outputs[0];
    return GD_OK;
}
