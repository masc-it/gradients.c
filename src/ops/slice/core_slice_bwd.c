#include "../op_impl.h"
#include "../meta_common.h"

#include "../../core/internal.h"
#include "../../graph/graph_internal.h"

#include <stdint.h>

#define GD_SLICE_BWD_N_INPUTS 2
#define GD_SLICE_BWD_N_OUTPUTS 1

static gd_status slice_bwd_meta(const gd_tensor_desc *const *inputs,
                                int n_inputs,
                                _gd_op_attrs *attrs,
                                gd_tensor_desc *outputs,
                                int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *go = NULL;
    const gd_tensor_desc *x = NULL;
    int dim = 0;
    int i = 0;

    if (inputs == NULL || outputs == NULL || n_outputs == NULL || attrs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice_bwd meta arguments are NULL");
    }
    if (n_inputs != GD_SLICE_BWD_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice_bwd input count mismatch");
    }
    go = inputs[0];
    x = inputs[1];
    if (go == NULL || x == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "slice_bwd input desc is NULL");
    }
    status = _gd_meta_set_output_count(GD_SLICE_BWD_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (go->dtype != x->dtype) {
        return _gd_error(GD_ERR_DTYPE, "slice_bwd grad and input must share dtype");
    }
    if (gd_dtype_sizeof(x->dtype) == 0U || x->quant != NULL || go->quant != NULL) {
        return _gd_error(GD_ERR_DTYPE, "slice_bwd requires unquantized fixed-size dtypes");
    }
    status = _gd_meta_require_same_device(go, x, "slice_bwd inputs must share device");
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_normalize_dim(attrs->dim, x->ndim, &dim);
    if (status != GD_OK) {
        return status;
    }
    if (attrs->slice_start < 0 || attrs->slice_len <= 0 ||
        attrs->slice_start > x->sizes[dim] ||
        attrs->slice_len > x->sizes[dim] - attrs->slice_start) {
        return _gd_error(GD_ERR_SHAPE, "slice_bwd range is invalid");
    }
    if (go->ndim != x->ndim) {
        return _gd_error(GD_ERR_SHAPE, "slice_bwd grad rank mismatch");
    }
    for (i = 0; i < x->ndim; ++i) {
        int64_t want = i == dim ? attrs->slice_len : x->sizes[i];
        if (go->sizes[i] != want) {
            return _gd_error(GD_ERR_SHAPE, "slice_bwd grad shape mismatch");
        }
    }
    attrs->dim = dim;
    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, x->sizes, &outputs[0]);
}

const _gd_op_def _gd_opdef_slice_bwd = {
    .kind = _GD_OP_SLICE_BWD,
    .name = "slice_bwd",
    .min_inputs = GD_SLICE_BWD_N_INPUTS,
    .max_inputs = GD_SLICE_BWD_N_INPUTS,
    .n_outputs = GD_SLICE_BWD_N_OUTPUTS,
    .flags = GD_OPF_INTERNAL,
    .meta = slice_bwd_meta,
};
