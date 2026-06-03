#include "../op_impl.h"
#include "../meta_common.h"

#include "../../core/internal.h"

#define GD_SDPA_VARLEN_BWD_N_INPUTS 5
#define GD_SDPA_VARLEN_BWD_N_OUTPUTS 3

static gd_status sdpa_varlen_bwd_meta(const gd_tensor_desc *const *inputs,
                                      int n_inputs,
                                      _gd_op_attrs *attrs,
                                      gd_tensor_desc *outputs,
                                      int *n_outputs)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *go = NULL;
    const gd_tensor_desc *q = NULL;
    const gd_tensor_desc *k = NULL;
    const gd_tensor_desc *v = NULL;
    const gd_tensor_desc *cu = NULL;

    (void)attrs;
    if (inputs == NULL || outputs == NULL || n_outputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen_bwd meta arguments are NULL");
    }
    if (n_inputs != GD_SDPA_VARLEN_BWD_N_INPUTS) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen_bwd input count mismatch");
    }
    go = inputs[0];
    q = inputs[1];
    k = inputs[2];
    v = inputs[3];
    cu = inputs[4];
    if (go == NULL || q == NULL || k == NULL || v == NULL || cu == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen_bwd input desc is NULL");
    }
    status = _gd_meta_set_output_count(GD_SDPA_VARLEN_BWD_N_OUTPUTS, n_outputs);
    if (status != GD_OK) {
        return status;
    }
    if (go->dtype != q->dtype || k->dtype != q->dtype || v->dtype != q->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_varlen_bwd requires matching q/k/v/go dtype");
    }
    if (go->ndim != 3 || q->ndim != 3 || k->ndim != 3 || v->ndim != 3 || cu->ndim != 1) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen_bwd shape rank mismatch");
    }
    if (go->sizes[0] != q->sizes[0] || go->sizes[1] != q->sizes[1] ||
        go->sizes[2] != q->sizes[2]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen_bwd go must match q/output shape");
    }
    if (k->sizes[0] != q->sizes[0] || v->sizes[0] != q->sizes[0] ||
        v->sizes[1] != k->sizes[1] || k->sizes[2] != q->sizes[2] ||
        v->sizes[2] != q->sizes[2]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen_bwd q/k/v shape mismatch");
    }
    if (cu->dtype != GD_DTYPE_I32) {
        return _gd_error(GD_ERR_DTYPE, "sdpa_varlen_bwd cu_seqlens must be I32");
    }
    outputs[0] = *q;
    outputs[1] = *k;
    outputs[2] = *v;
    return GD_OK;
}

const _gd_op_def _gd_opdef_sdpa_varlen_bwd = {
    .kind = _GD_OP_SDPA_VARLEN_BWD,
    .name = "sdpa_varlen_bwd",
    .min_inputs = GD_SDPA_VARLEN_BWD_N_INPUTS,
    .max_inputs = GD_SDPA_VARLEN_BWD_N_INPUTS,
    .n_outputs = GD_SDPA_VARLEN_BWD_N_OUTPUTS,
    .flags = GD_OPF_INTERNAL,
    .meta = sdpa_varlen_bwd_meta,
};
