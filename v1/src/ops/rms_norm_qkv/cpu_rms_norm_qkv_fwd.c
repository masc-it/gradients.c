#include "../../backends/cpu_ref/cpu_op.h"
#include "../../core/internal.h"

#include <math.h>
#include <stdlib.h>

static gd_status project_one(const gd_tensor_desc *out_desc,
                             void *out,
                             const gd_tensor_desc *w_desc,
                             const void *w,
                             const float *normed,
                             int64_t row,
                             int64_t d)
{
    int64_t out_features = out_desc->sizes[out_desc->ndim - 1];
    int64_t o = 0;

    for (o = 0; o < out_features; ++o) {
        double acc = 0.0;
        int64_t c = 0;
        for (c = 0; c < d; ++c) {
            float wf = 0.0F;
            gd_status status = _gd_cpu_load_float(w_desc, w, c * out_features + o, &wf);
            if (status != GD_OK) {
                return status;
            }
            acc += (double)normed[c] * (double)wf;
        }
        {
            gd_status status = _gd_cpu_store_float(out_desc, out,
                                                   row * out_features + o, (float)acc);
            if (status != GD_OK) {
                return status;
            }
        }
    }
    return GD_OK;
}

static gd_status rms_norm_qkv_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *x_desc = NULL;
    const gd_tensor_desc *weight_desc = NULL;
    const gd_tensor_desc *wq_desc = NULL;
    const gd_tensor_desc *wk_desc = NULL;
    const gd_tensor_desc *wv_desc = NULL;
    const gd_tensor_desc *norm_desc = NULL;
    const gd_tensor_desc *q_desc = NULL;
    const gd_tensor_desc *k_desc = NULL;
    const gd_tensor_desc *v_desc = NULL;
    void *x = NULL;
    void *weight = NULL;
    void *wq = NULL;
    void *wk = NULL;
    void *wv = NULL;
    void *norm_out = NULL;
    void *q_out = NULL;
    void *k_out = NULL;
    void *v_out = NULL;
    float *normed = NULL;
    int64_t d = 0;
    int64_t rows = 0;
    int64_t r = 0;

    status = _gd_cpu_exec_input(exec, node, 0, &x, &x_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, &weight, &weight_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, &wq, &wq_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, &wk, &wk_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 4, &wv, &wv_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 0, &norm_out, &norm_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 1, &q_out, &q_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 2, &k_out, &k_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 3, &v_out, &v_desc);
    if (status != GD_OK) { return status; }

    d = x_desc->sizes[x_desc->ndim - 1];
    rows = d > 0 ? _gd_cpu_desc_numel(x_desc) / d : 0;
    if (d <= 0) {
        return GD_OK;
    }
    normed = (float *)malloc((size_t)d * sizeof(*normed));
    if (normed == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "cpu rms_norm_qkv row allocation failed");
    }
    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;
        int64_t c = 0;

        for (c = 0; c < d; ++c) {
            float xf = 0.0F;
            status = _gd_cpu_load_float(x_desc, x, r * d + c, &xf);
            if (status != GD_OK) { goto done; }
            sumsq += (double)xf * (double)xf;
        }
        inv = 1.0 / sqrt(sumsq / (double)d + (double)node->attrs.eps);
        for (c = 0; c < d; ++c) {
            float xf = 0.0F;
            float wf = 0.0F;
            status = _gd_cpu_load_float(x_desc, x, r * d + c, &xf);
            if (status != GD_OK) { goto done; }
            status = _gd_cpu_load_float(weight_desc, weight, c, &wf);
            if (status != GD_OK) { goto done; }
            normed[c] = (float)((double)xf * inv * (double)wf);
            status = _gd_cpu_store_float(norm_desc, norm_out, r * d + c, normed[c]);
            if (status != GD_OK) { goto done; }
            /* Projection inputs are the materialized norm output. For F16 this
             * reload applies the same rounding as unfused rms_norm -> matmul and
             * the Metal/MPS path. */
            status = _gd_cpu_load_float(norm_desc, norm_out, r * d + c, &normed[c]);
            if (status != GD_OK) { goto done; }
        }
        status = project_one(q_desc, q_out, wq_desc, wq, normed, r, d);
        if (status != GD_OK) { goto done; }
        status = project_one(k_desc, k_out, wk_desc, wk, normed, r, d);
        if (status != GD_OK) { goto done; }
        status = project_one(v_desc, v_out, wv_desc, wv, normed, r, d);
        if (status != GD_OK) { goto done; }
    }

done:
    free(normed);
    return status;
}

const _gd_cpu_op _gd_cpu_op_rms_norm_qkv = {
    .kind = _GD_OP_RMS_NORM_QKV,
    .name = "rms_norm_qkv",
    .support = _gd_cpu_support_default,
    .run = rms_norm_qkv_run,
};
