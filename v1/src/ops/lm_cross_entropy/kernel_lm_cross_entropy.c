#include "../../backends/cpu_ref/cpu_backend.h"

#include <math.h>
#include <stdint.h>

#include "../../core/internal.h"

static int64_t desc_numel(const gd_tensor_desc *desc)
{
    int64_t numel = 1;
    int i = 0;

    for (i = 0; i < desc->ndim; ++i) {
        numel *= desc->sizes[i];
    }
    return numel;
}

static int64_t lmce_target_at(const gd_tensor_desc *targets_desc, const void *targets, int64_t row)
{
    return targets_desc->dtype == GD_DTYPE_I64 ? (int64_t)((const int64_t *)targets)[row]
                                               : (int64_t)((const int32_t *)targets)[row];
}

gd_status _gd_cpu_k_lm_cross_entropy(float *out,
                                     float *row_max,
                                     float *row_sum,
                                     const gd_tensor_desc *hidden_desc,
                                     const void *hidden,
                                     const gd_tensor_desc *weight_desc,
                                     const void *weight,
                                     const gd_tensor_desc *targets_desc,
                                     const void *targets,
                                     bool has_ignore_index,
                                     int ignore_index)
{
    int64_t D = hidden_desc->sizes[hidden_desc->ndim - 1];
    int64_t V = weight_desc->sizes[0];
    int64_t N = desc_numel(hidden_desc) / D;
    int64_t valid = 0;
    double loss = 0.0;
    int64_t n = 0;

    for (n = 0; n < N; ++n) {
        int64_t target = lmce_target_at(targets_desc, targets, n);
        double max_val = -HUGE_VAL;
        double sum = 0.0;
        double target_logit = 0.0;
        int64_t v = 0;
        if (has_ignore_index && target == (int64_t)ignore_index) {
            row_max[n] = 0.0F;
            row_sum[n] = 1.0F;
            continue;
        }
        if (target < 0 || target >= V) {
            return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy target out of range");
        }
        for (v = 0; v < V; ++v) {
            double s = 0.0;
            int64_t d = 0;
            for (d = 0; d < D; ++d) {
                float hv = 0.0F;
                float wv = 0.0F;
                gd_status status = _gd_cpu_load_float(hidden_desc, hidden, n * D + d, &hv);
                if (status != GD_OK) {
                    return status;
                }
                status = _gd_cpu_load_float(weight_desc, weight, v * D + d, &wv);
                if (status != GD_OK) {
                    return status;
                }
                s += (double)hv * (double)wv;
            }
            if (v == target) {
                target_logit = s;
            }
            if (s > max_val) {
                max_val = s;
            }
        }
        for (v = 0; v < V; ++v) {
            double s = 0.0;
            int64_t d = 0;
            for (d = 0; d < D; ++d) {
                float hv = 0.0F;
                float wv = 0.0F;
                gd_status status = _gd_cpu_load_float(hidden_desc, hidden, n * D + d, &hv);
                if (status != GD_OK) {
                    return status;
                }
                status = _gd_cpu_load_float(weight_desc, weight, v * D + d, &wv);
                if (status != GD_OK) {
                    return status;
                }
                s += (double)hv * (double)wv;
            }
            sum += exp(s - max_val);
        }
        row_max[n] = (float)max_val;
        row_sum[n] = (float)sum;
        loss += -(target_logit - max_val - log(sum));
        ++valid;
    }
    out[0] = valid > 0 ? (float)(loss / (double)valid) : 0.0F;
    return GD_OK;
}

gd_status _gd_cpu_k_lm_cross_entropy_bwd(const gd_tensor_desc *hidden_desc,
                                         const gd_tensor_desc *dhidden_desc,
                                         void *dhidden,
                                         const void *hidden,
                                         const gd_tensor_desc *weight_desc,
                                         const gd_tensor_desc *dweight_desc,
                                         void *dweight,
                                         const void *weight,
                                         const gd_tensor_desc *targets_desc,
                                         const void *targets,
                                         const float *go_scalar,
                                         const float *row_max,
                                         const float *row_sum,
                                         bool has_ignore_index,
                                         int ignore_index)
{
    int64_t D = hidden_desc->sizes[hidden_desc->ndim - 1];
    int64_t V = weight_desc->sizes[0];
    int64_t N = desc_numel(hidden_desc) / D;
    int64_t valid = 0;
    double scale = 0.0;
    int64_t i = 0;

    for (i = 0; i < N * D; ++i) {
        gd_status status = _gd_cpu_store_float(dhidden_desc, dhidden, i, 0.0F);
        if (status != GD_OK) { return status; }
    }
    for (i = 0; i < V * D; ++i) {
        gd_status status = _gd_cpu_store_float(dweight_desc, dweight, i, 0.0F);
        if (status != GD_OK) { return status; }
    }
    for (i = 0; i < N; ++i) {
        int64_t target = lmce_target_at(targets_desc, targets, i);
        if (has_ignore_index && target == (int64_t)ignore_index) {
            continue;
        }
        if (target < 0 || target >= V) {
            return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy target out of range");
        }
        ++valid;
    }
    if (valid == 0) {
        return GD_OK;
    }
    scale = (double)go_scalar[0] / (double)valid;

    for (int64_t n = 0; n < N; ++n) {
        int64_t target = lmce_target_at(targets_desc, targets, n);
        double max_val = (double)row_max[n];
        double sum = (double)row_sum[n];
        if (has_ignore_index && target == (int64_t)ignore_index) {
            continue;
        }
        for (int64_t v = 0; v < V; ++v) {
            double s = 0.0;
            double dl = 0.0;
            for (int64_t d = 0; d < D; ++d) {
                float hv = 0.0F;
                float wv = 0.0F;
                gd_status status = _gd_cpu_load_float(hidden_desc, hidden, n * D + d, &hv);
                if (status != GD_OK) { return status; }
                status = _gd_cpu_load_float(weight_desc, weight, v * D + d, &wv);
                if (status != GD_OK) { return status; }
                s += (double)hv * (double)wv;
            }
            dl = scale * (exp(s - max_val) / sum - (v == target ? 1.0 : 0.0));
            for (int64_t d = 0; d < D; ++d) {
                float hv = 0.0F;
                float wv = 0.0F;
                float old = 0.0F;
                gd_status status = _gd_cpu_load_float(hidden_desc, hidden, n * D + d, &hv);
                if (status != GD_OK) { return status; }
                status = _gd_cpu_load_float(weight_desc, weight, v * D + d, &wv);
                if (status != GD_OK) { return status; }
                status = _gd_cpu_load_float(dhidden_desc, dhidden, n * D + d, &old);
                if (status != GD_OK) { return status; }
                status = _gd_cpu_store_float(dhidden_desc, dhidden, n * D + d,
                                             old + (float)(dl * (double)wv));
                if (status != GD_OK) { return status; }
                status = _gd_cpu_load_float(dweight_desc, dweight, v * D + d, &old);
                if (status != GD_OK) { return status; }
                status = _gd_cpu_store_float(dweight_desc, dweight, v * D + d,
                                             old + (float)(dl * (double)hv));
                if (status != GD_OK) { return status; }
            }
        }
    }
    return GD_OK;
}
