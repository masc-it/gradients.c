#include "../../backends/cpu_ref/cpu_backend.h"

#include <math.h>
#include <stdint.h>

#include "../../core/internal.h"

static int64_t ce_target_at(const gd_tensor_desc *targets_desc, const void *targets, int64_t pos)
{
    return targets_desc->dtype == GD_DTYPE_I64 ? (int64_t)((const int64_t *)targets)[pos]
                                               : (int64_t)((const int32_t *)targets)[pos];
}

gd_status _gd_cpu_k_cross_entropy(float *out,
                                  const gd_tensor_desc *logits_desc,
                                  const void *logits,
                                  const gd_tensor_desc *targets_desc,
                                  const void *targets,
                                  int class_dim,
                                  bool has_ignore_index,
                                  int ignore_index)
{
    int64_t classes = logits_desc->sizes[class_dim];
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t positions = 0;
    int64_t valid = 0;
    int64_t o = 0;
    double loss = 0.0;
    int i = 0;

    for (i = 0; i < class_dim; ++i) {
        outer *= logits_desc->sizes[i];
    }
    for (i = class_dim + 1; i < logits_desc->ndim; ++i) {
        inner *= logits_desc->sizes[i];
    }
    positions = outer * inner;

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;
            int64_t pos = o * inner + in;
            int64_t target = ce_target_at(targets_desc, targets, pos);
            double logit_t = 0.0;

            if (has_ignore_index && target == (int64_t)ignore_index) {
                continue;
            }
            if (target < 0 || target >= classes) {
                return _gd_error(GD_ERR_SHAPE, "cross_entropy target out of range");
            }
            for (c = 0; c < classes; ++c) {
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(logits_desc, logits,
                                                       (o * classes + c) * inner + in, &f);
                if (status != GD_OK) {
                    return status;
                }
                if ((double)f > max_val) {
                    max_val = (double)f;
                }
            }
            for (c = 0; c < classes; ++c) {
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(logits_desc, logits,
                                                       (o * classes + c) * inner + in, &f);
                if (status != GD_OK) {
                    return status;
                }
                sum += exp((double)f - max_val);
            }
            {
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(logits_desc, logits,
                                                       (o * classes + target) * inner + in, &f);
                if (status != GD_OK) {
                    return status;
                }
                logit_t = (double)f;
            }
            loss += -(logit_t - max_val - log(sum));
            ++valid;
        }
    }

    (void)positions;
    out[0] = valid > 0 ? (float)(loss / (double)valid) : 0.0F;
    return GD_OK;
}

gd_status _gd_cpu_k_cross_entropy_bwd(const gd_tensor_desc *logits_desc,
                                      float *dlogits,
                                      const float *logits,
                                      const gd_tensor_desc *targets_desc,
                                      const void *targets,
                                      const float *go_scalar,
                                      int class_dim,
                                      bool has_ignore_index,
                                      int ignore_index)
{
    int64_t classes = logits_desc->sizes[class_dim];
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t positions = 0;
    int64_t valid = 0;
    int64_t o = 0;
    int64_t idx = 0;
    double scale = 0.0;
    int i = 0;

    for (i = 0; i < class_dim; ++i) {
        outer *= logits_desc->sizes[i];
    }
    for (i = class_dim + 1; i < logits_desc->ndim; ++i) {
        inner *= logits_desc->sizes[i];
    }
    positions = outer * inner;

    for (idx = 0; idx < outer * classes * inner; ++idx) {
        dlogits[idx] = 0.0F;
    }
    for (o = 0; o < positions; ++o) {
        int64_t target = ce_target_at(targets_desc, targets, o);
        if (has_ignore_index && target == (int64_t)ignore_index) {
            continue;
        }
        if (target < 0 || target >= classes) {
            return _gd_error(GD_ERR_SHAPE, "cross_entropy target out of range");
        }
        ++valid;
    }
    if (valid == 0) {
        return GD_OK;
    }
    scale = (double)go_scalar[0] / (double)valid;

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;
            int64_t pos = o * inner + in;
            int64_t target = ce_target_at(targets_desc, targets, pos);

            if (has_ignore_index && target == (int64_t)ignore_index) {
                continue;
            }
            for (c = 0; c < classes; ++c) {
                double v = (double)logits[(o * classes + c) * inner + in];
                if (v > max_val) {
                    max_val = v;
                }
            }
            for (c = 0; c < classes; ++c) {
                sum += exp((double)logits[(o * classes + c) * inner + in] - max_val);
            }
            for (c = 0; c < classes; ++c) {
                double p = exp((double)logits[(o * classes + c) * inner + in] - max_val) / sum;
                double onehot = (c == target) ? 1.0 : 0.0;
                dlogits[(o * classes + c) * inner + in] = (float)(scale * (p - onehot));
            }
        }
    }
    return GD_OK;
}
