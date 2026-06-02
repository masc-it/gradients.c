#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_clip_grad_norm(const gd_tensor_desc * const *grad_descs,
                                   float **grads,
                                   int n_grads,
                                   float max_norm,
                                   float eps,
                                   float *norm_out)
{
    double sumsq = 0.0;
    double norm;
    double scale = 1.0;
    int g = 0;

    if (grad_descs == NULL || grads == NULL || n_grads <= 0 || norm_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid clip_grad_norm arguments");
    }
    if (!isfinite(max_norm) || max_norm <= 0.0F || !isfinite(eps) || eps < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid clip_grad_norm config");
    }
    for (g = 0; g < n_grads; ++g) {
        int64_t total;
        int64_t i;
        if (grad_descs[g] == NULL || grads[g] == NULL) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "null clip_grad_norm input");
        }
        total = _gd_cpu_desc_numel(grad_descs[g]);
        for (i = 0; i < total; ++i) {
            double v = (double)grads[g][i];
            sumsq += v * v;
        }
    }
    norm = sqrt(sumsq);
    if (norm > (double)max_norm) {
        scale = (double)max_norm / (norm + (double)eps);
    }
    for (g = 0; g < n_grads; ++g) {
        int64_t total = _gd_cpu_desc_numel(grad_descs[g]);
        int64_t i;
        for (i = 0; i < total; ++i) {
            grads[g][i] = (float)((double)grads[g][i] * scale);
        }
    }
    norm_out[0] = (float)norm;
    return GD_OK;
}
