#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_adamw(const gd_tensor_desc *param_desc,
                          float *param,
                          const float *grad,
                          float *m,
                          float *v,
                          const float *step,
                          const float *lr_tensor,
                          float lr,
                          float lr_scale,
                          float beta1,
                          float beta2,
                          float eps,
                          float weight_decay)
{
    int64_t total = _gd_cpu_desc_numel(param_desc);
    int64_t i = 0;
    double t = (double)step[0];
    double bc1 = 1.0 - pow((double)beta1, t);
    double bc2 = 1.0 - pow((double)beta2, t);
    double step_lr = (lr_tensor != NULL ? (double)lr_tensor[0] : (double)lr) *
                     (double)lr_scale;

    if (t < 1.0) {
        return _gd_error(GD_ERR_INVALID_STATE, "adamw step counter must be >= 1");
    }
    if (!isfinite(step_lr) || step_lr < 0.0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "adamw lr must be finite and nonnegative");
    }
    for (i = 0; i < total; ++i) {
        double g = (double)grad[i];
        double mi = (double)beta1 * (double)m[i] + (1.0 - (double)beta1) * g;
        double vi = (double)beta2 * (double)v[i] + (1.0 - (double)beta2) * g * g;
        double mhat = mi / bc1;
        double vhat = vi / bc2;
        double p = (double)param[i];

        m[i] = (float)mi;
        v[i] = (float)vi;
        p -= step_lr * (double)weight_decay * p;        /* decoupled weight decay */
        p -= step_lr * mhat / (sqrt(vhat) + (double)eps);
        param[i] = (float)p;
    }
    return GD_OK;
}
