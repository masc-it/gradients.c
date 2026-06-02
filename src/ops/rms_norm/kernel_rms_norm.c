#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_rms_norm(const gd_tensor_desc *desc,
                             void *out,
                             const void *x,
                             const void *weight,
                             float eps)
{
    int64_t last = desc->sizes[desc->ndim - 1];
    int64_t rows = _gd_cpu_desc_numel(desc) / last;
    int64_t r = 0;

    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;
        int64_t c = 0;

        for (c = 0; c < last; ++c) {
            float f = 0.0F;
            gd_status status = _gd_cpu_load_float(desc, x, r * last + c, &f);
            if (status != GD_OK) {
                return status;
            }
            sumsq += (double)f * (double)f;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        for (c = 0; c < last; ++c) {
            float xv = 0.0F;
            float wv = 0.0F;
            gd_status status = _gd_cpu_load_float(desc, x, r * last + c, &xv);
            if (status != GD_OK) {
                return status;
            }
            status = _gd_cpu_load_float(desc, weight, c, &wv);
            if (status != GD_OK) {
                return status;
            }
            status = _gd_cpu_store_float(desc, out, r * last + c,
                                         (float)((double)xv * inv * (double)wv));
            if (status != GD_OK) {
                return status;
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rms_norm_bwd(const gd_tensor_desc *desc, void *dx,
                                 const void *x, const void *weight,
                                 const void *go, float eps)
{
    int64_t last = desc->sizes[desc->ndim - 1];
    int64_t rows = _gd_cpu_desc_numel(desc) / last;
    int64_t r = 0;

    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;
        double inv3 = 0.0;
        double A = 0.0; /* sum_c go_c * w_c * x_c */
        int64_t c = 0;

        for (c = 0; c < last; ++c) {
            float xf = 0.0F;
            gd_status status = _gd_cpu_load_float(desc, x, r * last + c, &xf);
            if (status != GD_OK) { return status; }
            sumsq += (double)xf * (double)xf;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        inv3 = inv * inv * inv;
        for (c = 0; c < last; ++c) {
            float gf = 0.0F;
            float wf = 0.0F;
            float xf = 0.0F;
            gd_status status = _gd_cpu_load_float(desc, go, r * last + c, &gf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, weight, c, &wf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, x, r * last + c, &xf);
            if (status != GD_OK) { return status; }
            A += (double)gf * (double)wf * (double)xf;
        }
        for (c = 0; c < last; ++c) {
            float gf = 0.0F;
            float xf = 0.0F;
            float wf = 0.0F;
            double d = 0.0;
            gd_status status = _gd_cpu_load_float(desc, go, r * last + c, &gf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, x, r * last + c, &xf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, weight, c, &wf);
            if (status != GD_OK) { return status; }
            d = inv * (double)gf * (double)wf -
                (double)xf * inv3 * A / (double)last;
            status = _gd_cpu_store_float(desc, dx, r * last + c, (float)d);
            if (status != GD_OK) { return status; }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_rms_norm_wbwd(const gd_tensor_desc *x_desc, float *dweight,
                                  const void *x, const void *go, float eps)
{
    int64_t last = x_desc->sizes[x_desc->ndim - 1];
    int64_t rows = _gd_cpu_desc_numel(x_desc) / last;
    int64_t r = 0;
    int64_t c = 0;

    for (c = 0; c < last; ++c) {
        dweight[c] = 0.0F;
    }
    for (r = 0; r < rows; ++r) {
        double sumsq = 0.0;
        double inv = 0.0;

        for (c = 0; c < last; ++c) {
            float xf = 0.0F;
            gd_status status = _gd_cpu_load_float(x_desc, x, r * last + c, &xf);
            if (status != GD_OK) { return status; }
            sumsq += (double)xf * (double)xf;
        }
        inv = 1.0 / sqrt(sumsq / (double)last + (double)eps);
        for (c = 0; c < last; ++c) {
            float gf = 0.0F;
            float xf = 0.0F;
            gd_status status = _gd_cpu_load_float(x_desc, go, r * last + c, &gf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(x_desc, x, r * last + c, &xf);
            if (status != GD_OK) { return status; }
            dweight[c] += (float)((double)gf * (double)xf * inv);
        }
    }
    return GD_OK;
}
