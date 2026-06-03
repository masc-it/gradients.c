#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_gelu(const gd_tensor_desc *desc, void *out, const void *x, int tanh_approx)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    if (tanh_approx) {
        const double c = 0.7978845608028654; /* sqrt(2/pi) */
        for (i = 0; i < total; ++i) {
            float f = 0.0F;
            double xv = 0.0;
            double inner = 0.0;
            gd_status status = _gd_cpu_load_float(desc, x, i, &f);
            if (status != GD_OK) {
                return status;
            }
            xv = (double)f;
            inner = c * (xv + 0.044715 * xv * xv * xv);
            status = _gd_cpu_store_float(desc, out, i,
                                         (float)(0.5 * xv * (1.0 + tanh(inner))));
            if (status != GD_OK) {
                return status;
            }
        }
    } else {
        const double inv_sqrt2 = 0.7071067811865476;
        for (i = 0; i < total; ++i) {
            float f = 0.0F;
            double xv = 0.0;
            gd_status status = _gd_cpu_load_float(desc, x, i, &f);
            if (status != GD_OK) {
                return status;
            }
            xv = (double)f;
            status = _gd_cpu_store_float(desc, out, i,
                                         (float)(0.5 * xv * (1.0 + erf(xv * inv_sqrt2))));
            if (status != GD_OK) {
                return status;
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_gelu_bwd(const gd_tensor_desc *desc, void *dx, const void *x,
                             const void *go, int tanh_approx)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    if (tanh_approx) {
        const double c = 0.7978845608028654; /* sqrt(2/pi) */
        for (i = 0; i < total; ++i) {
            float xf = 0.0F;
            float gf = 0.0F;
            double xv = 0.0;
            double u = 0.0;
            double t = 0.0;
            double du = 0.0;
            double grad = 0.0;
            gd_status status = _gd_cpu_load_float(desc, x, i, &xf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, go, i, &gf);
            if (status != GD_OK) { return status; }
            xv = (double)xf;
            u = c * (xv + 0.044715 * xv * xv * xv);
            t = tanh(u);
            du = c * (1.0 + 3.0 * 0.044715 * xv * xv);
            grad = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * du;
            status = _gd_cpu_store_float(desc, dx, i, (float)((double)gf * grad));
            if (status != GD_OK) { return status; }
        }
    } else {
        const double inv_sqrt2 = 0.7071067811865476;
        const double inv_sqrt2pi = 0.3989422804014327;
        for (i = 0; i < total; ++i) {
            float xf = 0.0F;
            float gf = 0.0F;
            double xv = 0.0;
            double cdf = 0.0;
            double pdf = 0.0;
            gd_status status = _gd_cpu_load_float(desc, x, i, &xf);
            if (status != GD_OK) { return status; }
            status = _gd_cpu_load_float(desc, go, i, &gf);
            if (status != GD_OK) { return status; }
            xv = (double)xf;
            cdf = 0.5 * (1.0 + erf(xv * inv_sqrt2));
            pdf = inv_sqrt2pi * exp(-0.5 * xv * xv);
            status = _gd_cpu_store_float(desc, dx, i,
                                         (float)((double)gf * (cdf + xv * pdf)));
            if (status != GD_OK) { return status; }
        }
    }
    return GD_OK;
}
