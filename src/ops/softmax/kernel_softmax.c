#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <math.h>

gd_status _gd_cpu_k_softmax(const gd_tensor_desc *desc, void *out, const void *x, int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        inner *= desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double max_val = -HUGE_VAL;
            double sum = 0.0;
            int64_t c = 0;

            for (c = 0; c < d; ++c) {
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(desc, x, (o * d + c) * inner + in, &f);
                if (status != GD_OK) {
                    return status;
                }
                if ((double)f > max_val) {
                    max_val = (double)f;
                }
            }
            for (c = 0; c < d; ++c) {
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(desc, x, (o * d + c) * inner + in, &f);
                if (status != GD_OK) {
                    return status;
                }
                sum += exp((double)f - max_val);
            }
            for (c = 0; c < d; ++c) {
                int64_t off = (o * d + c) * inner + in;
                float f = 0.0F;
                gd_status status = _gd_cpu_load_float(desc, x, off, &f);
                if (status != GD_OK) {
                    return status;
                }
                status = _gd_cpu_store_float(desc, out, off,
                                             (float)(exp((double)f - max_val) / sum));
                if (status != GD_OK) {
                    return status;
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_softmax_bwd(const gd_tensor_desc *desc,
                                float *dx,
                                const float *y,
                                const float *go,
                                int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= desc->sizes[i];
    }
    for (i = dim + 1; i < desc->ndim; ++i) {
        inner *= desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double dot = 0.0;
            int64_t c = 0;

            for (c = 0; c < d; ++c) {
                int64_t idx = (o * d + c) * inner + in;
                dot += (double)go[idx] * (double)y[idx];
            }
            for (c = 0; c < d; ++c) {
                int64_t idx = (o * d + c) * inner + in;
                dx[idx] = (float)((double)y[idx] * ((double)go[idx] - dot));
            }
        }
    }
    return GD_OK;
}
