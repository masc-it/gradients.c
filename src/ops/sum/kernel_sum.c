#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

gd_status _gd_cpu_k_sum(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = x_desc->sizes[dim];
    int64_t o = 0;
    int i = 0;

    (void)out_desc;
    for (i = 0; i < dim; ++i) {
        outer *= x_desc->sizes[i];
    }
    for (i = dim + 1; i < x_desc->ndim; ++i) {
        inner *= x_desc->sizes[i];
    }

    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double acc = 0.0;
            int64_t c = 0;
            for (c = 0; c < d; ++c) {
                acc += (double)x[(o * d + c) * inner + in];
            }
            out[o * inner + in] = (float)acc;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_sum_bwd(const gd_tensor_desc *x_desc,
                            float *dx,
                            const float *go,
                            int dim)
{
    int64_t inner = 1;
    int64_t outer = 1;
    int64_t d = x_desc->sizes[dim];
    int64_t o = 0;
    double scale = 1.0;
    int i = 0;

    for (i = 0; i < dim; ++i) {
        outer *= x_desc->sizes[i];
    }
    for (i = dim + 1; i < x_desc->ndim; ++i) {
        inner *= x_desc->sizes[i];
    }
    for (o = 0; o < outer; ++o) {
        int64_t in = 0;
        for (in = 0; in < inner; ++in) {
            double g = (double)go[o * inner + in] * scale;
            int64_t c = 0;
            for (c = 0; c < d; ++c) {
                dx[(o * d + c) * inner + in] = (float)g;
            }
        }
    }
    return GD_OK;
}
