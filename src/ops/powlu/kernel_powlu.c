#include "../../backends/cpu_ref/cpu_backend.h"
#include "../../core/internal.h"

#include <float.h>
#include <math.h>

static float sigmoidf_stable(float x)
{
    if (x >= 0.0F) {
        float e = expf(-x);
        return 1.0F / (1.0F + e);
    }
    {
        float e = expf(x);
        return e / (1.0F + e);
    }
}

static float powlu_gate(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0F) {
        return z * s;
    }
    {
        float r = sqrtf(z > 0.0F ? z : 0.0F);
        float a = m / (r + 1.0F);
        return powf(z, a) * s;
    }
}

static float powlu_gate_grad(float z, float m)
{
    float s = sigmoidf_stable(z);
    if (z <= 0.0F) {
        return s * (1.0F + z * (1.0F - s));
    }
    {
        float r = sqrtf(z > 0.0F ? z : 0.0F);
        float rp1 = r + 1.0F;
        float a = m / rp1;
        float g = powf(z, a);
        float da = -m / (2.0F * r * rp1 * rp1);
        float lz = logf(z > FLT_MIN ? z : FLT_MIN);
        return g * s * (a / z + da * lz + (1.0F - s));
    }
}

gd_status _gd_cpu_k_powlu(const gd_tensor_desc *desc, void *out,
                          const void *x1, const void *x2, float m)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v1 = 0.0F;
        float v2 = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x1, i, &v1);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_load_float(desc, x2, i, &v2);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v1 * powlu_gate(v2, m));
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_powlu_bwd(const gd_tensor_desc *desc, void *dx1, void *dx2,
                              const void *x1, const void *x2, const void *go,
                              float m)
{
    int64_t total = _gd_cpu_desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v1 = 0.0F;
        float v2 = 0.0F;
        float g = 0.0F;
        float gate = 0.0F;
        float grad = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x1, i, &v1);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_load_float(desc, x2, i, &v2);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_load_float(desc, go, i, &g);
        if (status != GD_OK) { return status; }
        gate = powlu_gate(v2, m);
        grad = powlu_gate_grad(v2, m);
        status = _gd_cpu_store_float(desc, dx1, i, g * gate);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_store_float(desc, dx2, i, g * v1 * grad);
        if (status != GD_OK) { return status; }
    }
    return GD_OK;
}
