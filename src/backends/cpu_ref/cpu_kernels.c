#include "cpu_backend.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static void unravel(int64_t linear, const gd_tensor_desc *desc, int64_t *index)
{
    int i = 0;

    for (i = desc->ndim - 1; i >= 0; --i) {
        index[i] = linear % desc->sizes[i];
        linear /= desc->sizes[i];
    }
}

/* Linear offset into a contiguous input, broadcasting against an output index. */
static int64_t broadcast_offset(const int64_t *out_index,
                                int out_ndim,
                                const gd_tensor_desc *in_desc)
{
    int64_t stride = 1;
    int64_t offset = 0;
    int i = 0;

    for (i = in_desc->ndim - 1; i >= 0; --i) {
        int out_pos = out_ndim - (in_desc->ndim - i);
        int64_t coord = in_desc->sizes[i] == 1 ? 0 : out_index[out_pos];

        offset += coord * stride;
        stride *= in_desc->sizes[i];
    }
    return offset;
}

gd_status _gd_cpu_load_float(const gd_tensor_desc *desc, const void *data,
                             int64_t i, float *out)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        *out = ((const float *)data)[i];
        return GD_OK;
    case GD_DTYPE_F16:
        *out = _gd_f16_bits_to_f32(((const uint16_t *)data)[i]);
        return GD_OK;
    case GD_DTYPE_I32:
        *out = (float)((const int32_t *)data)[i];
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF float load dtype is not implemented");
    }
}

gd_status _gd_cpu_store_float(const gd_tensor_desc *desc, void *data,
                              int64_t i, float value)
{
    switch (desc->dtype) {
    case GD_DTYPE_F32:
        ((float *)data)[i] = value;
        return GD_OK;
    case GD_DTYPE_F16:
        ((uint16_t *)data)[i] = _gd_f32_to_f16_bits(value);
        return GD_OK;
    case GD_DTYPE_I32:
        ((int32_t *)data)[i] = (int32_t)value;
        return GD_OK;
    default:
        return _gd_error(GD_ERR_UNSUPPORTED, "CPU_REF float store dtype is not implemented");
    }
}

gd_status _gd_cpu_k_elementwise(_gd_op_kind op,
                                const gd_tensor_desc *out_desc,
                                void *out,
                                const gd_tensor_desc *a_desc,
                                const void *a,
                                const gd_tensor_desc *b_desc,
                                const void *b)
{
    int64_t total = desc_numel(out_desc);
    int64_t lin = 0;
    int64_t index[GD_MAX_DIMS];

    for (lin = 0; lin < total; ++lin) {
        int64_t ao = 0;
        int64_t bo = 0;
        float av = 0.0F;
        float bv = 0.0F;
        float y = 0.0F;
        gd_status status = GD_OK;

        unravel(lin, out_desc, index);
        ao = broadcast_offset(index, out_desc->ndim, a_desc);
        bo = broadcast_offset(index, out_desc->ndim, b_desc);
        status = _gd_cpu_load_float(a_desc, a, ao, &av);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_load_float(b_desc, b, bo, &bv);
        if (status != GD_OK) {
            return status;
        }
        y = (op == _GD_OP_ADD) ? (av + bv) : (av * bv);
        status = _gd_cpu_store_float(out_desc, out, lin, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_scale(const gd_tensor_desc *desc, void *out, const void *x, float scale)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v * scale);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_relu(const gd_tensor_desc *desc, void *out, const void *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v > 0.0F ? v : 0.0F);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_silu(const gd_tensor_desc *desc, void *out, const void *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(desc, out, i, v / (1.0F + expf(-v)));
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static uint32_t dropout_hash(uint64_t seed, uint64_t run_id, uint64_t index)
{
    uint32_t h = (uint32_t)seed ^ ((uint32_t)(seed >> 32) * UINT32_C(0x9e3779b9));

    h ^= (uint32_t)run_id * UINT32_C(0x85ebca6b);
    h ^= (uint32_t)(run_id >> 32) * UINT32_C(0xc2b2ae35);
    h ^= (uint32_t)index * UINT32_C(0x27d4eb2d);
    h ^= (uint32_t)(index >> 32) * UINT32_C(0x165667b1);
    h ^= h >> 16;
    h *= UINT32_C(0x7feb352d);
    h ^= h >> 15;
    h *= UINT32_C(0x846ca68b);
    h ^= h >> 16;
    return h;
}

static int dropout_keep(uint64_t seed, uint64_t run_id, uint64_t index, float p)
{
    uint32_t r = dropout_hash(seed, run_id, index);
    float u = ((float)(r >> 8) + 0.5F) * (1.0F / 16777216.0F);

    return u >= p;
}

gd_status _gd_cpu_k_dropout(const gd_tensor_desc *desc,
                            void *out,
                            const void *x,
                            float p,
                            uint64_t seed,
                            uint64_t run_id)
{
    int64_t total = desc_numel(desc);
    float keep_scale = 1.0F / (1.0F - p);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        float y = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &v);
        if (status != GD_OK) {
            return status;
        }
        y = dropout_keep(seed, run_id, (uint64_t)i, p) ? v * keep_scale : 0.0F;
        status = _gd_cpu_store_float(desc, out, i, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_dropout_bwd(const gd_tensor_desc *desc,
                                void *dx,
                                const void *go,
                                float p,
                                uint64_t seed,
                                uint64_t run_id)
{
    int64_t total = desc_numel(desc);
    float keep_scale = 1.0F / (1.0F - p);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float v = 0.0F;
        float y = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, go, i, &v);
        if (status != GD_OK) {
            return status;
        }
        y = dropout_keep(seed, run_id, (uint64_t)i, p) ? v * keep_scale : 0.0F;
        status = _gd_cpu_store_float(desc, dx, i, y);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

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
    int64_t total = desc_numel(desc);
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
    int64_t total = desc_numel(desc);
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

gd_status _gd_cpu_k_reduce(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           int dim,
                           bool mean)
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
            if (mean) {
                acc /= (double)d;
            }
            out[o * inner + in] = (float)acc;
        }
    }
    return GD_OK;
}

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

gd_status _gd_cpu_k_rms_norm(const gd_tensor_desc *desc,
                             void *out,
                             const void *x,
                             const void *weight,
                             float eps)
{
    int64_t last = desc->sizes[desc->ndim - 1];
    int64_t rows = desc_numel(desc) / last;
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
    int64_t rows = desc_numel(desc) / last;
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
    int64_t rows = desc_numel(x_desc) / last;
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

gd_status _gd_cpu_k_cast(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *x_desc,
                         const void *x)
{
    int64_t total = desc_numel(out_desc);
    int64_t i = 0;

    if (desc_numel(x_desc) != total) {
        return _gd_error(GD_ERR_SHAPE, "cast requires equal element counts");
    }
    if (x_desc->dtype == out_desc->dtype) {
        size_t elem = gd_dtype_sizeof(out_desc->dtype);
        if (elem == 0U) {
            return _gd_error(GD_ERR_DTYPE, "cast requires a fixed-size dtype");
        }
        memcpy(out, x, (size_t)total * elem);
        return GD_OK;
    }
    for (i = 0; i < total; ++i) {
        float value = 0.0F;
        gd_status status = _gd_cpu_load_float(x_desc, x, i, &value);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_store_float(out_desc, out, i, value);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_copy(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *in_desc,
                         const void *in)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = desc_numel(out_desc);

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "copy requires a fixed-size dtype");
    }
    if (out_desc->dtype != in_desc->dtype) {
        return _gd_error(GD_ERR_DTYPE, "copy requires matching dtypes");
    }
    if (desc_numel(in_desc) != total) {
        return _gd_error(GD_ERR_SHAPE, "copy requires equal element counts");
    }
    memcpy(out, in, (size_t)total * elem);
    return GD_OK;
}

gd_status _gd_cpu_k_gelu(const gd_tensor_desc *desc, void *out, const void *x, int tanh_approx)
{
    int64_t total = desc_numel(desc);
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
    int64_t total = desc_numel(desc);
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

gd_status _gd_cpu_k_transpose(const gd_tensor_desc *out_desc, void *out,
                              const gd_tensor_desc *in_desc, const void *in,
                              const int *perm)
{
    size_t elem = gd_dtype_sizeof(out_desc->dtype);
    int64_t total = desc_numel(out_desc);
    int64_t in_strides[GD_MAX_DIMS];
    int64_t out_index[GD_MAX_DIMS];
    int ndim = out_desc->ndim;
    int64_t stride = 1;
    int64_t lin = 0;
    int k = 0;

    if (elem == 0U) {
        return _gd_error(GD_ERR_DTYPE, "transpose requires a fixed-size dtype");
    }
    if (in_desc->ndim != ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose rank mismatch");
    }
    for (k = ndim - 1; k >= 0; --k) {
        in_strides[k] = stride;
        stride *= in_desc->sizes[k];
    }
    for (lin = 0; lin < total; ++lin) {
        int64_t in_off = 0;
        unravel(lin, out_desc, out_index);
        for (k = 0; k < ndim; ++k) {
            in_off += out_index[k] * in_strides[perm[k]];
        }
        memcpy((unsigned char *)out + (size_t)lin * elem,
               (const unsigned char *)in + (size_t)in_off * elem, elem);
    }
    return GD_OK;
}

gd_status _gd_cpu_k_relu_bwd(const gd_tensor_desc *desc,
                             void *dx,
                             const void *x,
                             const void *go)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float xf = 0.0F;
        float gf = 0.0F;
        gd_status status = _gd_cpu_load_float(desc, x, i, &xf);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_load_float(desc, go, i, &gf);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_store_float(desc, dx, i, xf > 0.0F ? gf : 0.0F);
        if (status != GD_OK) { return status; }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_silu_bwd(const gd_tensor_desc *desc,
                             void *dx,
                             const void *x,
                             const void *go)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        float xf = 0.0F;
        float gf = 0.0F;
        double s = 0.0;
        double grad = 0.0;
        gd_status status = _gd_cpu_load_float(desc, x, i, &xf);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_load_float(desc, go, i, &gf);
        if (status != GD_OK) { return status; }
        s = 1.0 / (1.0 + exp(-(double)xf));
        grad = s * (1.0 + (double)xf * (1.0 - s));
        status = _gd_cpu_store_float(desc, dx, i, (float)((double)gf * grad));
        if (status != GD_OK) { return status; }
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

gd_status _gd_cpu_k_sum_bwd(const gd_tensor_desc *x_desc,
                            float *dx,
                            const float *go,
                            int dim,
                            bool mean)
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
    if (mean) {
        scale = 1.0 / (double)d;
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

gd_status _gd_cpu_k_assert_finite(const gd_tensor_desc *desc, const float *x)
{
    int64_t total = desc_numel(desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        if (!isfinite(x[i])) {
            return _gd_error(GD_ERR_INVALID_STATE, "assert_finite: non-finite value");
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_assert_close(const gd_tensor_desc *a_desc,
                                 const float *a,
                                 const float *b,
                                 float atol,
                                 float rtol)
{
    int64_t total = desc_numel(a_desc);
    int64_t i = 0;

    for (i = 0; i < total; ++i) {
        double diff = fabs((double)a[i] - (double)b[i]);
        double tol = (double)atol + (double)rtol * fabs((double)b[i]);
        if (diff > tol) {
            return _gd_error(GD_ERR_INVALID_STATE, "assert_close: tensors differ");
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_reduce_to(const gd_tensor_desc *target_desc,
                              void *out,
                              const gd_tensor_desc *go_desc,
                              const void *go)
{
    int64_t target_total = desc_numel(target_desc);
    int64_t go_total = desc_numel(go_desc);
    int64_t i = 0;
    int64_t index[GD_MAX_DIMS];
    float *acc = NULL;

    if (target_total <= 0) {
        return GD_OK;
    }
    acc = (float *)calloc((size_t)target_total, sizeof(float));
    if (acc == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "cpu reduce_to allocation failed");
    }
    for (i = 0; i < go_total; ++i) {
        int64_t off = 0;
        float v = 0.0F;
        gd_status status = _gd_cpu_load_float(go_desc, go, i, &v);
        if (status != GD_OK) {
            free(acc);
            return status;
        }
        unravel(i, go_desc, index);
        off = broadcast_offset(index, go_desc->ndim, target_desc);
        acc[off] += v;
    }
    for (i = 0; i < target_total; ++i) {
        gd_status status = _gd_cpu_store_float(target_desc, out, i, acc[i]);
        if (status != GD_OK) {
            free(acc);
            return status;
        }
    }
    free(acc);
    return GD_OK;
}

gd_status _gd_cpu_k_step_inc(float *step)
{
    step[0] += 1.0F;
    return GD_OK;
}

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
        total = desc_numel(grad_descs[g]);
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
        int64_t total = desc_numel(grad_descs[g]);
        int64_t i;
        for (i = 0; i < total; ++i) {
            grads[g][i] = (float)((double)grads[g][i] * scale);
        }
    }
    norm_out[0] = (float)norm;
    return GD_OK;
}

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
    int64_t total = desc_numel(param_desc);
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
