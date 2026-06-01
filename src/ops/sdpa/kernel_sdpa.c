#include "../../backends/cpu_ref/cpu_backend.h"

#include <math.h>
#include <stdint.h>

#include "../../core/internal.h"

#define GD_SDPA_MAX_HEAD_DIM 256

static int sdpa_allowed(int64_t i, int64_t j, int64_t Tq, int64_t Tk,
                        int causal, int window, int prefix_len)
{
    int64_t qpos = i + (Tk - Tq);

    if (causal) {
        if (prefix_len > 0) {
            if (qpos < prefix_len) {
                if (j >= prefix_len) {
                    return 0;
                }
            } else if (j > qpos) {
                return 0;
            }
        } else if (j > qpos) {
            return 0;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (qpos >= prefix_len && j >= prefix_len && (qpos - j) >= window) {
                return 0;
            }
        } else if ((qpos - j) >= window) {
            return 0;
        }
    }
    return 1;
}

static double sdpa_dot(const float *a, const float *b, int64_t n)
{
    double acc = 0.0;
    int64_t c = 0;
    for (c = 0; c < n; ++c) {
        acc += (double)a[c] * (double)b[c];
    }
    return acc;
}

/* Additive attention bias, broadcast over [B, Hq, Tq, Tk]. */
static double sdpa_bias_at(const gd_tensor_desc *bd, const float *bias,
                           int64_t b, int64_t hq, int64_t i, int64_t j)
{
    int64_t Bb = 0, Hb = 0, Tqb = 0, Tkb = 0, bb = 0, hb = 0, ib = 0, jb = 0;
    if (bias == NULL) {
        return 0.0;
    }
    Bb = bd->sizes[0];
    Hb = bd->sizes[1];
    Tqb = bd->sizes[2];
    Tkb = bd->sizes[3];
    bb = (Bb == 1) ? 0 : b;
    hb = (Hb == 1) ? 0 : hq;
    ib = (Tqb == 1) ? 0 : i;
    jb = (Tkb == 1) ? 0 : j;
    return (double)bias[((bb * Hb + hb) * Tqb + ib) * Tkb + jb];
}

static gd_status sdpa_dot_typed(const gd_tensor_desc *a_desc, const void *a, int64_t abase,
                                const gd_tensor_desc *b_desc, const void *b, int64_t bbase,
                                int64_t n, double *out)
{
    double acc = 0.0;
    int64_t c = 0;

    for (c = 0; c < n; ++c) {
        float av = 0.0F;
        float bv = 0.0F;
        gd_status status = _gd_cpu_load_float(a_desc, a, abase + c, &av);
        if (status != GD_OK) {
            return status;
        }
        status = _gd_cpu_load_float(b_desc, b, bbase + c, &bv);
        if (status != GD_OK) {
            return status;
        }
        acc += (double)av * (double)bv;
    }
    *out = acc;
    return GD_OK;
}

static gd_status sdpa_bias_at_typed(const gd_tensor_desc *bd, const void *bias,
                                    int64_t b, int64_t hq, int64_t i, int64_t j,
                                    double *out)
{
    int64_t Bb = 0, Hb = 0, Tqb = 0, Tkb = 0, bb = 0, hb = 0, ib = 0, jb = 0;
    float bv = 0.0F;

    if (bias == NULL) {
        *out = 0.0;
        return GD_OK;
    }
    Bb = bd->sizes[0];
    Hb = bd->sizes[1];
    Tqb = bd->sizes[2];
    Tkb = bd->sizes[3];
    bb = (Bb == 1) ? 0 : b;
    hb = (Hb == 1) ? 0 : hq;
    ib = (Tqb == 1) ? 0 : i;
    jb = (Tkb == 1) ? 0 : j;
    {
        gd_status status = _gd_cpu_load_float(bd, bias, ((bb * Hb + hb) * Tqb + ib) * Tkb + jb,
                                              &bv);
        if (status != GD_OK) {
            return status;
        }
    }
    *out = (double)bv;
    return GD_OK;
}

gd_status _gd_cpu_k_sdpa(const gd_tensor_desc *o_desc, void *o,
                         const gd_tensor_desc *q_desc, const void *q,
                         const gd_tensor_desc *k_desc, const void *k,
                         const gd_tensor_desc *v_desc, const void *v,
                         const gd_tensor_desc *bias_desc, const void *bias,
                         float scale, int causal, int window, int prefix_len)
{
    int64_t B = q_desc->sizes[0];
    int64_t Tq = q_desc->sizes[1];
    int64_t Hq = q_desc->sizes[2];
    int64_t Dh = q_desc->sizes[3];
    int64_t Tk = k_desc->sizes[1];
    int64_t Hkv = k_desc->sizes[2];
    int64_t group = Hkv > 0 ? Hq / Hkv : 0;
    int64_t b = 0, hq = 0, i = 0, j = 0, c = 0;

    (void)v_desc;
    if (Dh > GD_SDPA_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa head_dim exceeds reference limit");
    }
    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                int64_t qbase = ((b * Tq + i) * Hq + hq) * Dh;
                double acc[GD_SDPA_MAX_HEAD_DIM];
                double m = -HUGE_VAL;
                double sum = 0.0;

                for (c = 0; c < Dh; ++c) {
                    acc[c] = 0.0;
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        int64_t kbase = ((b * Tk + j) * Hkv + hkv) * Dh;
                        double dot = 0.0;
                        double bv = 0.0;
                        double s = 0.0;
                        gd_status status = sdpa_dot_typed(q_desc, q, qbase, k_desc, k, kbase,
                                                          Dh, &dot);
                        if (status != GD_OK) {
                            return status;
                        }
                        status = sdpa_bias_at_typed(bias_desc, bias, b, hq, i, j, &bv);
                        if (status != GD_OK) {
                            return status;
                        }
                        s = (double)scale * dot + bv;
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        int64_t kbase = ((b * Tk + j) * Hkv + hkv) * Dh;
                        double dot = 0.0;
                        double bv = 0.0;
                        double e = 0.0;
                        gd_status status = sdpa_dot_typed(q_desc, q, qbase, k_desc, k, kbase,
                                                          Dh, &dot);
                        if (status != GD_OK) {
                            return status;
                        }
                        status = sdpa_bias_at_typed(bias_desc, bias, b, hq, i, j, &bv);
                        if (status != GD_OK) {
                            return status;
                        }
                        e = exp((double)scale * dot + bv - m);
                        sum += e;
                        for (c = 0; c < Dh; ++c) {
                            float vv = 0.0F;
                            status = _gd_cpu_load_float(v_desc, v, kbase + c, &vv);
                            if (status != GD_OK) {
                                return status;
                            }
                            acc[c] += e * (double)vv;
                        }
                    }
                }
                for (c = 0; c < Dh; ++c) {
                    gd_status status = _gd_cpu_store_float(o_desc, o, qbase + c,
                                                           sum > 0.0 ? (float)(acc[c] / sum) : 0.0F);
                    if (status != GD_OK) {
                        return status;
                    }
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_sdpa_bwd(const gd_tensor_desc *q_desc, const float *q,
                             const gd_tensor_desc *k_desc, const float *k,
                             const gd_tensor_desc *v_desc, const float *v,
                             const gd_tensor_desc *bias_desc, const float *bias,
                             const float *go,
                             float *dq, float *dk, float *dv,
                             float scale, int causal, int window, int prefix_len)
{
    int64_t B = q_desc->sizes[0];
    int64_t Tq = q_desc->sizes[1];
    int64_t Hq = q_desc->sizes[2];
    int64_t Dh = q_desc->sizes[3];
    int64_t Tk = k_desc->sizes[1];
    int64_t Hkv = k_desc->sizes[2];
    int64_t group = Hkv > 0 ? Hq / Hkv : 0;
    int64_t b = 0, hq = 0, i = 0, j = 0, c = 0;

    (void)v_desc;
    if (Dh > GD_SDPA_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa head_dim exceeds reference limit");
    }
    for (j = 0; j < B * Tk * Hkv * Dh; ++j) {
        dk[j] = 0.0F;
        dv[j] = 0.0F;
    }
    for (j = 0; j < B * Tq * Hq * Dh; ++j) {
        dq[j] = 0.0F;
    }

    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                const float *qr = q + (((b * Tq + i) * Hq + hq) * Dh);
                const float *gor = go + (((b * Tq + i) * Hq + hq) * Dh);
                float *dqr = dq + (((b * Tq + i) * Hq + hq) * Dh);
                double m = -HUGE_VAL;
                double sum = 0.0;
                double dsum = 0.0; /* D = sum_j p_j * dp_j */

                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        sum += exp(s - m);
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double p = exp(s - m) / sum;
                        dsum += p * sdpa_dot(gor, vr, Dh);
                    }
                }
                for (j = 0; j < Tk; ++j) {
                    if (sdpa_allowed(i, j, Tq, Tk, causal, window, prefix_len)) {
                        const float *kr = k + (((b * Tk + j) * Hkv + hkv) * Dh);
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        float *dkr = dk + (((b * Tk + j) * Hkv + hkv) * Dh);
                        float *dvr = dv + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double p = exp(s - m) / sum;
                        double dp = sdpa_dot(gor, vr, Dh);
                        double ds = p * (dp - dsum);
                        for (c = 0; c < Dh; ++c) {
                            dvr[c] += (float)(p * (double)gor[c]);
                            dqr[c] += (float)((double)scale * ds * (double)kr[c]);
                            dkr[c] += (float)((double)scale * ds * (double)qr[c]);
                        }
                    }
                }
            }
        }
    }
    return GD_OK;
}
