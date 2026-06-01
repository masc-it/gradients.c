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

gd_status _gd_cpu_k_sdpa(const gd_tensor_desc *o_desc, float *o,
                         const gd_tensor_desc *q_desc, const float *q,
                         const gd_tensor_desc *k_desc, const float *k,
                         const gd_tensor_desc *v_desc, const float *v,
                         const gd_tensor_desc *bias_desc, const float *bias,
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

    (void)o_desc;
    (void)v_desc;
    if (Dh > GD_SDPA_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa head_dim exceeds reference limit");
    }
    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                const float *qr = q + (((b * Tq + i) * Hq + hq) * Dh);
                float *orow = o + (((b * Tq + i) * Hq + hq) * Dh);
                double acc[GD_SDPA_MAX_HEAD_DIM];
                double m = -HUGE_VAL;
                double sum = 0.0;

                for (c = 0; c < Dh; ++c) {
                    acc[c] = 0.0;
                }
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
                        const float *vr = v + (((b * Tk + j) * Hkv + hkv) * Dh);
                        double s = (double)scale * sdpa_dot(qr, kr, Dh)
                                   + sdpa_bias_at(bias_desc, bias, b, hq, i, j);
                        double e = exp(s - m);
                        sum += e;
                        for (c = 0; c < Dh; ++c) {
                            acc[c] += e * (double)vr[c];
                        }
                    }
                }
                for (c = 0; c < Dh; ++c) {
                    orow[c] = sum > 0.0 ? (float)(acc[c] / sum) : 0.0F;
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
