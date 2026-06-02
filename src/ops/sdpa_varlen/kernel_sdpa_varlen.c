#include "sdpa_varlen_kernel.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "../../core/internal.h"

static int64_t sdpa_varlen_numel(const gd_tensor_desc *desc)
{
    int64_t n = 1;
    for (int i = 0; i < desc->ndim; ++i) {
        n *= desc->sizes[i];
    }
    return n;
}

static gd_status validate_cu_seqlens(const gd_tensor_desc *cu_desc,
                                     const void *cu_data,
                                     int64_t total_tokens,
                                     int max_seqlen,
                                     int *batch_out)
{
    const int32_t *cu = (const int32_t *)cu_data;
    int B = 0;

    if (cu_desc == NULL || cu_data == NULL || batch_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen cu_seqlens argument is NULL");
    }
    if (cu_desc->dtype != GD_DTYPE_I32 || cu_desc->ndim != 1 || cu_desc->sizes[0] < 2) {
        return _gd_error(GD_ERR_SHAPE, "sdpa_varlen cu_seqlens must be I32 [B+1]");
    }
    B = (int)(cu_desc->sizes[0] - 1);
    if (cu[0] != 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen cu_seqlens[0] must be 0");
    }
    for (int b = 0; b < B; ++b) {
        int start = cu[b];
        int end = cu[b + 1];
        if (start < 0 || end < start || (int64_t)end > total_tokens) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen cu_seqlens must be monotonic and within N");
        }
        if (max_seqlen > 0 && end - start > max_seqlen) {
            return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen sequence length exceeds max_seqlen");
        }
    }
    if ((int64_t)cu[B] != total_tokens) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_varlen cu_seqlens[B] must equal packed token count");
    }
    *batch_out = B;
    return GD_OK;
}

gd_status _gd_cpu_k_sdpa_varlen(const gd_tensor_desc *out_desc,
                                void *out,
                                const gd_tensor_desc *q_desc,
                                const void *q,
                                const gd_tensor_desc *k_desc,
                                const void *k,
                                const gd_tensor_desc *v_desc,
                                const void *v,
                                const gd_tensor_desc *cu_desc,
                                const void *cu_data,
                                float scale,
                                int causal,
                                int window,
                                int prefix_len,
                                int max_seqlen)
{
    const int32_t *cu = (const int32_t *)cu_data;
    int64_t N = q_desc->sizes[0];
    int Hq = (int)q_desc->sizes[1];
    int Hkv = (int)k_desc->sizes[1];
    int Dh = (int)q_desc->sizes[2];
    int group = Hq / Hkv;
    int B = 0;
    gd_status status = GD_OK;

    status = validate_cu_seqlens(cu_desc, cu_data, N, max_seqlen, &B);
    if (status != GD_OK) {
        return status;
    }
    for (int64_t idx = 0; idx < sdpa_varlen_numel(out_desc); ++idx) {
        status = _gd_cpu_store_float(out_desc, out, idx, 0.0F);
        if (status != GD_OK) { return status; }
    }
    for (int b = 0; b < B; ++b) {
        int start = cu[b];
        int end = cu[b + 1];
        int T = end - start;
        for (int hq = 0; hq < Hq; ++hq) {
            int hkv = hq / group;
            for (int i = 0; i < T; ++i) {
                double m = -HUGE_VAL;
                double l = 0.0;
                int qg = start + i;
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F;
                        float kv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { return status; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { return status; }
                        s += (double)qv * (double)kv;
                    }
                    s *= (double)scale;
                    if (s > m) {
                        m = s;
                    }
                }
                if (!isfinite(m)) {
                    continue;
                }
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F;
                        float kv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { return status; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { return status; }
                        s += (double)qv * (double)kv;
                    }
                    l += exp((double)scale * s - m);
                }
                if (l <= 0.0) {
                    continue;
                }
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    double p = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F;
                        float kv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { return status; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { return status; }
                        s += (double)qv * (double)kv;
                    }
                    p = exp((double)scale * s - m) / l;
                    for (int c = 0; c < Dh; ++c) {
                        float vv = 0.0F;
                        float old = 0.0F;
                        int64_t obase = ((int64_t)qg * Hq + hq) * Dh + c;
                        status = _gd_cpu_load_float(v_desc, v, ((int64_t)kg * Hkv + hkv) * Dh + c, &vv);
                        if (status != GD_OK) { return status; }
                        status = _gd_cpu_load_float(out_desc, out, obase, &old);
                        if (status != GD_OK) { return status; }
                        status = _gd_cpu_store_float(out_desc, out, obase, old + (float)(p * (double)vv));
                        if (status != GD_OK) { return status; }
                    }
                }
            }
        }
    }
    return GD_OK;
}

gd_status _gd_cpu_k_sdpa_varlen_bwd(const gd_tensor_desc *dq_desc,
                                    void *dq,
                                    const gd_tensor_desc *dk_desc,
                                    void *dk,
                                    const gd_tensor_desc *dv_desc,
                                    void *dv,
                                    const gd_tensor_desc *go_desc,
                                    const void *go,
                                    const gd_tensor_desc *q_desc,
                                    const void *q,
                                    const gd_tensor_desc *k_desc,
                                    const void *k,
                                    const gd_tensor_desc *v_desc,
                                    const void *v,
                                    const gd_tensor_desc *cu_desc,
                                    const void *cu_data,
                                    float scale,
                                    int causal,
                                    int window,
                                    int prefix_len,
                                    int max_seqlen)
{
    const int32_t *cu = (const int32_t *)cu_data;
    int64_t N = q_desc->sizes[0];
    int Hq = (int)q_desc->sizes[1];
    int Hkv = (int)k_desc->sizes[1];
    int Dh = (int)q_desc->sizes[2];
    int group = Hq / Hkv;
    int B = 0;
    int64_t rows = N * (int64_t)Hq;
    float *m = NULL;
    float *l = NULL;
    float *Drow = NULL;
    gd_status status = GD_OK;

    status = validate_cu_seqlens(cu_desc, cu_data, N, max_seqlen, &B);
    if (status != GD_OK) {
        return status;
    }
    for (int64_t idx = 0; idx < sdpa_varlen_numel(dq_desc); ++idx) {
        status = _gd_cpu_store_float(dq_desc, dq, idx, 0.0F);
        if (status != GD_OK) { return status; }
    }
    for (int64_t idx = 0; idx < sdpa_varlen_numel(dk_desc); ++idx) {
        status = _gd_cpu_store_float(dk_desc, dk, idx, 0.0F);
        if (status != GD_OK) { return status; }
        status = _gd_cpu_store_float(dv_desc, dv, idx, 0.0F);
        if (status != GD_OK) { return status; }
    }
    m = malloc((size_t)rows * sizeof(float));
    l = malloc((size_t)rows * sizeof(float));
    Drow = malloc((size_t)rows * sizeof(float));
    if (m == NULL || l == NULL || Drow == NULL) {
        free(m);
        free(l);
        free(Drow);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "sdpa_varlen_bwd stats allocation failed");
    }
    for (int64_t r = 0; r < rows; ++r) {
        m[r] = -FLT_MAX;
        l[r] = 0.0F;
        Drow[r] = 0.0F;
    }

    for (int b = 0; b < B; ++b) {
        int start = cu[b];
        int T = cu[b + 1] - start;
        for (int hq = 0; hq < Hq; ++hq) {
            int hkv = hq / group;
            for (int i = 0; i < T; ++i) {
                int qg = start + i;
                int64_t row = (int64_t)qg * Hq + hq;
                double mm = -HUGE_VAL;
                double ll = 0.0;
                double raw = 0.0;
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F;
                        float kv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { goto done; }
                        s += (double)qv * (double)kv;
                    }
                    s *= (double)scale;
                    if (s > mm) {
                        mm = s;
                    }
                }
                if (!isfinite(mm)) {
                    continue;
                }
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    double dp = 0.0;
                    double e = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F, kv = 0.0F, gov = 0.0F, vv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(go_desc, go, ((int64_t)qg * Hq + hq) * Dh + c, &gov);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(v_desc, v, ((int64_t)kg * Hkv + hkv) * Dh + c, &vv);
                        if (status != GD_OK) { goto done; }
                        s += (double)qv * (double)kv;
                        dp += (double)gov * (double)vv;
                    }
                    e = exp((double)scale * s - mm);
                    ll += e;
                    raw += e * dp;
                }
                m[row] = (float)mm;
                l[row] = (float)ll;
                Drow[row] = ll > 0.0 ? (float)(raw / ll) : 0.0F;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        int start = cu[b];
        int T = cu[b + 1] - start;
        for (int hq = 0; hq < Hq; ++hq) {
            int hkv = hq / group;
            for (int i = 0; i < T; ++i) {
                int qg = start + i;
                int64_t row = (int64_t)qg * Hq + hq;
                if (l[row] <= 0.0F) {
                    continue;
                }
                for (int j = 0; j < T; ++j) {
                    double s = 0.0;
                    double dp = 0.0;
                    double pj = 0.0;
                    double ds = 0.0;
                    int kg = start + j;
                    if (!gd_sdpa_varlen_allowed(i, j, causal, window, prefix_len)) {
                        continue;
                    }
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F, kv = 0.0F, gov = 0.0F, vv = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(go_desc, go, ((int64_t)qg * Hq + hq) * Dh + c, &gov);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(v_desc, v, ((int64_t)kg * Hkv + hkv) * Dh + c, &vv);
                        if (status != GD_OK) { goto done; }
                        s += (double)qv * (double)kv;
                        dp += (double)gov * (double)vv;
                    }
                    pj = exp((double)scale * s - (double)m[row]) / (double)l[row];
                    ds = pj * (dp - (double)Drow[row]);
                    for (int c = 0; c < Dh; ++c) {
                        float qv = 0.0F, kv = 0.0F, gov = 0.0F;
                        float old = 0.0F;
                        status = _gd_cpu_load_float(q_desc, q, ((int64_t)qg * Hq + hq) * Dh + c, &qv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(k_desc, k, ((int64_t)kg * Hkv + hkv) * Dh + c, &kv);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(go_desc, go, ((int64_t)qg * Hq + hq) * Dh + c, &gov);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(dq_desc, dq, ((int64_t)qg * Hq + hq) * Dh + c, &old);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_store_float(dq_desc, dq, ((int64_t)qg * Hq + hq) * Dh + c,
                                                     old + (float)((double)scale * ds * (double)kv));
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(dk_desc, dk, ((int64_t)kg * Hkv + hkv) * Dh + c, &old);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_store_float(dk_desc, dk, ((int64_t)kg * Hkv + hkv) * Dh + c,
                                                     old + (float)((double)scale * ds * (double)qv));
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_load_float(dv_desc, dv, ((int64_t)kg * Hkv + hkv) * Dh + c, &old);
                        if (status != GD_OK) { goto done; }
                        status = _gd_cpu_store_float(dv_desc, dv, ((int64_t)kg * Hkv + hkv) * Dh + c,
                                                     old + (float)(pj * (double)gov));
                        if (status != GD_OK) { goto done; }
                    }
                }
            }
        }
    }

done:
    free(m);
    free(l);
    free(Drow);
    return status;
}
