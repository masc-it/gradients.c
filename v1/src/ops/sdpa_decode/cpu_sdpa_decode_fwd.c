#include "../../backends/cpu_ref/cpu_op.h"

#include <math.h>
#include <stdint.h>

#define GD_SDPA_DECODE_MAX_HEAD_DIM 256

static int64_t sdpa_decode_pos_value(const gd_tensor_desc *pos_desc, const void *pos)
{
    return pos_desc->dtype == GD_DTYPE_I64 ? ((const int64_t *)pos)[0]
                                          : (int64_t)((const int32_t *)pos)[0];
}

static int sdpa_decode_allowed(int64_t qpos, int64_t j, int window, int prefix_len)
{
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

static gd_status sdpa_decode_run(_gd_cpu_exec *exec, const _gd_node *node)
{
    gd_status status = GD_OK;
    void *q_data = NULL;
    void *k_data = NULL;
    void *v_data = NULL;
    void *pos_data = NULL;
    void *out_data = NULL;
    const gd_tensor_desc *q_desc = NULL;
    const gd_tensor_desc *k_desc = NULL;
    const gd_tensor_desc *v_desc = NULL;
    const gd_tensor_desc *pos_desc = NULL;
    const gd_tensor_desc *out_desc = NULL;
    int64_t B = 0;
    int64_t Tq = 0;
    int64_t Hq = 0;
    int64_t Dh = 0;
    int64_t Tmax = 0;
    int64_t Hkv = 0;
    int64_t group = 0;
    int64_t cache_pos = 0;
    int64_t live_len = 0;
    int64_t b = 0;
    int64_t hq = 0;
    int64_t i = 0;
    int64_t j = 0;
    int64_t c = 0;

    status = _gd_cpu_exec_input(exec, node, 0, &q_data, &q_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 1, &k_data, &k_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 2, &v_data, &v_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_input(exec, node, 3, &pos_data, &pos_desc);
    if (status != GD_OK) { return status; }
    status = _gd_cpu_exec_output(exec, node, 0, &out_data, &out_desc);
    if (status != GD_OK) { return status; }

    B = q_desc->sizes[0];
    Tq = q_desc->sizes[1];
    Hq = q_desc->sizes[2];
    Dh = q_desc->sizes[3];
    Tmax = k_desc->sizes[1];
    Hkv = k_desc->sizes[2];
    group = Hkv > 0 ? Hq / Hkv : 0;
    cache_pos = sdpa_decode_pos_value(pos_desc, pos_data);
    live_len = cache_pos + Tq;
    if (Dh > GD_SDPA_DECODE_MAX_HEAD_DIM) {
        return _gd_error(GD_ERR_UNSUPPORTED, "sdpa_decode head_dim exceeds reference limit");
    }
    if (cache_pos < 0 || live_len < 0 || live_len > Tmax) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa_decode live length exceeds cache length");
    }

    for (b = 0; b < B; ++b) {
        for (hq = 0; hq < Hq; ++hq) {
            int64_t hkv = hq / group;
            for (i = 0; i < Tq; ++i) {
                int64_t qpos = cache_pos + i;
                int64_t qbase = ((b * Tq + i) * Hq + hq) * Dh;
                double acc[GD_SDPA_DECODE_MAX_HEAD_DIM];
                double m = -HUGE_VAL;
                double sum = 0.0;
                for (c = 0; c < Dh; ++c) {
                    acc[c] = 0.0;
                }
                for (j = 0; j < live_len; ++j) {
                    if (sdpa_decode_allowed(qpos, j, node->attrs.sliding_window,
                                            node->attrs.prefix_len)) {
                        int64_t kbase = ((b * Tmax + j) * Hkv + hkv) * Dh;
                        double dot = 0.0;
                        double s = 0.0;
                        for (c = 0; c < Dh; ++c) {
                            float qv = 0.0F;
                            float kv = 0.0F;
                            status = _gd_cpu_load_float(q_desc, q_data, qbase + c, &qv);
                            if (status != GD_OK) { return status; }
                            status = _gd_cpu_load_float(k_desc, k_data, kbase + c, &kv);
                            if (status != GD_OK) { return status; }
                            dot += (double)qv * (double)kv;
                        }
                        s = (double)node->attrs.attn_scale * dot;
                        if (s > m) { m = s; }
                    }
                }
                for (j = 0; j < live_len; ++j) {
                    if (sdpa_decode_allowed(qpos, j, node->attrs.sliding_window,
                                            node->attrs.prefix_len)) {
                        int64_t kbase = ((b * Tmax + j) * Hkv + hkv) * Dh;
                        double dot = 0.0;
                        double e = 0.0;
                        for (c = 0; c < Dh; ++c) {
                            float qv = 0.0F;
                            float kv = 0.0F;
                            status = _gd_cpu_load_float(q_desc, q_data, qbase + c, &qv);
                            if (status != GD_OK) { return status; }
                            status = _gd_cpu_load_float(k_desc, k_data, kbase + c, &kv);
                            if (status != GD_OK) { return status; }
                            dot += (double)qv * (double)kv;
                        }
                        e = exp((double)node->attrs.attn_scale * dot - m);
                        sum += e;
                        for (c = 0; c < Dh; ++c) {
                            float vv = 0.0F;
                            status = _gd_cpu_load_float(v_desc, v_data, kbase + c, &vv);
                            if (status != GD_OK) { return status; }
                            acc[c] += e * (double)vv;
                        }
                    }
                }
                for (c = 0; c < Dh; ++c) {
                    status = _gd_cpu_store_float(out_desc, out_data, qbase + c,
                                                 sum > 0.0 ? (float)(acc[c] / sum) : 0.0F);
                    if (status != GD_OK) { return status; }
                }
            }
        }
    }
    return GD_OK;
}

const _gd_cpu_op _gd_cpu_op_sdpa_decode = {
    .kind = _GD_OP_SDPA_DECODE,
    .name = "sdpa_decode",
    .support = _gd_cpu_support_default,
    .run = sdpa_decode_run,
};
