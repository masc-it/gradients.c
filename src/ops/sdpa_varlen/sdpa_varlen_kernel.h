#ifndef GD_SDPA_VARLEN_KERNEL_H
#define GD_SDPA_VARLEN_KERNEL_H

#include "../../backends/cpu_ref/cpu_backend.h"

static inline bool gd_sdpa_varlen_allowed(int i, int j, int causal,
                                          int window, int prefix_len)
{
    if (causal) {
        if (prefix_len > 0) {
            if (i < prefix_len) {
                if (j >= prefix_len) {
                    return false;
                }
            } else if (j > i) {
                return false;
            }
        } else if (j > i) {
            return false;
        }
    }
    if (window > 0) {
        if (prefix_len > 0) {
            if (i >= prefix_len && j >= prefix_len && (i - j) >= window) {
                return false;
            }
        } else if ((i - j) >= window) {
            return false;
        }
    }
    return true;
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
                                const void *cu,
                                float scale,
                                int causal,
                                int window,
                                int prefix_len,
                                int max_seqlen);

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
                                    const void *cu,
                                    float scale,
                                    int causal,
                                    int window,
                                    int prefix_len,
                                    int max_seqlen);

#endif /* GD_SDPA_VARLEN_KERNEL_H */
