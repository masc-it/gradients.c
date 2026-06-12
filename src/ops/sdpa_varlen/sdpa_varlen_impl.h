#ifndef GD_OP_SDPA_VARLEN_IMPL_H
#define GD_OP_SDPA_VARLEN_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

#define GD_SDPA_VARLEN_MAX_HEAD_DIM 64

typedef struct gd_sdpa_varlen_attrs {
    float scale;
    int32_t causal;
    int32_t sliding_window;
    int32_t prefix_len;
    int32_t max_seqlen;
} gd_sdpa_varlen_attrs;

gd_status gd_sdpa_varlen_backward_with_stats(gd_context *ctx,
                                             const gd_tensor *q,
                                             const gd_tensor *k,
                                             const gd_tensor *v,
                                             const gd_tensor *cu_seqlens,
                                             const gd_tensor *forward_stats,
                                             const gd_tensor *grad_out,
                                             const gd_sdpa_varlen_config *config,
                                             gd_tensor *grad_q,
                                             gd_tensor *grad_k,
                                             gd_tensor *grad_v);

#endif /* GD_OP_SDPA_VARLEN_IMPL_H */
