#ifndef GD_OP_QKV_SPLIT_ROPE_IMPL_H
#define GD_OP_QKV_SPLIT_ROPE_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

typedef struct gd_qkv_split_rope_attrs {
    gd_rope_config rope;
    int32_t n_heads;
    int32_t head_dim;
} gd_qkv_split_rope_attrs;

gd_status gd_qkv_split_rope_backward(gd_context *ctx,
                                      const gd_tensor *qkv,
                                      const gd_tensor *pos_ids,
                                      const gd_tensor *grad_q,
                                      const gd_tensor *grad_k,
                                      const gd_tensor *grad_v,
                                      int32_t n_heads,
                                      int32_t head_dim,
                                      const gd_rope_config *config,
                                      gd_tensor *grad_qkv);

#endif /* GD_OP_QKV_SPLIT_ROPE_IMPL_H */
