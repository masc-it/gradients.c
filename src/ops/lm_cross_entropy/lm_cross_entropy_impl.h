#ifndef GD_OP_LM_CROSS_ENTROPY_IMPL_H
#define GD_OP_LM_CROSS_ENTROPY_IMPL_H

#include <gradients/ops.h>

typedef struct gd_lm_cross_entropy_attrs {
    float logits_softcap;
} gd_lm_cross_entropy_attrs;

_Static_assert(sizeof(gd_lm_cross_entropy_attrs) == 4U,
               "gd_lm_cross_entropy_attrs ABI mismatch");

gd_status gd_lm_cross_entropy_backward_with_stats(gd_context *ctx,
                                                  const gd_tensor *hidden,
                                                  const gd_tensor *weight,
                                                  const gd_tensor *bias,
                                                  const gd_tensor *targets,
                                                  const gd_tensor *row_max,
                                                  const gd_tensor *row_inv_sum,
                                                  const gd_tensor *inv_valid_count,
                                                  const gd_tensor *saved_logits,
                                                  float logits_softcap,
                                                  const gd_tensor *grad_out,
                                                  gd_tensor *grad_hidden,
                                                  gd_tensor *grad_weight,
                                                  gd_tensor *grad_bias);

#endif /* GD_OP_LM_CROSS_ENTROPY_IMPL_H */
