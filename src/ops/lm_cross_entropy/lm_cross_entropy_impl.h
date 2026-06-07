#ifndef GD_OP_LM_CROSS_ENTROPY_IMPL_H
#define GD_OP_LM_CROSS_ENTROPY_IMPL_H

#include <gradients/ops.h>

gd_status gd_lm_cross_entropy_backward_with_stats(gd_context *ctx,
                                                  const gd_tensor *hidden,
                                                  const gd_tensor *weight,
                                                  const gd_tensor *targets,
                                                  const gd_tensor *logits,
                                                  const gd_tensor *row_max,
                                                  const gd_tensor *row_inv_sum,
                                                  const gd_tensor *grad_out,
                                                  gd_tensor *grad_hidden,
                                                  gd_tensor *grad_weight);

#endif /* GD_OP_LM_CROSS_ENTROPY_IMPL_H */
