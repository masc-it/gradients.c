#ifndef GD_OP_CROSS_ENTROPY_IMPL_H
#define GD_OP_CROSS_ENTROPY_IMPL_H

#include <gradients/ops.h>

/* Internal autograd fast path: uses forward-saved row statistics. */
gd_status gd_cross_entropy_backward_with_stats(gd_context *ctx,
                                               const gd_tensor *logits,
                                               const gd_tensor *targets,
                                               const gd_tensor *row_max,
                                               const gd_tensor *row_inv_sum,
                                               const gd_tensor *grad_out,
                                               gd_tensor *grad_logits);

#endif /* GD_OP_CROSS_ENTROPY_IMPL_H */
