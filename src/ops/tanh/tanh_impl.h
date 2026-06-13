#ifndef GD_OP_TANH_IMPL_H
#define GD_OP_TANH_IMPL_H

#include <gradients/status.h>
#include <gradients/tensor.h>

/* Internal autograd fast path: consumes the forward tanh output instead of
 * recomputing tanh(x). Public gd_tanh_backward keeps the generated signature
 * and recomputes from x for direct callers.
 */
gd_status gd_tanh_backward_from_output(gd_context *ctx,
                                       const gd_tensor *tanh_out,
                                       const gd_tensor *grad_out,
                                       gd_tensor *grad_x);

#endif /* GD_OP_TANH_IMPL_H */
