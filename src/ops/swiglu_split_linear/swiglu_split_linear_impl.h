#ifndef GD_OP_SWIGLU_SPLIT_LINEAR_IMPL_H
#define GD_OP_SWIGLU_SPLIT_LINEAR_IMPL_H

#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

gd_status gd_swiglu_split_linear_backward_with_activation(gd_context *ctx,
                                                          const gd_tensor *x12,
                                                          const gd_tensor *activation,
                                                          const gd_tensor *w,
                                                          const gd_tensor *bias,
                                                          const gd_tensor *grad_out,
                                                          gd_tensor *grad_x12,
                                                          gd_tensor *grad_w,
                                                          gd_tensor *grad_bias);

#ifdef __cplusplus
}
#endif

#endif /* GD_OP_SWIGLU_SPLIT_LINEAR_IMPL_H */
