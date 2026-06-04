#ifndef GRADIENTS_OPS_H
#define GRADIENTS_OPS_H

#include <gradients/status.h>
#include <gradients/tensor.h>
#include <gradients/ops_generated.h>

#ifdef __cplusplus
extern "C" {
#endif

gd_status gd_matmul(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    gd_tensor *out);

/* Computes out = x @ w + bias. Pass bias=NULL for out = x @ w. */
gd_status gd_linear(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    const gd_tensor *bias,
                    gd_tensor *out);

gd_status gd_matmul_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w);

gd_status gd_linear_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *bias,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w,
                             gd_tensor *grad_bias);

/* Reduces a single axis. Negative axes are accepted Python-style. */
gd_status gd_reduce_sum_axis(gd_context *ctx,
                             const gd_tensor *x,
                             int32_t axis,
                             bool keepdims,
                             gd_tensor *out);
gd_status gd_reduce_sum_axis_backward(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *grad_out,
                                      int32_t axis,
                                      bool keepdims,
                                      gd_tensor *grad_x);
gd_status gd_reduce_mean_axis(gd_context *ctx,
                              const gd_tensor *x,
                              int32_t axis,
                              bool keepdims,
                              gd_tensor *out);
gd_status gd_reduce_mean_axis_backward(gd_context *ctx,
                                       const gd_tensor *x,
                                       const gd_tensor *grad_out,
                                       int32_t axis,
                                       bool keepdims,
                                       gd_tensor *grad_x);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
