#ifndef GRADIENTS_AUTOGRAD_H
#define GRADIENTS_AUTOGRAD_H

#include <stdbool.h>
#include <stdint.h>

#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reverse-mode vector-Jacobian product. If grad_output is NULL, a tensor of
   ones with output's shape/dtype is used. Gradients are scope-local scratch
   tensors and remain valid until the next gd_begin() reuses their arena slot. */
gd_status gd_backward(gd_context *ctx,
                      const gd_tensor *output,
                      const gd_tensor *grad_output);

gd_status gd_backward_many(gd_context *ctx,
                           uint32_t n_outputs,
                           const gd_tensor *const *outputs,
                           const gd_tensor *const *grad_outputs);

/* Returns the accumulated gradient descriptor for tensor. The returned tensor
   aliases internal scratch storage; copy/read it before the next gd_begin(). */
gd_status gd_tensor_grad(gd_context *ctx,
                         const gd_tensor *tensor,
                         gd_tensor *out_grad);

/* Drops accumulated gradient descriptors from the current tape. */
gd_status gd_zero_grad(gd_context *ctx);

/* Enable/disable recording for subsequent differentiable ops. Recording is
   automatically enabled for GD_SCOPE_TRAIN and disabled during backward. */
gd_status gd_set_grad_enabled(gd_context *ctx, bool enabled);
bool gd_is_grad_enabled(const gd_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_AUTOGRAD_H */
