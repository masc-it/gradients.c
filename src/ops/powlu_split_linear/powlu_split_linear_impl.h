#ifndef GD_OP_POWLU_SPLIT_LINEAR_IMPL_H
#define GD_OP_POWLU_SPLIT_LINEAR_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

typedef struct gd_powlu_split_linear_attrs {
    float m;
} gd_powlu_split_linear_attrs;

#ifndef __METAL_VERSION__
_Static_assert(sizeof(gd_powlu_split_linear_attrs) == 4U,
               "gd_powlu_split_linear_attrs ABI mismatch");
#endif

gd_status gd_powlu_split_linear_backward_with_activation(gd_context *ctx,
                                                         const gd_tensor *x12,
                                                         const gd_tensor *activation,
                                                         const gd_tensor *w,
                                                         const gd_tensor *bias,
                                                         const gd_tensor *grad_out,
                                                         float m,
                                                         gd_tensor *grad_x12,
                                                         gd_tensor *grad_w,
                                                         gd_tensor *grad_bias);

#endif /* GD_OP_POWLU_SPLIT_LINEAR_IMPL_H */
