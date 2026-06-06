#ifndef GD_OP_DROPOUT_IMPL_H
#define GD_OP_DROPOUT_IMPL_H

#include <gradients/status.h>
#include <gradients/tensor.h>

typedef struct gd_dropout_attrs {
    float p;
    float scale;
} gd_dropout_attrs;

gd_status gd_dropout_backward_from_mask(gd_context *ctx,
                                        const gd_tensor *mask,
                                        const gd_tensor *grad_out,
                                        float scale,
                                        gd_tensor *grad_x);

#endif /* GD_OP_DROPOUT_IMPL_H */
