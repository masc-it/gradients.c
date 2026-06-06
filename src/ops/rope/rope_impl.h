#ifndef GD_OP_ROPE_IMPL_H
#define GD_OP_ROPE_IMPL_H

#include <gradients/ops.h>
#include <gradients/status.h>
#include <gradients/tensor.h>

#include <stdint.h>

typedef struct gd_rope_attrs {
    float theta;
    int32_t n_dims;
    int32_t interleaved;
} gd_rope_attrs;

gd_status gd_rope_backward_from_attrs(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *pos_ids,
                                      const gd_tensor *grad_out,
                                      const gd_rope_attrs *attrs,
                                      gd_tensor *grad_x);

#endif /* GD_OP_ROPE_IMPL_H */
