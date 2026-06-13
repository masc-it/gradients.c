#ifndef GD_OP_RMS_NORM_IMPL_H
#define GD_OP_RMS_NORM_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

typedef struct gd_rms_norm_attrs {
    float eps;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} gd_rms_norm_attrs;

gd_status gd_rms_norm_backward_with_stats(gd_context *ctx,
                                          const gd_tensor *x,
                                          const gd_tensor *weight,
                                          const gd_tensor *inv_rms,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x,
                                          gd_tensor *grad_weight);

#endif /* GD_OP_RMS_NORM_IMPL_H */
