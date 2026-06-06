#ifndef GD_OP_SPLIT_IMPL_H
#define GD_OP_SPLIT_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

#define GD_SPLIT_MAX_OUTPUTS 256U

typedef struct gd_split_attrs {
    int32_t axis;
    uint32_t n_outputs;
    uint32_t reserved0;
    uint32_t reserved1;
} gd_split_attrs;

gd_status gd_split_forward_output_impl(gd_context *ctx,
                                       const gd_tensor *x,
                                       uint32_t axis,
                                       int64_t axis_offset,
                                       gd_tensor *out);

gd_status gd_split_backward_output_to_full_impl(gd_context *ctx,
                                                const gd_tensor *grad_output,
                                                const gd_tensor *grad_x,
                                                uint32_t axis,
                                                int64_t axis_offset,
                                                int64_t full_axis);

#endif /* GD_OP_SPLIT_IMPL_H */
