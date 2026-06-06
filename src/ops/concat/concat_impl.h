#ifndef GD_OP_CONCAT_IMPL_H
#define GD_OP_CONCAT_IMPL_H

#include <gradients/ops.h>

#include <stdint.h>

#define GD_CONCAT_MAX_INPUTS 256U

typedef struct gd_concat_attrs {
    int32_t axis;
    uint32_t n_inputs;
    uint32_t reserved0;
    uint32_t reserved1;
} gd_concat_attrs;

gd_status gd_concat_backward_input_impl(gd_context *ctx,
                                        const gd_tensor *grad_out,
                                        const gd_tensor *like_input,
                                        uint32_t axis,
                                        int64_t axis_offset,
                                        int64_t full_axis,
                                        gd_tensor *grad_input);

#endif /* GD_OP_CONCAT_IMPL_H */
