#ifndef GD_OPS_RESHAPE_IMPL_H
#define GD_OPS_RESHAPE_IMPL_H

#include <stdbool.h>

#include <gradients/ops.h>

/* Internal view helper. record_autograd=false is used by direct/autograd backward
 * helpers so backward helpers never create second-order tape nodes. */
gd_status gd_reshape_view_impl(gd_context *ctx,
                               const gd_tensor *x,
                               gd_shape shape,
                               bool record_autograd,
                               gd_tensor *out);

#endif /* GD_OPS_RESHAPE_IMPL_H */
