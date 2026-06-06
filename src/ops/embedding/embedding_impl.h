#ifndef GD_OP_EMBEDDING_IMPL_H
#define GD_OP_EMBEDDING_IMPL_H

#include <gradients/ops.h>

/* Internal helper used by the autograd rule. */
gd_status gd_embedding_backward_impl(gd_context *ctx,
                                     const gd_tensor *table,
                                     const gd_tensor *ids,
                                     const gd_tensor *grad_out,
                                     gd_tensor *grad_table);

#endif /* GD_OP_EMBEDDING_IMPL_H */
