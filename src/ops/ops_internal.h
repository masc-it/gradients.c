#ifndef GRADIENTS_OPS_INTERNAL_H
#define GRADIENTS_OPS_INTERNAL_H

#include <stdbool.h>

#include "gradients/dtype.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

bool _gd_dtype_is_float(gd_dtype dtype);
bool _gd_dtype_is_integer(gd_dtype dtype);

/* Shape/dtype inference helpers. Each fills `out` with a contiguous output
 * descriptor and validates v1 dtype/device rules. */
gd_status _gd_infer_elementwise(gd_tensor *a, gd_tensor *b, gd_tensor_desc *out);
gd_status _gd_infer_unary_float(gd_tensor *x, gd_tensor_desc *out);
gd_status _gd_infer_matmul(gd_tensor *a,
                           gd_tensor *b,
                           bool trans_a,
                           bool trans_b,
                           gd_tensor_desc *out);
gd_status _gd_infer_linear(gd_tensor *x,
                           gd_tensor *w,
                           gd_tensor *bias,
                           bool trans_w,
                           gd_tensor_desc *out);
gd_status _gd_infer_reduce(gd_tensor *x,
                           int dim,
                           bool keepdim,
                           int *norm_dim_out,
                           gd_tensor_desc *out);
gd_status _gd_infer_softmax(gd_tensor *x, int dim, int *norm_dim_out, gd_tensor_desc *out);
gd_status _gd_infer_cross_entropy(gd_tensor *logits,
                                  gd_tensor *targets,
                                  int class_dim,
                                  int *norm_dim_out,
                                  gd_tensor_desc *out);
gd_status _gd_infer_cast(gd_tensor *x, gd_dtype dtype, gd_tensor_desc *out);
gd_status _gd_infer_transpose(gd_tensor *x, const int *perm, int ndim, gd_tensor_desc *out);
gd_status _gd_infer_embedding(gd_tensor *table, gd_tensor *ids, gd_tensor_desc *out);
gd_status _gd_infer_rope(gd_tensor *x, gd_tensor *pos_ids, gd_tensor_desc *out);

#endif /* GRADIENTS_OPS_INTERNAL_H */
