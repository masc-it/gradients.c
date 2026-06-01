#ifndef GRADIENTS_META_COMMON_H
#define GRADIENTS_META_COMMON_H

#include <stdbool.h>
#include <stdint.h>

#include "gradients/dtype.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

bool _gd_dtype_is_float(gd_dtype dtype);
bool _gd_dtype_is_integer(gd_dtype dtype);

gd_status _gd_meta_normalize_dim(int dim, int ndim, int *out);
gd_status _gd_meta_broadcast_shapes(const int64_t *a,
                                    int a_ndim,
                                    const int64_t *b,
                                    int b_ndim,
                                    int64_t *out,
                                    int *out_ndim);
gd_status _gd_meta_require_same_device(const gd_tensor_desc *a,
                                       const gd_tensor_desc *b,
                                       const char *message);
gd_status _gd_meta_set_output_count(int expected, int *n_outputs);

gd_status _gd_meta_elementwise(const gd_tensor_desc *a,
                               const gd_tensor_desc *b,
                               gd_tensor_desc *out);
gd_status _gd_meta_unary_float(const gd_tensor_desc *x, gd_tensor_desc *out);
gd_status _gd_meta_powlu(const gd_tensor_desc *x1,
                         const gd_tensor_desc *x2,
                         gd_tensor_desc *out);
gd_status _gd_meta_matmul(const gd_tensor_desc *a,
                          const gd_tensor_desc *b,
                          bool trans_a,
                          bool trans_b,
                          gd_tensor_desc *out);
gd_status _gd_meta_linear(const gd_tensor_desc *x,
                          const gd_tensor_desc *w,
                          const gd_tensor_desc *bias,
                          bool trans_w,
                          gd_tensor_desc *out);
gd_status _gd_meta_reduce(const gd_tensor_desc *x,
                          int dim,
                          bool keepdim,
                          int *norm_dim_out,
                          gd_tensor_desc *out);
gd_status _gd_meta_reduce_to(const gd_tensor_desc *x,
                             const gd_tensor_desc *target,
                             gd_tensor_desc *out);
gd_status _gd_meta_softmax(const gd_tensor_desc *x,
                           int dim,
                           int *norm_dim_out,
                           gd_tensor_desc *out);
gd_status _gd_meta_cross_entropy(const gd_tensor_desc *logits,
                                 const gd_tensor_desc *targets,
                                 int class_dim,
                                 int *norm_dim_out,
                                 gd_tensor_desc *out);
gd_status _gd_meta_lm_cross_entropy(const gd_tensor_desc *hidden,
                                    const gd_tensor_desc *weight,
                                    const gd_tensor_desc *targets,
                                    gd_tensor_desc *out);
gd_status _gd_meta_cast(const gd_tensor_desc *x, gd_dtype dtype, gd_tensor_desc *out);
gd_status _gd_meta_transpose(const gd_tensor_desc *x,
                             const int *perm,
                             int ndim,
                             gd_tensor_desc *out);
gd_status _gd_meta_embedding(const gd_tensor_desc *table,
                             const gd_tensor_desc *ids,
                             gd_tensor_desc *out);
gd_status _gd_meta_rope(const gd_tensor_desc *x,
                        const gd_tensor_desc *pos_ids,
                        gd_tensor_desc *out);
gd_status _gd_meta_sdpa(const gd_tensor_desc *q,
                        const gd_tensor_desc *k,
                        const gd_tensor_desc *v,
                        gd_tensor_desc *out);

#endif /* GRADIENTS_META_COMMON_H */
