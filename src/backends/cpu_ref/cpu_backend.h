#ifndef GRADIENTS_CPU_BACKEND_H
#define GRADIENTS_CPU_BACKEND_H

#include "gradients/status.h"
#include "gradients/tensor.h"

#include "../../graph/graph_internal.h"

/* Scalar F32 (and minimal int) reference kernels. All tensors are contiguous. */
gd_status _gd_cpu_k_elementwise(_gd_op_kind op,
                                const gd_tensor_desc *out_desc,
                                float *out,
                                const gd_tensor_desc *a_desc,
                                const float *a,
                                const gd_tensor_desc *b_desc,
                                const float *b);
gd_status _gd_cpu_k_scale(const gd_tensor_desc *desc, float *out, const float *x, float scale);
gd_status _gd_cpu_k_relu(const gd_tensor_desc *desc, float *out, const float *x);
gd_status _gd_cpu_k_silu(const gd_tensor_desc *desc, float *out, const float *x);
gd_status _gd_cpu_k_matmul(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *a_desc,
                           const float *a,
                           bool trans_a,
                           const gd_tensor_desc *b_desc,
                           const float *b,
                           bool trans_b);
gd_status _gd_cpu_k_linear(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           const gd_tensor_desc *w_desc,
                           const float *w,
                           bool trans_w,
                           const float *bias);
gd_status _gd_cpu_k_reduce(const gd_tensor_desc *out_desc,
                           float *out,
                           const gd_tensor_desc *x_desc,
                           const float *x,
                           int dim,
                           bool mean);
gd_status _gd_cpu_k_softmax(const gd_tensor_desc *desc, float *out, const float *x, int dim);
gd_status _gd_cpu_k_rms_norm(const gd_tensor_desc *desc,
                             float *out,
                             const float *x,
                             const float *weight,
                             float eps);
gd_status _gd_cpu_k_cross_entropy(float *out,
                                  const gd_tensor_desc *logits_desc,
                                  const float *logits,
                                  const gd_tensor_desc *targets_desc,
                                  const void *targets,
                                  int class_dim);
gd_status _gd_cpu_k_cast(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *x_desc,
                         const void *x);

/* Identity/reshape copy of contiguous data (any fixed-size dtype, equal numel). */
gd_status _gd_cpu_k_copy(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *in_desc,
                         const void *in);
gd_status _gd_cpu_k_relu_bwd(const gd_tensor_desc *desc,
                             float *dx,
                             const float *x,
                             const float *go);
gd_status _gd_cpu_k_silu_bwd(const gd_tensor_desc *desc,
                             float *dx,
                             const float *x,
                             const float *go);
gd_status _gd_cpu_k_softmax_bwd(const gd_tensor_desc *desc,
                                float *dx,
                                const float *y,
                                const float *go,
                                int dim);
gd_status _gd_cpu_k_sum_bwd(const gd_tensor_desc *x_desc,
                            float *dx,
                            const float *go,
                            int dim,
                            bool mean);
gd_status _gd_cpu_k_cross_entropy_bwd(const gd_tensor_desc *logits_desc,
                                      float *dlogits,
                                      const float *logits,
                                      const gd_tensor_desc *targets_desc,
                                      const void *targets,
                                      const float *go_scalar,
                                      int class_dim);

/* Sums `go` (broadcasted output shape) down into `out` (target shape). */
gd_status _gd_cpu_k_reduce_to(const gd_tensor_desc *target_desc,
                              float *out,
                              const gd_tensor_desc *go_desc,
                              const float *go);

/* Optimizer kernels. */
gd_status _gd_cpu_k_assert_finite(const gd_tensor_desc *desc, const float *x);
gd_status _gd_cpu_k_assert_close(const gd_tensor_desc *a_desc,
                                 const float *a,
                                 const float *b,
                                 float atol,
                                 float rtol);
gd_status _gd_cpu_k_step_inc(float *step);
gd_status _gd_cpu_k_adamw(const gd_tensor_desc *param_desc,
                          float *param,
                          const float *grad,
                          float *m,
                          float *v,
                          const float *step,
                          float lr,
                          float beta1,
                          float beta2,
                          float eps,
                          float weight_decay);

#endif /* GRADIENTS_CPU_BACKEND_H */
