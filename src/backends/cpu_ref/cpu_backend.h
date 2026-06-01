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
gd_status _gd_cpu_k_powlu(const gd_tensor_desc *desc, float *out,
                          const float *x1, const float *x2, float m);
gd_status _gd_cpu_k_powlu_bwd(const gd_tensor_desc *desc, float *dx1, float *dx2,
                              const float *x1, const float *x2, const float *go,
                              float m);
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
/* RMSNorm backward: dx (input gradient) and dweight (weight gradient). */
gd_status _gd_cpu_k_rms_norm_bwd(const gd_tensor_desc *desc, float *dx,
                                 const float *x, const float *weight,
                                 const float *go, float eps);
gd_status _gd_cpu_k_rms_norm_wbwd(const gd_tensor_desc *x_desc, float *dweight,
                                  const float *x, const float *go, float eps);
gd_status _gd_cpu_k_cross_entropy(float *out,
                                  const gd_tensor_desc *logits_desc,
                                  const float *logits,
                                  const gd_tensor_desc *targets_desc,
                                  const void *targets,
                                  int class_dim);
gd_status _gd_cpu_k_lm_cross_entropy(float *out,
                                     float *row_max,
                                     float *row_sum,
                                     const gd_tensor_desc *hidden_desc,
                                     const float *hidden,
                                     const gd_tensor_desc *weight_desc,
                                     const float *weight,
                                     const gd_tensor_desc *targets_desc,
                                     const void *targets);
gd_status _gd_cpu_k_cast(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *x_desc,
                         const void *x);

/* Identity/reshape copy of contiguous data (any fixed-size dtype, equal numel). */
gd_status _gd_cpu_k_copy(const gd_tensor_desc *out_desc,
                         void *out,
                         const gd_tensor_desc *in_desc,
                         const void *in);

/* GELU (exact erf, or tanh approximation). */
gd_status _gd_cpu_k_gelu(const gd_tensor_desc *desc, float *out, const float *x, int tanh_approx);
gd_status _gd_cpu_k_gelu_bwd(const gd_tensor_desc *desc, float *dx, const float *x,
                             const float *go, int tanh_approx);

/* Physical axis permutation into a contiguous result (any fixed-size dtype). */
gd_status _gd_cpu_k_transpose(const gd_tensor_desc *out_desc, void *out,
                              const gd_tensor_desc *in_desc, const void *in,
                              const int *perm);

/* Row gather and its scatter-add backward. */
gd_status _gd_cpu_k_embedding(const gd_tensor_desc *out_desc, float *out,
                              const gd_tensor_desc *table_desc, const float *table,
                              const gd_tensor_desc *ids_desc, const void *ids);
gd_status _gd_cpu_k_embedding_bwd(const gd_tensor_desc *table_desc, float *dtable,
                                  const gd_tensor_desc *go_desc, const float *go,
                                  const gd_tensor_desc *ids_desc, const void *ids);

/* Rotary position embedding. sin_sign = +1 forward, -1 backward (transpose
 * rotation). x/out are [.., heads, head_dim]; positions index leading rows. */
gd_status _gd_cpu_k_rope(const gd_tensor_desc *desc, float *out, const float *x,
                         const gd_tensor_desc *pos_desc, const void *pos,
                         float theta, int n_dims, int interleaved, float sin_sign);

/* Scaled dot-product attention (reference). q[B,Tq,Hq,Dh], k/v[B,Tk,Hkv,Dh],
 * o[B,Tq,Hq,Dh]; grouped-query via Hq/Hkv. */
gd_status _gd_cpu_k_sdpa(const gd_tensor_desc *o_desc, float *o,
                         const gd_tensor_desc *q_desc, const float *q,
                         const gd_tensor_desc *k_desc, const float *k,
                         const gd_tensor_desc *v_desc, const float *v,
                         const gd_tensor_desc *bias_desc, const float *bias,
                         float scale, int causal, int window, int prefix_len);
gd_status _gd_cpu_k_sdpa_bwd(const gd_tensor_desc *q_desc, const float *q,
                             const gd_tensor_desc *k_desc, const float *k,
                             const gd_tensor_desc *v_desc, const float *v,
                             const gd_tensor_desc *bias_desc, const float *bias,
                             const float *go,
                             float *dq, float *dk, float *dv,
                             float scale, int causal, int window, int prefix_len);
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
gd_status _gd_cpu_k_lm_cross_entropy_bwd(const gd_tensor_desc *hidden_desc,
                                         float *dhidden,
                                         const float *hidden,
                                         const gd_tensor_desc *weight_desc,
                                         float *dweight,
                                         const float *weight,
                                         const gd_tensor_desc *targets_desc,
                                         const void *targets,
                                         const float *go_scalar,
                                         const float *row_max,
                                         const float *row_sum);

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
gd_status _gd_cpu_k_clip_grad_norm(const gd_tensor_desc * const *grad_descs,
                                   float **grads,
                                   int n_grads,
                                   float max_norm,
                                   float eps,
                                   float *norm_out);
gd_status _gd_cpu_k_adamw(const gd_tensor_desc *param_desc,
                          float *param,
                          const float *grad,
                          float *m,
                          float *v,
                          const float *step,
                          const float *lr_tensor,
                          float lr,
                          float beta1,
                          float beta2,
                          float eps,
                          float weight_decay);

#endif /* GRADIENTS_CPU_BACKEND_H */
