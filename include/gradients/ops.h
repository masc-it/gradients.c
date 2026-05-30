#ifndef GRADIENTS_OPS_H
#define GRADIENTS_OPS_H

#include <stdbool.h>

#include "gradients/context.h"
#include "gradients/dtype.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_matmul_desc {
    bool trans_a;
    bool trans_b;
    gd_compute_policy compute;
} gd_matmul_desc;

typedef struct gd_linear_desc {
    bool trans_w;
    gd_compute_policy compute;
} gd_linear_desc;

gd_status gd_add(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_mul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_scale(gd_context *ctx, gd_tensor *x, float scale, gd_tensor **out);

gd_status gd_matmul(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_tensor **out);
gd_status gd_matmul_ex(gd_context *ctx,
                       const gd_matmul_desc *desc,
                       gd_tensor *a,
                       gd_tensor *b,
                       gd_tensor **out);

gd_status gd_linear(gd_context *ctx,
                    gd_tensor *x,
                    gd_tensor *w,
                    gd_tensor *bias,
                    gd_tensor **out);
gd_status gd_linear_ex(gd_context *ctx,
                       const gd_linear_desc *desc,
                       gd_tensor *x,
                       gd_tensor *w,
                       gd_tensor *bias,
                       gd_tensor **out);

gd_status gd_relu(gd_context *ctx, gd_tensor *x, gd_tensor **out);
gd_status gd_silu(gd_context *ctx, gd_tensor *x, gd_tensor **out);
gd_status gd_sum(gd_context *ctx, gd_tensor *x, int dim, bool keepdim, gd_tensor **out);
gd_status gd_mean(gd_context *ctx, gd_tensor *x, int dim, bool keepdim, gd_tensor **out);
gd_status gd_rms_norm(gd_context *ctx,
                      gd_tensor *x,
                      gd_tensor *weight,
                      float eps,
                      gd_tensor **out);
gd_status gd_softmax(gd_context *ctx, gd_tensor *x, int dim, gd_tensor **out);
gd_status gd_cross_entropy(gd_context *ctx,
                           gd_tensor *logits,
                           gd_tensor *targets,
                           int class_dim,
                           gd_tensor **loss);
gd_status gd_cast(gd_context *ctx, gd_tensor *x, gd_dtype dtype, gd_tensor **out);

gd_status gd_backward(gd_context *ctx, gd_tensor *loss);
gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
