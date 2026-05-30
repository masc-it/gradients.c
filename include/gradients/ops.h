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

/* GELU activation. tanh_approx selects the tanh approximation; otherwise the
 * exact erf form. */
gd_status gd_gelu(gd_context *ctx, gd_tensor *x, bool tanh_approx, gd_tensor **out);

/* Physically permute axes into a contiguous result. `perm` has `ndim` entries
 * and must be a permutation of [0, ndim). out.sizes[i] = x.sizes[perm[i]]. */
gd_status gd_transpose(gd_context *ctx,
                       gd_tensor *x,
                       const int *perm,
                       int ndim,
                       gd_tensor **out);

/* Row gather: out[..., :] = table[ids[...], :]. table is [vocab, dim] (float),
 * ids is integer; out is ids.shape ++ [dim]. */
gd_status gd_embedding(gd_context *ctx,
                       gd_tensor *table,
                       gd_tensor *ids,
                       gd_tensor **out);

/* Rotary position embedding applied to x[.., heads, head_dim]. Positions are
 * given by pos_ids (integer), one per leading-index row (product of dims before
 * heads). NULL config uses theta=10000, full head_dim, NeoX half-split. */
typedef struct gd_rope_config {
    float theta;       /* 0 => 10000 */
    int   n_dims;      /* rotary dims; 0 => head_dim; must be even and <= head_dim */
    bool  interleaved; /* true: GPT-J (2i,2i+1); false: NeoX (i, i+n_dims/2) */
} gd_rope_config;
gd_status gd_rope(gd_context *ctx,
                  gd_tensor *x,
                  gd_tensor *pos_ids,
                  const gd_rope_config *config,
                  gd_tensor **out);

gd_status gd_backward(gd_context *ctx, gd_tensor *loss);
gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
