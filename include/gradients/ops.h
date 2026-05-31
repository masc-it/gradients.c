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
/* PowLU gated activation: out = x1 * f(x2, m). m must satisfy 0 < m < 10. */
gd_status gd_powlu(gd_context *ctx,
                   gd_tensor *x1,
                   gd_tensor *x2,
                   float m,
                   gd_tensor **out);
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

/* Fused tied-LM-head cross entropy. Computes mean CE of
 * logits = hidden @ weight^T without materializing logits. hidden is [..., D],
 * weight is [V, D], targets has shape hidden.shape without the last dim. */
gd_status gd_lm_cross_entropy(gd_context *ctx,
                              gd_tensor *hidden,
                              gd_tensor *weight,
                              gd_tensor *targets,
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

/* Scaled dot-product attention. q[B,Tq,Hq,Dh], k/v[B,Tk,Hkv,Dh] (head-major);
 * Hq must be a multiple of Hkv (grouped-query attention). out is q's shape.
 * NULL config => scale 1/sqrt(Dh), non-causal, no window.
 *
 * Layout note: this is head-major [B,T,H,Dh] -- exactly what a QKV projection
 * produces after reshaping [B,T,d] -> [B,T,H,Dh]. No transpose to [B,H,T,Dh] is
 * needed (or wanted): RoPE and this op index heads by stride. The physical K/V
 * tiling preferred by a fused FlashAttention kernel is a backend concern
 * (internal repack), not a caller responsibility.
 *
 * `bias` (nullable) is an additive attention bias added to the scaled scores
 * before softmax: scores[b,h,i,j] += bias[b,h,i,j]. Its shape is 4D and
 * broadcast over [B, Hq, Tq, Tk] (any axis may be 1). This expresses padding
 * masks (0 / large-negative), ALiBi (per-head linear bias), and arbitrary
 * relative-position bias, and composes with the causal/window fast path. Bias
 * is treated as a constant (no gradient) in v1.
 *
 * `prefix_len` enables VLM prefix-causal attention when `causal=true`: query
 * positions before prefix_len attend bidirectionally within the prefix only;
 * later positions attend causally to prefix + prior/current suffix tokens. */
typedef struct gd_sdpa_config {
    float scale;          /* 0 => 1/sqrt(head_dim) */
    bool  causal;
    int   sliding_window; /* 0 => none */
    int   prefix_len;     /* >0 with causal=true => prefix-causal */
} gd_sdpa_config;
gd_status gd_sdpa(gd_context *ctx,
                  gd_tensor *q,
                  gd_tensor *k,
                  gd_tensor *v,
                  gd_tensor *bias,
                  const gd_sdpa_config *config,
                  gd_tensor **out);

gd_status gd_backward(gd_context *ctx, gd_tensor *loss);
gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
