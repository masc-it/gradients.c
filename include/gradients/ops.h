#ifndef GRADIENTS_OPS_H
#define GRADIENTS_OPS_H

#include <stdbool.h>
#include <stdint.h>

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
/* Inverted dropout. Training mode applies y=x*mask/(1-p) with deterministic
 * per-graph-run masks from seed; eval mode or p==0 returns x unchanged. */
gd_status gd_dropout(gd_context *ctx,
                     gd_tensor *x,
                     float p,
                     uint64_t seed,
                     bool training,
                     gd_tensor **out);
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

typedef struct gd_cross_entropy_desc {
    int class_dim;
    bool has_ignore_index;
    int ignore_index;
} gd_cross_entropy_desc;

typedef struct gd_lm_cross_entropy_desc {
    bool has_ignore_index;
    int ignore_index;
} gd_lm_cross_entropy_desc;

gd_status gd_cross_entropy(gd_context *ctx,
                           gd_tensor *logits,
                           gd_tensor *targets,
                           int class_dim,
                           gd_tensor **loss);
gd_status gd_cross_entropy_ex(gd_context *ctx,
                              const gd_cross_entropy_desc *desc,
                              gd_tensor *logits,
                              gd_tensor *targets,
                              gd_tensor **loss);

/* Fused LM-head cross entropy. Computes mean CE of logits = hidden @ weight^T
 * without materializing logits. hidden is [..., D], weight is [V, D], targets
 * has shape hidden.shape without the last dim. */
gd_status gd_lm_cross_entropy(gd_context *ctx,
                              gd_tensor *hidden,
                              gd_tensor *weight,
                              gd_tensor *targets,
                              gd_tensor **loss);
gd_status gd_lm_cross_entropy_ex(gd_context *ctx,
                                 const gd_lm_cross_entropy_desc *desc,
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
 * later positions attend causally to prefix + prior/current suffix tokens. When
 * `sliding_window > 0`, the window applies only to suffix/suffix attention;
 * prefix keys stay visible to all suffix queries. */
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

/* Packed variable-length self-attention. q is [N,Hq,Dh], k/v are
 * [N,Hkv,Dh], and cu_seqlens is I32 [B+1] with cu_seqlens[0]=0 and
 * cu_seqlens[B]=N. Each row range [cu[b], cu[b+1]) is one independent sequence;
 * no padded keys are materialized or visited. max_seqlen is an optional dispatch
 * bound (0 => N) and must be >= every sequence length. Mask semantics match
 * gd_sdpa: causal/window/prefix_len apply within each packed sequence. */
typedef struct gd_sdpa_varlen_config {
    float scale;          /* 0 => 1/sqrt(head_dim) */
    bool  causal;
    int   sliding_window; /* 0 => none */
    int   prefix_len;     /* >0 with causal=true => prefix-causal */
    int   max_seqlen;     /* 0 => N; dispatch bound for packed sequence length */
} gd_sdpa_varlen_config;
gd_status gd_sdpa_varlen(gd_context *ctx,
                         gd_tensor *q,
                         gd_tensor *k,
                         gd_tensor *v,
                         gd_tensor *cu_seqlens,
                         const gd_sdpa_varlen_config *config,
                         gd_tensor **out);

/* Compact slice along one axis. `dim` may be negative; `start` is zero-based
 * after normalization and `len` must be positive. Output is contiguous with the
 * same rank as `x` and size[dim] = len. Backward scatters the slice gradient
 * into a zero-filled full-shape gradient. */
gd_status gd_slice(gd_context *ctx,
                   gd_tensor *x,
                   int dim,
                   int64_t start,
                   int64_t len,
                   gd_tensor **out);

/* Concatenate 1..256 tensors along `dim` (negative dims allowed). Inputs must
 * share dtype/device/rank and match in every non-concat dimension. Output is a
 * compact contiguous tensor. Backward slices the output gradient back to each
 * input span. */
gd_status gd_concat(gd_context *ctx,
                    gd_tensor *const *inputs,
                    int n_inputs,
                    int dim,
                    gd_tensor **out);

/* Fused pre-attention projection: norm = rms_norm(x, weight), then
 * q/k/v = linear(norm, wq/wk/wv). Outputs are norm, q, k, v. */
gd_status gd_rms_norm_qkv(gd_context *ctx,
                          gd_tensor *x,
                          gd_tensor *weight,
                          gd_tensor *wq,
                          gd_tensor *wk,
                          gd_tensor *wv,
                          float eps,
                          gd_tensor **norm,
                          gd_tensor **q,
                          gd_tensor **k,
                          gd_tensor **v);

gd_status gd_backward(gd_context *ctx, gd_tensor *loss);
gd_status gd_zero_grad(gd_context *ctx, gd_tensor **params, int n_params);

gd_status gd_clip_grad_norm(gd_context *ctx,
                            gd_tensor **params,
                            int n_params,
                            float max_norm,
                            gd_tensor **norm_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
