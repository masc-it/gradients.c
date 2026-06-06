#ifndef GRADIENTS_OPS_H
#define GRADIENTS_OPS_H

#include <gradients/status.h>
#include <gradients/tensor.h>
#include <gradients/ops_generated.h>

#ifdef __cplusplus
extern "C" {
#endif

gd_status gd_matmul(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    gd_tensor *out);

/* Computes out = x @ w + bias. Pass bias=NULL for out = x @ w. */
gd_status gd_linear(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    const gd_tensor *bias,
                    gd_tensor *out);

gd_status gd_matmul_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w);

gd_status gd_linear_backward(gd_context *ctx,
                             const gd_tensor *x,
                             const gd_tensor *w,
                             const gd_tensor *bias,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x,
                             gd_tensor *grad_w,
                             gd_tensor *grad_bias);

/* Materialized PyTorch-style concat along axis. Inputs must be contiguous,
 * non-scalar tensors with matching dtype/rank and equal non-axis dimensions.
 * Negative axes are accepted. Output is a new contiguous tensor. */
gd_status gd_concat(gd_context *ctx,
                    const gd_tensor *const *inputs,
                    uint32_t n_inputs,
                    int32_t axis,
                    gd_tensor *out);

gd_status gd_concat_backward(gd_context *ctx,
                             const gd_tensor *grad_out,
                             const gd_tensor *const *inputs,
                             uint32_t n_inputs,
                             int32_t axis,
                             gd_tensor *grad_inputs);

/* Metadata-only PyTorch-style reshape view. Input must be contiguous and the
 * requested shape must preserve element count. One requested dimension may be
 * -1 to infer it. Zero-size dimensions are not supported by the tensor runtime.
 * Output aliases input storage and is marked as a view. */
gd_status gd_reshape(gd_context *ctx,
                     const gd_tensor *x,
                     gd_shape shape,
                     gd_tensor *out);

/* Direct backward returns a metadata-only view of grad_out with x's shape. */
gd_status gd_reshape_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              gd_tensor *grad_x);

/* Packed variable-length scaled dot-product attention.
 * q/k/v are contiguous [N, Hq|Hkv, Dh], cu_seqlens is I32 [B+1].
 * Hq must be a multiple of Hkv. Output has q's shape and dtype.
 * scale <= 0 selects 1 / sqrt(Dh). */
typedef struct gd_sdpa_varlen_config {
    float scale;
    bool causal;
    int32_t sliding_window;
    int32_t prefix_len;
    int32_t max_seqlen;
} gd_sdpa_varlen_config;

gd_status gd_sdpa_varlen(gd_context *ctx,
                         const gd_tensor *q,
                         const gd_tensor *k,
                         const gd_tensor *v,
                         const gd_tensor *cu_seqlens,
                         const gd_sdpa_varlen_config *config,
                         gd_tensor *out);

gd_status gd_sdpa_varlen_backward(gd_context *ctx,
                                  const gd_tensor *q,
                                  const gd_tensor *k,
                                  const gd_tensor *v,
                                  const gd_tensor *cu_seqlens,
                                  const gd_tensor *grad_out,
                                  const gd_sdpa_varlen_config *config,
                                  gd_tensor *grad_q,
                                  gd_tensor *grad_k,
                                  gd_tensor *grad_v);

/* Decode-time attention over a fixed K/V cache. q is [B,Tq,Hq,Dh], k/v cache
 * are [B,Tmax,Hkv,Dh], cache_pos is an I32 scalar, and live keys are
 * [0, cache_pos + Tq). Attention is causal. */
typedef struct gd_sdpa_decode_config {
    float scale;
    int32_t sliding_window;
    int32_t prefix_len;
} gd_sdpa_decode_config;

gd_status gd_sdpa_decode(gd_context *ctx,
                         const gd_tensor *q,
                         const gd_tensor *k_cache,
                         const gd_tensor *v_cache,
                         const gd_tensor *cache_pos,
                         const gd_sdpa_decode_config *config,
                         gd_tensor *out);

/* Reduces a single axis. Negative axes are accepted Python-style. */
gd_status gd_reduce_sum_axis(gd_context *ctx,
                             const gd_tensor *x,
                             int32_t axis,
                             bool keepdims,
                             gd_tensor *out);
gd_status gd_reduce_sum_axis_backward(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *grad_out,
                                      int32_t axis,
                                      bool keepdims,
                                      gd_tensor *grad_x);
gd_status gd_reduce_mean_axis(gd_context *ctx,
                              const gd_tensor *x,
                              int32_t axis,
                              bool keepdims,
                              gd_tensor *out);
gd_status gd_reduce_mean_axis_backward(gd_context *ctx,
                                       const gd_tensor *x,
                                       const gd_tensor *grad_out,
                                       int32_t axis,
                                       bool keepdims,
                                       gd_tensor *grad_x);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_OPS_H */
