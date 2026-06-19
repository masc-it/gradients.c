#ifndef GRADIENTS_OPS_H
#define GRADIENTS_OPS_H

#include <gradients/status.h>
#include <gradients/tensor.h>
#include <gradients/ops_generated.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Direct backward helpers accept NULL grad_* output pointers to omit gradients.
 * If every grad_* output is NULL, helpers validate inputs/grad_out and return
 * GD_OK without enqueueing backward work. Requesting gradients for
 * non-differentiable inputs still returns GD_ERR_UNSUPPORTED. */

/* PyTorch-style F16 matrix multiplication with full batch broadcasting:
 * x [..., M, K] @ w [..., K, N] -> out [broadcast(...), M, N].
 * Inputs must be row-strided in their innermost matrix dimensions. */
gd_status gd_matmul(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    gd_tensor *out);

/* Fully-connected projection with optional bias:
 * x [..., K], w [K, N], optional bias [N] -> out [..., N].
 * Rank-N inputs are flattened over leading dimensions for GEMM; rank-2
 * row-strided inputs are accepted, while other rank-N inputs must be contiguous. */
gd_status gd_linear(gd_context *ctx,
                    const gd_tensor *x,
                    const gd_tensor *w,
                    const gd_tensor *bias,
                    gd_tensor *out);

/* Fully-connected projection using a row-major transposed weight, optimized for
 * tied embedding / LM-head weights:
 * x [..., K], w [N, K], optional bias [N] -> out [..., N].
 * Forward dispatches directly to x @ w^T without materializing w^T. */
gd_status gd_linear_transposed_weight(gd_context *ctx,
                                      const gd_tensor *x,
                                      const gd_tensor *w,
                                      const gd_tensor *bias,
                                      gd_tensor *out);

/* Fused tied LM head + cross entropy:
 * hidden [..., D], weight [V, D], targets [rows] -> scalar F32 mean loss.
 * Negative target ids are ignored and the loss is averaged over valid rows.
 * Uses the same F16 logits semantics as linear_transposed_weight followed by
 * cross_entropy, but records one training op for the combined LM-head loss. */
gd_status gd_lm_cross_entropy(gd_context *ctx,
                              const gd_tensor *hidden,
                              const gd_tensor *weight,
                              const gd_tensor *targets,
                              gd_tensor *loss);

/* Fused tied LM head + cross entropy with optional final-logits softcap:
 * soft_logit = logits_softcap * tanh(logit / logits_softcap).
 * Pass logits_softcap=0 to disable softcapping and use the exact gd_lm_cross_entropy path. */
gd_status gd_lm_cross_entropy_softcapped(gd_context *ctx,
                                         const gd_tensor *hidden,
                                         const gd_tensor *weight,
                                         const gd_tensor *targets,
                                         float logits_softcap,
                                         gd_tensor *loss);

/* Fused residual add with inverted dropout on the branch:
 * out = residual + dropout(x, p, seed) in training, or residual + x otherwise.
 * Training autograd records equivalent dropout and add nodes while using one
 * fused forward kernel. Currently optimized for same-shape contiguous F16. */
gd_status gd_dropout_add(gd_context *ctx,
                         const gd_tensor *residual,
                         const gd_tensor *x,
                         float p,
                         bool training,
                         uint64_t seed,
                         gd_tensor *out);

/* Direct matmul backward. Broadcasted batch dimensions are reduced back to
 * x/w's original shapes. Pass grad_x or grad_w as NULL to skip that gradient. */
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

gd_status gd_linear_transposed_weight_backward(gd_context *ctx,
                                               const gd_tensor *x,
                                               const gd_tensor *w,
                                               const gd_tensor *bias,
                                               const gd_tensor *grad_out,
                                               gd_tensor *grad_x,
                                               gd_tensor *grad_w,
                                               gd_tensor *grad_bias);

/* PoWLU gated activation ported from v1:
 * gate(z, m) = z * sigmoid(z) for z <= 0, otherwise
 *              z^(m / (sqrt(z) + 1)) * sigmoid(z).
 * out = x1 * gate(x2, m). m must satisfy 0 < m < 10.
 * Current v2 implementation is optimized for contiguous F16 tensors. */
gd_status gd_powlu(gd_context *ctx,
                   const gd_tensor *x1,
                   const gd_tensor *x2,
                   float m,
                   gd_tensor *out);

gd_status gd_powlu_backward(gd_context *ctx,
                            const gd_tensor *x1,
                            const gd_tensor *x2,
                            const gd_tensor *grad_out,
                            float m,
                            gd_tensor *grad_x1,
                            gd_tensor *grad_x2);

/* Fused split+PoWLU for gated MLP projections:
 * x12 [..., 2H] -> out [..., H], where x1=x12[..., :H] and x2=x12[..., H:].
 * Avoids materializing the two split halves and fuses the backward concat. */
gd_status gd_powlu_split(gd_context *ctx,
                         const gd_tensor *x12,
                         float m,
                         gd_tensor *out);

gd_status gd_powlu_split_backward(gd_context *ctx,
                                  const gd_tensor *x12,
                                  const gd_tensor *grad_out,
                                  float m,
                                  gd_tensor *grad_x12);

/* Fused gated-MLP projection:
 * y = linear(powlu_split(x12, m), w, bias), with x12 [..., 2H], w [H, N].
 * Training autograd fuses the down-projection input-gradient GEMM with the
 * PoWLU split backward epilogue to avoid materializing d_powlu. */
gd_status gd_powlu_split_linear(gd_context *ctx,
                                const gd_tensor *x12,
                                const gd_tensor *w,
                                const gd_tensor *bias,
                                float m,
                                gd_tensor *out);

gd_status gd_powlu_split_linear_backward(gd_context *ctx,
                                         const gd_tensor *x12,
                                         const gd_tensor *w,
                                         const gd_tensor *bias,
                                         const gd_tensor *grad_out,
                                         float m,
                                         gd_tensor *grad_x12,
                                         gd_tensor *grad_w,
                                         gd_tensor *grad_bias);

/* SwiGLU gated activation for MLP projections:
 * out = x1 * (x2 * sigmoid(x2)). Current implementation is optimized for
 * contiguous F16 tensors. */
gd_status gd_swiglu(gd_context *ctx,
                    const gd_tensor *x1,
                    const gd_tensor *x2,
                    gd_tensor *out);

gd_status gd_swiglu_backward(gd_context *ctx,
                             const gd_tensor *x1,
                             const gd_tensor *x2,
                             const gd_tensor *grad_out,
                             gd_tensor *grad_x1,
                             gd_tensor *grad_x2);

/* Fused split+SwiGLU for gated MLP projections:
 * x12 [..., 2H] -> out [..., H], where x1=x12[..., :H] and x2=x12[..., H:]. */
gd_status gd_swiglu_split(gd_context *ctx,
                          const gd_tensor *x12,
                          gd_tensor *out);

gd_status gd_swiglu_split_backward(gd_context *ctx,
                                   const gd_tensor *x12,
                                   const gd_tensor *grad_out,
                                   gd_tensor *grad_x12);

/* Fused SwiGLU gated-MLP projection:
 * y = linear(swiglu_split(x12), w, bias), with x12 [..., 2H], w [H, N]. */
gd_status gd_swiglu_split_linear(gd_context *ctx,
                                 const gd_tensor *x12,
                                 const gd_tensor *w,
                                 const gd_tensor *bias,
                                 gd_tensor *out);

gd_status gd_swiglu_split_linear_backward(gd_context *ctx,
                                          const gd_tensor *x12,
                                          const gd_tensor *w,
                                          const gd_tensor *bias,
                                          const gd_tensor *grad_out,
                                          gd_tensor *grad_x12,
                                          gd_tensor *grad_w,
                                          gd_tensor *grad_bias);

/* Embedding lookup. table is contiguous F16/F32 [vocab, dim], ids is a
 * contiguous I32 tensor with rank >= 1. Output is contiguous with shape
 * ids.shape + [dim] and table dtype. Invalid ids produce NaN output values;
 * backward ignores invalid ids. */
gd_status gd_embedding(gd_context *ctx,
                       const gd_tensor *table,
                       const gd_tensor *ids,
                       gd_tensor *out);

gd_status gd_embedding_backward(gd_context *ctx,
                                const gd_tensor *table,
                                const gd_tensor *ids,
                                const gd_tensor *grad_out,
                                gd_tensor *grad_table);

/* Root-mean-square normalization over the last dimension:
 * out[row, c] = x[row, c] * weight[c] / sqrt(mean_c(x[row, c]^2) + eps).
 * x and weight must be contiguous F16/F32 tensors with matching dtype; weight
 * is [last_dim]. Output is contiguous and has x's shape/dtype. */
gd_status gd_rms_norm(gd_context *ctx,
                      const gd_tensor *x,
                      const gd_tensor *weight,
                      float eps,
                      gd_tensor *out);

gd_status gd_rms_norm_backward(gd_context *ctx,
                               const gd_tensor *x,
                               const gd_tensor *weight,
                               const gd_tensor *grad_out,
                               float eps,
                               gd_tensor *grad_x,
                               gd_tensor *grad_weight);

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

/* Materialized PyTorch-style split along axis. Input must be contiguous and
 * non-scalar. sizes has length n_outputs and positive entries summing to the
 * input axis dimension. Negative axes are accepted. Outputs are new contiguous
 * tensors, suitable for downstream kernels that require contiguous inputs. */
gd_status gd_split(gd_context *ctx,
                   const gd_tensor *x,
                   const int64_t *sizes,
                   uint32_t n_outputs,
                   int32_t axis,
                   gd_tensor *outputs);

gd_status gd_split_backward(gd_context *ctx,
                            const gd_tensor *x,
                            const gd_tensor *const *grad_outputs,
                            const int64_t *sizes,
                            uint32_t n_outputs,
                            int32_t axis,
                            gd_tensor *grad_x);

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

/* Materialized PyTorch-style axis permutation. `axes` has length n_axes ==
 * x->rank and names the input axis used for each output axis. Negative axes are
 * accepted. Input and grad_out must be contiguous; outputs are contiguous. */
gd_status gd_permute(gd_context *ctx,
                     const gd_tensor *x,
                     const int32_t *axes,
                     uint32_t n_axes,
                     gd_tensor *out);

gd_status gd_permute_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              const int32_t *axes,
                              uint32_t n_axes,
                              gd_tensor *grad_x);

/* Rotary positional embedding over tensors shaped [.., heads, head_dim].
 * pos_ids is a contiguous I32 tensor with element count equal to the product of
 * the leading dimensions before heads. theta <= 0 selects 10000. n_dims <= 0
 * rotates the full head_dim; otherwise n_dims must be even and <= head_dim.
 * interleaved=false uses NeoX half-split pairs; true uses GPT-J even/odd pairs. */
typedef struct gd_rope_config {
    float theta;
    int32_t n_dims;
    bool interleaved;
} gd_rope_config;

gd_status gd_rope(gd_context *ctx,
                  const gd_tensor *x,
                  const gd_tensor *pos_ids,
                  const gd_rope_config *config,
                  gd_tensor *out);

gd_status gd_rope_backward(gd_context *ctx,
                           const gd_tensor *x,
                           const gd_tensor *pos_ids,
                           const gd_tensor *grad_out,
                           const gd_rope_config *config,
                           gd_tensor *grad_x);

/* Fused GPT-style QKV unpack + full-head RoPE for attention projections.
 * qkv is contiguous F16 [N, 3 * H * Dh], pos_ids is contiguous I32 [N].
 * Outputs are contiguous F16 q/k/v [N, H, Dh], with RoPE applied to q and k
 * and v copied directly. This avoids materializing split Q/K/V tensors and
 * computing identical Q/K rotary angles in separate kernels. */
gd_status gd_qkv_split_rope(gd_context *ctx,
                            const gd_tensor *qkv,
                            const gd_tensor *pos_ids,
                            int32_t n_heads,
                            int32_t head_dim,
                            const gd_rope_config *config,
                            gd_tensor *q,
                            gd_tensor *k,
                            gd_tensor *v);

gd_status gd_qkv_split_rope_backward(gd_context *ctx,
                                      const gd_tensor *qkv,
                                      const gd_tensor *pos_ids,
                                      const gd_tensor *grad_q,
                                      const gd_tensor *grad_k,
                                      const gd_tensor *grad_v,
                                      int32_t n_heads,
                                      int32_t head_dim,
                                      const gd_rope_config *config,
                                      gd_tensor *grad_qkv);

/* MiniMax M3 lightning-indexer + block-sparse attention.
 * The indexer scores each visible key block by max_j(q_i · k_j), then selects
 * top-k block ids with optional forced initial and local blocks. Sparse
 * attention consumes those block ids and applies causal attention only inside
 * selected blocks. Tensors are contiguous [N,H,Dh], cu_seqlens is I32 [B+1],
 * topk_idx is I32 [Hkv,N,topk]. Current Metal kernels are optimized for F16,
 * head_dim <= 64, topk <= 16, and block_size <= 128 (M3 uses 128). */
typedef struct gd_minimax_m3_sparse_config {
    float scale;        /* <= 0 selects 1 / sqrt(Dh) for attention. */
    int32_t block_size; /* <= 0 selects 128. */
    int32_t topk;       /* <= 0 selects 4. */
    int32_t init_blocks;
    int32_t local_blocks;
    int32_t max_seqlen; /* <= 0 selects N. */
} gd_minimax_m3_sparse_config;

gd_status gd_minimax_m3_index_topk(gd_context *ctx,
                                   const gd_tensor *index_q,
                                   const gd_tensor *index_k,
                                   const gd_tensor *cu_seqlens,
                                   const gd_minimax_m3_sparse_config *config,
                                   gd_tensor *topk_idx);

gd_status gd_minimax_m3_sparse_attention(gd_context *ctx,
                                         const gd_tensor *q,
                                         const gd_tensor *k,
                                         const gd_tensor *v,
                                         const gd_tensor *cu_seqlens,
                                         const gd_tensor *topk_idx,
                                         const gd_minimax_m3_sparse_config *config,
                                         gd_tensor *out);

gd_status gd_minimax_m3_sparse_attention_backward(gd_context *ctx,
                                                  const gd_tensor *q,
                                                  const gd_tensor *k,
                                                  const gd_tensor *v,
                                                  const gd_tensor *cu_seqlens,
                                                  const gd_tensor *topk_idx,
                                                  const gd_tensor *grad_out,
                                                  const gd_minimax_m3_sparse_config *config,
                                                  gd_tensor *grad_q,
                                                  gd_tensor *grad_k,
                                                  gd_tensor *grad_v);

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
 * are [B,Tmax,Hkv,Dh], cache_pos is the starting absolute position for q,
 * and live keys are [0, cache_pos + Tq). Attention is causal. */
typedef struct gd_sdpa_decode_config {
    float scale;
    int32_t sliding_window;
    int32_t prefix_len;
} gd_sdpa_decode_config;

/* Compatibility variant: cache_pos is an I32 scalar tensor. */
gd_status gd_sdpa_decode(gd_context *ctx,
                         const gd_tensor *q,
                         const gd_tensor *k_cache,
                         const gd_tensor *v_cache,
                         const gd_tensor *cache_pos,
                         const gd_sdpa_decode_config *config,
                         gd_tensor *out);

/* Fast decode variant: cache_pos is passed as a host scalar kernel argument,
 * avoiding a per-step scalar tensor/upload in autoregressive generation. */
gd_status gd_sdpa_decode_at(gd_context *ctx,
                            const gd_tensor *q,
                            const gd_tensor *k_cache,
                            const gd_tensor *v_cache,
                            int32_t cache_pos,
                            const gd_sdpa_decode_config *config,
                            gd_tensor *out);

/* Batched decode variant: cache_pos is I32 [B], one absolute start position per
 * batch row. */
gd_status gd_sdpa_decode_positions(gd_context *ctx,
                                   const gd_tensor *q,
                                   const gd_tensor *k_cache,
                                   const gd_tensor *v_cache,
                                   const gd_tensor *cache_pos,
                                   const gd_sdpa_decode_config *config,
                                   gd_tensor *out);

/* In-place append of new K/V rows into a fixed decode cache. Cache tensors are
 * [B,Tmax,Hkv,Dh], new tensors are [B,Tnew,Hkv,Dh], and writes target
 * cache[:, cache_pos:cache_pos+Tnew, :, :]. Inference/eval only. */
gd_status gd_kv_cache_append_at(gd_context *ctx,
                                gd_tensor *k_cache,
                                gd_tensor *v_cache,
                                int32_t cache_pos,
                                const gd_tensor *k_new,
                                const gd_tensor *v_new);

/* Batched append with one cache position per batch row. cache_pos is I32 [B]. */
gd_status gd_kv_cache_append_positions(gd_context *ctx,
                                       gd_tensor *k_cache,
                                       gd_tensor *v_cache,
                                       const gd_tensor *cache_pos,
                                       const gd_tensor *k_new,
                                       const gd_tensor *v_new);

/* Packed prefill append for variable-length prompt batches. k_new/v_new are
 * [N,Hkv,Dh], cu_seqlens is I32 [B+1], and cache_pos is I32 [B]. */
gd_status gd_kv_cache_append_packed(gd_context *ctx,
                                    gd_tensor *k_cache,
                                    gd_tensor *v_cache,
                                    const gd_tensor *cache_pos,
                                    const gd_tensor *cu_seqlens,
                                    const gd_tensor *k_new,
                                    const gd_tensor *v_new);

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
