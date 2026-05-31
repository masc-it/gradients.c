#ifndef GRADIENTS_NN_H
#define GRADIENTS_NN_H

#include <stdbool.h>
#include <stdint.h>

#include "gradients/context.h"
#include "gradients/status.h"
#include "gradients/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decoder-only transformer (GPT) assembled from core ops. Pure composition:
 * embedding -> [pre-norm attention + MLP] x L -> final norm -> (tied) LM head.
 * Parameters are materialized, deterministically initialized from `seed`, and
 * collected for the optimizer via gd_gpt_parameters. */

typedef enum gd_gpt_mlp_kind {
    GD_GPT_MLP_SWIGLU = 0, /* down(silu(gate(x)) * up(x)) */
    GD_GPT_MLP_GELU        /* proj(gelu(fc(x))) */
} gd_gpt_mlp_kind;

typedef struct gd_gpt_config {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;       /* query heads */
    int n_kv_heads;    /* key/value heads (GQA); 0 => n_heads */
    int head_dim;      /* d_model must equal n_heads * head_dim */
    int d_ff;          /* MLP hidden size */
    int max_seq_len;
    float rope_theta;  /* 0 => 10000 */
    float norm_eps;    /* 0 => 1e-5 */
    gd_gpt_mlp_kind mlp_kind;
    bool tie_embeddings; /* LM head shares the token embedding */
} gd_gpt_config;

typedef struct gd_gpt gd_gpt;

gd_status gd_gpt_create(gd_context *ctx, const gd_gpt_config *config,
                        uint64_t seed, gd_gpt **out);
void gd_gpt_destroy(gd_gpt *gpt);

/* Trainable parameters (owned by the model; valid until destroy). */
gd_status gd_gpt_parameters(gd_gpt *gpt, gd_tensor ***params_out, int *n_out);

/* Records the forward pass into the active graph. `tokens` is int32 [B,T];
 * `positions` is int32 [B,T] (RoPE/causal positions). Produces logits[B,T,V]
 * (a new virtual tensor owned by the caller). */
gd_status gd_gpt_forward(gd_context *ctx, gd_gpt *gpt,
                         gd_tensor *tokens, gd_tensor *positions,
                         gd_tensor **logits_out);

/* Records forward + mean CE loss. For tied embeddings this may use a fused
 * LM-head CE op that avoids materializing logits. */
gd_status gd_gpt_forward_loss(gd_context *ctx, gd_gpt *gpt,
                              gd_tensor *tokens, gd_tensor *positions,
                              gd_tensor *targets, gd_tensor **loss_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_NN_H */
