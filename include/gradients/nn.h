#ifndef GRADIENTS_NN_H
#define GRADIENTS_NN_H

#include <stdbool.h>
#include <stdint.h>

#include "gradients/context.h"
#include "gradients/status.h"
#include "gradients/tensor.h"
#include "gradients/optim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Decoder-only transformer (GPT) assembled from core ops. Pure composition:
 * embedding -> [pre-norm attention + MLP] x L -> final norm -> LM head.
 * Parameters are materialized, deterministically initialized from `seed`, and
 * collected for the optimizer via gd_gpt_parameters. */

typedef enum gd_gpt_mlp_kind {
    GD_GPT_MLP_POWLU = 0, /* down(powlu(up(x), gate(x), m)) */
    GD_GPT_MLP_SWIGLU,    /* legacy: down(silu(gate(x)) * up(x)) */
    GD_GPT_MLP_GELU       /* proj(gelu(fc(x))) */
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
    float powlu_m;     /* 0 => 3.0; valid 0 < m < 10 */
    int attention_window; /* 0 => full causal; >0 => causal sliding-window */
    gd_dtype param_dtype; /* 0/GD_DTYPE_INVALID => F32; F16 enables forward-only v1 */
    bool tie_embeddings; /* LM head shares the token embedding */
    float dropout_p; /* 0 => disabled; inverted dropout on embedding/residual branches */
} gd_gpt_config;

typedef struct gd_gpt gd_gpt;

gd_status gd_gpt_create(gd_context *ctx, const gd_gpt_config *config,
                        uint64_t seed, gd_gpt **out);
void gd_gpt_destroy(gd_gpt *gpt);
void gd_gpt_set_training(gd_gpt *gpt, bool training);
bool gd_gpt_is_training(const gd_gpt *gpt);

/* Trainable parameters (owned by the model; valid until destroy). */
gd_status gd_gpt_parameters(gd_gpt *gpt, gd_tensor ***params_out, int *n_out);
gd_status gd_gpt_parameter_groups(gd_gpt *gpt,
                                  float weight_decay,
                                  gd_param_group **groups_out,
                                  int *n_groups_out);
void gd_gpt_parameter_groups_free(gd_param_group *groups, int n_groups);

/* Shared decoder config for token-id and inputs_embeds GPT forwards.
 * prefix_len=0 gives normal causal attention. prefix_len>0 gives prefix-causal
 * attention: prefix tokens attend bidirectionally inside the prefix, suffix
 * tokens attend all prefix tokens plus causal suffix history. Loss ignore fields
 * are used only by *_loss variants. */
typedef struct gd_gpt_forward_config {
    int prefix_len;
    bool has_ignore_index;
    int ignore_index;
    int max_seqlen; /* varlen only: 0 => packed token count */
} gd_gpt_forward_config;

/* Token embedding lookup only: tokens int32/int64 with any shape -> embeds
 * tokens.shape + [d_model]. The returned tensor is suitable for
 * gd_gpt_forward_embeds*. */
gd_status gd_gpt_embed_tokens(gd_context *ctx, gd_gpt *gpt,
                              gd_tensor *tokens, gd_tensor **embeds_out);

/* Decoder trunk from already-embedded inputs. `inputs_embeds` is [B,T,d_model],
 * `positions` is int32/int64 [B,T]. The trunk applies GPT embedding dropout,
 * transformer blocks, and final norm, returning hidden [B,T,d_model]. */
gd_status gd_gpt_decode_embeds(gd_context *ctx, gd_gpt *gpt,
                               gd_tensor *inputs_embeds, gd_tensor *positions,
                               const gd_gpt_forward_config *config,
                               gd_tensor **hidden_out);

/* Packed variable-length decoder. inputs_embeds is [N,d_model], positions is
 * I32/I64 [N], and cu_seqlens is I32 [B+1]. Attention uses gd_sdpa_varlen so
 * padded tokens are not materialized/visited. Set prefix_len>0,
 * attention_window>0, param_dtype=F16, and head_dim=64 to hit optimized Metal
 * prefix-window kernels. */
gd_status gd_gpt_decode_embeds_varlen(gd_context *ctx,
                                      gd_gpt *gpt,
                                      gd_tensor *inputs_embeds,
                                      gd_tensor *positions,
                                      gd_tensor *cu_seqlens,
                                      const gd_gpt_forward_config *config,
                                      gd_tensor **hidden_out);

/* Records the forward pass into the active graph. `tokens` is int32 [B,T];
 * `positions` is int32 [B,T] (RoPE/causal positions). Produces logits[B,T,V]
 * (a new virtual tensor owned by the caller). */
gd_status gd_gpt_forward(gd_context *ctx, gd_gpt *gpt,
                         gd_tensor *tokens, gd_tensor *positions,
                         gd_tensor **logits_out);

gd_status gd_gpt_forward_embeds(gd_context *ctx, gd_gpt *gpt,
                                gd_tensor *inputs_embeds, gd_tensor *positions,
                                const gd_gpt_forward_config *config,
                                gd_tensor **logits_out);

/* Records forward + mean CE loss via fused LMCE, avoiding materialized logits. */
gd_status gd_gpt_forward_loss(gd_context *ctx, gd_gpt *gpt,
                              gd_tensor *tokens, gd_tensor *positions,
                              gd_tensor *targets, gd_tensor **loss_out);

gd_status gd_gpt_forward_embeds_loss(gd_context *ctx, gd_gpt *gpt,
                                     gd_tensor *inputs_embeds, gd_tensor *positions,
                                     gd_tensor *targets,
                                     const gd_gpt_forward_config *config,
                                     gd_tensor **loss_out);

gd_status gd_gpt_forward_embeds_varlen_loss(gd_context *ctx,
                                            gd_gpt *gpt,
                                            gd_tensor *inputs_embeds,
                                            gd_tensor *positions,
                                            gd_tensor *cu_seqlens,
                                            gd_tensor *targets,
                                            const gd_gpt_forward_config *config,
                                            gd_tensor **loss_out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_NN_H */
