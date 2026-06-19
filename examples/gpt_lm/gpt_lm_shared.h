#ifndef GPT_LM_SHARED_H
#define GPT_LM_SHARED_H

#include <gradients/gradients.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define GPT_VOCAB_SIZE 2048
#define GPT_CONTEXT_LENGTH 512
#define GPT_D_MODEL 512
#define GPT_N_HEADS 8
#define GPT_HEAD_DIM 64
#define GPT_SDPA_WINDOW 256
#define GPT_MLP_HIDDEN (2 * GPT_D_MODEL)
#define GPT_MINIMAX_M3_BLOCK_SIZE 128
#define GPT_MINIMAX_M3_TOPK_BLOCKS 3
#define GPT_MINIMAX_M3_INIT_BLOCKS 1
#define GPT_MINIMAX_M3_LOCAL_BLOCKS 1

#define GPT_DEFAULT_LAYERS 3
#define GPT_DEFAULT_EPOCHS 2
#define GPT_DEFAULT_BATCH_SIZE 64
#define GPT_DEFAULT_REPORT_EVERY 10
#define GPT_DEFAULT_EVAL_EVERY_N_EPOCHS 1
#define GPT_DEFAULT_EARLY_STOPPING_PATIENCE 10
#define GPT_DEFAULT_DROPOUT_P 0.05f
#define GPT_DEFAULT_LR_MAX 3.0e-4f
#define GPT_DEFAULT_LR_MIN 5.0e-5f
#define GPT_DEFAULT_WEIGHT_DECAY 1.0e-4f
#define GPT_DEFAULT_GRAD_CLIP_NORM 1.0f
#define GPT_DEFAULT_RMS_EPS 1.0e-5f
#define GPT_DEFAULT_MIN_P 0.0f
#define GPT_DEFAULT_REPETITION_PENALTY 1.0f
#define GPT_DEFAULT_SEED UINT64_C(0x6750746c6d5eed00)

#define GPT_ALIGNMENT 256U
#define GPT_MIN_PARAMS_BYTES (64ULL * 1024ULL * 1024ULL)
#define GPT_MIN_STATE_BYTES (128ULL * 1024ULL * 1024ULL)
#define GPT_SCRATCH_SLOT_BYTES (3ULL * 1024ULL * 1024ULL * 1024ULL)
#define GPT_MIN_DATA_SLOT_BYTES (128ULL * 1024ULL * 1024ULL)

#define GPT_DROPOUT_EMBED UINT64_C(0x9e3779b97f4a7c15)
#define GPT_DROPOUT_ATTN UINT64_C(0xbf58476d1ce4e5b9)
#define GPT_DROPOUT_MLP UINT64_C(0x94d049bb133111eb)

#define TRY(ctx, expr)                                                        \
    do {                                                                      \
        gd_status _st = (expr);                                                \
        if (_st != GD_OK) {                                                    \
            gpt_fail_status((ctx), _st, #expr, __LINE__);                     \
        }                                                                     \
    } while (0)

typedef enum gpt_architecture {
    GPT_ARCH_GPT = 0,
    GPT_ARCH_MINIMAX_M3 = 1,
} gpt_architecture;

typedef struct gpt_block {
    gd_module mod;
    gd_tensor attn_norm_w;
    gd_tensor mlp_norm_w;
    gd_linear_layer qkv_proj;
    gd_linear_layer attn_proj;
    gd_linear_layer up_gate;
    gd_linear_layer down_proj;
} gpt_block;

typedef struct gpt_kv_cache {
    int batch_size;
    int batch_capacity;
    int max_seq;
    int n_layers;
    int n_heads;
    int head_dim;
    int32_t *pos;
    gd_tensor *k;
    gd_tensor *v;
} gpt_kv_cache;

typedef struct gpt_lm {
    gd_module mod;
    int n_layers;
    int vocab_size;
    int context_length;
    int d_model;
    int n_heads;
    int head_dim;
    int mlp_hidden;
    int sdpa_window;
    gpt_architecture architecture;
    int minimax_m3_block_size;
    int minimax_m3_topk_blocks;
    int minimax_m3_init_blocks;
    int minimax_m3_local_blocks;
    float dropout_p;
    float rms_eps;
    float logits_softcap;
    uint64_t dropout_seed;
    gd_tensor token_embedding;
    gd_tensor final_norm_w;
    gd_module_list blocks;
    gpt_block *block_items;
    gpt_kv_cache generation_cache;
} gpt_lm;

typedef struct gpt_config {
    const char *data_dir;
    const char *tokenizer_path;
    const char *generate_prompt;
    const char *checkpoint_path;
    const char *latest_checkpoint_path;
    const char *load_checkpoint_path;
    const char *resume_checkpoint_path;
    const char *val_split;
    const char *metrics_dir;
    const char *metrics_project;
    const char *metrics_run_id;
    int epochs;
    int batch_size;
    int n_layers;
    int report_every;
    int eval_every_n_epochs;
    int early_stopping_patience;
    int lr_warmup_steps;
    int max_new_tokens;
    int generate_every_n_steps;
    gpt_architecture architecture;
    int minimax_m3_topk_blocks;
    int minimax_m3_init_blocks;
    int minimax_m3_local_blocks;
    bool epochs_set;
    bool save_best;
    bool save_latest;
    bool metrics_enabled;
    uint64_t overfit_num_samples;
    uint64_t seed;
    float dropout_p;
    float lr_max;
    float lr_min;
    float weight_decay;
    float grad_clip_norm;
    float temperature;
    float min_p;
    float repetition_penalty;
    float logits_softcap;
} gpt_config;

typedef struct gpt_generation_tokenizer {
    gd_tokenizer *tok;
    const char *path;
    char *owned_path;
    int32_t stop_token;
} gpt_generation_tokenizer;

void gpt_fail_status(gd_context *ctx, gd_status st, const char *expr, int line);
double gpt_wall_seconds(void);
size_t gpt_param_count_for_layers(int n_layers);
gd_memory_config gpt_memory_config(const gpt_config *config);
void gpt_lm_init(gd_context *ctx, gpt_lm *model, const gpt_config *config);
void gpt_lm_deinit(gpt_lm *model);
gd_status gpt_lm_forward(gd_context *ctx,
                         gpt_lm *model,
                         const gd_tensor *input_ids,
                         const gd_tensor *target_ids,
                         const gd_tensor *positions,
                         const gd_tensor *cu_seqlens,
                         uint64_t step,
                         gd_tensor *loss);
char *gpt_default_tokenizer_path(const char *data_dir);
void gpt_generation_tokenizer_init(gd_context *ctx,
                                   const gpt_config *config,
                                   gpt_generation_tokenizer *out);
void gpt_generation_tokenizer_deinit(gpt_generation_tokenizer *tokenizer);
int gpt_generate(gd_context *ctx, gpt_lm *model, const gpt_config *config);
int gpt_generate_with_tokenizer(gd_context *ctx,
                                gpt_lm *model,
                                const gpt_config *config,
                                const gpt_generation_tokenizer *tokenizer);
int gpt_generate_vowels(gd_context *ctx,
                        gpt_lm *model,
                        const gpt_config *config,
                        size_t step);
int gpt_generate_vowels_with_tokenizer(gd_context *ctx,
                                       gpt_lm *model,
                                       const gpt_config *config,
                                       size_t step,
                                       const gpt_generation_tokenizer *tokenizer);

#endif /* GPT_LM_SHARED_H */
