/*
 * Tiny GPT training demo with gradients.c.
 *
 * Trains a small decoder-only transformer (embedding -> [RMSNorm -> GQA
 * attention with RoPE -> RMSNorm -> SwiGLU MLP] x L -> RMSNorm -> tied LM head)
 * to predict the next token of a fixed sequence, then reports per-position
 * accuracy via a forward pass.
 *
 * Device: set GD_DEVICE=metal to train on the GPU (falls back to CPU if Metal
 * is unavailable); defaults to CPU.
 */

#include "gradients/gradients.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 32
#define SEQ 16
#define N_STEPS 600

#define CHECK(expr)                                                              \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                            \
        }                                                                        \
    } while (0)

int main(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");

    gd_gpt *model = NULL;
    gd_gpt_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor **params = NULL;
    int n_params = 0;

    gd_tensor *tokens = NULL;
    gd_tensor *positions = NULL;
    gd_tensor *targets = NULL;
    gd_tensor_desc idesc;
    int64_t shape[2] = {1, SEQ};

    int32_t tok[SEQ];
    int32_t tgt[SEQ];
    int32_t pos[SEQ];
    float logits[SEQ * VOCAB];

    gd_graph *train = NULL;
    gd_graph *eval = NULL;
    gd_tensor *train_logits = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *eval_logits = NULL;
    gd_adamw_config acfg = {0};
    int step = 0;
    int i = 0;
    int correct = 0;
    float loss_val = 0.0F;

    CHECK(gd_context_create(&ctx));
    if (dev_env != NULL && strcmp(dev_env, "metal") == 0) {
        gd_device metal = {GD_DEVICE_METAL, 0};
        if (gd_synchronize(ctx, metal) == GD_OK) {
            target = metal;
            printf("device: metal\n");
        } else {
            printf("device: cpu (metal unavailable)\n");
        }
    } else {
        printf("device: cpu\n");
    }
    CHECK(gd_context_set_default_device(ctx, target));

    /* Fixed sequence; target is the next token (last wraps to the first). */
    srand(7u);
    for (i = 0; i < SEQ; ++i) {
        tok[i] = rand() % VOCAB;
        pos[i] = i;
    }
    for (i = 0; i < SEQ; ++i) {
        tgt[i] = tok[(i + 1) % SEQ];
    }

    CHECK(gd_tensor_desc_contiguous(GD_DTYPE_I32, target, 2, shape, &idesc));
    CHECK(gd_tensor_empty(ctx, &idesc, &tokens));
    CHECK(gd_tensor_copy_from_cpu(ctx, tokens, tok, sizeof(tok)));
    CHECK(gd_tensor_empty(ctx, &idesc, &positions));
    CHECK(gd_tensor_copy_from_cpu(ctx, positions, pos, sizeof(pos)));
    CHECK(gd_tensor_empty(ctx, &idesc, &targets));
    CHECK(gd_tensor_copy_from_cpu(ctx, targets, tgt, sizeof(tgt)));

    cfg.vocab_size = VOCAB;
    cfg.d_model = 32;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2; /* grouped-query attention */
    cfg.head_dim = 8;   /* 4 * 8 == 32 */
    cfg.d_ff = 64;
    cfg.max_seq_len = SEQ;
    cfg.rope_theta = 10000.0F;
    cfg.norm_eps = 1e-5F;
    cfg.mlp_kind = GD_GPT_MLP_SWIGLU;
    cfg.tie_embeddings = true;
    CHECK(gd_gpt_create(ctx, &cfg, 0xABCDEFu, &model));
    CHECK(gd_gpt_parameters(model, &params, &n_params));

    acfg.lr = 0.02F;
    acfg.beta1 = 0.9F;
    acfg.beta2 = 0.999F;
    acfg.eps = 1e-8F;
    acfg.weight_decay = 0.0F;
    CHECK(gd_adamw_create(ctx, params, n_params, &acfg, &opt));

    /* Build the train-step graph once (forward + cross-entropy + backward +
     * AdamW), reused every step. */
    CHECK(gd_graph_create(ctx, &train));
    CHECK(gd_graph_begin(ctx, train));
    CHECK(gd_gpt_forward(ctx, model, tokens, positions, &train_logits));
    CHECK(gd_cross_entropy(ctx, train_logits, targets, 2, &loss));
    CHECK(gd_backward(ctx, loss));
    CHECK(gd_optimizer_step(ctx, opt));
    CHECK(gd_graph_end(ctx));
    CHECK(gd_graph_compile(train, target));

    for (step = 1; step <= N_STEPS; ++step) {
        CHECK(gd_graph_run(train));
        if (step == 1 || step % 100 == 0) {
            CHECK(gd_tensor_copy_to_cpu(ctx, loss, &loss_val, sizeof(loss_val)));
            printf("step %4d  loss %.6f\n", step, (double)loss_val);
        }
    }

    gd_tensor_release(train_logits);
    gd_tensor_release(loss);
    CHECK(gd_graph_reset(train));
    CHECK(gd_graph_destroy(train));

    /* Eval: forward pass with the trained weights, argmax next-token accuracy. */
    CHECK(gd_graph_create(ctx, &eval));
    CHECK(gd_graph_begin(ctx, eval));
    CHECK(gd_gpt_forward(ctx, model, tokens, positions, &eval_logits));
    CHECK(gd_graph_end(ctx));
    CHECK(gd_graph_compile(eval, target));
    CHECK(gd_graph_run(eval));
    CHECK(gd_tensor_copy_to_cpu(ctx, eval_logits, logits, sizeof(logits)));

    for (i = 0; i < SEQ; ++i) {
        int best = 0;
        int c = 0;
        for (c = 1; c < VOCAB; ++c) {
            if (logits[i * VOCAB + c] > logits[i * VOCAB + best]) {
                best = c;
            }
        }
        if (best == tgt[i]) {
            correct += 1;
        }
    }
    printf("next-token accuracy %d/%d\n", correct, SEQ);

    gd_tensor_release(eval_logits);
    CHECK(gd_graph_reset(eval));
    CHECK(gd_graph_destroy(eval));

    gd_optimizer_destroy(opt);
    gd_gpt_destroy(model);
    gd_tensor_release(tokens);
    gd_tensor_release(positions);
    gd_tensor_release(targets);
    gd_context_destroy(ctx);
    return correct == SEQ ? 0 : 1;
}
