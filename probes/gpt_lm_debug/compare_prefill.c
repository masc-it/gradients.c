#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../examples/gpt_lm/gpt_lm_shared.h"
#define static
#include "../../examples/gpt_lm/gpt_lm_shared.c"
#undef static

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    long n;
    char *s;
    if (!f) { perror(path); exit(2); }
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    s = (char *)malloc((size_t)n + 1U); if (!s) exit(2);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { perror("fread"); exit(2); }
    fclose(f); s[n] = '\0'; return s;
}

static gd_status full_logits(gd_context *ctx, gpt_lm *model, const gd_tensor *ids, const gd_tensor *pos, const gd_tensor *cu, gd_tensor *logits) {
    gd_tensor x, block_out, final_norm;
    gd_status st;
    uint32_t i;
    st = gd_embedding(ctx, &model->token_embedding, ids, &x); if (st != GD_OK) return st;
    for (i = 0; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_forward(ctx, model, &model->block_items[i], i, &x, pos, cu, 0U, &block_out);
        if (st != GD_OK) return st;
        x = block_out;
    }
    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm); if (st != GD_OK) return st;
    return gd_linear_transposed_weight(ctx, &final_norm, &model->lm_head, &model->lm_head_bias, logits);
}

int main(int argc, char **argv) {
    const char *ckpt = argc > 1 ? argv[1] : "examples/gpt_lm/checkpoints/gpt_lm_latest.gdckpt";
    const char *tok_path = argc > 2 ? argv[2] : "examples/gpt_lm/data/tokenizer-v2048.json";
    const char *prompt_path = argc > 3 ? argv[3] : "probes/gpt_lm_debug/prefix_ndrangheta_special_long.txt";
    char *prompt = read_file(prompt_path);
    gpt_config cfg; memset(&cfg, 0, sizeof(cfg));
    cfg.data_dir = "examples/gpt_lm/data"; cfg.tokenizer_path = tok_path; cfg.generate_prompt = prompt; cfg.checkpoint_path = ckpt;
    cfg.n_layers = 3; cfg.architecture = GPT_ARCH_MINIMAX_M3; cfg.minimax_m3_topk_blocks = GPT_MINIMAX_M3_TOPK_BLOCKS; cfg.minimax_m3_init_blocks = GPT_MINIMAX_M3_INIT_BLOCKS; cfg.minimax_m3_local_blocks = GPT_MINIMAX_M3_LOCAL_BLOCKS;
    cfg.batch_size = GPT_DEFAULT_BATCH_SIZE; cfg.max_new_tokens = 8; cfg.dropout_p = GPT_DEFAULT_DROPOUT_P; cfg.logits_softcap = 20.0f; cfg.seed = GPT_DEFAULT_SEED;
    gd_memory_config mem = gpt_memory_config(&cfg);
    gd_context *ctx = NULL; gd_status st = gd_context_create(&mem, &ctx); if (st != GD_OK) { fprintf(stderr, "ctx %s\n", gd_status_string(st)); return 1; }
    gpt_lm model; gpt_lm_init(ctx, &model, &cfg);
    gd_module_load_options lo; lo.strict = true; lo.load_buffers = true; TRY(ctx, gd_module_load_state(ctx, &model.mod, ckpt, &lo));
    TRY(ctx, gd_context_seal_params(ctx)); gd_module_set_training(&model.mod, false);
    gd_tokenizer_config tc; tc.split_digits = 1; tc.allow_special = 1; gd_tokenizer *tok = NULL; TRY(ctx, gd_bpe_tokenizer_load(tok_path, &tc, &tok));
    int32_t *ids = NULL; int n = 0; TRY(ctx, gd_tokenizer_encode(tok, prompt, &ids, &n));
    if (n <= 0 || n > GPT_CONTEXT_LENGTH) { fprintf(stderr, "bad n=%d\n", n); return 2; }
    int32_t *pos = calloc((size_t)n, sizeof(int32_t)); int32_t cuv[2] = {0, n}; int32_t cp[1] = {0};
    for (int i=0;i<n;i++) pos[i]=i;
    gpt_kv_cache *cache = NULL; TRY(ctx, gpt_lm_prepare_generation_cache(ctx, &model, &cfg, 1, GPT_CONTEXT_LENGTH, &cache));
    gd_tensor ids_t,pos_t,cu_t,cp_t,logits_cached,logits_full;
    TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
    TRY(ctx, gpt_data_i32_tensor(ctx, ids, n, &ids_t));
    TRY(ctx, gpt_data_i32_tensor(ctx, pos, n, &pos_t));
    TRY(ctx, gpt_data_i32_tensor(ctx, cuv, 2, &cu_t));
    TRY(ctx, gpt_data_i32_tensor(ctx, cp, 1, &cp_t));
    TRY(ctx, gpt_lm_prefill_logits(ctx, &model, cache, &ids_t, &pos_t, &cu_t, &cp_t, &logits_cached));
    TRY(ctx, full_logits(ctx, &model, &ids_t, &pos_t, &cu_t, &logits_full));
    TRY(ctx, gd_end_step(ctx));
    size_t count = (size_t)n * (size_t)GPT_VOCAB_SIZE;
    float *a = malloc(count * sizeof(float)); float *b = malloc(count * sizeof(float));
    TRY(ctx, gd_tensor_read_f32(ctx, &logits_cached, a, count));
    TRY(ctx, gd_tensor_read_f32(ctx, &logits_full, b, count));
    double max_abs=0, mean_abs=0, max_rel=0; size_t arg=0;
    for (size_t i=0;i<count;i++) { double d=fabs((double)a[i]-(double)b[i]); mean_abs += d; if (d>max_abs){max_abs=d;arg=i;} double denom=fmax(1.0, fabs((double)b[i])); if (d/denom>max_rel) max_rel=d/denom; }
    mean_abs /= (double)count;
    printf("n=%d count=%zu max_abs=%.9g mean_abs=%.9g max_rel=%.9g arg_row=%zu arg_vocab=%zu cached=%.9g full=%.9g\n", n, count, max_abs, mean_abs, max_rel, arg/(size_t)GPT_VOCAB_SIZE, arg%(size_t)GPT_VOCAB_SIZE, (double)a[arg], (double)b[arg]);
    gd_tokenizer_free(ids); gd_tokenizer_destroy(tok); free(pos); free(prompt); free(a); free(b); gpt_lm_deinit(&model); gd_context_destroy(ctx); return 0;
}
