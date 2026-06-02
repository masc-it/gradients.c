#include "gradients/gradients.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());        \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s failed\n", #expr);                            \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static const gd_device CPU = {GD_DEVICE_CPU, 0};

static int tensor_f32(gd_context *ctx, int ndim, const int64_t *sizes,
                      const float *data, gd_tensor **out)
{
    gd_tensor_desc desc;
    int64_t n = 1;
    int i;
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, CPU, ndim, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, out));
    for (i = 0; i < ndim; ++i) { n *= sizes[i]; }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float)));
    return 0;
}

static int tensor_i32(gd_context *ctx, int ndim, const int64_t *sizes,
                      const int32_t *data, gd_tensor **out)
{
    gd_tensor_desc desc;
    int64_t n = 1;
    int i;
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, CPU, ndim, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, out));
    for (i = 0; i < ndim; ++i) { n *= sizes[i]; }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(int32_t)));
    return 0;
}

static int test_kv_cache_append_cpu(gd_context *ctx)
{
    int64_t cache_sizes[4] = {1, 5, 1, 2};
    int64_t new_sizes[4] = {1, 2, 1, 2};
    float zeros[10] = {0};
    float k_new[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_new[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    int32_t pos_value = 2;
    gd_tensor *kc = NULL, *vc = NULL, *kn = NULL, *vn = NULL, *pos = NULL;
    gd_graph *g = NULL;
    float got_k[10];
    float got_v[10];

    CHECK_TRUE(tensor_f32(ctx, 4, cache_sizes, zeros, &kc) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, cache_sizes, zeros, &vc) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, new_sizes, k_new, &kn) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, new_sizes, v_new, &vn) == 0);
    CHECK_TRUE(tensor_i32(ctx, 0, NULL, &pos_value, &pos) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_kv_cache_append(ctx, kc, vc, pos, kn, vn));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, kc, got_k, sizeof(got_k)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, vc, got_v, sizeof(got_v)));
    CHECK_TRUE(got_k[4] == 1.0f && got_k[5] == 2.0f && got_k[6] == 3.0f && got_k[7] == 4.0f);
    CHECK_TRUE(got_v[4] == 5.0f && got_v[5] == 6.0f && got_v[6] == 7.0f && got_v[7] == 8.0f);

    gd_graph_destroy(g);
    gd_tensor_release(kc); gd_tensor_release(vc); gd_tensor_release(kn); gd_tensor_release(vn);
    gd_tensor_release(pos);
    return 0;
}

static int test_sdpa_decode_matches_sdpa_cpu(gd_context *ctx)
{
    int64_t q_sizes[4] = {1, 1, 2, 3};
    int64_t full_sizes[4] = {1, 4, 1, 3};
    int64_t cache_sizes[4] = {1, 6, 1, 3};
    float q_data[6] = {0.2f, -0.4f, 0.7f, 0.1f, 0.3f, -0.2f};
    float k_full[12] = {0.1f, 0.0f, 0.2f, 0.3f, -0.2f, 0.4f,
                        0.5f, 0.1f, -0.1f, -0.3f, 0.2f, 0.6f};
    float v_full[12] = {1.0f, 0.5f, -0.2f, 0.3f, -0.7f, 0.4f,
                        -0.1f, 0.8f, 0.2f, 0.6f, 0.1f, -0.5f};
    float k_cache[18] = {0};
    float v_cache[18] = {0};
    int32_t pos_value = 3;
    gd_tensor *q = NULL, *kf = NULL, *vf = NULL, *kc = NULL, *vc = NULL, *pos = NULL;
    gd_tensor *y_full = NULL, *y_dec = NULL;
    gd_graph *g = NULL;
    gd_sdpa_config cfg = {0.0f, true, 0, 0};
    gd_sdpa_decode_config dcfg = {0.0f, 0, 0};
    float full[6];
    float dec[6];
    int i;

    memcpy(k_cache, k_full, sizeof(k_full));
    memcpy(v_cache, v_full, sizeof(v_full));
    CHECK_TRUE(tensor_f32(ctx, 4, q_sizes, q_data, &q) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, full_sizes, k_full, &kf) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, full_sizes, v_full, &vf) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, cache_sizes, k_cache, &kc) == 0);
    CHECK_TRUE(tensor_f32(ctx, 4, cache_sizes, v_cache, &vc) == 0);
    CHECK_TRUE(tensor_i32(ctx, 0, NULL, &pos_value, &pos) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa(ctx, q, kf, vf, NULL, &cfg, &y_full));
    CHECK_OK(gd_sdpa_decode(ctx, q, kc, vc, pos, &dcfg, &y_dec));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y_full, full, sizeof(full)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y_dec, dec, sizeof(dec)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(fabsf(full[i] - dec[i]) < 1.0e-5f);
    }

    gd_graph_destroy(g);
    gd_tensor_release(q); gd_tensor_release(kf); gd_tensor_release(vf);
    gd_tensor_release(kc); gd_tensor_release(vc); gd_tensor_release(pos);
    gd_tensor_release(y_full); gd_tensor_release(y_dec);
    return 0;
}

static int test_gpt_kv_decode_matches_full_cpu(gd_context *ctx)
{
    enum { B = 1, T = 4, V = 12 };
    gd_gpt_config cfg;
    gd_gpt *gpt = NULL;
    gd_kv_cache *cache = NULL;
    int64_t tok_sizes[2] = {B, T};
    int64_t step_sizes[2] = {B, 1};
    int32_t toks_data[T] = {1, 5, 2, 7};
    int32_t pos_data[T] = {0, 1, 2, 3};
    gd_tensor *tokens = NULL, *positions = NULL, *full_logits = NULL;
    gd_graph *g = NULL;
    float full[T * V];
    int t;

    memset(&cfg, 0, sizeof(cfg));
    cfg.vocab_size = V;
    cfg.d_model = 8;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 16;
    cfg.max_seq_len = T;
    cfg.mlp_kind = GD_GPT_MLP_POWLU;
    cfg.param_dtype = GD_DTYPE_F32;
    cfg.tie_embeddings = true;
    CHECK_OK(gd_gpt_create(ctx, &cfg, UINT64_C(123), &gpt));
    gd_gpt_set_training(gpt, false);
    CHECK_TRUE(tensor_i32(ctx, 2, tok_sizes, toks_data, &tokens) == 0);
    CHECK_TRUE(tensor_i32(ctx, 2, tok_sizes, pos_data, &positions) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_forward(ctx, gpt, tokens, positions, &full_logits));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, full_logits, full, sizeof(full)));
    gd_graph_destroy(g);
    gd_tensor_release(full_logits);
    gd_tensor_release(tokens);
    gd_tensor_release(positions);

    CHECK_OK(gd_kv_cache_create(ctx, &cfg, B, T, CPU, &cache));
    for (t = 0; t < T; ++t) {
        gd_tensor *step_tok = NULL;
        gd_tensor *step_pos = NULL;
        gd_tensor *emb = NULL;
        gd_tensor *hidden = NULL;
        gd_tensor *logits = NULL;
        gd_graph *step_g = NULL;
        int32_t tok1[1] = {toks_data[t]};
        int32_t pos1[1] = {pos_data[t]};
        float cached[V];
        int v;

        CHECK_TRUE(tensor_i32(ctx, 2, step_sizes, tok1, &step_tok) == 0);
        CHECK_TRUE(tensor_i32(ctx, 2, step_sizes, pos1, &step_pos) == 0);
        CHECK_OK(gd_kv_cache_set_len(ctx, cache, t));
        CHECK_OK(gd_graph_create(ctx, &step_g));
        CHECK_OK(gd_graph_begin(ctx, step_g));
        CHECK_OK(gd_gpt_embed_tokens(ctx, gpt, step_tok, &emb));
        if (t == 0) {
            CHECK_OK(gd_gpt_kv_prefill_embeds(ctx, gpt, cache, emb, step_pos, NULL, &hidden));
        } else {
            CHECK_OK(gd_gpt_kv_decode_step_embeds(ctx, gpt, cache, emb, step_pos, NULL, &hidden));
        }
        CHECK_OK(gd_gpt_logits(ctx, gpt, hidden, &logits));
        CHECK_OK(gd_graph_end(ctx));
        CHECK_OK(gd_graph_compile(step_g, CPU));
        CHECK_OK(gd_graph_run(step_g));
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, logits, cached, sizeof(cached)));
        for (v = 0; v < V; ++v) {
            float a = full[t * V + v];
            float b = cached[v];
            CHECK_TRUE(fabsf(a - b) < 2.0e-5f);
        }
        CHECK_OK(gd_kv_cache_set_len(ctx, cache, t + 1));
        gd_graph_destroy(step_g);
        gd_tensor_release(step_tok);
        gd_tensor_release(step_pos);
        gd_tensor_release(emb);
        gd_tensor_release(hidden);
        gd_tensor_release(logits);
    }
    gd_kv_cache_destroy(cache);
    gd_gpt_destroy(gpt);
    return 0;
}

static int test_gpt_kv_decode_prefix_matches_full_cpu(gd_context *ctx)
{
    enum { B = 1, T = 4, P = 2, PRE = 3, V = 12 };
    gd_gpt_config cfg;
    gd_gpt_forward_config fwd;
    gd_gpt *gpt = NULL;
    gd_kv_cache *cache = NULL;
    int64_t tok_sizes[2] = {B, T};
    int64_t pre_sizes[2] = {B, PRE};
    int64_t step_sizes[2] = {B, 1};
    int32_t toks_data[T] = {1, 5, 2, 7};
    int32_t pos_data[T] = {0, 1, 2, 3};
    int32_t pre_toks[PRE] = {1, 5, 2};
    int32_t pre_pos[PRE] = {0, 1, 2};
    int32_t dec_tok[1] = {7};
    int32_t dec_pos[1] = {3};
    gd_tensor *tokens = NULL, *positions = NULL, *emb = NULL, *full_logits = NULL;
    gd_tensor *pre_tokens = NULL, *pre_positions = NULL, *pre_emb = NULL, *pre_hidden = NULL;
    gd_tensor *pre_logits = NULL, *dec_tokens = NULL, *dec_positions = NULL, *dec_emb = NULL;
    gd_tensor *dec_hidden = NULL, *dec_logits = NULL;
    gd_graph *g = NULL;
    float full[T * V];
    float pre[PRE * V];
    float dec[V];
    int v;

    memset(&cfg, 0, sizeof(cfg));
    cfg.vocab_size = V;
    cfg.d_model = 8;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 4;
    cfg.d_ff = 16;
    cfg.max_seq_len = T;
    cfg.mlp_kind = GD_GPT_MLP_POWLU;
    cfg.param_dtype = GD_DTYPE_F32;
    cfg.tie_embeddings = true;
    memset(&fwd, 0, sizeof(fwd));
    fwd.prefix_len = P;

    CHECK_OK(gd_gpt_create(ctx, &cfg, UINT64_C(456), &gpt));
    gd_gpt_set_training(gpt, false);
    CHECK_TRUE(tensor_i32(ctx, 2, tok_sizes, toks_data, &tokens) == 0);
    CHECK_TRUE(tensor_i32(ctx, 2, tok_sizes, pos_data, &positions) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_embed_tokens(ctx, gpt, tokens, &emb));
    CHECK_OK(gd_gpt_forward_embeds(ctx, gpt, emb, positions, &fwd, &full_logits));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, full_logits, full, sizeof(full)));
    gd_graph_destroy(g);
    g = NULL;
    gd_tensor_release(tokens); gd_tensor_release(positions); gd_tensor_release(emb);
    gd_tensor_release(full_logits);

    CHECK_OK(gd_kv_cache_create(ctx, &cfg, B, T, CPU, &cache));
    CHECK_OK(gd_kv_cache_reset(ctx, cache));
    CHECK_TRUE(tensor_i32(ctx, 2, pre_sizes, pre_toks, &pre_tokens) == 0);
    CHECK_TRUE(tensor_i32(ctx, 2, pre_sizes, pre_pos, &pre_positions) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_embed_tokens(ctx, gpt, pre_tokens, &pre_emb));
    CHECK_OK(gd_gpt_kv_prefill_embeds(ctx, gpt, cache, pre_emb, pre_positions, &fwd,
                                      &pre_hidden));
    CHECK_OK(gd_gpt_logits(ctx, gpt, pre_hidden, &pre_logits));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, pre_logits, pre, sizeof(pre)));
    gd_graph_destroy(g);
    g = NULL;
    for (v = 0; v < V; ++v) {
        CHECK_TRUE(fabsf(full[(PRE - 1) * V + v] - pre[(PRE - 1) * V + v]) < 2.0e-5f);
    }
    CHECK_OK(gd_kv_cache_set_len(ctx, cache, PRE));

    CHECK_TRUE(tensor_i32(ctx, 2, step_sizes, dec_tok, &dec_tokens) == 0);
    CHECK_TRUE(tensor_i32(ctx, 2, step_sizes, dec_pos, &dec_positions) == 0);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_gpt_embed_tokens(ctx, gpt, dec_tokens, &dec_emb));
    CHECK_OK(gd_gpt_kv_decode_step_embeds(ctx, gpt, cache, dec_emb, dec_positions, &fwd,
                                          &dec_hidden));
    CHECK_OK(gd_gpt_logits(ctx, gpt, dec_hidden, &dec_logits));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, dec_logits, dec, sizeof(dec)));
    for (v = 0; v < V; ++v) {
        CHECK_TRUE(fabsf(full[(T - 1) * V + v] - dec[v]) < 2.0e-5f);
    }

    gd_graph_destroy(g);
    gd_tensor_release(pre_tokens); gd_tensor_release(pre_positions); gd_tensor_release(pre_emb);
    gd_tensor_release(pre_hidden); gd_tensor_release(pre_logits);
    gd_tensor_release(dec_tokens); gd_tensor_release(dec_positions); gd_tensor_release(dec_emb);
    gd_tensor_release(dec_hidden); gd_tensor_release(dec_logits);
    gd_kv_cache_destroy(cache);
    gd_gpt_destroy(gpt);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    CHECK_OK(gd_context_create(&ctx));
    if (test_kv_cache_append_cpu(ctx) != 0 ||
        test_sdpa_decode_matches_sdpa_cpu(ctx) != 0 ||
        test_gpt_kv_decode_matches_full_cpu(ctx) != 0 ||
        test_gpt_kv_decode_prefix_matches_full_cpu(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    printf("test_kv_cache: ok\n");
    return 0;
}
