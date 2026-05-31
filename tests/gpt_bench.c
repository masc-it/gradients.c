/*
 * GPT profiling/benchmark harness (not a correctness test).
 *
 * Builds a configurable decoder-only transformer, runs a fixed number of
 * train (or forward-only) steps on the chosen device, and reports parameters,
 * wall time per step, and an estimated GFLOP/s. Intended for backend perf work
 * (Metal GPU_SAFE) without waiting on the CPU reference for long runs.
 *
 * It is intentionally excluded from `make check`; run it via `make gpt-bench`.
 *
 * Environment knobs (all optional):
 *   GD_DEVICE=metal           train on the GPU (default CPU; metal falls back to
 *                             CPU if unavailable)
 *   GD_BENCH_MODE=train|fwd   training step (fwd+bwd+AdamW) or forward only
 *   GD_BENCH_ITERS=N          measured iterations (default 20)
 *   GD_BENCH_WARMUP=N         warmup iterations excluded from timing (default 3)
 *   GD_BENCH_B / GD_BENCH_T   batch / sequence length (default 1 / 128)
 *   GD_BENCH_VOCAB            vocab size (default 8000)
 *   GD_BENCH_DMODEL           model width (default 320)
 *   GD_BENCH_LAYERS           number of blocks (default 6)
 *   GD_BENCH_HEADS            query heads (default 5)
 *   GD_BENCH_KV_HEADS         key/value heads (default = heads)
 *   GD_BENCH_HEAD_DIM         per-head dim (default 64; d_model==heads*head_dim)
 *   GD_BENCH_DFF              MLP hidden size (default 1280)
 *
 * Defaults describe a ~12M-parameter model. GD_PROFILE=summary works as usual.
 */

#include "gradients/gradients.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CHECK(expr)                                                              \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    return (int)strtol(v, NULL, 10);
}

static double now_ms(void)
{
    struct timeval tv;
    (void)gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

/* Counts elements across `params`. Note: gd_gpt_parameters rebuilds the module's
 * flat array on each call (invalidating prior pointers), so callers must reuse a
 * single fetched list rather than calling it again. */
static long count_params(gd_tensor **params, int n)
{
    long total = 0;
    int i = 0;

    for (i = 0; i < n; ++i) {
        long k = 1;
        int d = 0;
        for (d = 0; d < gd_tensor_ndim(params[i]); ++d) {
            k *= (long)gd_tensor_size(params[i], d);
        }
        total += k;
    }
    return total;
}

/* Forward-pass GEMM/attention FLOPs (multiply-adds counted as 2 flops). */
static double forward_gflops(const gd_gpt_config *c, int B, int T)
{
    double d = (double)c->d_model;
    double Hq = (double)c->n_heads;
    double Hkv = (double)c->n_kv_heads;
    double Dh = (double)c->head_dim;
    double dff = (double)c->d_ff;
    double V = (double)c->vocab_size;
    double bt = (double)B * (double)T;
    double per_layer = 0.0;
    double attn_proj = 0.0;
    double attn_score = 0.0;
    double mlp = 0.0;
    double head = 0.0;

    /* q,k,v,out projections. */
    attn_proj = 2.0 * bt * d * (Hq * Dh)        /* q */
              + 2.0 * bt * d * (Hkv * Dh)       /* k */
              + 2.0 * bt * d * (Hkv * Dh)       /* v */
              + 2.0 * bt * (Hq * Dh) * d;       /* out proj */
    /* QK^T and P*V. */
    attn_score = 4.0 * (double)B * Hq * (double)T * (double)T * Dh;
    /* SwiGLU: gate, up, down. */
    mlp = 6.0 * bt * d * dff;
    per_layer = attn_proj + attn_score + mlp;
    head = 2.0 * bt * d * V;

    return ((double)c->n_layers * per_layer + head) / 1e9;
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");
    const char *mode_env = getenv("GD_BENCH_MODE");
    int train_mode = (mode_env == NULL) || (strcmp(mode_env, "fwd") != 0);

    int B = env_int("GD_BENCH_B", 1);
    int T = env_int("GD_BENCH_T", 128);
    int iters = env_int("GD_BENCH_ITERS", 20);
    int warmup = env_int("GD_BENCH_WARMUP", 3);

    gd_gpt_config cfg = {0};
    gd_gpt *model = NULL;
    gd_optimizer *opt = NULL;
    gd_tensor **params = NULL;
    int n_params = 0;
    long total_params = 0;

    gd_tensor *tokens = NULL;
    gd_tensor *positions = NULL;
    gd_tensor *targets = NULL;
    gd_tensor_desc idesc;
    int64_t shape[2];
    int32_t *tok = NULL;
    int32_t *pos = NULL;
    int32_t *tgt = NULL;

    gd_graph *g = NULL;
    gd_tensor *logits = NULL;
    gd_tensor *loss = NULL;
    gd_adamw_config acfg = {0};
    int i = 0;
    double ms_per_iter = 0.0;
    double fwd_gflop = 0.0;
    double step_gflop = 0.0;

    /* Unbuffered stdout: this harness is usually run through `make` (a pipe),
     * where libc would otherwise fully buffer output and show nothing until the
     * process exits, even though setup/iteration progress is being printed. */
    setvbuf(stdout, NULL, _IONBF, 0);

    CHECK(gd_context_create(&ctx));
    if (dev_env != NULL && strcmp(dev_env, "metal") == 0) {
        gd_device metal = {GD_DEVICE_METAL, 0};
        if (gd_synchronize(ctx, metal) == GD_OK) {
            target = metal;
        } else {
            printf("device: cpu (metal unavailable)\n");
        }
    }
    CHECK(gd_context_set_default_device(ctx, target));

    cfg.vocab_size = env_int("GD_BENCH_VOCAB", 8000);
    cfg.d_model = env_int("GD_BENCH_DMODEL", 320);
    cfg.n_layers = env_int("GD_BENCH_LAYERS", 6);
    cfg.n_heads = env_int("GD_BENCH_HEADS", 5);
    cfg.n_kv_heads = env_int("GD_BENCH_KV_HEADS", cfg.n_heads);
    cfg.head_dim = env_int("GD_BENCH_HEAD_DIM", 64);
    cfg.d_ff = env_int("GD_BENCH_DFF", 1280);
    cfg.max_seq_len = T;
    cfg.rope_theta = 10000.0F;
    cfg.norm_eps = 1e-5F;
    cfg.mlp_kind = GD_GPT_MLP_SWIGLU;
    cfg.tie_embeddings = true;

    if (cfg.d_model != cfg.n_heads * cfg.head_dim) {
        fprintf(stderr, "config error: d_model (%d) must equal n_heads*head_dim (%d*%d)\n",
                cfg.d_model, cfg.n_heads, cfg.head_dim);
        gd_context_destroy(ctx);
        return 1;
    }

    /* Deterministic dummy data; correctness is not the goal here. */
    shape[0] = B;
    shape[1] = T;
    tok = malloc((size_t)B * (size_t)T * sizeof(int32_t));
    pos = malloc((size_t)B * (size_t)T * sizeof(int32_t));
    tgt = malloc((size_t)B * (size_t)T * sizeof(int32_t));
    if (tok == NULL || pos == NULL || tgt == NULL) {
        fprintf(stderr, "alloc failed\n");
        free(tok);
        free(pos);
        free(tgt);
        gd_context_destroy(ctx);
        return 1;
    }
    for (i = 0; i < B * T; ++i) {
        tok[i] = i % cfg.vocab_size;
        pos[i] = i % T;
        tgt[i] = (i + 1) % cfg.vocab_size;
    }

    CHECK(gd_tensor_desc_contiguous(GD_DTYPE_I32, target, 2, shape, &idesc));
    CHECK(gd_tensor_empty(ctx, &idesc, &tokens));
    CHECK(gd_tensor_copy_from_cpu(ctx, tokens, tok, (size_t)B * (size_t)T * sizeof(int32_t)));
    CHECK(gd_tensor_empty(ctx, &idesc, &positions));
    CHECK(gd_tensor_copy_from_cpu(ctx, positions, pos, (size_t)B * (size_t)T * sizeof(int32_t)));
    CHECK(gd_tensor_empty(ctx, &idesc, &targets));
    CHECK(gd_tensor_copy_from_cpu(ctx, targets, tgt, (size_t)B * (size_t)T * sizeof(int32_t)));

    CHECK(gd_gpt_create(ctx, &cfg, 0xBEEFu, &model));
    CHECK(gd_gpt_parameters(model, &params, &n_params));
    total_params = count_params(params, n_params);

    fwd_gflop = forward_gflops(&cfg, B, T);
    step_gflop = train_mode ? 3.0 * fwd_gflop : fwd_gflop;

    /* Print the setup up front so a long/slow run is observable immediately. */
    printf("gpt_bench\n");
    printf("  device      : %s\n", target.type == GD_DEVICE_METAL ? "metal" : "cpu");
    printf("  mode        : %s\n", train_mode ? "train (fwd+bwd+adamw)" : "forward");
    printf("  params      : %ld (%.2fM)\n", total_params, (double)total_params / 1e6);
    printf("  config      : d_model=%d layers=%d heads=%d kv_heads=%d head_dim=%d d_ff=%d vocab=%d\n",
           cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.d_ff,
           cfg.vocab_size);
    printf("  batch x seq : %d x %d\n", B, T);
    printf("  iters       : %d (warmup %d)\n", iters, warmup);
    printf("  fwd GFLOP   : %.3f   step GFLOP: %.3f\n", fwd_gflop, step_gflop);

    if (train_mode) {
        acfg.lr = 0.001F;
        acfg.beta1 = 0.9F;
        acfg.beta2 = 0.999F;
        acfg.eps = 1e-8F;
        CHECK(gd_adamw_create(ctx, params, n_params, &acfg, &opt));
    }

    CHECK(gd_graph_create(ctx, &g));
    CHECK(gd_graph_begin(ctx, g));
    CHECK(gd_gpt_forward(ctx, model, tokens, positions, &logits));
    if (train_mode) {
        CHECK(gd_cross_entropy(ctx, logits, targets, 2, &loss));
        CHECK(gd_backward(ctx, loss));
        CHECK(gd_optimizer_step(ctx, opt));
    }
    CHECK(gd_graph_end(ctx));
    printf("  compiling graph...\n");
    CHECK(gd_graph_compile(g, target));

    /* Warmup absorbs compile/autorelease/first-run/pipeline costs; sync so it is
     * fully excluded from the measured window. */
    printf("  warmup (%d iters)...\n", warmup);
    for (i = 0; i < warmup; ++i) {
        CHECK(gd_graph_run(g));
    }
    CHECK(gd_synchronize(ctx, target));

    /* Measure each iteration with its own sync so timing is per-step, the GPU
     * queue cannot grow unboundedly, and progress is visible live. */
    {
        double best_ms = 0.0;
        double total_ms = 0.0;

        for (i = 0; i < iters; ++i) {
            double a = now_ms();
            double b = 0.0;
            CHECK(gd_graph_run(g));
            CHECK(gd_synchronize(ctx, target));
            b = now_ms();
            {
                double iter_ms = b - a;
                total_ms += iter_ms;
                if (i == 0 || iter_ms < best_ms) {
                    best_ms = iter_ms;
                }
                printf("  iter %3d/%d  %.3f ms\n", i + 1, iters, iter_ms);
            }
        }
        ms_per_iter = iters > 0 ? total_ms / (double)iters : 0.0;
        printf("  ms/iter     : %.3f (mean)  %.3f (best)\n", ms_per_iter, best_ms);
        if (best_ms > 0.0) {
            printf("  throughput  : %.1f GFLOP/s (mean)  %.1f GFLOP/s (best)\n",
                   step_gflop / (ms_per_iter / 1000.0), step_gflop / (best_ms / 1000.0));
            printf("  tokens/s    : %.1f (best)\n",
                   (double)B * (double)T / (best_ms / 1000.0));
        }
    }

    gd_tensor_release(logits);
    gd_tensor_release(loss);
    CHECK(gd_graph_reset(g));
    CHECK(gd_graph_destroy(g));
    if (opt != NULL) {
        gd_optimizer_destroy(opt);
    }
    gd_gpt_destroy(model);
    gd_tensor_release(tokens);
    gd_tensor_release(positions);
    gd_tensor_release(targets);
    free(tok);
    free(pos);
    free(tgt);
    gd_context_destroy(ctx);
    return 0;
}
