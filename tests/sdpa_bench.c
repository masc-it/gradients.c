/*
 * Raw SDPA profiling/benchmark harness (not a correctness test).
 *
 * Builds a single sdpa graph over q/k/v [B,T,H,Dh] and reports wall time plus
 * attention-pair math. Intended for prefix-causal VLM mask work without the rest
 * of a transformer in the profile.
 *
 * Environment knobs (all optional):
 *   GD_DEVICE=metal                 run on Metal when available (default CPU)
 *   GD_BENCH_MODE=fwd|train         forward only or fwd+bwd over q/k/v (default fwd)
 *   GD_BENCH_ITERS=N                measured iterations (default 20)
 *   GD_BENCH_WARMUP=N               warmup iterations (default 3)
 *   GD_BENCH_B / GD_BENCH_T         batch / total sequence (default 1 / 1024)
 *   GD_BENCH_HEADS / GD_BENCH_KV_HEADS / GD_BENCH_HEAD_DIM (default 8 / heads / 64)
 *   GD_BENCH_DTYPE=f32|f16          q/k/v dtype (default f32)
 *   GD_BENCH_MASK=dense|causal|window|prefix|prefix_window (default prefix_window)
 *   GD_BENCH_ATTN_WINDOW=N          suffix sliding-window size (default 128)
 *   GD_BENCH_PREFIX=N               override prefix/image token count
 *   GD_BENCH_IMAGE_SIZE=N           image side in pixels for prefix math (default 224)
 *   GD_BENCH_PATCH=N                square patch size in pixels (default 16)
 *   GD_BENCH_IMAGE_EXTRA_TOKENS=N   optional CLS/register tokens added to patches (default 0)
 *   GD_BENCH_TEXT=N                 text/suffix length; when set and T unset, T=prefix+text
 */

#include "gradients/gradients.h"

#include <stdbool.h>
#include <stdint.h>
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

static int env_present(const char *name)
{
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0';
}

static int env_int(const char *name, int fallback)
{
    const char *v = getenv(name);
    if (v == NULL || v[0] == '\0') {
        return fallback;
    }
    return (int)strtol(v, NULL, 10);
}

static int env_window(int fallback)
{
    if (env_present("GD_BENCH_ATTN_WINDOW")) {
        return env_int("GD_BENCH_ATTN_WINDOW", fallback);
    }
    return env_int("GD_BENCH_WINDOW", fallback);
}

static double now_ms(void)
{
    struct timeval tv;
    (void)gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static uint16_t f32_to_f16(float x)
{
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t sign = 0U;
    uint32_t mant = 0U;
    int exp = 0;

    v.f = x;
    sign = (v.u >> 16) & 0x8000U;
    exp = (int)((v.u >> 23) & 0xffU) - 127 + 15;
    mant = v.u & 0x7fffffU;

    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;
        }
        mant = (mant | 0x800000U) >> (uint32_t)(1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000U) >> 13));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000U) >> 13));
}

static int64_t numel4(int a, int b, int c, int d)
{
    return (int64_t)a * (int64_t)b * (int64_t)c * (int64_t)d;
}

static gd_status make_tensor(gd_context *ctx,
                             gd_device device,
                             gd_dtype dtype,
                             const int64_t *shape,
                             const float *data,
                             int64_t n,
                             gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(dtype, device, 4, shape, &desc);

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    if (dtype == GD_DTYPE_F32) {
        return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float));
    }
    if (dtype == GD_DTYPE_F16) {
        uint16_t *h = malloc((size_t)n * sizeof(uint16_t));
        int64_t i = 0;
        if (h == NULL) {
            return GD_ERR_OUT_OF_MEMORY;
        }
        for (i = 0; i < n; ++i) {
            h[i] = f32_to_f16(data[i]);
        }
        status = gd_tensor_copy_from_cpu(ctx, *out, h, (size_t)n * sizeof(uint16_t));
        free(h);
        return status;
    }
    return GD_ERR_DTYPE;
}

static void fill_data(float *x, int64_t n, int seed)
{
    int64_t i = 0;
    for (i = 0; i < n; ++i) {
        int v = (int)((i + (int64_t)seed * 17) % 29) - 14;
        x[i] = 0.02F * (float)v;
    }
}

static long long suffix_window_pairs(int n, int window)
{
    if (n <= 0) {
        return 0;
    }
    if (window <= 0 || n <= window) {
        return (long long)n * (long long)(n + 1) / 2LL;
    }
    return (long long)window * (long long)(window + 1) / 2LL +
           (long long)(n - window) * (long long)window;
}

static long long mask_pairs(const char *mask, int T, int prefix, int window)
{
    int N = T - prefix;
    if (strcmp(mask, "dense") == 0) {
        return (long long)T * (long long)T;
    }
    if (strcmp(mask, "causal") == 0) {
        return (long long)T * (long long)(T + 1) / 2LL;
    }
    if (strcmp(mask, "window") == 0) {
        return suffix_window_pairs(T, window);
    }
    if (strcmp(mask, "prefix") == 0) {
        return (long long)prefix * (long long)prefix +
               (long long)N * (long long)prefix + suffix_window_pairs(N, 0);
    }
    return (long long)prefix * (long long)prefix +
           (long long)N * (long long)prefix + suffix_window_pairs(N, window);
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device target = cpu;
    const char *dev_env = getenv("GD_DEVICE");
    const char *mode_env = getenv("GD_BENCH_MODE");
    const char *dtype_env = getenv("GD_BENCH_DTYPE");
    const char *mask_env = getenv("GD_BENCH_MASK");
    const char *mask = mask_env != NULL && mask_env[0] != '\0' ? mask_env : "prefix_window";
    bool train_mode = mode_env != NULL && strcmp(mode_env, "train") == 0;
    gd_dtype dtype = GD_DTYPE_F32;

    int image_size = env_int("GD_BENCH_IMAGE_SIZE", 224);
    int patch = env_int("GD_BENCH_PATCH", 16);
    int extra = env_int("GD_BENCH_IMAGE_EXTRA_TOKENS", 0);
    int grid = 0;
    int default_prefix = 0;
    int prefix = 0;
    int text = 0;
    int T = 0;
    int B = env_int("GD_BENCH_B", 1);
    int H = env_int("GD_BENCH_HEADS", 8);
    int Hkv = env_int("GD_BENCH_KV_HEADS", H);
    int Dh = env_int("GD_BENCH_HEAD_DIM", 64);
    int window = env_window(128);
    int iters = env_int("GD_BENCH_ITERS", 20);
    int warmup = env_int("GD_BENCH_WARMUP", 3);
    int64_t qshape[4];
    int64_t kshape[4];
    int64_t flat_shape[1];
    int64_t qn = 0;
    int64_t kn = 0;
    float *qd = NULL;
    float *kd = NULL;
    float *vd = NULL;
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *y = NULL;
    gd_tensor *y_f32 = NULL;
    gd_tensor *flat = NULL;
    gd_tensor *loss = NULL;
    gd_graph *g = NULL;
    gd_sdpa_config cfg = {0};
    gd_tensor *params[3];
    long long pairs = 0;
    long long dense_pairs = 0;
    long long causal_pairs = 0;
    double fwd_gflop = 0.0;
    double work_gflop = 0.0;
    double ms_mean = 0.0;
    int i = 0;

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

    if (dtype_env != NULL && dtype_env[0] != '\0') {
        if (strcmp(dtype_env, "f16") == 0 || strcmp(dtype_env, "fp16") == 0) {
            dtype = GD_DTYPE_F16;
        } else if (strcmp(dtype_env, "f32") == 0 || strcmp(dtype_env, "fp32") == 0) {
            dtype = GD_DTYPE_F32;
        } else {
            fprintf(stderr, "config error: GD_BENCH_DTYPE must be f32 or f16\n");
            gd_context_destroy(ctx);
            return 1;
        }
    }

    if (patch <= 0 || image_size <= 0 || image_size % patch != 0) {
        fprintf(stderr, "config error: image_size must be divisible by positive patch\n");
        gd_context_destroy(ctx);
        return 1;
    }
    grid = image_size / patch;
    default_prefix = grid * grid + extra;
    prefix = env_int("GD_BENCH_PREFIX", default_prefix);
    if (env_present("GD_BENCH_T")) {
        T = env_int("GD_BENCH_T", 1024);
    } else if (env_present("GD_BENCH_TEXT")) {
        T = prefix + env_int("GD_BENCH_TEXT", 0);
    } else {
        T = 1024;
    }
    text = T - prefix;

    if (B <= 0 || T <= 0 || H <= 0 || Hkv <= 0 || Dh <= 0 || H % Hkv != 0) {
        fprintf(stderr, "config error: invalid B/T/heads/kv_heads/head_dim\n");
        gd_context_destroy(ctx);
        return 1;
    }
    if (prefix < 0 || prefix > T || text < 0 || window < 0) {
        fprintf(stderr, "config error: require 0<=prefix<=T and window>=0\n");
        gd_context_destroy(ctx);
        return 1;
    }
    if (strcmp(mask, "dense") != 0 && strcmp(mask, "causal") != 0 &&
        strcmp(mask, "window") != 0 && strcmp(mask, "prefix") != 0 &&
        strcmp(mask, "prefix_window") != 0) {
        fprintf(stderr, "config error: GD_BENCH_MASK must be dense|causal|window|prefix|prefix_window\n");
        gd_context_destroy(ctx);
        return 1;
    }

    qshape[0] = B;
    qshape[1] = T;
    qshape[2] = H;
    qshape[3] = Dh;
    kshape[0] = B;
    kshape[1] = T;
    kshape[2] = Hkv;
    kshape[3] = Dh;
    qn = numel4(B, T, H, Dh);
    kn = numel4(B, T, Hkv, Dh);
    flat_shape[0] = qn;

    qd = malloc((size_t)qn * sizeof(float));
    kd = malloc((size_t)kn * sizeof(float));
    vd = malloc((size_t)kn * sizeof(float));
    if (qd == NULL || kd == NULL || vd == NULL) {
        fprintf(stderr, "alloc failed\n");
        free(qd);
        free(kd);
        free(vd);
        gd_context_destroy(ctx);
        return 1;
    }
    fill_data(qd, qn, 1);
    fill_data(kd, kn, 2);
    fill_data(vd, kn, 3);

    CHECK(make_tensor(ctx, target, dtype, qshape, qd, qn, &q));
    CHECK(make_tensor(ctx, target, dtype, kshape, kd, kn, &k));
    CHECK(make_tensor(ctx, target, dtype, kshape, vd, kn, &v));
    free(qd);
    free(kd);
    free(vd);
    qd = NULL;
    kd = NULL;
    vd = NULL;

    if (strcmp(mask, "dense") == 0) {
        cfg.causal = false;
    } else {
        cfg.causal = true;
    }
    if (strcmp(mask, "window") == 0 || strcmp(mask, "prefix_window") == 0) {
        cfg.sliding_window = window;
    }
    if (strcmp(mask, "prefix") == 0 || strcmp(mask, "prefix_window") == 0) {
        cfg.prefix_len = prefix;
    }

    if (train_mode) {
        CHECK(gd_tensor_set_requires_grad(q, true));
        CHECK(gd_tensor_set_requires_grad(k, true));
        CHECK(gd_tensor_set_requires_grad(v, true));
    }

    dense_pairs = mask_pairs("dense", T, prefix, window);
    causal_pairs = mask_pairs("causal", T, prefix, window);
    pairs = mask_pairs(mask, T, prefix, window);
    fwd_gflop = 4.0 * (double)B * (double)H * (double)pairs * (double)Dh / 1e9;
    work_gflop = train_mode ? 3.0 * fwd_gflop : fwd_gflop;

    printf("sdpa_bench\n");
    printf("  device      : %s\n", target.type == GD_DEVICE_METAL ? "metal" : "cpu");
    printf("  mode        : %s\n", train_mode ? "train (fwd+bwd q/k/v)" : "forward");
    printf("  dtype       : %s\n", gd_dtype_name(dtype));
    printf("  shape       : B=%d T=%d H=%d Hkv=%d Dh=%d\n", B, T, H, Hkv, Dh);
    printf("  image math  : %d/%d = %d patches per side => %d tokens%s\n",
           image_size, patch, grid, grid * grid, extra != 0 ? " + extra" : "");
    printf("  prefix/text : P=%d N=%d\n", prefix, text);
    printf("  mask        : %s", mask);
    if (cfg.sliding_window > 0) {
        printf(" window=%d", cfg.sliding_window);
    }
    printf("\n");
    printf("  pairs       : %lld (dense %.3fx, causal %.3fx)\n", pairs,
           dense_pairs > 0 ? (double)pairs / (double)dense_pairs : 0.0,
           causal_pairs > 0 ? (double)pairs / (double)causal_pairs : 0.0);
    printf("  attn GFLOP  : %.3f fwd, %.3f measured-work\n", fwd_gflop, work_gflop);
    printf("  iters       : %d (warmup %d)\n", iters, warmup);

    CHECK(gd_graph_create(ctx, &g));
    CHECK(gd_graph_begin(ctx, g));
    CHECK(gd_sdpa(ctx, q, k, v, NULL, &cfg, &y));
    if (train_mode) {
        if (dtype == GD_DTYPE_F32) {
            y_f32 = y;
        } else {
            CHECK(gd_cast(ctx, y, GD_DTYPE_F32, &y_f32));
        }
        CHECK(gd_tensor_reshape(y_f32, 1, flat_shape, &flat));
        CHECK(gd_mean(ctx, flat, 0, false, &loss));
        CHECK(gd_backward(ctx, loss));
    }
    CHECK(gd_graph_end(ctx));
    printf("  compiling graph...\n");
    CHECK(gd_graph_compile(g, target));

    params[0] = q;
    params[1] = k;
    params[2] = v;
    printf("  warmup (%d iters)...\n", warmup);
    for (i = 0; i < warmup; ++i) {
        if (train_mode) {
            CHECK(gd_zero_grad(ctx, params, 3));
        }
        CHECK(gd_graph_run(g));
    }
    CHECK(gd_synchronize(ctx, target));

    {
        double total_ms = 0.0;
        double best_ms = 0.0;
        for (i = 0; i < iters; ++i) {
            double a = 0.0;
            double b = 0.0;
            if (train_mode) {
                CHECK(gd_zero_grad(ctx, params, 3));
            }
            a = now_ms();
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
        ms_mean = iters > 0 ? total_ms / (double)iters : 0.0;
        printf("  ms/iter     : %.3f (mean)  %.3f (best)\n", ms_mean, best_ms);
        if (best_ms > 0.0) {
            printf("  throughput  : %.1f attn-GFLOP/s (mean)  %.1f (best)\n",
                   work_gflop / (ms_mean / 1000.0), work_gflop / (best_ms / 1000.0));
            printf("  tokens/s    : %.1f (best)\n",
                   (double)B * (double)T / (best_ms / 1000.0));
        }
    }

    gd_tensor_release(loss);
    gd_tensor_release(flat);
    if (y_f32 != y) {
        gd_tensor_release(y_f32);
    }
    gd_tensor_release(y);
    CHECK(gd_graph_destroy(g));
    gd_tensor_release(q);
    gd_tensor_release(k);
    gd_tensor_release(v);
    gd_context_destroy(ctx);
    return 0;
}
