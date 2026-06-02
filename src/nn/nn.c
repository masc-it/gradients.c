#include "gradients/nn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "gradients/module.h"
#include "gradients/ops.h"

#include "../core/internal.h"

struct gd_gpt {
    gd_context *ctx;
    gd_gpt_config cfg;
    gd_module *module; /* owns all parameter tensors */
    uint64_t dropout_seed;
    bool training;

    gd_tensor *wte;    /* [V, d] token embedding (also tied LM head) */
    gd_tensor *ln_f;   /* [d] final norm */
    gd_tensor *w_head; /* [d, V] untied head, or NULL when tied */

    /* per-layer (length n_layers; weak pointers owned by `module`) */
    gd_tensor **ln1;
    gd_tensor **wq;
    gd_tensor **wk;
    gd_tensor **wv;
    gd_tensor **wo;
    gd_tensor **ln2;
    gd_tensor **w_gate; /* powlu/swiglu gate / gelu fc */
    gd_tensor **w_up;   /* powlu/swiglu value projection (NULL for gelu) */
    gd_tensor **w_down; /* powlu/swiglu/gelu down (proj) */
};

/* Deterministic PRNG (splitmix64-ish) so two models built from the same seed
 * have identical initial weights -- required for CPU<->Metal parity tests. */
static uint64_t next_rand_u64(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static float rand_uniform01(uint64_t *state)
{
    uint64_t z = next_rand_u64(state);
    /* 24 random mantissa bits, open interval avoids log(0) in normal init. */
    return ((float)(z >> 40) + 0.5F) * (1.0F / 16777216.0F);
}

static float rand_normal(uint64_t *state)
{
    float u = 0.0F;
    float v = 0.0F;
    float s = 0.0F;

    do {
        u = 2.0F * rand_uniform01(state) - 1.0F;
        v = 2.0F * rand_uniform01(state) - 1.0F;
        s = u * u + v * v;
    } while (s >= 1.0F || s <= 0.0F);
    return u * sqrtf(-2.0F * logf(s) / s);
}

static gd_status create_param(gd_gpt *g, const char *name, int ndim, const int64_t *sizes,
                              float scale, int ones, uint64_t *rng, gd_tensor **slot)
{
    gd_device device = gd_context_default_device(g->ctx);
    gd_dtype dtype = g->cfg.param_dtype;
    gd_tensor_desc desc;
    gd_tensor *t = NULL;
    int64_t numel = 1;
    size_t elem_size = gd_dtype_sizeof(dtype);
    size_t nbytes = 0U;
    void *buf = NULL;
    int i = 0;
    int64_t j = 0;
    gd_status status = gd_tensor_desc_contiguous(dtype, device, ndim, sizes, &desc);

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(g->ctx, &desc, &t);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        numel *= sizes[i];
    }
    if (numel < 0 || elem_size == 0U || (uint64_t)numel > (uint64_t)SIZE_MAX / elem_size) {
        gd_tensor_release(t);
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gpt parameter size overflows");
    }
    nbytes = (size_t)numel * elem_size;
    buf = malloc(nbytes > 0U ? nbytes : 1U);
    if (buf == NULL) {
        gd_tensor_release(t);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate parameter init buffer");
    }
    if (dtype == GD_DTYPE_F32) {
        float *f = (float *)buf;
        for (j = 0; j < numel; ++j) {
            f[j] = ones ? 1.0f : scale * rand_normal(rng);
        }
    } else {
        uint16_t *h = (uint16_t *)buf;
        for (j = 0; j < numel; ++j) {
            h[j] = _gd_f32_to_f16_bits(ones ? 1.0f : scale * rand_normal(rng));
        }
    }
    status = gd_tensor_copy_from_cpu(g->ctx, t, buf, nbytes);
    free(buf);
    if (status == GD_OK) {
        status = gd_tensor_set_requires_grad(t, true);
    }
    if (status == GD_OK) {
        status = gd_module_param(g->module, name, t);
    }
    if (status != GD_OK) {
        gd_tensor_release(t);
        return status;
    }
    gd_tensor_release(t); /* module is now the sole owner */
    *slot = t;
    return GD_OK;
}

static gd_status alloc_layers(gd_gpt *g)
{
    int L = g->cfg.n_layers;
    gd_tensor ***arrays[9] = {&g->ln1, &g->wq, &g->wk, &g->wv, &g->wo,
                              &g->ln2, &g->w_gate, &g->w_up, &g->w_down};
    int i = 0;

    for (i = 0; i < 9; ++i) {
        *arrays[i] = calloc((size_t)(L > 0 ? L : 1), sizeof(gd_tensor *));
        if (*arrays[i] == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate gpt layer arrays");
        }
    }
    return GD_OK;
}

gd_status gd_gpt_create(gd_context *ctx, const gd_gpt_config *config,
                        uint64_t seed, gd_gpt **out)
{
    gd_gpt *g = NULL;
    gd_gpt_config cfg;
    gd_status status = GD_OK;
    uint64_t rng = seed ? seed : 0x123456789ULL;
    float wscale = 0.02f;
    float residual_wscale = 0.02f;
    int d = 0, dff = 0, Hq = 0, Hkv = 0, Dh = 0, V = 0, L = 0, l = 0;
    char name[64];

    if (ctx == NULL || config == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gpt_create argument is NULL");
    }
    *out = NULL;
    cfg = *config;
    if (cfg.n_kv_heads <= 0) {
        cfg.n_kv_heads = cfg.n_heads;
    }
    if (cfg.rope_theta <= 0.0f) {
        cfg.rope_theta = 10000.0f;
    }
    if (cfg.norm_eps <= 0.0f) {
        cfg.norm_eps = 1e-5f;
    }
    if (cfg.powlu_m == 0.0f) {
        cfg.powlu_m = 3.0f;
    }
    if (cfg.attention_window < 0) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "attention_window must be non-negative");
    }
    if (!isfinite(cfg.dropout_p) || cfg.dropout_p < 0.0F || cfg.dropout_p >= 1.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "dropout_p must satisfy 0 <= p < 1");
    }
    if (cfg.param_dtype == GD_DTYPE_INVALID) {
        cfg.param_dtype = GD_DTYPE_F32;
    }
    if (cfg.param_dtype != GD_DTYPE_F32 && cfg.param_dtype != GD_DTYPE_F16) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gpt param_dtype must be F32 or F16 in v1");
    }
    if (cfg.mlp_kind != GD_GPT_MLP_POWLU && cfg.mlp_kind != GD_GPT_MLP_SWIGLU &&
        cfg.mlp_kind != GD_GPT_MLP_GELU) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid gpt mlp kind");
    }
    if (cfg.mlp_kind == GD_GPT_MLP_POWLU &&
        (!isfinite(cfg.powlu_m) || cfg.powlu_m <= 0.0f || cfg.powlu_m >= 10.0f)) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "powlu_m must satisfy 0 < m < 10");
    }
    d = cfg.d_model;
    dff = cfg.d_ff;
    Hq = cfg.n_heads;
    Hkv = cfg.n_kv_heads;
    Dh = cfg.head_dim;
    V = cfg.vocab_size;
    L = cfg.n_layers;
    if (d <= 0 || dff <= 0 || Hq <= 0 || Hkv <= 0 || Dh <= 0 || V <= 0 || L <= 0 ||
        Hq % Hkv != 0 || Hq * Dh != d) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "invalid gpt config (need d_model==n_heads*head_dim, n_heads%n_kv_heads==0)");
    }
    /* GPT-2-style init: normal(0, 0.02) for weights; residual output
     * projections use 0.02/sqrt(2*n_layers) to control residual-path growth. */
    residual_wscale = wscale / sqrtf(2.0F * (float)L);

    g = calloc(1u, sizeof(*g));
    if (g == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate gpt");
    }
    g->ctx = ctx;
    g->cfg = cfg;
    g->dropout_seed = rng ^ UINT64_C(0xd1b54a32d192ed03);
    g->training = false;
    status = gd_module_create(ctx, "gpt", &g->module);
    if (status != GD_OK) {
        free(g);
        return status;
    }
    status = alloc_layers(g);
    if (status != GD_OK) {
        gd_gpt_destroy(g);
        return status;
    }

    {
        int64_t s_wte[2] = {V, d};
        int64_t s_d[1] = {d};
        status = create_param(g, "wte", 2, s_wte, wscale, 0, &rng, &g->wte);
        if (status == GD_OK) {
            status = create_param(g, "ln_f", 1, s_d, 0.0f, 1, &rng, &g->ln_f);
        }
        if (status == GD_OK && !cfg.tie_embeddings) {
            int64_t s_head[2] = {d, V};
            status = create_param(g, "head", 2, s_head, wscale, 0, &rng, &g->w_head);
        }
    }

    for (l = 0; l < L && status == GD_OK; ++l) {
        int64_t s_d[1] = {d};
        int64_t s_wq[2] = {d, Hq * Dh};
        int64_t s_wkv[2] = {d, Hkv * Dh};
        int64_t s_wo[2] = {Hq * Dh, d};
        int64_t s_in_ff[2] = {d, dff};
        int64_t s_ff_out[2] = {dff, d};

        (void)snprintf(name, sizeof(name), "blk.%d.ln1", l);
        status = create_param(g, name, 1, s_d, 0.0f, 1, &rng, &g->ln1[l]);
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.wq", l);
            status = create_param(g, name, 2, s_wq, wscale, 0, &rng, &g->wq[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.wk", l);
            status = create_param(g, name, 2, s_wkv, wscale, 0, &rng, &g->wk[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.wv", l);
            status = create_param(g, name, 2, s_wkv, wscale, 0, &rng, &g->wv[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.wo", l);
            status = create_param(g, name, 2, s_wo, residual_wscale, 0, &rng, &g->wo[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.ln2", l);
            status = create_param(g, name, 1, s_d, 0.0f, 1, &rng, &g->ln2[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.w_gate", l);
            status = create_param(g, name, 2, s_in_ff, wscale, 0, &rng, &g->w_gate[l]);
        }
        if (status == GD_OK && cfg.mlp_kind != GD_GPT_MLP_GELU) {
            (void)snprintf(name, sizeof(name), "blk.%d.w_up", l);
            status = create_param(g, name, 2, s_in_ff, wscale, 0, &rng, &g->w_up[l]);
        }
        if (status == GD_OK) {
            (void)snprintf(name, sizeof(name), "blk.%d.w_down", l);
            status = create_param(g, name, 2, s_ff_out, residual_wscale, 0, &rng, &g->w_down[l]);
        }
    }

    if (status != GD_OK) {
        gd_gpt_destroy(g);
        return status;
    }
    *out = g;
    return GD_OK;
}

void gd_gpt_destroy(gd_gpt *gpt)
{
    if (gpt == NULL) {
        return;
    }
    /* module owns the parameter tensors; destroying it releases them. */
    gd_module_destroy(gpt->module);
    free(gpt->ln1);
    free(gpt->wq);
    free(gpt->wk);
    free(gpt->wv);
    free(gpt->wo);
    free(gpt->ln2);
    free(gpt->w_gate);
    free(gpt->w_up);
    free(gpt->w_down);
    free(gpt);
}

void gd_gpt_set_training(gd_gpt *gpt, bool training)
{
    if (gpt != NULL) {
        gpt->training = training;
    }
}

bool gd_gpt_is_training(const gd_gpt *gpt)
{
    return gpt != NULL && gpt->training;
}

gd_status gd_gpt_parameters(gd_gpt *gpt, gd_tensor ***params_out, int *n_out)
{
    if (gpt == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gpt_parameters gpt is NULL");
    }
    return gd_module_parameters(gpt->module, params_out, n_out);
}

gd_status gd_gpt_parameter_groups(gd_gpt *gpt,
                                  float weight_decay,
                                  gd_param_group **groups_out,
                                  int *n_groups_out)
{
    gd_param_group *groups = NULL;
    int decay_count = 0;
    int no_decay_count = 0;
    int decay_i = 0;
    int no_decay_i = 0;
    int l = 0;

    if (gpt == NULL || groups_out == NULL || n_groups_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gpt_parameter_groups argument is NULL");
    }
    if (!isfinite(weight_decay) || weight_decay < 0.0F) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT,
                         "gpt parameter group weight_decay must be finite and nonnegative");
    }
    *groups_out = NULL;
    *n_groups_out = 0;

    decay_count = 1 + (gpt->w_head != NULL ? 1 : 0);
    no_decay_count = 1;
    for (l = 0; l < gpt->cfg.n_layers; ++l) {
        decay_count += gpt->w_up[l] != NULL ? 7 : 6;
        no_decay_count += 2;
    }

    groups = calloc(2U, sizeof(*groups));
    if (groups == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate gpt parameter groups");
    }
    groups[0].params = calloc((size_t)decay_count, sizeof(*groups[0].params));
    groups[1].params = calloc((size_t)no_decay_count, sizeof(*groups[1].params));
    if (groups[0].params == NULL || groups[1].params == NULL) {
        gd_gpt_parameter_groups_free(groups, 2);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate gpt group params");
    }

    groups[0].weight_decay = weight_decay;
    groups[0].lr_scale = 1.0F;
    groups[1].weight_decay = 0.0F;
    groups[1].lr_scale = 1.0F;

    groups[0].params[decay_i++] = gpt->wte;
    if (gpt->w_head != NULL) {
        groups[0].params[decay_i++] = gpt->w_head;
    }
    groups[1].params[no_decay_i++] = gpt->ln_f;
    for (l = 0; l < gpt->cfg.n_layers; ++l) {
        groups[1].params[no_decay_i++] = gpt->ln1[l];
        groups[0].params[decay_i++] = gpt->wq[l];
        groups[0].params[decay_i++] = gpt->wk[l];
        groups[0].params[decay_i++] = gpt->wv[l];
        groups[0].params[decay_i++] = gpt->wo[l];
        groups[1].params[no_decay_i++] = gpt->ln2[l];
        groups[0].params[decay_i++] = gpt->w_gate[l];
        if (gpt->w_up[l] != NULL) {
            groups[0].params[decay_i++] = gpt->w_up[l];
        }
        groups[0].params[decay_i++] = gpt->w_down[l];
    }
    groups[0].n_params = decay_i;
    groups[1].n_params = no_decay_i;

    *groups_out = groups;
    *n_groups_out = 2;
    return GD_OK;
}

void gd_gpt_parameter_groups_free(gd_param_group *groups, int n_groups)
{
    int i = 0;

    if (groups == NULL) {
        return;
    }
    for (i = 0; i < n_groups; ++i) {
        free(groups[i].params);
    }
    free(groups);
}

static uint64_t mix_dropout_seed(uint64_t seed, uint64_t site)
{
    uint64_t x = seed + site * UINT64_C(0x9E3779B97F4A7C15);

    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static gd_status gpt_dropout(gd_context *ctx,
                             gd_gpt *g,
                             gd_tensor *x,
                             uint64_t site,
                             gd_tensor **out)
{
    return gd_dropout(ctx, x, g->cfg.dropout_p, mix_dropout_seed(g->dropout_seed, site),
                      g->training, out);
}

/* x[B,T,d] -> x + attn(rms_norm(x)). Returns a new virtual tensor. */
static gd_status attention_block(gd_context *ctx, gd_gpt *g, int l,
                                 gd_tensor *x, gd_tensor *positions,
                                 int64_t B, int64_t T, gd_tensor **out_p)
{
    const gd_gpt_config *c = &g->cfg;
    int64_t Hq = c->n_heads, Hkv = c->n_kv_heads, Dh = c->head_dim;
    int64_t q4[4] = {B, T, Hq, Dh};
    int64_t kv4[4] = {B, T, Hkv, Dh};
    int64_t o3[3] = {B, T, Hq * Dh};
    gd_rope_config rope = {c->rope_theta, 0, false};
    gd_sdpa_config sdpa = {0.0f, true, c->attention_window, 0};
    gd_tensor *n = NULL, *qf = NULL, *q = NULL, *kf = NULL, *k = NULL;
    gd_tensor *vf = NULL, *v = NULL, *qr = NULL, *kr = NULL, *o = NULL;
    gd_tensor *om = NULL, *op = NULL, *drop = NULL, *out = NULL;
    gd_status s = GD_OK;

    *out_p = NULL;
    s = gd_rms_norm(ctx, x, g->ln1[l], c->norm_eps, &n);
    if (s == GD_OK) { s = gd_linear(ctx, n, g->wq[l], NULL, &qf); }
    if (s == GD_OK) { s = gd_tensor_reshape(qf, 4, q4, &q); }
    if (s == GD_OK) { s = gd_linear(ctx, n, g->wk[l], NULL, &kf); }
    if (s == GD_OK) { s = gd_tensor_reshape(kf, 4, kv4, &k); }
    if (s == GD_OK) { s = gd_linear(ctx, n, g->wv[l], NULL, &vf); }
    if (s == GD_OK) { s = gd_tensor_reshape(vf, 4, kv4, &v); }
    if (s == GD_OK) { s = gd_rope(ctx, q, positions, &rope, &qr); }
    if (s == GD_OK) { s = gd_rope(ctx, k, positions, &rope, &kr); }
    if (s == GD_OK) { s = gd_sdpa(ctx, qr, kr, v, NULL, &sdpa, &o); }
    if (s == GD_OK) { s = gd_tensor_reshape(o, 3, o3, &om); }
    if (s == GD_OK) { s = gd_linear(ctx, om, g->wo[l], NULL, &op); }
    if (s == GD_OK) { s = gpt_dropout(ctx, g, op, UINT64_C(0x1000) + (uint64_t)l, &drop); }
    if (s == GD_OK) { s = gd_add(ctx, x, drop, &out); }

    gd_tensor_release(n);
    gd_tensor_release(qf);
    gd_tensor_release(q);
    gd_tensor_release(kf);
    gd_tensor_release(k);
    gd_tensor_release(vf);
    gd_tensor_release(v);
    gd_tensor_release(qr);
    gd_tensor_release(kr);
    gd_tensor_release(o);
    gd_tensor_release(om);
    gd_tensor_release(op);
    gd_tensor_release(drop);
    *out_p = out;
    return s;
}

/* h[B,T,d] -> h + mlp(rms_norm(h)). Returns a new virtual tensor. */
static gd_status mlp_block(gd_context *ctx, gd_gpt *g, int l, gd_tensor *h,
                           gd_tensor **out_p)
{
    const gd_gpt_config *c = &g->cfg;
    gd_tensor *n = NULL, *gate = NULL, *act = NULL, *up = NULL, *hh = NULL;
    gd_tensor *down = NULL, *drop = NULL, *out = NULL;
    gd_status s = GD_OK;

    *out_p = NULL;
    s = gd_rms_norm(ctx, h, g->ln2[l], c->norm_eps, &n);
    if (s == GD_OK) { s = gd_linear(ctx, n, g->w_gate[l], NULL, &gate); }
    if (c->mlp_kind == GD_GPT_MLP_POWLU) {
        if (s == GD_OK) { s = gd_linear(ctx, n, g->w_up[l], NULL, &up); }
        if (s == GD_OK) { s = gd_powlu(ctx, up, gate, c->powlu_m, &hh); }
    } else if (c->mlp_kind == GD_GPT_MLP_SWIGLU) {
        if (s == GD_OK) { s = gd_silu(ctx, gate, &act); }
        if (s == GD_OK) { s = gd_linear(ctx, n, g->w_up[l], NULL, &up); }
        if (s == GD_OK) { s = gd_mul(ctx, act, up, &hh); }
    } else {
        if (s == GD_OK) { s = gd_gelu(ctx, gate, false, &hh); }
    }
    if (s == GD_OK) { s = gd_linear(ctx, hh, g->w_down[l], NULL, &down); }
    if (s == GD_OK) { s = gpt_dropout(ctx, g, down, UINT64_C(0x2000) + (uint64_t)l, &drop); }
    if (s == GD_OK) { s = gd_add(ctx, h, drop, &out); }

    gd_tensor_release(n);
    gd_tensor_release(gate);
    gd_tensor_release(act);
    gd_tensor_release(up);
    gd_tensor_release(hh);
    gd_tensor_release(down);
    gd_tensor_release(drop);
    *out_p = out;
    return s;
}

static gd_status gpt_forward_normed(gd_context *ctx, gd_gpt *gpt,
                                    gd_tensor *tokens, gd_tensor *positions,
                                    gd_tensor **normed_out)
{
    gd_status s = GD_OK;
    int64_t B = 0, T = 0;
    gd_tensor *emb = NULL;
    gd_tensor *x = NULL;
    gd_tensor *normed = NULL;
    int l = 0;

    if (ctx == NULL || gpt == NULL || tokens == NULL || positions == NULL ||
        normed_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gpt_forward_normed argument is NULL");
    }
    *normed_out = NULL;
    if (gd_tensor_ndim(tokens) != 2) {
        return _gd_error(GD_ERR_SHAPE, "gpt tokens must be [batch, seq]");
    }
    B = gd_tensor_size(tokens, 0);
    T = gd_tensor_size(tokens, 1);

    s = gd_embedding(ctx, gpt->wte, tokens, &emb);
    if (s == GD_OK) {
        s = gpt_dropout(ctx, gpt, emb, UINT64_C(0x1), &x);
    }
    for (l = 0; l < gpt->cfg.n_layers && s == GD_OK; ++l) {
        gd_tensor *attn = NULL;
        gd_tensor *mlp = NULL;

        s = attention_block(ctx, gpt, l, x, positions, B, T, &attn);
        gd_tensor_release(x);
        x = attn;
        if (s != GD_OK) {
            break;
        }
        s = mlp_block(ctx, gpt, l, x, &mlp);
        gd_tensor_release(x);
        x = mlp;
    }
    if (s == GD_OK) {
        s = gd_rms_norm(ctx, x, gpt->ln_f, gpt->cfg.norm_eps, &normed);
    }
    gd_tensor_release(emb);
    gd_tensor_release(x);
    if (s != GD_OK) {
        gd_tensor_release(normed);
        return s;
    }
    *normed_out = normed;
    return GD_OK;
}

gd_status gd_gpt_forward(gd_context *ctx, gd_gpt *gpt,
                         gd_tensor *tokens, gd_tensor *positions,
                         gd_tensor **logits_out)
{
    gd_status s = GD_OK;
    gd_tensor *normed = NULL;
    gd_tensor *logits = NULL;

    if (logits_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gpt_forward argument is NULL");
    }
    *logits_out = NULL;
    s = gpt_forward_normed(ctx, gpt, tokens, positions, &normed);
    if (s == GD_OK) {
        if (gpt->cfg.tie_embeddings) {
            gd_matmul_desc md = {false, true, gd_compute_policy_default()};
            s = gd_matmul_ex(ctx, &md, normed, gpt->wte, &logits); /* x @ Wte^T */
        } else {
            s = gd_linear(ctx, normed, gpt->w_head, NULL, &logits);
        }
    }
    gd_tensor_release(normed);
    if (s != GD_OK) {
        gd_tensor_release(logits);
        return s;
    }
    *logits_out = logits;
    return GD_OK;
}

gd_status gd_gpt_forward_loss(gd_context *ctx, gd_gpt *gpt,
                              gd_tensor *tokens, gd_tensor *positions,
                              gd_tensor *targets, gd_tensor **loss_out)
{
    gd_status s = GD_OK;
    gd_tensor *normed = NULL;
    gd_tensor *logits = NULL;

    if (loss_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_gpt_forward_loss argument is NULL");
    }
    *loss_out = NULL;
    s = gpt_forward_normed(ctx, gpt, tokens, positions, &normed);
    if (s == GD_OK) {
        if (gpt->cfg.tie_embeddings) {
            s = gd_lm_cross_entropy(ctx, normed, gpt->wte, targets, loss_out);
        } else {
            s = gd_linear(ctx, normed, gpt->w_head, NULL, &logits);
            if (s == GD_OK) {
                s = gd_cross_entropy(ctx, logits, targets, 2, loss_out);
            }
        }
    }
    gd_tensor_release(normed);
    gd_tensor_release(logits);
    return s;
}
