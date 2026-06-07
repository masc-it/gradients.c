#include "gpt_lm_shared.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void gpt_fail_status(gd_context *ctx, gd_status st, const char *expr, int line)
{
    fprintf(stderr,
            "gpt_lm failed at line %d: %s -> %s (%s)\n",
            line,
            expr,
            gd_status_string(st),
            ctx != NULL ? gd_context_error(ctx) : "no context");
    exit(1);
}

double gpt_wall_seconds(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static uint64_t splitmix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}


static size_t size_max2(size_t a, size_t b)
{
    return a > b ? a : b;
}

static size_t checked_mul_size(size_t a, size_t b, const char *what)
{
    if (a != 0U && b > SIZE_MAX / a) {
        fprintf(stderr, "gpt_lm: size overflow while computing %s\n", what);
        exit(2);
    }
    return a * b;
}

static size_t checked_add_size(size_t a, size_t b, const char *what)
{
    if (b > SIZE_MAX - a) {
        fprintf(stderr, "gpt_lm: size overflow while computing %s\n", what);
        exit(2);
    }
    return a + b;
}

size_t gpt_param_count_for_layers(int n_layers)
{
    size_t per_block = 0U;
    size_t total = 0U;
    total += checked_mul_size((size_t)GPT_VOCAB_SIZE, (size_t)GPT_D_MODEL, "embedding params");
    total += (size_t)GPT_D_MODEL;
    per_block += (size_t)(2 * GPT_D_MODEL);
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)(3 * GPT_D_MODEL), "qkv params");
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)GPT_D_MODEL, "attn out params");
    per_block += checked_mul_size((size_t)GPT_D_MODEL, (size_t)(2 * GPT_MLP_HIDDEN), "up/gate params");
    per_block += checked_mul_size((size_t)GPT_MLP_HIDDEN, (size_t)GPT_D_MODEL, "down params");
    total += checked_mul_size(per_block, (size_t)n_layers, "block params");
    return total;
}

gd_memory_config gpt_memory_config(const gpt_config *config)
{
    const size_t param_count = gpt_param_count_for_layers(config->n_layers);
    const size_t param_bytes = checked_mul_size(param_count, gd_dtype_size(GD_DTYPE_F16), "param bytes");
    const size_t adam_bytes = checked_mul_size(param_count, 3U * gd_dtype_size(GD_DTYPE_F32), "adam state bytes");
    const size_t tokens_per_batch = checked_mul_size((size_t)config->batch_size,
                                                     (size_t)GPT_CONTEXT_LENGTH,
                                                     "batch tokens");
    const size_t token_field_bytes = checked_mul_size(tokens_per_batch,
                                                      gd_dtype_size(GD_DTYPE_I32),
                                                      "packed token field bytes");
    gd_memory_config mem = gd_memory_config_default();
    mem.params_bytes = size_max2((size_t)GPT_MIN_PARAMS_BYTES,
                                 checked_add_size(param_bytes,
                                                  32U * 1024U * 1024U,
                                                  "param arena bytes"));
    mem.state_bytes = size_max2((size_t)GPT_MIN_STATE_BYTES,
                                checked_add_size(adam_bytes,
                                                 32U * 1024U * 1024U,
                                                 "state arena bytes"));
    mem.scratch_slot_bytes = (size_t)GPT_SCRATCH_SLOT_BYTES;
    mem.data_slot_bytes = size_max2(
        (size_t)GPT_MIN_DATA_SLOT_BYTES,
        checked_add_size(checked_mul_size(token_field_bytes,
                                          8U,
                                          "data arena packed token fields"),
                         64U * 1024U * 1024U,
                         "data arena bytes"));
    mem.scratch_slots = 1U;
    mem.data_slots = 2U;
    mem.default_alignment = GPT_ALIGNMENT;
    return mem;
}

static gd_tensor_spec tensor_spec_1d(gd_dtype dtype, int64_t dim)
{
    int64_t shape[1];
    shape[0] = dim;
    return gd_tensor_spec_make(dtype, gd_shape_make(1U, shape), GPT_ALIGNMENT);
}

static gd_tensor_spec tensor_spec_2d(gd_dtype dtype, int64_t dim0, int64_t dim1)
{
    int64_t shape[2];
    shape[0] = dim0;
    shape[1] = dim1;
    return gd_tensor_spec_make(dtype, gd_shape_make(2U, shape), GPT_ALIGNMENT);
}

static gd_linear_layer_config linear_config(int64_t in_features,
                                            int64_t out_features,
                                            uint64_t seed,
                                            float init_scale)
{
    gd_linear_layer_config cfg = gd_linear_layer_config_make(in_features,
                                                             out_features,
                                                             GD_DTYPE_F16,
                                                             seed);
    cfg.use_bias = false;
    cfg.alignment = GPT_ALIGNMENT;
    cfg.weight_low = -init_scale;
    cfg.weight_high = init_scale;
    return cfg;
}

static void gpt_block_init(gd_context *ctx,
                           gpt_block *block,
                           uint32_t index,
                           int n_layers,
                           uint64_t seed)
{
    const float base_scale = 0.02f;
    const float residual_scale = base_scale / sqrtf(2.0f * (float)n_layers);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec one = gd_init_one();
    const gd_linear_layer_config qkv_cfg = linear_config(GPT_D_MODEL,
                                                         3 * GPT_D_MODEL,
                                                         splitmix64(seed ^ ((uint64_t)index << 8) ^ 1U),
                                                         base_scale);
    const gd_linear_layer_config attn_cfg = linear_config(GPT_D_MODEL,
                                                          GPT_D_MODEL,
                                                          splitmix64(seed ^ ((uint64_t)index << 8) ^ 2U),
                                                          residual_scale);
    const gd_linear_layer_config up_gate_cfg = linear_config(GPT_D_MODEL,
                                                             2 * GPT_MLP_HIDDEN,
                                                             splitmix64(seed ^ ((uint64_t)index << 8) ^ 3U),
                                                             base_scale);
    const gd_linear_layer_config down_cfg = linear_config(GPT_MLP_HIDDEN,
                                                          GPT_D_MODEL,
                                                          splitmix64(seed ^ ((uint64_t)index << 8) ^ 4U),
                                                          residual_scale);
    char name[32];
    memset(block, 0, sizeof(*block));
    (void)snprintf(name, sizeof(name), "block_%u", (unsigned)index);
    TRY(ctx, gd_module_init(ctx, &block->mod, name));
    TRY(ctx, gd_module_param(ctx, &block->mod, "attn_norm_w", &norm_spec, &one, &block->attn_norm_w));
    TRY(ctx, gd_module_param(ctx, &block->mod, "mlp_norm_w", &norm_spec, &one, &block->mlp_norm_w));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "qkv_proj", &block->qkv_proj, &qkv_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "attn_proj", &block->attn_proj, &attn_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "up_gate", &block->up_gate, &up_gate_cfg));
    TRY(ctx, gd_linear_layer_init_child(ctx, &block->mod, "down_proj", &block->down_proj, &down_cfg));
}

void gpt_lm_init(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    const gd_tensor_spec embed_spec = tensor_spec_2d(GD_DTYPE_F16, GPT_VOCAB_SIZE, GPT_D_MODEL);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec embed_init = gd_init_rand_uniform(splitmix64(config->seed ^ UINT64_C(0xabc001)),
                                                         -0.02f,
                                                         0.02f);
    const gd_init_spec one = gd_init_one();
    uint32_t i;
    memset(model, 0, sizeof(*model));
    model->n_layers = config->n_layers;
    model->vocab_size = GPT_VOCAB_SIZE;
    model->context_length = GPT_CONTEXT_LENGTH;
    model->d_model = GPT_D_MODEL;
    model->n_heads = GPT_N_HEADS;
    model->head_dim = GPT_HEAD_DIM;
    model->mlp_hidden = GPT_MLP_HIDDEN;
    model->sdpa_window = GPT_SDPA_WINDOW;
    model->dropout_p = config->dropout_p;
    model->rms_eps = GPT_DEFAULT_RMS_EPS;
    model->powlu_m = GPT_DEFAULT_POWLU_M;
    model->dropout_seed = splitmix64(config->seed ^ UINT64_C(0xd00d1234));

    TRY(ctx, gd_module_init(ctx, &model->mod, "gpt_lm"));
    TRY(ctx, gd_module_param(ctx,
                             &model->mod,
                             "token_embedding",
                             &embed_spec,
                             &embed_init,
                             &model->token_embedding));
    TRY(ctx, gd_module_param(ctx, &model->mod, "final_norm_w", &norm_spec, &one, &model->final_norm_w));
    model->block_items = (gpt_block *)calloc((size_t)model->n_layers, sizeof(model->block_items[0]));
    if (model->block_items == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "calloc blocks", __LINE__);
    }
    TRY(ctx, gd_module_list_init_child(ctx, &model->mod, "blocks", &model->blocks, (uint32_t)model->n_layers));
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        gpt_block_init(ctx, &model->block_items[i], i, model->n_layers, config->seed);
        TRY(ctx, gd_module_list_set(&model->blocks, i, &model->block_items[i].mod));
    }
}

static void gpt_block_deinit(gpt_block *block)
{
    if (block == NULL) {
        return;
    }
    gd_linear_layer_deinit(&block->down_proj);
    gd_linear_layer_deinit(&block->up_gate);
    gd_linear_layer_deinit(&block->attn_proj);
    gd_linear_layer_deinit(&block->qkv_proj);
    gd_module_deinit(&block->mod);
}

void gpt_lm_deinit(gpt_lm *model)
{
    uint32_t i;
    if (model == NULL) {
        return;
    }
    if (model->block_items != NULL) {
        for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
            gpt_block_deinit(&model->block_items[i]);
        }
    }
    gd_module_list_deinit(&model->blocks);
    free(model->block_items);
    gd_module_deinit(&model->mod);
    memset(model, 0, sizeof(*model));
}

static gd_status gpt_kv_cache_init(gd_context *ctx,
                                   const gpt_lm *model,
                                   int batch_size,
                                   int max_seq,
                                   gpt_kv_cache *cache)
{
    int64_t shape[4];
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || batch_size <= 0 || max_seq <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(cache, 0, sizeof(*cache));
    cache->k = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->k[0]));
    cache->v = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->v[0]));
    cache->pos = (int32_t *)calloc((size_t)batch_size, sizeof(cache->pos[0]));
    if (cache->k == NULL || cache->v == NULL || cache->pos == NULL) {
        free(cache->k);
        free(cache->v);
        free(cache->pos);
        memset(cache, 0, sizeof(*cache));
        return GD_ERR_OUT_OF_MEMORY;
    }
    cache->batch_size = batch_size;
    cache->max_seq = max_seq;
    cache->n_layers = model->n_layers;
    cache->n_heads = model->n_heads;
    cache->head_dim = model->head_dim;
    shape[0] = batch_size;
    shape[1] = max_seq;
    shape[2] = model->n_heads;
    shape[3] = model->head_dim;
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        gd_status st = gd_tensor_empty(ctx,
                                       GD_ARENA_STATE,
                                       GD_DTYPE_F16,
                                       gd_shape_make(4U, shape),
                                       GPT_ALIGNMENT,
                                       &cache->k[i]);
        if (st != GD_OK) {
            free(cache->k);
            free(cache->v);
            free(cache->pos);
            memset(cache, 0, sizeof(*cache));
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F16,
                             gd_shape_make(4U, shape),
                             GPT_ALIGNMENT,
                             &cache->v[i]);
        if (st != GD_OK) {
            free(cache->k);
            free(cache->v);
            free(cache->pos);
            memset(cache, 0, sizeof(*cache));
            return st;
        }
        cache->k[i].is_leaf = false;
        cache->v[i].is_leaf = false;
    }
    return GD_OK;
}

static void gpt_kv_cache_deinit(gpt_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }
    free(cache->k);
    free(cache->v);
    free(cache->pos);
    memset(cache, 0, sizeof(*cache));
}

static gd_status gpt_block_forward(gd_context *ctx,
                                   gpt_lm *model,
                                   gpt_block *block,
                                   uint32_t block_index,
                                   const gd_tensor *x,
                                   const gd_tensor *positions,
                                   const gd_tensor *cu_seqlens,
                                   uint64_t step,
                                   gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_varlen_config sdpa_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_drop;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_tensor mlp_drop;
    gd_status st;
    int64_t n_tokens;
    int64_t q_shape[3];
    int64_t flat_shape[2];
    uint64_t site_seed;

    if (ctx == NULL || model == NULL || block == NULL || x == NULL || positions == NULL ||
        cu_seqlens == NULL || out == NULL || x->rank != 2U || x->shape[1] != GPT_D_MODEL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_tokens = x->shape[0];
    q_shape[0] = n_tokens;
    q_shape[1] = GPT_N_HEADS;
    q_shape[2] = GPT_HEAD_DIM;
    flat_shape[0] = n_tokens;
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(3U, q_shape), &q_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(3U, q_shape), &k_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(3U, q_shape), &v_view);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) {
        return st;
    }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) {
        return st;
    }
    st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
    if (st != GD_OK) {
        return st;
    }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_ATTN ^ ((uint64_t)block_index << 32) ^ step);
    st = gd_dropout(ctx, &attn_proj, model->dropout_p, model->mod.training, site_seed, &attn_drop);
    if (st != GD_OK) {
        return st;
    }
    st = gd_add(ctx, &residual, &attn_drop, &attn_resid);
    if (st != GD_OK) {
        return st;
    }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) {
        return st;
    }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) {
        return st;
    }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) {
        return st;
    }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_MLP ^ ((uint64_t)block_index << 32) ^ step);
    st = gd_dropout(ctx, &mlp_proj, model->dropout_p, model->mod.training, site_seed, &mlp_drop);
    if (st != GD_OK) {
        return st;
    }
    return gd_add(ctx, &residual, &mlp_drop, out);
}

gd_status gpt_lm_forward(gd_context *ctx,
                                gpt_lm *model,
                                const gd_tensor *input_ids,
                                const gd_tensor *target_ids,
                                const gd_tensor *positions,
                                const gd_tensor *cu_seqlens,
                                uint64_t step,
                                gd_tensor *loss)
{
    gd_tensor x;
    gd_tensor dropped;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_status st;
    uint32_t i;
    uint64_t site_seed;

    if (ctx == NULL || model == NULL || input_ids == NULL || target_ids == NULL ||
        positions == NULL || cu_seqlens == NULL || loss == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (input_ids->rank != 1U || target_ids->rank != 1U || positions->rank != 1U ||
        input_ids->shape[0] != target_ids->shape[0] || input_ids->shape[0] != positions->shape[0]) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (input_ids->shape[0] <= 0 || (input_ids->shape[0] % GPT_CONTEXT_LENGTH) != 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (cu_seqlens->rank != 1U || cu_seqlens->shape[0] != input_ids->shape[0] / GPT_CONTEXT_LENGTH + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }

    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_EMBED ^ step);
    st = gd_dropout(ctx, &x, model->dropout_p, model->mod.training, site_seed, &dropped);
    if (st != GD_OK) {
        return st;
    }
    x = dropped;

    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_forward(ctx,
                               model,
                               &model->block_items[i],
                               i,
                               &x,
                               positions,
                               cu_seqlens,
                               step,
                               &block_out);
        if (st != GD_OK) {
            return st;
        }
        x = block_out;
    }

    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) {
        return st;
    }
    return gd_lm_cross_entropy(ctx, &final_norm, &model->token_embedding, target_ids, loss);
}

static gd_status gpt_block_prefill_cached(gd_context *ctx,
                                           gpt_lm *model,
                                           gpt_kv_cache *cache,
                                           gpt_block *block,
                                           uint32_t block_index,
                                           const gd_tensor *x,
                                           const gd_tensor *positions,
                                           const gd_tensor *cu_seqlens,
                                           const gd_tensor *cache_positions,
                                           gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_varlen_config sdpa_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t n_tokens;
    int64_t q_shape[3];
    int64_t flat_shape[2];

    if (ctx == NULL || model == NULL || cache == NULL || block == NULL || x == NULL ||
        positions == NULL || cu_seqlens == NULL || cache_positions == NULL || out == NULL ||
        x->rank != 2U || x->shape[1] != GPT_D_MODEL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_tokens = x->shape[0];
    if (n_tokens <= 0 || n_tokens > (int64_t)cache->batch_size * (int64_t)cache->max_seq ||
        block_index >= (uint32_t)cache->n_layers || cache_positions->rank != 1U ||
        cache_positions->shape[0] != cache->batch_size || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] != cache->batch_size + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    q_shape[0] = n_tokens;
    q_shape[1] = GPT_N_HEADS;
    q_shape[2] = GPT_HEAD_DIM;
    flat_shape[0] = n_tokens;
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(3U, q_shape), &q_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(3U, q_shape), &k_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(3U, q_shape), &v_view);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) { return st; }
    st = gd_kv_cache_append_packed(ctx,
                                    &cache->k[block_index],
                                    &cache->v[block_index],
                                    cache_positions,
                                    cu_seqlens,
                                    &k_rot,
                                    &v_view);
    if (st != GD_OK) { return st; }
    st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) { return st; }
    st = gd_add(ctx, &residual, &attn_proj, &attn_resid);
    if (st != GD_OK) { return st; }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) { return st; }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) { return st; }
    return gd_add(ctx, &residual, &mlp_proj, out);
}

static gd_status gpt_block_decode_cached(gd_context *ctx,
                                         gpt_lm *model,
                                         gpt_kv_cache *cache,
                                         gpt_block *block,
                                         uint32_t block_index,
                                         const gd_tensor *x,
                                         const gd_tensor *positions,
                                         const gd_tensor *cache_positions,
                                         gd_tensor *out)
{
    const gd_rope_config rope_cfg = {
        .theta = 10000.0f,
        .n_dims = GPT_HEAD_DIM,
        .interleaved = false,
    };
    const gd_sdpa_decode_config sdpa_cfg = {
        .scale = 0.0f,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
    };
    int64_t qkv_sizes[3] = {GPT_D_MODEL, GPT_D_MODEL, GPT_D_MODEL};
    int64_t mlp_sizes[2] = {GPT_MLP_HIDDEN, GPT_MLP_HIDDEN};
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor qkv_parts[3];
    gd_tensor q_view;
    gd_tensor k_view;
    gd_tensor v_view;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor up_gate_parts[2];
    gd_tensor powlu;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t q_shape[4];
    int64_t flat_shape[2];

    if (ctx == NULL || model == NULL || cache == NULL || block == NULL || x == NULL ||
        positions == NULL || cache_positions == NULL || out == NULL || x->rank != 2U ||
        x->shape[1] != GPT_D_MODEL || x->shape[0] != cache->batch_size ||
        positions->rank != 1U || positions->shape[0] != cache->batch_size ||
        cache_positions->rank != 1U || cache_positions->shape[0] != cache->batch_size ||
        block_index >= (uint32_t)cache->n_layers) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    q_shape[0] = x->shape[0];
    q_shape[1] = 1;
    q_shape[2] = GPT_N_HEADS;
    q_shape[3] = GPT_HEAD_DIM;
    flat_shape[0] = x->shape[0];
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &qkv, qkv_sizes, 3U, -1, qkv_parts);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[0], gd_shape_make(4U, q_shape), &q_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[1], gd_shape_make(4U, q_shape), &k_view);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &qkv_parts[2], gd_shape_make(4U, q_shape), &v_view);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &q_view, positions, &rope_cfg, &q_rot);
    if (st != GD_OK) { return st; }
    st = gd_rope(ctx, &k_view, positions, &rope_cfg, &k_rot);
    if (st != GD_OK) { return st; }
    st = gd_kv_cache_append_positions(ctx,
                                       &cache->k[block_index],
                                       &cache->v[block_index],
                                       cache_positions,
                                       &k_rot,
                                       &v_view);
    if (st != GD_OK) { return st; }
    st = gd_sdpa_decode_positions(ctx,
                                  &q_rot,
                                  &cache->k[block_index],
                                  &cache->v[block_index],
                                  cache_positions,
                                  &sdpa_cfg,
                                  &attn);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &attn, gd_shape_make(2U, flat_shape), &attn_flat);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->attn_proj, &attn_flat, &attn_proj);
    if (st != GD_OK) { return st; }
    st = gd_add(ctx, &residual, &attn_proj, &attn_resid);
    if (st != GD_OK) { return st; }

    residual = attn_resid;
    st = gd_rms_norm(ctx, &attn_resid, &block->mlp_norm_w, model->rms_eps, &mlp_normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->up_gate, &mlp_normed, &up_gate);
    if (st != GD_OK) { return st; }
    st = gd_split(ctx, &up_gate, mlp_sizes, 2U, -1, up_gate_parts);
    if (st != GD_OK) { return st; }
    st = gd_powlu(ctx, &up_gate_parts[0], &up_gate_parts[1], model->powlu_m, &powlu);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->down_proj, &powlu, &mlp_proj);
    if (st != GD_OK) { return st; }
    return gd_add(ctx, &residual, &mlp_proj, out);
}

static gd_status gpt_lm_prefill_logits(gd_context *ctx,
                                       gpt_lm *model,
                                       gpt_kv_cache *cache,
                                       const gd_tensor *input_ids,
                                       const gd_tensor *positions,
                                       const gd_tensor *cu_seqlens,
                                       const gd_tensor *cache_positions,
                                       gd_tensor *logits)
{
    gd_tensor x;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_status st;
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || input_ids == NULL || positions == NULL ||
        cu_seqlens == NULL || cache_positions == NULL || logits == NULL || input_ids->rank != 1U ||
        positions->rank != 1U || input_ids->shape[0] != positions->shape[0] ||
        input_ids->shape[0] <= 0 || cache_positions->rank != 1U ||
        cache_positions->shape[0] != cache->batch_size || cu_seqlens->rank != 1U ||
        cu_seqlens->shape[0] != cache->batch_size + 1) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_prefill_cached(ctx,
                                      model,
                                      cache,
                                      &model->block_items[i],
                                      i,
                                      &x,
                                      positions,
                                      cu_seqlens,
                                      cache_positions,
                                      &block_out);
        if (st != GD_OK) { return st; }
        x = block_out;
    }
    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) { return st; }
    st = gd_linear_transposed_weight(ctx, &final_norm, &model->token_embedding, NULL, logits);
    if (st != GD_OK) { return st; }
    return GD_OK;
}

static gd_status gpt_lm_decode_logits(gd_context *ctx,
                                      gpt_lm *model,
                                      gpt_kv_cache *cache,
                                      const gd_tensor *input_ids,
                                      const gd_tensor *positions,
                                      const gd_tensor *cache_positions,
                                      gd_tensor *logits)
{
    gd_tensor x;
    gd_tensor block_out;
    gd_tensor final_norm;
    gd_status st;
    uint32_t i;
    if (ctx == NULL || model == NULL || cache == NULL || input_ids == NULL || positions == NULL ||
        cache_positions == NULL || logits == NULL || input_ids->rank != 1U || positions->rank != 1U ||
        input_ids->shape[0] != positions->shape[0] || input_ids->shape[0] != cache->batch_size ||
        cache_positions->rank != 1U || cache_positions->shape[0] != cache->batch_size) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_embedding(ctx, &model->token_embedding, input_ids, &x);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gpt_block_decode_cached(ctx,
                                     model,
                                     cache,
                                     &model->block_items[i],
                                     i,
                                     &x,
                                     positions,
                                     cache_positions,
                                     &block_out);
        if (st != GD_OK) { return st; }
        x = block_out;
    }
    st = gd_rms_norm(ctx, &x, &model->final_norm_w, model->rms_eps, &final_norm);
    if (st != GD_OK) { return st; }
    st = gd_linear_transposed_weight(ctx, &final_norm, &model->token_embedding, NULL, logits);
    if (st != GD_OK) { return st; }
    for (i = 0U; i < (uint32_t)cache->batch_size; ++i) {
        cache->pos[i] += 1;
    }
    return GD_OK;
}


char *gpt_default_tokenizer_path(const char *data_dir)
{
    const char *file = "tokenizer-v2048.json";
    const size_t data_len = strlen(data_dir);
    const bool need_sep = data_len == 0U || data_dir[data_len - 1U] != '/';
    const size_t file_len = strlen(file);
    char *path;
    if (data_len > SIZE_MAX - file_len - (need_sep ? 2U : 1U)) {
        return NULL;
    }
    path = (char *)malloc(data_len + file_len + (need_sep ? 2U : 1U));
    if (path == NULL) {
        return NULL;
    }
    (void)sprintf(path, "%s%s%s", data_dir, need_sep ? "/" : "", file);
    return path;
}

static gd_status gpt_data_i32_tensor(gd_context *ctx,
                                     const int32_t *values,
                                     int64_t count,
                                     gd_tensor *out)
{
    gd_status st;
    if (ctx == NULL || values == NULL || out == NULL || count <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_DATA, GD_DTYPE_I32, GD_SHAPE(count), GPT_ALIGNMENT, out);
    if (st != GD_OK) {
        return st;
    }
    return gd_tensor_write(ctx, out, values, (size_t)count * sizeof(values[0]));
}

static int gpt_sample_next_token(const float *logits,
                                 int vocab_size,
                                 float temperature,
                                 uint64_t *rng)
{
    int best = 0;
    float best_value = logits[0];
    int i;
    for (i = 1; i < vocab_size; ++i) {
        if (logits[i] > best_value) {
            best_value = logits[i];
            best = i;
        }
    }
    if (temperature <= 0.0f) {
        return best;
    }
    {
        double sum = 0.0;
        double target;
        uint64_t r;
        for (i = 0; i < vocab_size; ++i) {
            const double z = ((double)logits[i] - (double)best_value) / (double)temperature;
            sum += exp(z);
        }
        if (!(sum > 0.0) || !isfinite(sum)) {
            return best;
        }
        *rng = splitmix64(*rng);
        r = *rng >> 11;
        target = ((double)r * (1.0 / 9007199254740992.0)) * sum;
        for (i = 0; i < vocab_size; ++i) {
            const double z = ((double)logits[i] - (double)best_value) / (double)temperature;
            target -= exp(z);
            if (target <= 0.0) {
                return i;
            }
        }
    }
    return best;
}

static void gpt_generate_prompts(gd_context *ctx,
                                 gpt_lm *model,
                                 const gpt_config *config,
                                 const char *const *prompts,
                                 int n_prompts,
                                 const char *tag,
                                 size_t step,
                                 bool restore_training)
{
    gd_tokenizer *tok = NULL;
    gd_tokenizer_config tok_cfg;
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    int32_t **encoded = NULL;
    int32_t **seq_ids = NULL;
    int *n_encoded = NULL;
    int *prompt_offset = NULL;
    int *prompt_len = NULL;
    int *seq_len = NULL;
    int32_t *packed_ids = NULL;
    int32_t *packed_positions = NULL;
    int32_t *cu = NULL;
    int32_t *cache_pos_values = NULL;
    int32_t *decode_ids = NULL;
    int32_t *decode_positions = NULL;
    int *next_ids = NULL;
    gd_tensor *last_logits = NULL;
    float *logits_host = NULL;
    gpt_kv_cache cache;
    int total_prompt_tokens = 0;
    int max_new;
    int room_for_prompt;
    int generated = 0;
    int b;
    int i;
    uint64_t rng;
    double start;
    double elapsed;

    memset(&cache, 0, sizeof(cache));
    if (ctx == NULL || model == NULL || config == NULL || prompts == NULL || n_prompts <= 0) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "generation arguments", __LINE__);
    }
    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "tokenizer path allocation", __LINE__);
    }
    TRY(ctx, gd_bpe_tokenizer_load(tokenizer_path, &tok_cfg, &tok));

    encoded = (int32_t **)calloc((size_t)n_prompts, sizeof(encoded[0]));
    seq_ids = (int32_t **)calloc((size_t)n_prompts, sizeof(seq_ids[0]));
    n_encoded = (int *)calloc((size_t)n_prompts, sizeof(n_encoded[0]));
    prompt_offset = (int *)calloc((size_t)n_prompts, sizeof(prompt_offset[0]));
    prompt_len = (int *)calloc((size_t)n_prompts, sizeof(prompt_len[0]));
    seq_len = (int *)calloc((size_t)n_prompts, sizeof(seq_len[0]));
    cu = (int32_t *)calloc((size_t)n_prompts + 1U, sizeof(cu[0]));
    cache_pos_values = (int32_t *)calloc((size_t)n_prompts, sizeof(cache_pos_values[0]));
    decode_ids = (int32_t *)calloc((size_t)n_prompts, sizeof(decode_ids[0]));
    decode_positions = (int32_t *)calloc((size_t)n_prompts, sizeof(decode_positions[0]));
    next_ids = (int *)calloc((size_t)n_prompts, sizeof(next_ids[0]));
    last_logits = (gd_tensor *)calloc((size_t)n_prompts, sizeof(last_logits[0]));
    logits_host = (float *)malloc((size_t)n_prompts * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (encoded == NULL || seq_ids == NULL || n_encoded == NULL || prompt_offset == NULL ||
        prompt_len == NULL || seq_len == NULL || cu == NULL || cache_pos_values == NULL ||
        decode_ids == NULL || decode_positions == NULL || next_ids == NULL || last_logits == NULL ||
        logits_host == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation allocation", __LINE__);
    }

    max_new = config->max_new_tokens;
    room_for_prompt = GPT_CONTEXT_LENGTH - max_new;
    if (room_for_prompt <= 0) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "no context room for generation", __LINE__);
    }
    for (b = 0; b < n_prompts; ++b) {
        TRY(ctx, gd_tokenizer_encode(tok, prompts[b], &encoded[b], &n_encoded[b]));
        if (n_encoded[b] <= 0) {
            gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "empty generation prompt", __LINE__);
        }
        if (n_encoded[b] > room_for_prompt) {
            prompt_offset[b] = n_encoded[b] - room_for_prompt;
            printf("generate%s%s: prompt[%d] tokens=%d exceeds context budget; using last %d tokens\n",
                   tag != NULL ? ":" : "",
                   tag != NULL ? tag : "",
                   b,
                   n_encoded[b],
                   room_for_prompt);
        }
        prompt_len[b] = n_encoded[b] - prompt_offset[b];
        cu[b] = (int32_t)total_prompt_tokens;
        total_prompt_tokens += prompt_len[b];
        seq_ids[b] = (int32_t *)calloc((size_t)GPT_CONTEXT_LENGTH, sizeof(seq_ids[b][0]));
        if (seq_ids[b] == NULL) {
            gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation sequence allocation", __LINE__);
        }
        memcpy(seq_ids[b], encoded[b] + prompt_offset[b], (size_t)prompt_len[b] * sizeof(seq_ids[b][0]));
        seq_len[b] = prompt_len[b];
    }
    cu[n_prompts] = (int32_t)total_prompt_tokens;
    packed_ids = (int32_t *)calloc((size_t)total_prompt_tokens, sizeof(packed_ids[0]));
    packed_positions = (int32_t *)calloc((size_t)total_prompt_tokens, sizeof(packed_positions[0]));
    if (packed_ids == NULL || packed_positions == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation packed prompt allocation", __LINE__);
    }
    for (b = 0; b < n_prompts; ++b) {
        for (i = 0; i < prompt_len[b]; ++i) {
            const int dst = (int)cu[b] + i;
            packed_ids[dst] = seq_ids[b][i];
            packed_positions[dst] = cache_pos_values[b] + i;
        }
    }

    rng = splitmix64(config->seed ^ UINT64_C(0xdec0de1234567890) ^ (uint64_t)n_prompts ^ (uint64_t)step);
    TRY(ctx, gpt_kv_cache_init(ctx, model, n_prompts, GPT_CONTEXT_LENGTH, &cache));
    gd_module_set_training(&model->mod, false);
    start = gpt_wall_seconds();

    {
        gd_tensor ids_t;
        gd_tensor pos_t;
        gd_tensor cu_t;
        gd_tensor cache_pos_t;
        gd_tensor logits;
        TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
        TRY(ctx, gpt_data_i32_tensor(ctx, packed_ids, total_prompt_tokens, &ids_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, packed_positions, total_prompt_tokens, &pos_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, cu, n_prompts + 1, &cu_t));
        TRY(ctx, gpt_data_i32_tensor(ctx, cache_pos_values, n_prompts, &cache_pos_t));
        TRY(ctx, gpt_lm_prefill_logits(ctx,
                                       model,
                                       &cache,
                                       &ids_t,
                                       &pos_t,
                                       &cu_t,
                                       &cache_pos_t,
                                       &logits));
        for (b = 0; b < n_prompts; ++b) {
            TRY(ctx, gd_tensor_slice(ctx, &logits, 0U, (int64_t)cu[b + 1] - 1, 1, &last_logits[b]));
        }
        TRY(ctx, gd_end_step(ctx));
        for (b = 0; b < n_prompts; ++b) {
            TRY(ctx, gd_tensor_read_f32(ctx,
                                        &last_logits[b],
                                        logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                        GPT_VOCAB_SIZE));
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                config->temperature,
                                                &rng);
            cache.pos[b] = prompt_len[b];
        }
    }

    while (generated < max_new) {
        for (b = 0; b < n_prompts; ++b) {
            seq_ids[b][seq_len[b]] = (int32_t)next_ids[b];
            seq_len[b] += 1;
        }
        generated += 1;
        if (generated >= max_new) {
            break;
        }
        for (b = 0; b < n_prompts; ++b) {
            decode_ids[b] = (int32_t)next_ids[b];
            decode_positions[b] = cache.pos[b];
            cache_pos_values[b] = cache.pos[b];
        }
        {
            gd_tensor ids_t;
            gd_tensor pos_t;
            gd_tensor cache_pos_t;
            gd_tensor logits;
            TRY(ctx, gd_begin_step(ctx, GD_SCOPE_INFER, gd_batch_empty()));
            TRY(ctx, gpt_data_i32_tensor(ctx, decode_ids, n_prompts, &ids_t));
            TRY(ctx, gpt_data_i32_tensor(ctx, decode_positions, n_prompts, &pos_t));
            TRY(ctx, gpt_data_i32_tensor(ctx, cache_pos_values, n_prompts, &cache_pos_t));
            TRY(ctx, gpt_lm_decode_logits(ctx, model, &cache, &ids_t, &pos_t, &cache_pos_t, &logits));
            TRY(ctx, gd_end_step(ctx));
            TRY(ctx, gd_tensor_read_f32(ctx,
                                        &logits,
                                        logits_host,
                                        (size_t)n_prompts * (size_t)GPT_VOCAB_SIZE));
        }
        for (b = 0; b < n_prompts; ++b) {
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                config->temperature,
                                                &rng);
        }
    }

    elapsed = gpt_wall_seconds() - start;
    printf("generate%s%s: tokenizer=%s batch=%d generated=%d temperature=%.3f elapsed=%.3fs tok/s=%.1f",
           tag != NULL ? ":" : "",
           tag != NULL ? tag : "",
           tokenizer_path,
           n_prompts,
           generated,
           (double)config->temperature,
           elapsed,
           elapsed > 0.0 ? (double)((size_t)n_prompts * (size_t)generated) / elapsed : 0.0);
    if (step != 0U) {
        printf(" step=%zu", step);
    }
    printf("\n");
    for (b = 0; b < n_prompts; ++b) {
        char *decoded = NULL;
        char *decoded_new = NULL;
        TRY(ctx, gd_tokenizer_decode(tok, seq_ids[b], seq_len[b], &decoded));
        TRY(ctx, gd_tokenizer_decode(tok, seq_ids[b] + prompt_len[b], generated, &decoded_new));
        printf("  prefix=\"%s\" prompt_tokens=%d generated_text=%s\n",
               prompts[b],
               prompt_len[b],
               decoded_new != NULL ? decoded_new : "");
        if (n_prompts == 1) {
            printf("full_text:\n%s\n", decoded != NULL ? decoded : "");
        }
        gd_tokenizer_free(decoded_new);
        gd_tokenizer_free(decoded);
    }

    if (restore_training) {
        gd_module_set_training(&model->mod, true);
    }
    gpt_kv_cache_deinit(&cache);
    for (b = 0; b < n_prompts; ++b) {
        gd_tokenizer_free(encoded[b]);
        free(seq_ids[b]);
    }
    free(logits_host);
    free(last_logits);
    free(next_ids);
    free(decode_positions);
    free(decode_ids);
    free(cache_pos_values);
    free(cu);
    free(packed_positions);
    free(packed_ids);
    free(seq_len);
    free(prompt_len);
    free(prompt_offset);
    free(n_encoded);
    free(seq_ids);
    free(encoded);
    gd_tokenizer_destroy(tok);
    free(default_tok_path);
}

void gpt_generate(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    const char *prompts[1];
    prompts[0] = config->generate_prompt;
    gpt_generate_prompts(ctx, model, config, prompts, 1, "user", 0U, false);
}

void gpt_generate_vowels(gd_context *ctx,
                         gpt_lm *model,
                         const gpt_config *config,
                         size_t step)
{
    static const char *const prompts[5] = {"a", "e", "i", "o", "u"};
    gpt_generate_prompts(ctx, model, config, prompts, 5, "vowels", step, true);
}
