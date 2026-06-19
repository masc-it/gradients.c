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

#define GPT_GENERATE_VOWEL_PROMPT_COUNT 5U

static const float GPT_WEIGHT_INIT_STD = 0.02f;
static const double GPT_TWO_PI = 6.283185307179586476925286766559;

static uint64_t splitmix64_scramble(uint64_t x)
{
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t splitmix64(uint64_t x)
{
    return splitmix64_scramble(x + UINT64_C(0x9e3779b97f4a7c15));
}

static uint64_t splitmix64_next(uint64_t *state)
{
    *state += UINT64_C(0x9e3779b97f4a7c15);
    return splitmix64_scramble(*state);
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

static int int_max2(int a, int b)
{
    return a > b ? a : b;
}

static int gpt_generation_cache_batch_capacity(const gpt_config *config)
{
    int capacity = 0;
    if (config == NULL) {
        return 0;
    }
    if (config->generate_prompt != NULL) {
        capacity = int_max2(capacity, 1);
    }
    if (config->generate_every_n_steps > 0) {
        capacity = int_max2(capacity, (int)GPT_GENERATE_VOWEL_PROMPT_COUNT);
    }
    return capacity;
}

static size_t gpt_kv_cache_state_bytes(int n_layers, int batch_capacity, int max_seq)
{
    size_t count;
    if (n_layers <= 0 || batch_capacity <= 0 || max_seq <= 0) {
        return 0U;
    }
    count = checked_mul_size((size_t)n_layers, 2U, "KV cache K/V tensors");
    count = checked_mul_size(count, (size_t)batch_capacity, "KV cache batch");
    count = checked_mul_size(count, (size_t)max_seq, "KV cache sequence");
    count = checked_mul_size(count, (size_t)GPT_N_HEADS, "KV cache heads");
    count = checked_mul_size(count, (size_t)GPT_HEAD_DIM, "KV cache head dim");
    return checked_mul_size(count, gd_dtype_size(GD_DTYPE_F16), "KV cache bytes");
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
    const int generation_batch_capacity = gpt_generation_cache_batch_capacity(config);
    const size_t generation_cache_bytes = gpt_kv_cache_state_bytes(config->n_layers,
                                                                    generation_batch_capacity,
                                                                    GPT_CONTEXT_LENGTH);
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
                                checked_add_size(checked_add_size(adam_bytes,
                                                                  generation_cache_bytes,
                                                                  "state arena optimizer+KV bytes"),
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

static double gpt_rng_uniform_open_closed(uint64_t *state)
{
    const uint64_t bits = splitmix64_next(state) >> 11;
    return ((double)bits + 1.0) * (1.0 / 9007199254740992.0);
}

static void gpt_fill_normal(float *values, size_t count, uint64_t seed, float stddev)
{
    uint64_t state = seed;
    size_t i = 0U;
    while (i < count) {
        const double u1 = gpt_rng_uniform_open_closed(&state);
        const double u2 = gpt_rng_uniform_open_closed(&state);
        const double radius = sqrt(-2.0 * log(u1));
        const double theta = GPT_TWO_PI * u2;
        values[i] = (float)(radius * cos(theta) * (double)stddev);
        i += 1U;
        if (i < count) {
            values[i] = (float)(radius * sin(theta) * (double)stddev);
            i += 1U;
        }
    }
}

static void gpt_tensor_init_normal(gd_context *ctx, gd_tensor *tensor, uint64_t seed, float stddev)
{
    int64_t numel_i64;
    size_t count;
    float *values;
    if (!(stddev >= 0.0f) || !isfinite(stddev)) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "invalid GPT normal init stddev", __LINE__);
    }
    TRY(ctx, gd_tensor_numel(tensor, &numel_i64));
    if (numel_i64 <= 0 || (uint64_t)numel_i64 > (uint64_t)SIZE_MAX) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "invalid GPT init tensor shape", __LINE__);
    }
    count = (size_t)numel_i64;
    if (count > SIZE_MAX / sizeof(values[0])) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "GPT init allocation overflow", __LINE__);
    }
    values = (float *)malloc(count * sizeof(values[0]));
    if (values == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "malloc GPT init values", __LINE__);
    }
    gpt_fill_normal(values, count, seed, stddev);
    {
        const gd_status st = gd_tensor_write_f32(ctx, tensor, values, count);
        free(values);
        if (st != GD_OK) {
            gpt_fail_status(ctx, st, "gd_tensor_write_f32 GPT normal init", __LINE__);
        }
    }
}

static void gpt_linear_layer_init_normal_child(gd_context *ctx,
                                               gd_module *parent,
                                               const char *name,
                                               gd_linear_layer *layer,
                                               int64_t in_features,
                                               int64_t out_features,
                                               bool transposed_weight,
                                               uint64_t seed,
                                               float stddev)
{
    int64_t weight_shape[2];
    gd_tensor_spec weight_spec;
    const gd_init_spec empty = gd_init_empty();
    memset(layer, 0, sizeof(*layer));
    weight_shape[0] = transposed_weight ? out_features : in_features;
    weight_shape[1] = transposed_weight ? in_features : out_features;
    weight_spec = gd_tensor_spec_make(GD_DTYPE_F16, gd_shape_make(2U, weight_shape), GPT_ALIGNMENT);
    TRY(ctx, gd_module_init(ctx, &layer->mod, name));
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->has_bias = false;
    layer->weight_transposed = transposed_weight;
    TRY(ctx, gd_module_param(ctx, &layer->mod, "weight", &weight_spec, &empty, &layer->weight));
    gpt_tensor_init_normal(ctx, &layer->weight, seed, stddev);
    TRY(ctx, gd_module_add_child(parent, name, &layer->mod));
}

static gd_status gpt_kv_cache_init(gd_context *ctx,
                                   const gpt_lm *model,
                                   int batch_capacity,
                                   int max_seq,
                                   gpt_kv_cache *cache);
static void gpt_kv_cache_deinit(gpt_kv_cache *cache);

static void gpt_block_init(gd_context *ctx,
                           gpt_block *block,
                           uint32_t index,
                           int n_layers,
                           uint64_t seed)
{
    const float residual_scale = GPT_WEIGHT_INIT_STD / sqrtf(2.0f * (float)n_layers);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec rms_norm_init = gd_init_one();
    const uint64_t block_seed = seed ^ ((uint64_t)index << 8);
    char name[32];
    memset(block, 0, sizeof(*block));
    (void)snprintf(name, sizeof(name), "block_%u", (unsigned)index);
    TRY(ctx, gd_module_init(ctx, &block->mod, name));
    /* GPT-2/NeoX/LLaMA-style pre-norm initialization: RMSNorm scale starts
     * at one, QKV/MLP input projections use N(0, 0.02), and residual-branch
     * output projections are scaled by 1/sqrt(2 * n_layers). */
    TRY(ctx, gd_module_param(ctx, &block->mod, "attn_norm_w", &norm_spec, &rms_norm_init, &block->attn_norm_w));
    TRY(ctx, gd_module_param(ctx, &block->mod, "mlp_norm_w", &norm_spec, &rms_norm_init, &block->mlp_norm_w));
    gpt_linear_layer_init_normal_child(ctx,
                                       &block->mod,
                                       "qkv_proj",
                                       &block->qkv_proj,
                                       GPT_D_MODEL,
                                       3 * GPT_D_MODEL,
                                       true,
                                       splitmix64(block_seed ^ 1U),
                                       GPT_WEIGHT_INIT_STD);
    gpt_linear_layer_init_normal_child(ctx,
                                       &block->mod,
                                       "attn_proj",
                                       &block->attn_proj,
                                       GPT_D_MODEL,
                                       GPT_D_MODEL,
                                       true,
                                       splitmix64(block_seed ^ 2U),
                                       residual_scale);
    gpt_linear_layer_init_normal_child(ctx,
                                       &block->mod,
                                       "up_gate",
                                       &block->up_gate,
                                       GPT_D_MODEL,
                                       2 * GPT_MLP_HIDDEN,
                                       true,
                                       splitmix64(block_seed ^ 3U),
                                       GPT_WEIGHT_INIT_STD);
    gpt_linear_layer_init_normal_child(ctx,
                                       &block->mod,
                                       "down_proj",
                                       &block->down_proj,
                                       GPT_MLP_HIDDEN,
                                       GPT_D_MODEL,
                                       false,
                                       splitmix64(block_seed ^ 4U),
                                       residual_scale);
}

void gpt_lm_init(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    const gd_tensor_spec embed_spec = tensor_spec_2d(GD_DTYPE_F16, GPT_VOCAB_SIZE, GPT_D_MODEL);
    const gd_tensor_spec norm_spec = tensor_spec_1d(GD_DTYPE_F16, GPT_D_MODEL);
    const gd_init_spec empty = gd_init_empty();
    const gd_init_spec rms_norm_init = gd_init_one();
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
    model->architecture = config->architecture;
    model->minimax_m3_block_size = GPT_MINIMAX_M3_BLOCK_SIZE;
    model->minimax_m3_topk_blocks = config->minimax_m3_topk_blocks > 0 ?
                                        config->minimax_m3_topk_blocks :
                                        GPT_MINIMAX_M3_TOPK_BLOCKS;
    model->minimax_m3_init_blocks = config->minimax_m3_init_blocks >= 0 ?
                                        config->minimax_m3_init_blocks :
                                        GPT_MINIMAX_M3_INIT_BLOCKS;
    model->minimax_m3_local_blocks = config->minimax_m3_local_blocks >= 0 ?
                                         config->minimax_m3_local_blocks :
                                         GPT_MINIMAX_M3_LOCAL_BLOCKS;
    model->dropout_p = config->dropout_p;
    model->rms_eps = GPT_DEFAULT_RMS_EPS;
    model->logits_softcap = config->logits_softcap;
    model->dropout_seed = splitmix64(config->seed ^ UINT64_C(0xd00d1234));

    TRY(ctx, gd_module_init(ctx, &model->mod, "gpt_lm"));
    TRY(ctx, gd_module_param(ctx,
                             &model->mod,
                             "token_embedding",
                             &embed_spec,
                             &empty,
                             &model->token_embedding));
    gpt_tensor_init_normal(ctx,
                           &model->token_embedding,
                           splitmix64(config->seed ^ UINT64_C(0xabc001)),
                           GPT_WEIGHT_INIT_STD);
    TRY(ctx, gd_module_param(ctx,
                             &model->mod,
                             "final_norm_w",
                             &norm_spec,
                             &rms_norm_init,
                             &model->final_norm_w));
    model->block_items = (gpt_block *)calloc((size_t)model->n_layers, sizeof(model->block_items[0]));
    if (model->block_items == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "calloc blocks", __LINE__);
    }
    TRY(ctx, gd_module_list_init_child(ctx, &model->mod, "blocks", &model->blocks, (uint32_t)model->n_layers));
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        gpt_block_init(ctx, &model->block_items[i], i, model->n_layers, config->seed);
        TRY(ctx, gd_module_list_set(&model->blocks, i, &model->block_items[i].mod));
    }
    {
        const int generation_batch_capacity = gpt_generation_cache_batch_capacity(config);
        if (generation_batch_capacity > 0) {
            TRY(ctx, gpt_kv_cache_init(ctx,
                                       model,
                                       generation_batch_capacity,
                                       GPT_CONTEXT_LENGTH,
                                       &model->generation_cache));
        }
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
    gpt_kv_cache_deinit(&model->generation_cache);
    gd_module_list_deinit(&model->blocks);
    free(model->block_items);
    gd_module_deinit(&model->mod);
    memset(model, 0, sizeof(*model));
}

static gd_status gpt_kv_cache_set_active_batch(gpt_kv_cache *cache, int batch_size)
{
    uint32_t i;
    if (cache == NULL || cache->k == NULL || cache->v == NULL || cache->pos == NULL ||
        batch_size <= 0 || batch_size > cache->batch_capacity) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    cache->batch_size = batch_size;
    memset(cache->pos, 0, (size_t)batch_size * sizeof(cache->pos[0]));
    for (i = 0U; i < (uint32_t)cache->n_layers; ++i) {
        if (cache->k[i].rank != 4U || cache->v[i].rank != 4U ||
            cache->k[i].shape[1] != cache->max_seq || cache->v[i].shape[1] != cache->max_seq) {
            return GD_ERR_BAD_STATE;
        }
        cache->k[i].shape[0] = batch_size;
        cache->v[i].shape[0] = batch_size;
    }
    return GD_OK;
}

static gd_status gpt_kv_cache_init(gd_context *ctx,
                                   const gpt_lm *model,
                                   int batch_capacity,
                                   int max_seq,
                                   gpt_kv_cache *cache)
{
    int64_t shape[4];
    uint32_t i;
    gd_status st;
    if (ctx == NULL || model == NULL || cache == NULL || batch_capacity <= 0 || max_seq <= 0 ||
        model->n_layers <= 0 || model->n_heads <= 0 || model->head_dim <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(cache, 0, sizeof(*cache));
    cache->k = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->k[0]));
    cache->v = (gd_tensor *)calloc((size_t)model->n_layers, sizeof(cache->v[0]));
    cache->pos = (int32_t *)calloc((size_t)batch_capacity, sizeof(cache->pos[0]));
    if (cache->k == NULL || cache->v == NULL || cache->pos == NULL) {
        free(cache->k);
        free(cache->v);
        free(cache->pos);
        memset(cache, 0, sizeof(*cache));
        return GD_ERR_OUT_OF_MEMORY;
    }
    cache->batch_capacity = batch_capacity;
    cache->max_seq = max_seq;
    cache->n_layers = model->n_layers;
    cache->n_heads = model->n_heads;
    cache->head_dim = model->head_dim;
    shape[0] = batch_capacity;
    shape[1] = max_seq;
    shape[2] = model->n_heads;
    shape[3] = model->head_dim;
    for (i = 0U; i < (uint32_t)model->n_layers; ++i) {
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F16,
                             gd_shape_make(4U, shape),
                             GPT_ALIGNMENT,
                             &cache->k[i]);
        if (st != GD_OK) {
            gpt_kv_cache_deinit(cache);
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F16,
                             gd_shape_make(4U, shape),
                             GPT_ALIGNMENT,
                             &cache->v[i]);
        if (st != GD_OK) {
            gpt_kv_cache_deinit(cache);
            return st;
        }
        cache->k[i].is_leaf = false;
        cache->v[i].is_leaf = false;
    }
    return gpt_kv_cache_set_active_batch(cache, batch_capacity);
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

static gd_status gpt_lm_prepare_generation_cache(gd_context *ctx,
                                                 gpt_lm *model,
                                                 const gpt_config *config,
                                                 int batch_size,
                                                 int max_seq,
                                                 gpt_kv_cache **out)
{
    gpt_kv_cache *cache;
    int capacity;
    gd_status st;
    if (out != NULL) {
        *out = NULL;
    }
    if (ctx == NULL || model == NULL || out == NULL || batch_size <= 0 || max_seq <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    cache = &model->generation_cache;
    capacity = int_max2(batch_size, gpt_generation_cache_batch_capacity(config));
    if (cache->batch_capacity == 0) {
        st = gpt_kv_cache_init(ctx, model, capacity, max_seq, cache);
        if (st != GD_OK) {
            return st;
        }
    }
    if (cache->batch_capacity < batch_size || cache->max_seq != max_seq ||
        cache->n_layers != model->n_layers || cache->n_heads != model->n_heads ||
        cache->head_dim != model->head_dim) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    st = gpt_kv_cache_set_active_batch(cache, batch_size);
    if (st != GD_OK) {
        return st;
    }
    *out = cache;
    return GD_OK;
}

static bool gpt_minimax_m3_should_use_sparse(const gpt_lm *model)
{
    int64_t sparse_span;
    int64_t min_sparse_context;
    if (model == NULL || model->architecture != GPT_ARCH_MINIMAX_M3 ||
        model->minimax_m3_block_size <= 0 || model->minimax_m3_topk_blocks <= 0) {
        return false;
    }
    sparse_span = (int64_t)model->minimax_m3_block_size * (int64_t)model->minimax_m3_topk_blocks;
    min_sparse_context = (int64_t)model->minimax_m3_block_size * 8;
    return (int64_t)model->context_length > min_sparse_context &&
           sparse_span * 2 < (int64_t)model->context_length;
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
    const gd_minimax_m3_sparse_config minimax_cfg = {
        .scale = 0.0f,
        .block_size = model != NULL ? model->minimax_m3_block_size : GPT_MINIMAX_M3_BLOCK_SIZE,
        .topk = model != NULL ? model->minimax_m3_topk_blocks : GPT_MINIMAX_M3_TOPK_BLOCKS,
        .init_blocks = model != NULL ? model->minimax_m3_init_blocks : GPT_MINIMAX_M3_INIT_BLOCKS,
        .local_blocks = model != NULL ? model->minimax_m3_local_blocks : GPT_MINIMAX_M3_LOCAL_BLOCKS,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    const gd_sdpa_varlen_config minimax_dense_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor v_view;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t n_tokens;
    int64_t flat_shape[2];
    uint64_t site_seed;

    if (ctx == NULL || model == NULL || block == NULL || x == NULL || positions == NULL ||
        cu_seqlens == NULL || out == NULL || x->rank != 2U || x->shape[1] != GPT_D_MODEL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n_tokens = x->shape[0];
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
    st = gd_qkv_split_rope(ctx,
                            &qkv,
                            positions,
                            GPT_N_HEADS,
                            GPT_HEAD_DIM,
                            &rope_cfg,
                            &q_rot,
                            &k_rot,
                            &v_view);
    if (st != GD_OK) {
        return st;
    }
    if (model->architecture == GPT_ARCH_MINIMAX_M3) {
        if (gpt_minimax_m3_should_use_sparse(model)) {
            gd_tensor topk_idx;
            st = gd_minimax_m3_index_topk(ctx, &q_rot, &k_rot, cu_seqlens, &minimax_cfg, &topk_idx);
            if (st != GD_OK) {
                return st;
            }
            st = gd_minimax_m3_sparse_attention(ctx,
                                                &q_rot,
                                                &k_rot,
                                                &v_view,
                                                cu_seqlens,
                                                &topk_idx,
                                                &minimax_cfg,
                                                &attn);
            if (st != GD_OK) {
                return st;
            }
        } else {
            st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &minimax_dense_cfg, &attn);
            if (st != GD_OK) {
                return st;
            }
        }
    } else {
        st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
        if (st != GD_OK) {
            return st;
        }
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
    st = gd_dropout_add(ctx, &residual, &attn_proj, model->dropout_p, model->mod.training, site_seed, &attn_resid);
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
    st = gd_swiglu_split_linear(ctx,
                                 &up_gate,
                                 &block->down_proj.weight,
                                 block->down_proj.has_bias ? &block->down_proj.bias : NULL,
                                 &mlp_proj);
    if (st != GD_OK) {
        return st;
    }
    site_seed = splitmix64(model->dropout_seed ^ GPT_DROPOUT_MLP ^ ((uint64_t)block_index << 32) ^ step);
    return gd_dropout_add(ctx, &residual, &mlp_proj, model->dropout_p, model->mod.training, site_seed, out);
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
    if (cu_seqlens->rank != 1U || cu_seqlens->shape[0] < 2 ||
        cu_seqlens->shape[0] > input_ids->shape[0] + 1) {
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
    return gd_lm_cross_entropy_softcapped(ctx,
                                          &final_norm,
                                          &model->token_embedding,
                                          target_ids,
                                          model->logits_softcap,
                                          loss);
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
    const gd_minimax_m3_sparse_config minimax_cfg = {
        .scale = 0.0f,
        .block_size = model != NULL ? model->minimax_m3_block_size : GPT_MINIMAX_M3_BLOCK_SIZE,
        .topk = model != NULL ? model->minimax_m3_topk_blocks : GPT_MINIMAX_M3_TOPK_BLOCKS,
        .init_blocks = model != NULL ? model->minimax_m3_init_blocks : GPT_MINIMAX_M3_INIT_BLOCKS,
        .local_blocks = model != NULL ? model->minimax_m3_local_blocks : GPT_MINIMAX_M3_LOCAL_BLOCKS,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    const gd_sdpa_varlen_config minimax_dense_cfg = {
        .scale = 0.0f,
        .causal = true,
        .sliding_window = GPT_SDPA_WINDOW,
        .prefix_len = 0,
        .max_seqlen = GPT_CONTEXT_LENGTH,
    };
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor v_view;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
    gd_tensor mlp_proj;
    gd_status st;
    int64_t n_tokens;
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
    flat_shape[0] = n_tokens;
    flat_shape[1] = GPT_D_MODEL;

    residual = *x;
    st = gd_rms_norm(ctx, x, &block->attn_norm_w, model->rms_eps, &normed);
    if (st != GD_OK) { return st; }
    st = gd_linear_layer_forward(ctx, &block->qkv_proj, &normed, &qkv);
    if (st != GD_OK) { return st; }
    st = gd_qkv_split_rope(ctx,
                            &qkv,
                            positions,
                            GPT_N_HEADS,
                            GPT_HEAD_DIM,
                            &rope_cfg,
                            &q_rot,
                            &k_rot,
                            &v_view);
    if (st != GD_OK) { return st; }
    st = gd_kv_cache_append_packed(ctx,
                                    &cache->k[block_index],
                                    &cache->v[block_index],
                                    cache_positions,
                                    cu_seqlens,
                                    &k_rot,
                                    &v_view);
    if (st != GD_OK) { return st; }
    if (model->architecture == GPT_ARCH_MINIMAX_M3) {
        if (gpt_minimax_m3_should_use_sparse(model)) {
            gd_tensor topk_idx;
            st = gd_minimax_m3_index_topk(ctx, &q_rot, &k_rot, cu_seqlens, &minimax_cfg, &topk_idx);
            if (st != GD_OK) { return st; }
            st = gd_minimax_m3_sparse_attention(ctx,
                                                &q_rot,
                                                &k_rot,
                                                &v_view,
                                                cu_seqlens,
                                                &topk_idx,
                                                &minimax_cfg,
                                                &attn);
            if (st != GD_OK) { return st; }
        } else {
            st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &minimax_dense_cfg, &attn);
            if (st != GD_OK) { return st; }
        }
    } else {
        st = gd_sdpa_varlen(ctx, &q_rot, &k_rot, &v_view, cu_seqlens, &sdpa_cfg, &attn);
        if (st != GD_OK) { return st; }
    }
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
    st = gd_swiglu_split_linear(ctx,
                                 &up_gate,
                                 &block->down_proj.weight,
                                 block->down_proj.has_bias ? &block->down_proj.bias : NULL,
                                 &mlp_proj);
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
    gd_tensor residual;
    gd_tensor normed;
    gd_tensor qkv;
    gd_tensor q_rot_3d;
    gd_tensor k_rot_3d;
    gd_tensor v_view_3d;
    gd_tensor q_rot;
    gd_tensor k_rot;
    gd_tensor v_view;
    gd_tensor attn;
    gd_tensor attn_flat;
    gd_tensor attn_proj;
    gd_tensor attn_resid;
    gd_tensor mlp_normed;
    gd_tensor up_gate;
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
    st = gd_qkv_split_rope(ctx,
                            &qkv,
                            positions,
                            GPT_N_HEADS,
                            GPT_HEAD_DIM,
                            &rope_cfg,
                            &q_rot_3d,
                            &k_rot_3d,
                            &v_view_3d);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &q_rot_3d, gd_shape_make(4U, q_shape), &q_rot);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &k_rot_3d, gd_shape_make(4U, q_shape), &k_rot);
    if (st != GD_OK) { return st; }
    st = gd_reshape(ctx, &v_view_3d, gd_shape_make(4U, q_shape), &v_view);
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
    st = gd_swiglu_split_linear(ctx,
                                 &up_gate,
                                 &block->down_proj.weight,
                                 block->down_proj.has_bias ? &block->down_proj.bias : NULL,
                                 &mlp_proj);
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

void gpt_generation_tokenizer_init(gd_context *ctx,
                                   const gpt_config *config,
                                   gpt_generation_tokenizer *out)
{
    gd_tokenizer_config tok_cfg;
    char *default_tok_path = NULL;
    const char *tokenizer_path;
    if (config == NULL || out == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "generation tokenizer arguments", __LINE__);
    }
    memset(out, 0, sizeof(*out));
    tok_cfg.split_digits = 1;
    tok_cfg.allow_special = 1;
    default_tok_path = config->tokenizer_path == NULL ? gpt_default_tokenizer_path(config->data_dir) : NULL;
    tokenizer_path = config->tokenizer_path != NULL ? config->tokenizer_path : default_tok_path;
    if (tokenizer_path == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "tokenizer path allocation", __LINE__);
    }
    TRY(ctx, gd_bpe_tokenizer_load(tokenizer_path, &tok_cfg, &out->tok));
    out->path = tokenizer_path;
    out->owned_path = default_tok_path;
    if (gd_tokenizer_id(out->tok, "<|im_end|>", &out->stop_token) != GD_OK) {
        out->stop_token = -1;
    }
}

void gpt_generation_tokenizer_deinit(gpt_generation_tokenizer *tokenizer)
{
    if (tokenizer == NULL) {
        return;
    }
    gd_tokenizer_destroy(tokenizer->tok);
    free(tokenizer->owned_path);
    memset(tokenizer, 0, sizeof(*tokenizer));
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

static void gpt_apply_logits_softcap(float *logits, int vocab_size, float logits_softcap)
{
    int i;
    if (logits == NULL || vocab_size <= 0 || logits_softcap <= 0.0f) {
        return;
    }
    for (i = 0; i < vocab_size; ++i) {
        logits[i] = logits_softcap * tanhf(logits[i] / logits_softcap);
    }
}

static bool gpt_history_contains_token(const int32_t *history, int start, int end, int token)
{
    int i;
    if (history == NULL || start < 0 || end < start) {
        return false;
    }
    for (i = start; i < end; ++i) {
        if (history[i] == token) {
            return true;
        }
    }
    return false;
}

static void gpt_apply_repetition_penalty(float *logits,
                                         int vocab_size,
                                         const int32_t *history,
                                         int history_len,
                                         float repetition_penalty)
{
    int i;
    if (logits == NULL || history == NULL || vocab_size <= 0 || history_len <= 0 ||
        !isfinite(repetition_penalty) || repetition_penalty <= 1.0f) {
        return;
    }
    for (i = 0; i < history_len; ++i) {
        const int token = (int)history[i];
        if (token < 0 || token >= vocab_size || gpt_history_contains_token(history, 0, i, token)) {
            continue;
        }
        if (logits[token] > 0.0f) {
            logits[token] /= repetition_penalty;
        } else {
            logits[token] *= repetition_penalty;
        }
    }
}

static int gpt_sample_next_token(float *logits,
                                 int vocab_size,
                                 const int32_t *history,
                                 int history_len,
                                 float temperature,
                                 float min_p,
                                 float repetition_penalty,
                                 uint64_t *rng)
{
    int best = 0;
    double best_value = -HUGE_VAL;
    bool have_best = false;
    int i;
    if (logits == NULL || vocab_size <= 0 || rng == NULL) {
        return 0;
    }
    gpt_apply_repetition_penalty(logits, vocab_size, history, history_len, repetition_penalty);
    for (i = 0; i < vocab_size; ++i) {
        const double value = (double)logits[i];
        if (!isfinite(value)) {
            continue;
        }
        if (!have_best || value > best_value) {
            best_value = value;
            best = i;
            have_best = true;
        }
    }
    if (!have_best || !isfinite(temperature) || temperature <= 0.0f) {
        return best;
    }
    {
        const double temp = (double)temperature;
        const double threshold = min_p > 0.0f ? best_value + temp * log((double)min_p) : -HUGE_VAL;
        double sum = 0.0;
        double target;
        uint64_t r;
        for (i = 0; i < vocab_size; ++i) {
            const double value = (double)logits[i];
            double p;
            if (!isfinite(value) || value < threshold) {
                continue;
            }
            p = exp((value - best_value) / temp);
            if (isfinite(p)) {
                sum += p;
            }
        }
        if (!(sum > 0.0) || !isfinite(sum)) {
            return best;
        }
        *rng = splitmix64(*rng);
        r = *rng >> 11;
        target = ((double)r * (1.0 / 9007199254740992.0)) * sum;
        for (i = 0; i < vocab_size; ++i) {
            const double value = (double)logits[i];
            double p;
            if (!isfinite(value) || value < threshold) {
                continue;
            }
            p = exp((value - best_value) / temp);
            if (!isfinite(p)) {
                continue;
            }
            target -= p;
            if (target <= 0.0) {
                return i;
            }
        }
    }
    return best;
}

static int gpt_generate_prompts_loaded(gd_context *ctx,
                                       gpt_lm *model,
                                       const gpt_config *config,
                                       const char *const *prompts,
                                       int n_prompts,
                                       const char *tag,
                                       size_t step,
                                       bool restore_training,
                                       const gpt_generation_tokenizer *tokenizer)
{
    gd_tokenizer *tok;
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
    bool *finished = NULL;
    gd_tensor *last_logits = NULL;
    float *logits_host = NULL;
    gpt_kv_cache *cache = NULL;
    int total_prompt_tokens = 0;
    int max_new;
    int room_for_prompt;
    int generated = 0;
    int finished_count = 0;
    int32_t stop_token;
    bool stream_tokens;
    int b;
    int i;
    uint64_t rng;
    double start;
    double elapsed;

    if (ctx == NULL || model == NULL || config == NULL || prompts == NULL || n_prompts <= 0 ||
        tokenizer == NULL || tokenizer->tok == NULL || tokenizer->path == NULL) {
        gpt_fail_status(ctx, GD_ERR_INVALID_ARGUMENT, "generation arguments", __LINE__);
    }
    tok = tokenizer->tok;
    tokenizer_path = tokenizer->path;
    stop_token = tokenizer->stop_token;
    stream_tokens = n_prompts == 1;

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
    finished = (bool *)calloc((size_t)n_prompts, sizeof(finished[0]));
    last_logits = (gd_tensor *)calloc((size_t)n_prompts, sizeof(last_logits[0]));
    logits_host = (float *)malloc((size_t)n_prompts * (size_t)GPT_VOCAB_SIZE * sizeof(logits_host[0]));
    if (encoded == NULL || seq_ids == NULL || n_encoded == NULL || prompt_offset == NULL ||
        prompt_len == NULL || seq_len == NULL || cu == NULL || cache_pos_values == NULL ||
        decode_ids == NULL || decode_positions == NULL || next_ids == NULL || finished == NULL ||
        last_logits == NULL || logits_host == NULL) {
        gpt_fail_status(ctx, GD_ERR_OUT_OF_MEMORY, "generation allocation", __LINE__);
    }

    max_new = config->max_new_tokens;
    room_for_prompt = max_new >= GPT_CONTEXT_LENGTH ? GPT_CONTEXT_LENGTH - 1 : GPT_CONTEXT_LENGTH - max_new;
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
    TRY(ctx, gpt_lm_prepare_generation_cache(ctx, model, config, n_prompts, GPT_CONTEXT_LENGTH, &cache));
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
                                       cache,
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
            gpt_apply_logits_softcap(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                     GPT_VOCAB_SIZE,
                                     model->logits_softcap);
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                seq_ids[b],
                                                prompt_len[b],
                                                config->temperature,
                                                config->min_p,
                                                config->repetition_penalty,
                                                &rng);
            cache->pos[b] = prompt_len[b];
        }
    }

    if (stream_tokens) {
        printf("  prefix=\"%s\" prompt_tokens=%d generated_text:", prompts[0], prompt_len[0]);
        fflush(stdout);
    }

    while (generated < max_new && finished_count < n_prompts) {
        bool appended_any = false;
        for (b = 0; b < n_prompts; ++b) {
            if (finished[b]) {
                continue;
            }
            if (stop_token >= 0 && next_ids[b] == stop_token) {
                finished[b] = true;
                finished_count += 1;
                continue;
            }
            if (seq_len[b] >= GPT_CONTEXT_LENGTH) {
                finished[b] = true;
                finished_count += 1;
                continue;
            }
            seq_ids[b][seq_len[b]] = (int32_t)next_ids[b];
            seq_len[b] += 1;
            if (stream_tokens) {
                char *piece = NULL;
                const int32_t token_id = (int32_t)next_ids[b];
                TRY(ctx, gd_tokenizer_decode(tok, &token_id, 1, &piece));
                printf("%s", piece != NULL ? piece : "");
                fflush(stdout);
                gd_tokenizer_free(piece);
            }
            appended_any = true;
        }
        if (!appended_any) {
            break;
        }
        generated += 1;
        if (generated >= max_new || finished_count >= n_prompts) {
            break;
        }
        for (b = 0; b < n_prompts; ++b) {
            decode_ids[b] = finished[b] ? (stop_token >= 0 ? stop_token : 0) : (int32_t)next_ids[b];
            decode_positions[b] = cache->pos[b];
            cache_pos_values[b] = cache->pos[b];
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
            TRY(ctx, gpt_lm_decode_logits(ctx, model, cache, &ids_t, &pos_t, &cache_pos_t, &logits));
            TRY(ctx, gd_end_step(ctx));
            TRY(ctx, gd_tensor_read_f32(ctx,
                                        &logits,
                                        logits_host,
                                        (size_t)n_prompts * (size_t)GPT_VOCAB_SIZE));
        }
        for (b = 0; b < n_prompts; ++b) {
            if (finished[b]) {
                continue;
            }
            gpt_apply_logits_softcap(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                     GPT_VOCAB_SIZE,
                                     model->logits_softcap);
            next_ids[b] = gpt_sample_next_token(logits_host + (size_t)b * (size_t)GPT_VOCAB_SIZE,
                                                GPT_VOCAB_SIZE,
                                                seq_ids[b],
                                                seq_len[b],
                                                config->temperature,
                                                config->min_p,
                                                config->repetition_penalty,
                                                &rng);
        }
    }

    elapsed = gpt_wall_seconds() - start;
    if (stream_tokens) {
        printf("\n");
    } else {
        for (b = 0; b < n_prompts; ++b) {
            char *decoded_new = NULL;
            const int generated_len = seq_len[b] - prompt_len[b];
            TRY(ctx, gd_tokenizer_decode(tok, seq_ids[b] + prompt_len[b], generated_len, &decoded_new));
            printf("  prefix=\"%s\" prompt_tokens=%d generated_text=%s\n",
                   prompts[b],
                   prompt_len[b],
                   decoded_new != NULL ? decoded_new : "");
            gd_tokenizer_free(decoded_new);
        }
    }
    printf("generate%s%s: tokenizer=%s batch=%d generated=%d temperature=%.3f min_p=%.3f repetition_penalty=%.3f logits_softcap=%.3f elapsed=%.3fs tok/s=%.1f",
           tag != NULL ? ":" : "",
           tag != NULL ? tag : "",
           tokenizer_path,
           n_prompts,
           generated,
           (double)config->temperature,
           (double)config->min_p,
           (double)config->repetition_penalty,
           (double)model->logits_softcap,
           elapsed,
           elapsed > 0.0 ? (double)((size_t)n_prompts * (size_t)generated) / elapsed : 0.0);
    if (step != 0U) {
        printf(" step=%zu", step);
    }
    printf("\n");

    if (restore_training) {
        gd_module_set_training(&model->mod, true);
    }
    for (b = 0; b < n_prompts; ++b) {
        gd_tokenizer_free(encoded[b]);
        free(seq_ids[b]);
    }
    free(logits_host);
    free(last_logits);
    free(finished);
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
    return generated;
}

int gpt_generate_with_tokenizer(gd_context *ctx,
                                gpt_lm *model,
                                const gpt_config *config,
                                const gpt_generation_tokenizer *tokenizer)
{
    const char *prompts[1];
    prompts[0] = config->generate_prompt;
    return gpt_generate_prompts_loaded(ctx, model, config, prompts, 1, "user", 0U, false, tokenizer);
}

int gpt_generate(gd_context *ctx, gpt_lm *model, const gpt_config *config)
{
    int generated;
    gpt_generation_tokenizer tokenizer;
    gpt_generation_tokenizer_init(ctx, config, &tokenizer);
    generated = gpt_generate_with_tokenizer(ctx, model, config, &tokenizer);
    gpt_generation_tokenizer_deinit(&tokenizer);
    return generated;
}

int gpt_generate_vowels_with_tokenizer(gd_context *ctx,
                                       gpt_lm *model,
                                       const gpt_config *config,
                                       size_t step,
                                       const gpt_generation_tokenizer *tokenizer)
{
    static const char *const prompts[GPT_GENERATE_VOWEL_PROMPT_COUNT] = {"a", "e", "i", "o", "u"};
    return gpt_generate_prompts_loaded(ctx,
                                       model,
                                       config,
                                       prompts,
                                       (int)GPT_GENERATE_VOWEL_PROMPT_COUNT,
                                       "vowels",
                                       step,
                                       true,
                                       tokenizer);
}

int gpt_generate_vowels(gd_context *ctx,
                        gpt_lm *model,
                        const gpt_config *config,
                        size_t step)
{
    int generated;
    gpt_generation_tokenizer tokenizer;
    gpt_generation_tokenizer_init(ctx, config, &tokenizer);
    generated = gpt_generate_vowels_with_tokenizer(ctx, model, config, step, &tokenizer);
    gpt_generation_tokenizer_deinit(&tokenizer);
    return generated;
}
