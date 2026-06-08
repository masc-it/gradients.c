#include <gradients/optimizer.h>
#include <gradients/autograd.h>
#include <gradients/transfer.h>

#include "../core/backend.h"
#include "../core/memory_internal.h"

#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GD_OPTIM_GRAD_CLIP_BLOCK_ELEMS 1024U
#define GD_OPTIM_GRAD_CLIP_SCALE_EPS 1.0e-6f

typedef struct gd_optimizer_param {
    char path[GD_MODULE_PATH_MAX];
    gd_tensor *param;
    gd_tensor master;
    gd_tensor m;
    gd_tensor v;
    size_t count;
    uint64_t step;
    float beta1_power;
    float beta2_power;
    float lr_mult;
    float weight_decay;
    bool trainable;
    bool has_master;
} gd_optimizer_param;

struct gd_optimizer {
    gd_optimizer_param *params;
    gd_tensor *step_grads;
    bool *step_has_grad;
    gd_backend_amp_unscale_desc *amp_descs;
    gd_backend_grad_norm_desc *grad_norm_descs;
    gd_backend_adamw_desc *adamw_descs;
    uint32_t param_count;
    size_t grad_clip_partial_capacity;
    gd_tensor found_inf;
    gd_tensor grad_clip_partials;
    gd_tensor grad_clip_scale;
    gd_adamw_config config;
    uint64_t step;
};

struct gd_amp_scaler {
    gd_amp_config config;
    float scale;
    uint32_t growth_tracker;
    bool last_found_inf;
};

static bool gd_adamw_lr_valid(float lr)
{
    return lr == lr && lr >= 0.0f && lr <= FLT_MAX;
}

static bool gd_optimizer_grad_clip_norm_valid(float max_norm)
{
    return max_norm == max_norm && max_norm > 0.0f && max_norm <= FLT_MAX;
}

static size_t gd_optimizer_grad_clip_group_count(size_t count)
{
    const size_t block = (size_t)GD_OPTIM_GRAD_CLIP_BLOCK_ELEMS;
    return (count / block) + ((count % block) != 0U ? (size_t)1U : (size_t)0U);
}

static bool gd_optimizer_size_add_overflow(size_t a, size_t b, size_t *out)
{
    if (out == NULL || a > SIZE_MAX - b) {
        return true;
    }
    *out = a + b;
    return false;
}

static bool gd_optimizer_size_mul_overflow(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0U && b > SIZE_MAX / a)) {
        return true;
    }
    *out = a * b;
    return false;
}

static bool gd_adamw_config_valid(const gd_adamw_config *config)
{
    return config != NULL && gd_adamw_lr_valid(config->lr) && config->beta1 >= 0.0f &&
           config->beta1 < 1.0f && config->beta2 >= 0.0f && config->beta2 < 1.0f &&
           config->eps > 0.0f && config->weight_decay >= 0.0f;
}

static bool gd_amp_float_finite_positive(float value)
{
    return value > 0.0f && value == value && value <= 3.402823466e+38f;
}

static bool gd_amp_config_valid(const gd_amp_config *config)
{
    return config != NULL && gd_amp_float_finite_positive(config->init_scale) &&
           gd_amp_float_finite_positive(config->growth_factor) && config->growth_factor > 1.0f &&
           gd_amp_float_finite_positive(config->backoff_factor) && config->backoff_factor < 1.0f &&
           gd_amp_float_finite_positive(config->min_scale) &&
           gd_amp_float_finite_positive(config->max_scale) && config->max_scale >= config->min_scale &&
           config->growth_interval > 0U;
}

static float gd_amp_clamp_scale(float value, const gd_amp_config *config)
{
    if (value < config->min_scale) {
        return config->min_scale;
    }
    if (value > config->max_scale) {
        return config->max_scale;
    }
    return value;
}

static float gd_adamw_f16_to_f32(uint16_t bits)
{
    uint32_t sign = ((uint32_t)bits & 0x8000U) << 16;
    int32_t exp = (int32_t)(((uint32_t)bits >> 10) & 0x1fU);
    uint32_t mant = (uint32_t)bits & 0x3ffU;
    union {
        uint32_t u;
        float f;
    } v;
    if (exp == 0) {
        if (mant == 0U) {
            v.u = sign;
            return v.f;
        }
        while ((mant & 0x400U) == 0U) {
            mant <<= 1U;
            exp -= 1;
        }
        mant &= 0x3ffU;
        exp += 1;
    } else if (exp == 31) {
        v.u = sign | 0x7f800000U | (mant << 13);
        return v.f;
    }
    v.u = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    return v.f;
}

static bool gd_adamw_dtype_supported(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_optimizer_same_shape(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL || a->rank != b->rank) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_optimizer_tensor_count(gd_context *ctx,
                                           const gd_tensor *tensor,
                                           size_t *out_count)
{
    int64_t numel;
    gd_status st;
    if (ctx == NULL || tensor == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_numel(tensor, &numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "optimizer tensor has invalid shape");
    }
    if ((uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "optimizer tensor count overflow");
    }
    *out_count = (size_t)numel;
    return GD_OK;
}

static gd_status gd_adamw_init_master_from_param(gd_context *ctx,
                                                 const gd_tensor *param,
                                                 gd_tensor *master,
                                                 size_t count)
{
    gd_status st;
    if (ctx == NULL || param == NULL || master == NULL || count == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (param->dtype == GD_DTYPE_F32) {
        return GD_OK;
    }
    if (param->dtype == GD_DTYPE_F16) {
        uint16_t *src;
        float *dst;
        size_t i;
        if (count > SIZE_MAX / sizeof(*src) || count > SIZE_MAX / sizeof(*dst)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "adamw master init overflow");
        }
        src = (uint16_t *)malloc(count * sizeof(*src));
        dst = (float *)malloc(count * sizeof(*dst));
        if (src == NULL || dst == NULL) {
            free(src);
            free(dst);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "adamw master init allocation failed");
        }
        st = gd_tensor_read(ctx, param, src, count * sizeof(*src));
        if (st != GD_OK) {
            free(src);
            free(dst);
            return st;
        }
        for (i = 0U; i < count; ++i) {
            dst[i] = gd_adamw_f16_to_f32(src[i]);
        }
        st = gd_tensor_write(ctx, master, dst, count * sizeof(*dst));
        free(src);
        free(dst);
        return st;
    }
    return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "adamw master init unsupported dtype");
}

static gd_status gd_optimizer_validate_param(gd_context *ctx,
                                             const gd_tensor *param,
                                             size_t *out_count)
{
    gd_status st;
    if (ctx == NULL || param == NULL || out_count == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, param);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_adamw_dtype_supported(param->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "adamw supports f16/f32 params only");
    }
    if (!gd_tensor_is_contiguous(param)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "adamw requires contiguous params");
    }
    return gd_optimizer_tensor_count(ctx, param, out_count);
}

static gd_status gd_optimizer_validate_grad(gd_context *ctx,
                                            const gd_optimizer_param *slot,
                                            const gd_tensor *grad)
{
    gd_status st;
    if (ctx == NULL || slot == NULL || slot->param == NULL || grad == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, grad);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_adamw_dtype_supported(grad->dtype)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "adamw supports f16/f32 grads only");
    }
    if (!gd_tensor_is_contiguous(grad)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED,
                                    "adamw requires contiguous grads");
    }
    if (!gd_optimizer_same_shape(slot->param, grad)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "adamw grad shape mismatch");
    }
    return GD_OK;
}

static gd_status gd_adamw_init_slot(gd_context *ctx,
                                    gd_optimizer_param *slot,
                                    const gd_param_ref *ref,
                                    const gd_adamw_config *config)
{
    gd_status st;
    size_t count;
    if (ctx == NULL || slot == NULL || ref == NULL || ref->tensor == NULL || config == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (ref->lr_mult < 0.0f || ref->weight_decay < 0.0f) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "invalid optimizer param group options");
    }
    st = gd_optimizer_validate_param(ctx, ref->tensor, &count);
    if (st != GD_OK) {
        return st;
    }
    memset(slot, 0, sizeof(*slot));
    (void)snprintf(slot->path, sizeof(slot->path), "%s", ref->path);
    slot->param = ref->tensor;
    slot->count = count;
    slot->step = 0U;
    slot->beta1_power = 1.0f;
    slot->beta2_power = 1.0f;
    slot->lr_mult = ref->lr_mult;
    slot->weight_decay = ref->group_index >= 0 ? ref->weight_decay : config->weight_decay;
    slot->trainable = ref->trainable;
    slot->has_master = ref->tensor->dtype == GD_DTYPE_F16;
    if (slot->has_master) {
        st = gd_tensor_empty(ctx, GD_ARENA_STATE, GD_DTYPE_F32, gd_shape_make(ref->tensor->rank, ref->tensor->shape), 256U, &slot->master);
        if (st != GD_OK) {
            return st;
        }
        st = gd_adamw_init_master_from_param(ctx, ref->tensor, &slot->master, count);
        if (st != GD_OK) {
            return st;
        }
        slot->master.requires_grad = false;
        slot->master.is_leaf = false;
    }
    st = gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32, gd_shape_make(ref->tensor->rank, ref->tensor->shape), 256U, &slot->m);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_F32, gd_shape_make(ref->tensor->rank, ref->tensor->shape), 256U, &slot->v);
    if (st != GD_OK) {
        return st;
    }
    slot->m.requires_grad = false;
    slot->m.is_leaf = false;
    slot->v.requires_grad = false;
    slot->v.is_leaf = false;
    return GD_OK;
}

static gd_status gd_optimizer_build_adamw_desc(gd_context *ctx,
                                                gd_optimizer_param *slot,
                                                const gd_tensor *grad,
                                                const gd_tensor *grad_scale,
                                                const gd_tensor *found_inf,
                                                const gd_adamw_config *config,
                                                float base_lr,
                                                bool grad_already_validated,
                                                gd_backend_adamw_desc *out_desc)
{
    float beta1_power;
    float beta2_power;
    float bias_correction1;
    float bias_correction2;
    gd_status st;
    if (ctx == NULL || slot == NULL || slot->param == NULL || grad == NULL ||
        config == NULL || out_desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!grad_already_validated) {
        st = gd_optimizer_validate_grad(ctx, slot, grad);
        if (st != GD_OK) {
            return st;
        }
    }
    beta1_power = slot->beta1_power * config->beta1;
    beta2_power = slot->beta2_power * config->beta2;
    bias_correction1 = config->bias_correction ? 1.0f - beta1_power : 1.0f;
    bias_correction2 = config->bias_correction ? 1.0f - beta2_power : 1.0f;
    if (!(bias_correction1 > 0.0f) || !(bias_correction2 > 0.0f)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "invalid adamw bias correction state");
    }
    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->param_buffer = (gd_backend_buffer *)slot->param->storage.buffer;
    out_desc->param_offset = gd_tensor_storage_offset(slot->param);
    if (slot->has_master) {
        out_desc->master_buffer = (gd_backend_buffer *)slot->master.storage.buffer;
        out_desc->master_offset = gd_tensor_storage_offset(&slot->master);
        out_desc->has_master = 1U;
    }
    out_desc->grad_buffer = (gd_backend_buffer *)grad->storage.buffer;
    out_desc->grad_offset = gd_tensor_storage_offset(grad);
    if (grad_scale != NULL) {
        out_desc->grad_scale_buffer = (gd_backend_buffer *)grad_scale->storage.buffer;
        out_desc->grad_scale_offset = gd_tensor_storage_offset(grad_scale);
        out_desc->has_grad_scale = 1U;
    }
    if (found_inf != NULL) {
        out_desc->found_inf_buffer = (gd_backend_buffer *)found_inf->storage.buffer;
        out_desc->found_inf_offset = gd_tensor_storage_offset(found_inf);
        out_desc->has_found_inf = 1U;
    }
    out_desc->m_buffer = (gd_backend_buffer *)slot->m.storage.buffer;
    out_desc->m_offset = gd_tensor_storage_offset(&slot->m);
    out_desc->v_buffer = (gd_backend_buffer *)slot->v.storage.buffer;
    out_desc->v_offset = gd_tensor_storage_offset(&slot->v);
    out_desc->count = slot->count;
    out_desc->param_dtype = (uint32_t)slot->param->dtype;
    out_desc->grad_dtype = (uint32_t)grad->dtype;
    out_desc->lr = base_lr * slot->lr_mult;
    out_desc->beta1 = config->beta1;
    out_desc->beta2 = config->beta2;
    out_desc->eps = config->eps;
    out_desc->weight_decay = slot->weight_decay;
    out_desc->bias_correction1 = bias_correction1;
    out_desc->bias_correction2 = bias_correction2;
    return GD_OK;
}

static void gd_optimizer_commit_step_param(gd_optimizer_param *slot,
                                           const gd_adamw_config *config)
{
    if (slot == NULL || slot->param == NULL || config == NULL) {
        return;
    }
    slot->step += 1U;
    slot->beta1_power *= config->beta1;
    slot->beta2_power *= config->beta2;
    slot->param->version += 1U;
    if (slot->param->version == 0U) {
        slot->param->version = 1U;
    }
    if (slot->has_master) {
        slot->master.version += 1U;
        if (slot->master.version == 0U) {
            slot->master.version = 1U;
        }
    }
}

gd_adamw_config gd_adamw_config_default(void)
{
    gd_adamw_config config;
    config.lr = 1.0e-3f;
    config.beta1 = 0.9f;
    config.beta2 = 0.999f;
    config.eps = 1.0e-8f;
    config.weight_decay = 0.01f;
    config.bias_correction = true;
    return config;
}

gd_amp_config gd_amp_config_default(void)
{
    gd_amp_config config;
    config.enabled = true;
    config.defer_found_inf = false;
    config.init_scale = 32768.0f;
    config.growth_factor = 2.0f;
    config.backoff_factor = 0.5f;
    config.min_scale = 1.0f;
    config.max_scale = 2147483648.0f;
    config.growth_interval = 2000U;
    return config;
}

gd_status gd_amp_scaler_create(const gd_amp_config *config, gd_amp_scaler **out)
{
    gd_amp_config cfg;
    gd_amp_scaler *scaler;
    if (out != NULL) {
        *out = NULL;
    }
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    cfg = config != NULL ? *config : gd_amp_config_default();
    if (!gd_amp_config_valid(&cfg)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    scaler = (gd_amp_scaler *)calloc(1U, sizeof(*scaler));
    if (scaler == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    scaler->config = cfg;
    scaler->scale = gd_amp_clamp_scale(cfg.init_scale, &cfg);
    scaler->growth_tracker = 0U;
    scaler->last_found_inf = false;
    *out = scaler;
    return GD_OK;
}

void gd_amp_scaler_destroy(gd_amp_scaler *scaler)
{
    if (scaler == NULL) {
        return;
    }
    memset(scaler, 0, sizeof(*scaler));
    free(scaler);
}

float gd_amp_scaler_scale(const gd_amp_scaler *scaler)
{
    return scaler != NULL ? scaler->scale : 1.0f;
}

bool gd_amp_scaler_enabled(const gd_amp_scaler *scaler)
{
    return scaler != NULL && scaler->config.enabled;
}

bool gd_amp_scaler_last_found_inf(const gd_amp_scaler *scaler)
{
    return scaler != NULL && scaler->last_found_inf;
}

uint32_t gd_amp_scaler_growth_tracker(const gd_amp_scaler *scaler)
{
    return scaler != NULL ? scaler->growth_tracker : 0U;
}

static void gd_amp_scaler_update(gd_amp_scaler *scaler, bool found_inf)
{
    if (scaler == NULL || !scaler->config.enabled) {
        return;
    }
    scaler->last_found_inf = found_inf;
    if (found_inf) {
        scaler->scale = gd_amp_clamp_scale(scaler->scale * scaler->config.backoff_factor,
                                           &scaler->config);
        scaler->growth_tracker = 0U;
        return;
    }
    scaler->growth_tracker += 1U;
    if (scaler->growth_tracker >= scaler->config.growth_interval) {
        scaler->scale = gd_amp_clamp_scale(scaler->scale * scaler->config.growth_factor,
                                           &scaler->config);
        scaler->growth_tracker = 0U;
    }
}

gd_status gd_adamw_create(gd_context *ctx,
                          const gd_param_set *params,
                          const gd_adamw_config *config,
                          gd_optimizer **out)
{
    gd_adamw_config cfg;
    gd_optimizer *optimizer;
    uint32_t trainable_count = 0U;
    uint32_t write_index = 0U;
    uint32_t i;
    gd_status st;
    if (out != NULL) {
        *out = NULL;
    }
    if (ctx == NULL || params == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    cfg = config != NULL ? *config : gd_adamw_config_default();
    if (!gd_adamw_config_valid(&cfg)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid adamw config");
    }
    for (i = 0U; i < params->count; ++i) {
        if (params->items[i].trainable && params->items[i].tensor != NULL) {
            trainable_count += 1U;
        }
    }
    optimizer = (gd_optimizer *)calloc(1U, sizeof(*optimizer));
    if (optimizer == NULL) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "adamw allocation failed");
    }
    optimizer->config = cfg;
    if (trainable_count != 0U) {
        optimizer->params = (gd_optimizer_param *)calloc(trainable_count, sizeof(optimizer->params[0]));
        optimizer->step_grads = (gd_tensor *)calloc(trainable_count, sizeof(optimizer->step_grads[0]));
        optimizer->step_has_grad = (bool *)calloc(trainable_count, sizeof(optimizer->step_has_grad[0]));
        optimizer->amp_descs = (gd_backend_amp_unscale_desc *)calloc(trainable_count, sizeof(optimizer->amp_descs[0]));
        optimizer->grad_norm_descs = (gd_backend_grad_norm_desc *)calloc(trainable_count, sizeof(optimizer->grad_norm_descs[0]));
        optimizer->adamw_descs = (gd_backend_adamw_desc *)calloc(trainable_count, sizeof(optimizer->adamw_descs[0]));
        if (optimizer->params == NULL || optimizer->step_grads == NULL ||
            optimizer->step_has_grad == NULL || optimizer->amp_descs == NULL ||
            optimizer->grad_norm_descs == NULL || optimizer->adamw_descs == NULL) {
            gd_optimizer_destroy(optimizer);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "adamw param allocation failed");
        }
    }
    for (i = 0U; i < params->count; ++i) {
        const gd_param_ref *ref = &params->items[i];
        if (!ref->trainable || ref->tensor == NULL) {
            continue;
        }
        st = gd_adamw_init_slot(ctx, &optimizer->params[write_index], ref, &cfg);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        write_index += 1U;
    }
    optimizer->param_count = write_index;
    {
        size_t partial_capacity = 0U;
        int64_t partial_shape[1];
        const int64_t scale_shape[1] = {2};
        for (i = 0U; i < optimizer->param_count; ++i) {
            size_t groups = gd_optimizer_grad_clip_group_count(optimizer->params[i].count);
            if (gd_optimizer_size_add_overflow(partial_capacity, groups, &partial_capacity)) {
                gd_optimizer_destroy(optimizer);
                return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                            "adamw grad clip partial count overflow");
            }
        }
        if (partial_capacity == 0U) {
            partial_capacity = 1U;
        }
        if (partial_capacity > (size_t)INT64_MAX) {
            gd_optimizer_destroy(optimizer);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "adamw grad clip partial shape overflow");
        }
        partial_shape[0] = (int64_t)partial_capacity;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, partial_shape),
                             256U,
                             &optimizer->grad_clip_partials);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, scale_shape),
                             256U,
                             &optimizer->grad_clip_scale);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        optimizer->grad_clip_partial_capacity = partial_capacity;
        optimizer->grad_clip_partials.requires_grad = false;
        optimizer->grad_clip_partials.is_leaf = false;
        optimizer->grad_clip_scale.requires_grad = false;
        optimizer->grad_clip_scale.is_leaf = false;
    }
    {
        const int64_t flag_shape[1] = {1};
        st = gd_tensor_zeros(ctx, GD_ARENA_STATE, GD_DTYPE_I32, gd_shape_make(1U, flag_shape), 256U, &optimizer->found_inf);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        optimizer->found_inf.requires_grad = false;
        optimizer->found_inf.is_leaf = false;
    }
    *out = optimizer;
    return GD_OK;
}

void gd_optimizer_destroy(gd_optimizer *optimizer)
{
    if (optimizer == NULL) {
        return;
    }
    free(optimizer->params);
    free(optimizer->step_grads);
    free(optimizer->step_has_grad);
    free(optimizer->amp_descs);
    free(optimizer->grad_norm_descs);
    free(optimizer->adamw_descs);
    memset(optimizer, 0, sizeof(*optimizer));
    free(optimizer);
}

gd_status gd_optimizer_zero_grad(gd_context *ctx, gd_optimizer *optimizer)
{
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_zero_grad(ctx);
}

static gd_status gd_optimizer_require_train_scope(gd_context *ctx)
{
    if (ctx == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_context_in_scope(ctx) || gd_context_scope_mode(ctx) != GD_SCOPE_TRAIN) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer step requires active train scope");
    }
    return GD_OK;
}

static gd_status gd_optimizer_build_unscale_desc(gd_context *ctx,
                                                  gd_optimizer *optimizer,
                                                  const gd_optimizer_param *slot,
                                                  const gd_tensor *grad,
                                                  float inv_scale,
                                                  gd_backend_amp_unscale_desc *out_desc)
{
    gd_status st;
    if (ctx == NULL || optimizer == NULL || slot == NULL || grad == NULL || out_desc == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_validate_grad(ctx, slot, grad);
    if (st != GD_OK) {
        return st;
    }
    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->grad_buffer = (gd_backend_buffer *)grad->storage.buffer;
    out_desc->grad_offset = gd_tensor_storage_offset(grad);
    out_desc->count = slot->count;
    out_desc->grad_dtype = (uint32_t)grad->dtype;
    out_desc->inv_scale = inv_scale;
    out_desc->found_inf_buffer = (gd_backend_buffer *)optimizer->found_inf.storage.buffer;
    out_desc->found_inf_offset = gd_tensor_storage_offset(&optimizer->found_inf);
    return GD_OK;
}

static gd_status gd_optimizer_build_grad_norm_desc(gd_context *ctx,
                                                   const gd_optimizer_param *slot,
                                                   const gd_tensor *grad,
                                                   float grad_scale,
                                                   const gd_tensor *found_inf,
                                                   size_t partial_index,
                                                   gd_backend_buffer *partial_buffer,
                                                   size_t partial_base_offset,
                                                   gd_backend_grad_norm_desc *out_desc,
                                                   size_t *out_groups)
{
    size_t groups;
    size_t partial_byte_delta;
    gd_status st;
    if (ctx == NULL || slot == NULL || grad == NULL || partial_buffer == NULL ||
        out_desc == NULL || out_groups == NULL || !(grad_scale > 0.0f) ||
        grad_scale > FLT_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_validate_grad(ctx, slot, grad);
    if (st != GD_OK) {
        return st;
    }
    groups = gd_optimizer_grad_clip_group_count(slot->count);
    if (gd_optimizer_size_mul_overflow(partial_index, sizeof(float), &partial_byte_delta) ||
        partial_base_offset > SIZE_MAX - partial_byte_delta) {
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "adamw grad clip partial offset overflow");
    }
    memset(out_desc, 0, sizeof(*out_desc));
    out_desc->grad_buffer = (gd_backend_buffer *)grad->storage.buffer;
    out_desc->grad_offset = gd_tensor_storage_offset(grad);
    out_desc->count = slot->count;
    out_desc->grad_dtype = (uint32_t)grad->dtype;
    out_desc->partial_buffer = partial_buffer;
    out_desc->partial_offset = partial_base_offset + partial_byte_delta;
    out_desc->partial_count = groups;
    out_desc->grad_scale = grad_scale;
    if (found_inf != NULL) {
        out_desc->found_inf_buffer = (gd_backend_buffer *)found_inf->storage.buffer;
        out_desc->found_inf_offset = gd_tensor_storage_offset(found_inf);
        out_desc->has_found_inf = 1U;
    }
    *out_groups = groups;
    return GD_OK;
}

static gd_status gd_optimizer_apply_grad_clip_scaled(gd_context *ctx,
                                                     gd_optimizer *optimizer,
                                                     float max_norm,
                                                     float grad_scale,
                                                     const gd_tensor *found_inf)
{
    gd_backend_buffer *partial_buffer;
    gd_backend_buffer *scale_buffer;
    size_t partial_base_offset;
    size_t scale_offset;
    size_t partial_index = 0U;
    uint32_t desc_count = 0U;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_optimizer_grad_clip_norm_valid(max_norm)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid grad clip norm");
    }
    if (!(grad_scale > 0.0f) || grad_scale > FLT_MAX) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid grad clip scale");
    }
    if (optimizer->param_count == 0U) {
        return GD_OK;
    }
    if (optimizer->step_grads == NULL || optimizer->step_has_grad == NULL ||
        optimizer->grad_norm_descs == NULL) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "adamw grad clip cache missing");
    }
    partial_buffer = (gd_backend_buffer *)optimizer->grad_clip_partials.storage.buffer;
    scale_buffer = (gd_backend_buffer *)optimizer->grad_clip_scale.storage.buffer;
    partial_base_offset = gd_tensor_storage_offset(&optimizer->grad_clip_partials);
    scale_offset = gd_tensor_storage_offset(&optimizer->grad_clip_scale);
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        size_t groups;
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_grad_norm_desc(ctx,
                                               slot,
                                               &optimizer->step_grads[i],
                                               grad_scale,
                                               found_inf,
                                               partial_index,
                                               partial_buffer,
                                               partial_base_offset,
                                               &optimizer->grad_norm_descs[desc_count],
                                               &groups);
        if (st != GD_OK) {
            return st;
        }
        if (partial_index > SIZE_MAX - groups ||
            partial_index + groups > optimizer->grad_clip_partial_capacity) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "adamw grad clip partial capacity exceeded");
        }
        partial_index += groups;
        desc_count += 1U;
    }
    if (desc_count == 0U) {
        return GD_OK;
    }
    st = gd_backend_grad_clip_scale(gd_context_backend(ctx),
                                    optimizer->grad_norm_descs,
                                    desc_count,
                                    scale_buffer,
                                    scale_offset,
                                    max_norm,
                                    GD_OPTIM_GRAD_CLIP_SCALE_EPS);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend grad clip failed");
    }
    optimizer->grad_clip_scale.version += 1U;
    if (optimizer->grad_clip_scale.version == 0U) {
        optimizer->grad_clip_scale.version = 1U;
    }
    return GD_OK;
}

gd_status gd_optimizer_step_lr(gd_context *ctx, gd_optimizer *optimizer, float lr)
{
    bool updated = false;
    uint32_t adamw_desc_count = 0U;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_adamw_lr_valid(lr)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid optimizer lr");
    }
    st = gd_optimizer_require_train_scope(ctx);
    if (st != GD_OK) {
        return st;
    }
    if (optimizer->param_count != 0U &&
        (optimizer->step_has_grad == NULL || optimizer->adamw_descs == NULL)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "adamw step cache missing");
    }
    if (optimizer->param_count != 0U) {
        memset(optimizer->step_has_grad,
               0,
               (size_t)optimizer->param_count * sizeof(optimizer->step_has_grad[0]));
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        gd_tensor grad;
        if (!slot->trainable || slot->param == NULL) {
            continue;
        }
        st = gd_tensor_grad(ctx, slot->param, &grad);
        if (st == GD_ERR_BAD_STATE) {
            gd_context_clear_error(ctx);
            continue;
        }
        if (st != GD_OK) {
            return st;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           slot,
                                           &grad,
                                           NULL,
                                           NULL,
                                           &optimizer->config,
                                           lr,
                                           false,
                                           &optimizer->adamw_descs[adamw_desc_count]);
        if (st != GD_OK) {
            return st;
        }
        optimizer->step_has_grad[i] = true;
        adamw_desc_count += 1U;
    }
    st = gd_backend_adamw_batch(gd_context_backend(ctx), optimizer->adamw_descs, adamw_desc_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend adamw failed");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        gd_optimizer_commit_step_param(&optimizer->params[i], &optimizer->config);
        updated = true;
    }
    if (updated) {
        optimizer->step += 1U;
    }
    return GD_OK;
}

gd_status gd_optimizer_step_clip_lr(gd_context *ctx,
                                    gd_optimizer *optimizer,
                                    float lr,
                                    float max_grad_norm)
{
    bool updated = false;
    uint32_t adamw_desc_count = 0U;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_adamw_lr_valid(lr)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid optimizer lr");
    }
    if (!gd_optimizer_grad_clip_norm_valid(max_grad_norm)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid grad clip norm");
    }
    st = gd_optimizer_require_train_scope(ctx);
    if (st != GD_OK) {
        return st;
    }
    if (optimizer->param_count != 0U &&
        (optimizer->step_grads == NULL || optimizer->step_has_grad == NULL ||
         optimizer->grad_norm_descs == NULL || optimizer->adamw_descs == NULL)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "adamw step cache missing");
    }
    if (optimizer->param_count != 0U) {
        memset(optimizer->step_has_grad,
               0,
               (size_t)optimizer->param_count * sizeof(optimizer->step_has_grad[0]));
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        gd_tensor grad;
        if (!slot->trainable || slot->param == NULL) {
            continue;
        }
        st = gd_tensor_grad(ctx, slot->param, &grad);
        if (st == GD_ERR_BAD_STATE) {
            gd_context_clear_error(ctx);
            continue;
        }
        if (st != GD_OK) {
            return st;
        }
        st = gd_optimizer_validate_grad(ctx, slot, &grad);
        if (st != GD_OK) {
            return st;
        }
        optimizer->step_grads[i] = grad;
        optimizer->step_has_grad[i] = true;
    }
    st = gd_optimizer_apply_grad_clip_scaled(ctx, optimizer, max_grad_norm, 1.0f, NULL);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           slot,
                                           &optimizer->step_grads[i],
                                           &optimizer->grad_clip_scale,
                                           NULL,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           &optimizer->adamw_descs[adamw_desc_count]);
        if (st != GD_OK) {
            return st;
        }
        adamw_desc_count += 1U;
    }
    st = gd_backend_adamw_batch(gd_context_backend(ctx), optimizer->adamw_descs, adamw_desc_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend adamw failed");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        gd_optimizer_commit_step_param(&optimizer->params[i], &optimizer->config);
        updated = true;
    }
    if (updated) {
        optimizer->step += 1U;
    }
    return GD_OK;
}

gd_status gd_optimizer_step_clip(gd_context *ctx,
                                 gd_optimizer *optimizer,
                                 float max_grad_norm)
{
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_optimizer_step_clip_lr(ctx, optimizer, optimizer->config.lr, max_grad_norm);
}

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer)
{
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_optimizer_step_lr(ctx, optimizer, optimizer->config.lr);
}

gd_status gd_optimizer_step_amp_lr(gd_context *ctx,
                                   gd_optimizer *optimizer,
                                   gd_amp_scaler *scaler,
                                   float lr)
{
    bool updated = false;
    int32_t found_inf = 0;
    float inv_scale;
    uint32_t amp_desc_count = 0U;
    uint32_t adamw_desc_count = 0U;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_adamw_lr_valid(lr)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid optimizer lr");
    }
    if (scaler == NULL || !scaler->config.enabled) {
        return gd_optimizer_step_lr(ctx, optimizer, lr);
    }
    st = gd_optimizer_require_train_scope(ctx);
    if (st != GD_OK) {
        return st;
    }
    if (!(scaler->scale > 0.0f)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "invalid amp scale");
    }
    if (optimizer->param_count != 0U &&
        (optimizer->step_grads == NULL || optimizer->step_has_grad == NULL ||
         optimizer->amp_descs == NULL || optimizer->adamw_descs == NULL)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "adamw step cache missing");
    }
    if (optimizer->param_count != 0U) {
        memset(optimizer->step_has_grad,
               0,
               (size_t)optimizer->param_count * sizeof(optimizer->step_has_grad[0]));
    }
    inv_scale = 1.0f / scaler->scale;
    st = gd_tensor_zero_(ctx, &optimizer->found_inf);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        gd_tensor grad;
        if (!slot->trainable || slot->param == NULL) {
            continue;
        }
        st = gd_tensor_grad(ctx, slot->param, &grad);
        if (st == GD_ERR_BAD_STATE) {
            gd_context_clear_error(ctx);
            continue;
        }
        if (st != GD_OK) {
            return st;
        }
        st = gd_optimizer_build_unscale_desc(ctx,
                                             optimizer,
                                             slot,
                                             &grad,
                                             inv_scale,
                                             &optimizer->amp_descs[amp_desc_count]);
        if (st != GD_OK) {
            return st;
        }
        optimizer->step_grads[i] = grad;
        optimizer->step_has_grad[i] = true;
        amp_desc_count += 1U;
    }
    st = gd_backend_amp_unscale_batch(gd_context_backend(ctx), optimizer->amp_descs, amp_desc_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend amp unscale failed");
    }
    if (!scaler->config.defer_found_inf) {
        st = gd_tensor_read(ctx, &optimizer->found_inf, &found_inf, sizeof(found_inf));
        if (st != GD_OK) {
            return st;
        }
        if (found_inf != 0) {
            gd_amp_scaler_update(scaler, true);
            return GD_OK;
        }
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           slot,
                                           &optimizer->step_grads[i],
                                           NULL,
                                           scaler->config.defer_found_inf ? &optimizer->found_inf : NULL,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           &optimizer->adamw_descs[adamw_desc_count]);
        if (st != GD_OK) {
            return st;
        }
        adamw_desc_count += 1U;
    }
    st = gd_backend_adamw_batch(gd_context_backend(ctx), optimizer->adamw_descs, adamw_desc_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend adamw failed");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        gd_optimizer_commit_step_param(&optimizer->params[i], &optimizer->config);
        updated = true;
    }
    if (updated) {
        optimizer->step += 1U;
        gd_amp_scaler_update(scaler, false);
    }
    return GD_OK;
}

gd_status gd_optimizer_step_amp_clip_lr(gd_context *ctx,
                                        gd_optimizer *optimizer,
                                        gd_amp_scaler *scaler,
                                        float lr,
                                        float max_grad_norm)
{
    bool updated = false;
    int32_t found_inf = 0;
    float inv_scale;
    uint32_t adamw_desc_count = 0U;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_adamw_lr_valid(lr)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid optimizer lr");
    }
    if (!gd_optimizer_grad_clip_norm_valid(max_grad_norm)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid grad clip norm");
    }
    if (scaler == NULL || !scaler->config.enabled) {
        return gd_optimizer_step_clip_lr(ctx, optimizer, lr, max_grad_norm);
    }
    st = gd_optimizer_require_train_scope(ctx);
    if (st != GD_OK) {
        return st;
    }
    if (!(scaler->scale > 0.0f)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "invalid amp scale");
    }
    if (optimizer->param_count != 0U &&
        (optimizer->step_grads == NULL || optimizer->step_has_grad == NULL ||
         optimizer->grad_norm_descs == NULL || optimizer->adamw_descs == NULL)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE, "adamw step cache missing");
    }
    if (optimizer->param_count != 0U) {
        memset(optimizer->step_has_grad,
               0,
               (size_t)optimizer->param_count * sizeof(optimizer->step_has_grad[0]));
    }
    inv_scale = 1.0f / scaler->scale;
    st = gd_tensor_zero_(ctx, &optimizer->found_inf);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        gd_tensor grad;
        if (!slot->trainable || slot->param == NULL) {
            continue;
        }
        st = gd_tensor_grad(ctx, slot->param, &grad);
        if (st == GD_ERR_BAD_STATE) {
            gd_context_clear_error(ctx);
            continue;
        }
        if (st != GD_OK) {
            return st;
        }
        st = gd_optimizer_validate_grad(ctx, slot, &grad);
        if (st != GD_OK) {
            return st;
        }
        optimizer->step_grads[i] = grad;
        optimizer->step_has_grad[i] = true;
    }
    st = gd_optimizer_apply_grad_clip_scaled(ctx, optimizer, max_grad_norm, inv_scale, &optimizer->found_inf);
    if (st != GD_OK) {
        return st;
    }
    if (!scaler->config.defer_found_inf) {
        st = gd_tensor_read(ctx, &optimizer->found_inf, &found_inf, sizeof(found_inf));
        if (st != GD_OK) {
            return st;
        }
        if (found_inf != 0) {
            gd_amp_scaler_update(scaler, true);
            return GD_OK;
        }
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           slot,
                                           &optimizer->step_grads[i],
                                           &optimizer->grad_clip_scale,
                                           scaler->config.defer_found_inf ? &optimizer->found_inf : NULL,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           &optimizer->adamw_descs[adamw_desc_count]);
        if (st != GD_OK) {
            return st;
        }
        adamw_desc_count += 1U;
    }
    st = gd_backend_adamw_batch(gd_context_backend(ctx), optimizer->adamw_descs, adamw_desc_count);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend adamw failed");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        gd_optimizer_commit_step_param(&optimizer->params[i], &optimizer->config);
        updated = true;
    }
    if (updated) {
        optimizer->step += 1U;
        gd_amp_scaler_update(scaler, false);
    }
    return GD_OK;
}

gd_status gd_optimizer_step_amp_clip(gd_context *ctx,
                                     gd_optimizer *optimizer,
                                     gd_amp_scaler *scaler,
                                     float max_grad_norm)
{
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_optimizer_step_amp_clip_lr(ctx,
                                         optimizer,
                                         scaler,
                                         optimizer->config.lr,
                                         max_grad_norm);
}

gd_status gd_optimizer_step_amp(gd_context *ctx, gd_optimizer *optimizer, gd_amp_scaler *scaler)
{
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_optimizer_step_amp_lr(ctx, optimizer, scaler, optimizer->config.lr);
}

gd_status gd_optimizer_last_grad_norm(gd_context *ctx,
                                      const gd_optimizer *optimizer,
                                      float *out)
{
    float clip_stats[2] = {0.0f, 0.0f};
    gd_status st;
    if (ctx == NULL || optimizer == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_read_f32(ctx, &optimizer->grad_clip_scale, clip_stats, 2U);
    if (st != GD_OK) {
        return st;
    }
    *out = clip_stats[1];
    return GD_OK;
}

uint64_t gd_optimizer_step_count(const gd_optimizer *optimizer)
{
    return optimizer != NULL ? optimizer->step : 0U;
}
