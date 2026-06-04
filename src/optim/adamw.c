#include <gradients/optimizer.h>
#include <gradients/autograd.h>

#include "../core/backend.h"
#include "../core/memory_internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct gd_optimizer_param {
    char path[GD_MODULE_PATH_MAX];
    gd_tensor *param;
    gd_tensor m;
    gd_tensor v;
    size_t count;
    float lr_mult;
    float weight_decay;
    bool trainable;
} gd_optimizer_param;

struct gd_optimizer {
    gd_optimizer_param *params;
    uint32_t param_count;
    gd_adamw_config config;
    uint64_t step;
    float beta1_power;
    float beta2_power;
};

static bool gd_adamw_config_valid(const gd_adamw_config *config)
{
    return config != NULL && config->lr >= 0.0f && config->beta1 >= 0.0f &&
           config->beta1 < 1.0f && config->beta2 >= 0.0f && config->beta2 < 1.0f &&
           config->eps > 0.0f && config->weight_decay >= 0.0f;
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
    slot->lr_mult = ref->lr_mult;
    slot->weight_decay = ref->group_index >= 0 ? ref->weight_decay : config->weight_decay;
    slot->trainable = ref->trainable;
    st = gd_tensor_zeros(ctx,
                         GD_ARENA_PARAMS,
                         GD_DTYPE_F32,
                         ref->tensor->rank,
                         ref->tensor->shape,
                         256U,
                         &slot->m);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_zeros(ctx,
                         GD_ARENA_PARAMS,
                         GD_DTYPE_F32,
                         ref->tensor->rank,
                         ref->tensor->shape,
                         256U,
                         &slot->v);
    if (st != GD_OK) {
        return st;
    }
    slot->m.requires_grad = false;
    slot->m.is_leaf = false;
    slot->v.requires_grad = false;
    slot->v.is_leaf = false;
    return GD_OK;
}

static gd_status gd_optimizer_step_param(gd_context *ctx,
                                         const gd_optimizer_param *slot,
                                         const gd_tensor *grad,
                                         const gd_adamw_config *config,
                                         float bias_correction1,
                                         float bias_correction2)
{
    gd_backend_adamw_desc desc;
    gd_status st;
    if (ctx == NULL || slot == NULL || slot->param == NULL || grad == NULL || config == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_validate_grad(ctx, slot, grad);
    if (st != GD_OK) {
        return st;
    }
    memset(&desc, 0, sizeof(desc));
    desc.param_buffer = (gd_backend_buffer *)slot->param->storage.buffer;
    desc.param_offset = gd_tensor_storage_offset(slot->param);
    desc.grad_buffer = (gd_backend_buffer *)grad->storage.buffer;
    desc.grad_offset = gd_tensor_storage_offset(grad);
    desc.m_buffer = (gd_backend_buffer *)slot->m.storage.buffer;
    desc.m_offset = gd_tensor_storage_offset(&slot->m);
    desc.v_buffer = (gd_backend_buffer *)slot->v.storage.buffer;
    desc.v_offset = gd_tensor_storage_offset(&slot->v);
    desc.count = slot->count;
    desc.param_dtype = (uint32_t)slot->param->dtype;
    desc.grad_dtype = (uint32_t)grad->dtype;
    desc.lr = config->lr * slot->lr_mult;
    desc.beta1 = config->beta1;
    desc.beta2 = config->beta2;
    desc.eps = config->eps;
    desc.weight_decay = slot->weight_decay;
    desc.bias_correction1 = bias_correction1;
    desc.bias_correction2 = bias_correction2;
    st = gd_backend_adamw(gd_context_backend(ctx), &desc);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend adamw failed");
    }
    slot->param->version += 1U;
    if (slot->param->version == 0U) {
        slot->param->version = 1U;
    }
    return GD_OK;
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
    optimizer->beta1_power = 1.0f;
    optimizer->beta2_power = 1.0f;
    if (trainable_count != 0U) {
        optimizer->params = (gd_optimizer_param *)calloc(trainable_count, sizeof(optimizer->params[0]));
        if (optimizer->params == NULL) {
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
    *out = optimizer;
    return GD_OK;
}

void gd_optimizer_destroy(gd_optimizer *optimizer)
{
    if (optimizer == NULL) {
        return;
    }
    free(optimizer->params);
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

gd_status gd_optimizer_step(gd_context *ctx, gd_optimizer *optimizer)
{
    gd_adamw_config *config;
    float bias_correction1;
    float bias_correction2;
    uint32_t i;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_context_in_scope(ctx) || gd_context_scope_mode(ctx) != GD_SCOPE_TRAIN) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer step requires active train scope");
    }
    config = &optimizer->config;
    optimizer->step += 1U;
    optimizer->beta1_power *= config->beta1;
    optimizer->beta2_power *= config->beta2;
    bias_correction1 = config->bias_correction ? 1.0f - optimizer->beta1_power : 1.0f;
    bias_correction2 = config->bias_correction ? 1.0f - optimizer->beta2_power : 1.0f;
    if (!(bias_correction1 > 0.0f) || !(bias_correction2 > 0.0f)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "invalid adamw bias correction state");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        gd_tensor grad;
        gd_status st;
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
        st = gd_optimizer_step_param(ctx, slot, &grad, config, bias_correction1, bias_correction2);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

uint64_t gd_optimizer_step_count(const gd_optimizer *optimizer)
{
    return optimizer != NULL ? optimizer->step : 0U;
}
