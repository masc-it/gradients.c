#include <gradients/optimizer.h>
#include <gradients/autograd.h>
#include <gradients/transfer.h>

#include "../core/backend.h"
#include "../core/memory_internal.h"

#include <errno.h>
#include <float.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GD_OPTIM_GRAD_CLIP_BLOCK_ELEMS 1024U
#define GD_OPTIM_GRAD_CLIP_SCALE_EPS 1.0e-6f
#define GD_OPTIM_STATE_FORMAT "gd_optimizer_adamw_state_v1"
#define GD_OPTIM_STATE_MODULE_NAME "gd_optimizer"
#define GD_AMP_SCALE_VALUES 2U
#define GD_AMP_FLAG_VALUES 3U
#define GD_AMP_SCALE_INDEX 0U
#define GD_AMP_INV_SCALE_INDEX 1U
#define GD_AMP_GROWTH_TRACKER_INDEX 0U
#define GD_AMP_FOUND_INF_INDEX 1U
#define GD_AMP_LAST_FOUND_INF_INDEX 2U
#define GD_OPTIM_STATE_F32_FIELDS 5U
#define GD_OPTIM_BETA1_POWER_FIELD 0U
#define GD_OPTIM_BETA2_POWER_FIELD 1U
#define GD_OPTIM_STEP_SCALE_FIELD 2U
#define GD_OPTIM_DECAY_SCALE_FIELD 3U
#define GD_OPTIM_EPS_SCALED_FIELD 4U

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
    gd_tensor grad_clip_partials;
    gd_tensor grad_clip_scale;
    gd_tensor unit_scale;
    gd_tensor device_state_f32;
    gd_tensor device_state_i32;
    gd_adamw_config config;
    uint64_t step;
    bool host_state_dirty;
};

struct gd_amp_scaler {
    gd_amp_config config;
    gd_tensor scale_state;
    gd_tensor flag_state;
    float scale;
    uint32_t growth_tracker;
    bool last_found_inf;
    bool host_state_dirty;
    bool step_begun;
};

typedef struct gd_optim_string_builder {
    char *data;
    size_t len;
    size_t capacity;
} gd_optim_string_builder;

typedef struct gd_optimizer_saved_param_state {
    uint64_t step;
    float beta1_power;
    float beta2_power;
    float lr_mult;
    float weight_decay;
    bool trainable;
    bool has_master;
} gd_optimizer_saved_param_state;

typedef struct gd_optimizer_saved_state {
    gd_adamw_config config;
    uint64_t step;
    uint32_t param_count;
    gd_optimizer_saved_param_state *params;
} gd_optimizer_saved_state;

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

static size_t gd_optimizer_state_f32_offset(const gd_optimizer *optimizer,
                                            uint32_t field,
                                            uint32_t param_index)
{
    const size_t index = ((size_t)field * (size_t)optimizer->param_count) + (size_t)param_index;
    return gd_tensor_storage_offset(&optimizer->device_state_f32) + index * sizeof(float);
}

static size_t gd_optimizer_state_i32_param_offset(const gd_optimizer *optimizer,
                                                  uint32_t param_index)
{
    return gd_tensor_storage_offset(&optimizer->device_state_i32) +
           (size_t)param_index * sizeof(int32_t);
}

static size_t gd_optimizer_state_i32_step_offset(const gd_optimizer *optimizer)
{
    return gd_tensor_storage_offset(&optimizer->device_state_i32) +
           (size_t)optimizer->param_count * sizeof(int32_t);
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

static void gd_optim_string_builder_free(gd_optim_string_builder *builder)
{
    if (builder == NULL) {
        return;
    }
    free(builder->data);
    memset(builder, 0, sizeof(*builder));
}

static gd_status gd_optim_string_builder_reserve(gd_optim_string_builder *builder,
                                                 size_t required)
{
    char *grown;
    size_t new_capacity;
    if (builder == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (required <= builder->capacity) {
        return GD_OK;
    }
    new_capacity = builder->capacity != 0U ? builder->capacity : 1024U;
    while (new_capacity < required) {
        if (new_capacity > SIZE_MAX / 2U) {
            new_capacity = required;
        } else {
            new_capacity *= 2U;
        }
    }
    grown = (char *)realloc(builder->data, new_capacity);
    if (grown == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    builder->data = grown;
    builder->capacity = new_capacity;
    return GD_OK;
}

static gd_status gd_optim_string_builder_appendf(gd_optim_string_builder *builder,
                                                 const char *fmt,
                                                 ...)
{
    va_list args;
    va_list copy;
    int needed;
    size_t required;
    gd_status st;
    if (builder == NULL || fmt == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    va_start(args, fmt);
    va_copy(copy, args);
    needed = vsnprintf(NULL, 0U, fmt, args);
    va_end(args);
    if (needed < 0) {
        va_end(copy);
        return GD_ERR_INVALID_ARGUMENT;
    }
    if ((size_t)needed > SIZE_MAX - builder->len - 1U) {
        va_end(copy);
        return GD_ERR_OUT_OF_MEMORY;
    }
    required = builder->len + (size_t)needed + 1U;
    st = gd_optim_string_builder_reserve(builder, required);
    if (st != GD_OK) {
        va_end(copy);
        return st;
    }
    (void)vsnprintf(builder->data + builder->len,
                    builder->capacity - builder->len,
                    fmt,
                    copy);
    va_end(copy);
    builder->len += (size_t)needed;
    return GD_OK;
}

static gd_status gd_optim_snprintf_name(gd_context *ctx,
                                         char name[GD_MODULE_NAME_MAX],
                                         const char *fmt,
                                         uint32_t index)
{
    int n;
    if (name == NULL || fmt == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = snprintf(name, GD_MODULE_NAME_MAX, fmt, (unsigned)index);
    if (n < 0 || (size_t)n >= GD_MODULE_NAME_MAX) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "optimizer checkpoint buffer name too long");
    }
    return GD_OK;
}

static gd_status gd_optimizer_state_module_add_buffer(gd_context *ctx,
                                                      gd_module *module,
                                                      const char *name,
                                                      gd_tensor *tensor)
{
    gd_status st;
    if (ctx == NULL || module == NULL || name == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_add_buffer(module, name, tensor);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "optimizer checkpoint buffer registration failed");
    }
    return GD_OK;
}

static gd_status gd_optimizer_build_state_module(gd_context *ctx,
                                                 gd_optimizer *optimizer,
                                                 gd_module *module)
{
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL || module == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(module, 0, sizeof(*module));
    st = gd_module_init(ctx, module, GD_OPTIM_STATE_MODULE_NAME);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        char name[GD_MODULE_NAME_MAX];
        st = gd_optim_snprintf_name(ctx, name, "slot_%06u_m", i);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_state_module_add_buffer(ctx, module, name, &slot->m);
        if (st != GD_OK) { return st; }
        st = gd_optim_snprintf_name(ctx, name, "slot_%06u_v", i);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_state_module_add_buffer(ctx, module, name, &slot->v);
        if (st != GD_OK) { return st; }
        if (slot->has_master) {
            st = gd_optim_snprintf_name(ctx, name, "slot_%06u_master", i);
            if (st != GD_OK) { return st; }
            st = gd_optimizer_state_module_add_buffer(ctx, module, name, &slot->master);
            if (st != GD_OK) { return st; }
        }
    }
    return GD_OK;
}

static gd_status gd_optimizer_build_state_metadata(const gd_optimizer *optimizer,
                                                   char **metadata_out,
                                                   size_t *metadata_len_out)
{
    gd_optim_string_builder builder;
    uint32_t i;
    gd_status st;
    if (metadata_out != NULL) {
        *metadata_out = NULL;
    }
    if (metadata_len_out != NULL) {
        *metadata_len_out = 0U;
    }
    if (optimizer == NULL || metadata_out == NULL || metadata_len_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&builder, 0, sizeof(builder));
#define GD_OPTIM_META_APPEND(...)                                             \
    do {                                                                      \
        st = gd_optim_string_builder_appendf(&builder, __VA_ARGS__);           \
        if (st != GD_OK) {                                                    \
            gd_optim_string_builder_free(&builder);                            \
            return st;                                                        \
        }                                                                     \
    } while (0)
    GD_OPTIM_META_APPEND("format=%s\n", GD_OPTIM_STATE_FORMAT);
    GD_OPTIM_META_APPEND("optimizer_type=adamw\n");
    GD_OPTIM_META_APPEND("global_step=%llu\n", (unsigned long long)optimizer->step);
    GD_OPTIM_META_APPEND("param_count=%u\n", (unsigned)optimizer->param_count);
    GD_OPTIM_META_APPEND("config.lr=%.9g\n", (double)optimizer->config.lr);
    GD_OPTIM_META_APPEND("config.beta1=%.9g\n", (double)optimizer->config.beta1);
    GD_OPTIM_META_APPEND("config.beta2=%.9g\n", (double)optimizer->config.beta2);
    GD_OPTIM_META_APPEND("config.eps=%.9g\n", (double)optimizer->config.eps);
    GD_OPTIM_META_APPEND("config.weight_decay=%.9g\n", (double)optimizer->config.weight_decay);
    GD_OPTIM_META_APPEND("config.bias_correction=%u\n", optimizer->config.bias_correction ? 1U : 0U);
    for (i = 0U; i < optimizer->param_count; ++i) {
        const gd_optimizer_param *slot = &optimizer->params[i];
        GD_OPTIM_META_APPEND("param.%u.path=%s\n", (unsigned)i, slot->path);
        GD_OPTIM_META_APPEND("param.%u.step=%llu\n", (unsigned)i, (unsigned long long)slot->step);
        GD_OPTIM_META_APPEND("param.%u.beta1_power=%.9g\n", (unsigned)i, (double)slot->beta1_power);
        GD_OPTIM_META_APPEND("param.%u.beta2_power=%.9g\n", (unsigned)i, (double)slot->beta2_power);
        GD_OPTIM_META_APPEND("param.%u.lr_mult=%.9g\n", (unsigned)i, (double)slot->lr_mult);
        GD_OPTIM_META_APPEND("param.%u.weight_decay=%.9g\n", (unsigned)i, (double)slot->weight_decay);
        GD_OPTIM_META_APPEND("param.%u.trainable=%u\n", (unsigned)i, slot->trainable ? 1U : 0U);
        GD_OPTIM_META_APPEND("param.%u.has_master=%u\n", (unsigned)i, slot->has_master ? 1U : 0U);
    }
#undef GD_OPTIM_META_APPEND
    *metadata_out = builder.data;
    *metadata_len_out = builder.len;
    return GD_OK;
}

static gd_status gd_optimizer_metadata_value(const char *metadata,
                                             const char *key,
                                             const char **value_out,
                                             size_t *value_len_out)
{
    const char *line;
    const size_t key_len = key != NULL ? strlen(key) : 0U;
    if (metadata == NULL || key == NULL || value_out == NULL || value_len_out == NULL || key_len == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    line = metadata;
    while (*line != '\0') {
        const char *end = strchr(line, '\n');
        const size_t line_len = end != NULL ? (size_t)(end - line) : strlen(line);
        if (line_len > key_len && memcmp(line, key, key_len) == 0 && line[key_len] == '=') {
            *value_out = line + key_len + 1U;
            *value_len_out = line_len - key_len - 1U;
            return GD_OK;
        }
        if (end == NULL) {
            break;
        }
        line = end + 1;
    }
    return GD_ERR_BAD_STATE;
}

static gd_status gd_optimizer_metadata_copy_value(gd_context *ctx,
                                                  const char *metadata,
                                                  const char *key,
                                                  char *buffer,
                                                  size_t buffer_size)
{
    const char *value;
    size_t value_len;
    gd_status st;
    if (ctx == NULL || metadata == NULL || key == NULL || buffer == NULL || buffer_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_metadata_value(metadata, key, &value, &value_len);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "optimizer checkpoint metadata key missing");
    }
    if (value_len >= buffer_size) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "optimizer checkpoint metadata value too long");
    }
    memcpy(buffer, value, value_len);
    buffer[value_len] = '\0';
    return GD_OK;
}

static gd_status gd_optimizer_metadata_get_string(gd_context *ctx,
                                                  const char *metadata,
                                                  const char *key,
                                                  char *buffer,
                                                  size_t buffer_size)
{
    return gd_optimizer_metadata_copy_value(ctx, metadata, key, buffer, buffer_size);
}

static gd_status gd_optimizer_metadata_get_u64(gd_context *ctx,
                                               const char *metadata,
                                               const char *key,
                                               uint64_t *out)
{
    char buffer[128];
    char *end = NULL;
    unsigned long long parsed;
    gd_status st;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_metadata_copy_value(ctx, metadata, key, buffer, sizeof(buffer));
    if (st != GD_OK) {
        return st;
    }
    errno = 0;
    parsed = strtoull(buffer, &end, 10);
    if (errno != 0 || end == buffer || *end != '\0') {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "optimizer checkpoint metadata uint parse failed");
    }
    *out = (uint64_t)parsed;
    return GD_OK;
}

static gd_status gd_optimizer_metadata_get_bool(gd_context *ctx,
                                                const char *metadata,
                                                const char *key,
                                                bool *out)
{
    uint64_t value;
    gd_status st;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_metadata_get_u64(ctx, metadata, key, &value);
    if (st != GD_OK) {
        return st;
    }
    if (value > 1U) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "optimizer checkpoint metadata bool parse failed");
    }
    *out = value != 0U;
    return GD_OK;
}

static gd_status gd_optimizer_metadata_get_f32(gd_context *ctx,
                                               const char *metadata,
                                               const char *key,
                                               float *out)
{
    char buffer[128];
    char *end = NULL;
    float parsed;
    gd_status st;
    if (out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_metadata_copy_value(ctx, metadata, key, buffer, sizeof(buffer));
    if (st != GD_OK) {
        return st;
    }
    errno = 0;
    parsed = strtof(buffer, &end);
    /* strtof may set ERANGE for a finite subnormal result.  Adam beta powers
     * legitimately underflow into the subnormal range in long runs, so accept
     * ERANGE here and let the field-specific validation below reject invalid
     * ranges/overflows. */
    if ((errno != 0 && errno != ERANGE) || end == buffer || *end != '\0' || parsed != parsed ||
        parsed > FLT_MAX || parsed < -FLT_MAX) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "optimizer checkpoint metadata float parse failed");
    }
    *out = parsed;
    return GD_OK;
}

static gd_status gd_optimizer_metadata_param_key(char key[128],
                                                 uint32_t index,
                                                 const char *field)
{
    int n;
    if (key == NULL || field == NULL || field[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    n = snprintf(key, 128U, "param.%u.%s", (unsigned)index, field);
    if (n < 0 || (size_t)n >= 128U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_optimizer_saved_state_free(gd_optimizer_saved_state *saved)
{
    if (saved == NULL) {
        return;
    }
    free(saved->params);
    memset(saved, 0, sizeof(*saved));
}

static gd_status gd_optimizer_parse_state_metadata(gd_context *ctx,
                                                   const gd_optimizer *optimizer,
                                                   const char *metadata,
                                                   gd_optimizer_saved_state *saved)
{
    char text[GD_MODULE_PATH_MAX];
    uint64_t u64_value;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL || metadata == NULL || saved == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(saved, 0, sizeof(*saved));
    st = gd_optimizer_metadata_get_string(ctx, metadata, "format", text, sizeof(text));
    if (st != GD_OK) { return st; }
    if (strcmp(text, GD_OPTIM_STATE_FORMAT) != 0) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer checkpoint format mismatch");
    }
    st = gd_optimizer_metadata_get_string(ctx, metadata, "optimizer_type", text, sizeof(text));
    if (st != GD_OK) { return st; }
    if (strcmp(text, "adamw") != 0) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer checkpoint type mismatch");
    }
    st = gd_optimizer_metadata_get_u64(ctx, metadata, "global_step", &saved->step);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_u64(ctx, metadata, "param_count", &u64_value);
    if (st != GD_OK) { return st; }
    if (u64_value != (uint64_t)optimizer->param_count || u64_value > (uint64_t)UINT32_MAX) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer checkpoint param count mismatch");
    }
    saved->param_count = (uint32_t)u64_value;
    st = gd_optimizer_metadata_get_f32(ctx, metadata, "config.lr", &saved->config.lr);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_f32(ctx, metadata, "config.beta1", &saved->config.beta1);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_f32(ctx, metadata, "config.beta2", &saved->config.beta2);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_f32(ctx, metadata, "config.eps", &saved->config.eps);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_f32(ctx, metadata, "config.weight_decay", &saved->config.weight_decay);
    if (st != GD_OK) { return st; }
    st = gd_optimizer_metadata_get_bool(ctx, metadata, "config.bias_correction", &saved->config.bias_correction);
    if (st != GD_OK) { return st; }
    if (!gd_adamw_config_valid(&saved->config)) {
        return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                    "optimizer checkpoint config invalid");
    }
    if (saved->param_count != 0U) {
        saved->params = (gd_optimizer_saved_param_state *)calloc(saved->param_count,
                                                                 sizeof(saved->params[0]));
        if (saved->params == NULL) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "optimizer checkpoint saved state allocation failed");
        }
    }
    for (i = 0U; i < saved->param_count; ++i) {
        gd_optimizer_saved_param_state *param_state = &saved->params[i];
        const gd_optimizer_param *slot = &optimizer->params[i];
        char key[128];
        st = gd_optimizer_metadata_param_key(key, i, "path");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_string(ctx, metadata, key, text, sizeof(text));
        if (st != GD_OK) { return st; }
        if (strcmp(text, slot->path) != 0) {
            return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                        "optimizer checkpoint param path mismatch");
        }
        st = gd_optimizer_metadata_param_key(key, i, "step");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_u64(ctx, metadata, key, &param_state->step);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "beta1_power");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_f32(ctx, metadata, key, &param_state->beta1_power);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "beta2_power");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_f32(ctx, metadata, key, &param_state->beta2_power);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "lr_mult");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_f32(ctx, metadata, key, &param_state->lr_mult);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "weight_decay");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_f32(ctx, metadata, key, &param_state->weight_decay);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "trainable");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_bool(ctx, metadata, key, &param_state->trainable);
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_param_key(key, i, "has_master");
        if (st != GD_OK) { return st; }
        st = gd_optimizer_metadata_get_bool(ctx, metadata, key, &param_state->has_master);
        if (st != GD_OK) { return st; }
        if (param_state->has_master != slot->has_master) {
            return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                        "optimizer checkpoint master-state mismatch");
        }
        if (param_state->beta1_power < 0.0f || param_state->beta1_power > 1.0f ||
            param_state->beta2_power < 0.0f || param_state->beta2_power > 1.0f ||
            param_state->lr_mult < 0.0f || param_state->lr_mult > FLT_MAX ||
            param_state->weight_decay < 0.0f || param_state->weight_decay > FLT_MAX) {
            return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                        "optimizer checkpoint param options invalid");
        }
    }
    return GD_OK;
}

static void gd_optimizer_apply_saved_state(gd_optimizer *optimizer,
                                           const gd_optimizer_saved_state *saved)
{
    uint32_t i;
    if (optimizer == NULL || saved == NULL) {
        return;
    }
    optimizer->config = saved->config;
    optimizer->step = saved->step;
    for (i = 0U; i < saved->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        const gd_optimizer_saved_param_state *param_state = &saved->params[i];
        slot->step = param_state->step;
        slot->beta1_power = param_state->beta1_power;
        slot->beta2_power = param_state->beta2_power;
        slot->lr_mult = param_state->lr_mult;
        slot->weight_decay = param_state->weight_decay;
        slot->trainable = param_state->trainable;
    }
}

static gd_status gd_optimizer_write_device_scalar_state(gd_context *ctx,
                                                        gd_optimizer *optimizer)
{
    size_t state_f32_count;
    size_t state_i32_count;
    float *state_f32_values;
    uint32_t *state_i32_values;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    state_f32_count = (size_t)optimizer->param_count * (size_t)GD_OPTIM_STATE_F32_FIELDS;
    if (state_f32_count == 0U) {
        state_f32_count = 1U;
    }
    state_i32_count = (size_t)optimizer->param_count + 1U;
    state_f32_values = (float *)calloc(state_f32_count, sizeof(state_f32_values[0]));
    state_i32_values = (uint32_t *)calloc(state_i32_count, sizeof(state_i32_values[0]));
    if (state_f32_values == NULL || state_i32_values == NULL) {
        free(state_f32_values);
        free(state_i32_values);
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "adamw scalar state write allocation failed");
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        if (optimizer->params[i].step > (uint64_t)UINT32_MAX) {
            free(state_f32_values);
            free(state_i32_values);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "adamw scalar step exceeds device state range");
        }
        state_f32_values[(size_t)GD_OPTIM_BETA1_POWER_FIELD * optimizer->param_count + i] =
            optimizer->params[i].beta1_power;
        state_f32_values[(size_t)GD_OPTIM_BETA2_POWER_FIELD * optimizer->param_count + i] =
            optimizer->params[i].beta2_power;
        state_i32_values[i] = (uint32_t)optimizer->params[i].step;
    }
    if (optimizer->step > (uint64_t)UINT32_MAX) {
        free(state_f32_values);
        free(state_i32_values);
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "adamw optimizer step exceeds device state range");
    }
    state_i32_values[optimizer->param_count] = (uint32_t)optimizer->step;
    st = gd_tensor_write(ctx,
                         &optimizer->device_state_f32,
                         state_f32_values,
                         state_f32_count * sizeof(state_f32_values[0]));
    if (st == GD_OK) {
        st = gd_tensor_write(ctx,
                             &optimizer->device_state_i32,
                             state_i32_values,
                             state_i32_count * sizeof(state_i32_values[0]));
    }
    free(state_f32_values);
    free(state_i32_values);
    optimizer->host_state_dirty = false;
    return st;
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
                                                gd_optimizer *optimizer,
                                                uint32_t param_index,
                                                gd_optimizer_param *slot,
                                                const gd_tensor *grad,
                                                const gd_tensor *grad_scale,
                                                size_t grad_scale_extra_offset,
                                                const gd_tensor *found_inf,
                                                const gd_adamw_config *config,
                                                float base_lr,
                                                bool grad_already_validated,
                                                bool use_device_state,
                                                bool update_optimizer_step,
                                                gd_backend_adamw_desc *out_desc)
{
    float beta1_power = 1.0f;
    float beta2_power = 1.0f;
    float bias_correction1 = 1.0f;
    float bias_correction2 = 1.0f;
    gd_status st;
    if (ctx == NULL || optimizer == NULL || slot == NULL || slot->param == NULL ||
        grad == NULL || config == NULL || out_desc == NULL ||
        param_index >= optimizer->param_count) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!grad_already_validated) {
        st = gd_optimizer_validate_grad(ctx, slot, grad);
        if (st != GD_OK) {
            return st;
        }
    }
    if (!use_device_state) {
        beta1_power = slot->beta1_power * config->beta1;
        beta2_power = slot->beta2_power * config->beta2;
        bias_correction1 = config->bias_correction ? 1.0f - beta1_power : 1.0f;
        bias_correction2 = config->bias_correction ? 1.0f - beta2_power : 1.0f;
        if (!(bias_correction1 > 0.0f) || !(bias_correction2 > 0.0f)) {
            return gd_context_set_error(ctx, GD_ERR_BAD_STATE,
                                        "invalid adamw bias correction state");
        }
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
        out_desc->grad_scale_offset = gd_tensor_storage_offset(grad_scale) + grad_scale_extra_offset;
        out_desc->has_grad_scale = 1U;
    }
    if (found_inf != NULL) {
        out_desc->found_inf_buffer = (gd_backend_buffer *)found_inf->storage.buffer;
        out_desc->found_inf_offset = gd_tensor_storage_offset(found_inf) +
                                     (size_t)GD_AMP_FOUND_INF_INDEX * sizeof(int32_t);
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
    out_desc->bias_correction = config->bias_correction ? 1U : 0U;
    out_desc->bias_correction1 = bias_correction1;
    out_desc->bias_correction2 = bias_correction2;
    if (use_device_state) {
        out_desc->state_f32_buffer = (gd_backend_buffer *)optimizer->device_state_f32.storage.buffer;
        out_desc->beta1_power_offset = gd_optimizer_state_f32_offset(optimizer,
                                                                     GD_OPTIM_BETA1_POWER_FIELD,
                                                                     param_index);
        out_desc->beta2_power_offset = gd_optimizer_state_f32_offset(optimizer,
                                                                     GD_OPTIM_BETA2_POWER_FIELD,
                                                                     param_index);
        out_desc->step_scale_offset = gd_optimizer_state_f32_offset(optimizer,
                                                                    GD_OPTIM_STEP_SCALE_FIELD,
                                                                    param_index);
        out_desc->decay_scale_offset = gd_optimizer_state_f32_offset(optimizer,
                                                                     GD_OPTIM_DECAY_SCALE_FIELD,
                                                                     param_index);
        out_desc->eps_scaled_offset = gd_optimizer_state_f32_offset(optimizer,
                                                                    GD_OPTIM_EPS_SCALED_FIELD,
                                                                    param_index);
        out_desc->state_i32_buffer = (gd_backend_buffer *)optimizer->device_state_i32.storage.buffer;
        out_desc->param_step_offset = gd_optimizer_state_i32_param_offset(optimizer, param_index);
        out_desc->optimizer_step_offset = gd_optimizer_state_i32_step_offset(optimizer);
        out_desc->has_device_state = 1U;
        out_desc->update_optimizer_step = update_optimizer_step ? 1U : 0U;
    }
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
    config.init_scale = 32768.0f;
    config.growth_factor = 2.0f;
    config.backoff_factor = 0.5f;
    config.min_scale = 1.0f;
    config.max_scale = 2147483648.0f;
    config.growth_interval = 2000U;
    return config;
}

static gd_status gd_amp_scaler_device_desc(const gd_amp_scaler *scaler,
                                           gd_backend_amp_state_desc *out)
{
    if (scaler == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    out->scale_buffer = (gd_backend_buffer *)scaler->scale_state.storage.buffer;
    out->scale_offset = gd_tensor_storage_offset(&scaler->scale_state);
    out->flags_buffer = (gd_backend_buffer *)scaler->flag_state.storage.buffer;
    out->flags_offset = gd_tensor_storage_offset(&scaler->flag_state);
    out->growth_factor = scaler->config.growth_factor;
    out->backoff_factor = scaler->config.backoff_factor;
    out->min_scale = scaler->config.min_scale;
    out->max_scale = scaler->config.max_scale;
    out->growth_interval = scaler->config.growth_interval;
    return GD_OK;
}

static gd_status gd_amp_scaler_write_device_state(gd_context *ctx, gd_amp_scaler *scaler)
{
    float scale_values[GD_AMP_SCALE_VALUES];
    int32_t flag_values[GD_AMP_FLAG_VALUES];
    gd_status st;
    if (ctx == NULL || scaler == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    scale_values[GD_AMP_SCALE_INDEX] = scaler->scale;
    scale_values[GD_AMP_INV_SCALE_INDEX] = 1.0f / scaler->scale;
    flag_values[GD_AMP_GROWTH_TRACKER_INDEX] = (int32_t)scaler->growth_tracker;
    flag_values[GD_AMP_FOUND_INF_INDEX] = 0;
    flag_values[GD_AMP_LAST_FOUND_INF_INDEX] = scaler->last_found_inf ? 1 : 0;
    st = gd_tensor_write(ctx, &scaler->scale_state, scale_values, sizeof(scale_values));
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_write(ctx, &scaler->flag_state, flag_values, sizeof(flag_values));
    if (st != GD_OK) {
        return st;
    }
    scaler->host_state_dirty = false;
    return GD_OK;
}

gd_status gd_amp_scaler_create(gd_context *ctx,
                                const gd_amp_config *config,
                                gd_amp_scaler **out)
{
    gd_amp_config cfg;
    gd_amp_scaler *scaler;
    const int64_t scale_shape[1] = {(int64_t)GD_AMP_SCALE_VALUES};
    const int64_t flag_shape[1] = {(int64_t)GD_AMP_FLAG_VALUES};
    gd_status st;
    if (out != NULL) {
        *out = NULL;
    }
    if (ctx == NULL || out == NULL) {
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
    st = gd_tensor_empty(ctx,
                         GD_ARENA_STATE,
                         GD_DTYPE_F32,
                         gd_shape_make(1U, scale_shape),
                         256U,
                         &scaler->scale_state);
    if (st != GD_OK) {
        free(scaler);
        return st;
    }
    st = gd_tensor_empty(ctx,
                         GD_ARENA_STATE,
                         GD_DTYPE_I32,
                         gd_shape_make(1U, flag_shape),
                         256U,
                         &scaler->flag_state);
    if (st != GD_OK) {
        free(scaler);
        return st;
    }
    scaler->scale_state.requires_grad = false;
    scaler->scale_state.is_leaf = false;
    scaler->flag_state.requires_grad = false;
    scaler->flag_state.is_leaf = false;
    st = gd_amp_scaler_write_device_state(ctx, scaler);
    if (st != GD_OK) {
        free(scaler);
        return st;
    }
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

gd_status gd_amp_scaler_sync(gd_context *ctx, gd_amp_scaler *scaler)
{
    float scale_values[GD_AMP_SCALE_VALUES];
    int32_t flag_values[GD_AMP_FLAG_VALUES];
    gd_status st;
    if (ctx == NULL || scaler == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!scaler->host_state_dirty) {
        return GD_OK;
    }
    st = gd_tensor_read_f32(ctx, &scaler->scale_state, scale_values, GD_AMP_SCALE_VALUES);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_read(ctx, &scaler->flag_state, flag_values, sizeof(flag_values));
    if (st != GD_OK) {
        return st;
    }
    scaler->scale = gd_amp_clamp_scale(scale_values[GD_AMP_SCALE_INDEX], &scaler->config);
    scaler->growth_tracker = flag_values[GD_AMP_GROWTH_TRACKER_INDEX] < 0 ?
                                 0U :
                                 (uint32_t)flag_values[GD_AMP_GROWTH_TRACKER_INDEX];
    scaler->last_found_inf = flag_values[GD_AMP_LAST_FOUND_INF_INDEX] != 0;
    scaler->host_state_dirty = false;
    return GD_OK;
}

gd_status gd_amp_scaler_get_state(gd_context *ctx,
                                  gd_amp_scaler *scaler,
                                  gd_amp_scaler_state *out)
{
    gd_status st;
    if (ctx == NULL || scaler == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_amp_scaler_sync(ctx, scaler);
    if (st != GD_OK) {
        return st;
    }
    out->config = scaler->config;
    out->scale = scaler->scale;
    out->growth_tracker = scaler->growth_tracker;
    out->last_found_inf = scaler->last_found_inf;
    return GD_OK;
}

gd_status gd_amp_scaler_set_state(gd_context *ctx,
                                  gd_amp_scaler *scaler,
                                  const gd_amp_scaler_state *state)
{
    if (ctx == NULL || scaler == NULL || state == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_amp_config_valid(&state->config) ||
        !gd_amp_float_finite_positive(state->scale)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    scaler->config = state->config;
    scaler->scale = gd_amp_clamp_scale(state->scale, &state->config);
    scaler->growth_tracker = state->growth_tracker;
    scaler->last_found_inf = state->last_found_inf;
    scaler->step_begun = false;
    return gd_amp_scaler_write_device_state(ctx, scaler);
}

static gd_status gd_amp_scaler_begin_step(gd_context *ctx, gd_amp_scaler *scaler)
{
    gd_backend_amp_state_desc desc;
    gd_status st;
    if (ctx == NULL || scaler == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (scaler->step_begun) {
        return GD_OK;
    }
    st = gd_amp_scaler_device_desc(scaler, &desc);
    if (st != GD_OK) {
        return st;
    }
    st = gd_backend_amp_begin_step(gd_context_backend(ctx), &desc);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend amp begin failed");
    }
    scaler->host_state_dirty = true;
    scaler->step_begun = true;
    return GD_OK;
}

static gd_status gd_amp_scaler_finish_step(gd_context *ctx, gd_amp_scaler *scaler)
{
    gd_backend_amp_state_desc desc;
    gd_status st;
    if (ctx == NULL || scaler == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_amp_scaler_device_desc(scaler, &desc);
    if (st != GD_OK) {
        return st;
    }
    st = gd_backend_amp_finish_step(gd_context_backend(ctx), &desc);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend amp finish failed");
    }
    scaler->host_state_dirty = true;
    scaler->step_begun = false;
    return GD_OK;
}

gd_status gd_amp_scaler_fill_scale(gd_context *ctx,
                                   gd_amp_scaler *scaler,
                                   gd_tensor *dst)
{
    gd_backend_buffer *dst_buffer;
    gd_backend_buffer *scale_buffer;
    int64_t numel = 0;
    gd_status st;
    if (ctx == NULL || scaler == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, dst);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(dst)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "amp scale fill requires contiguous tensor");
    }
    st = gd_tensor_numel(dst, &numel);
    if (st != GD_OK || numel <= 0 || (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, st == GD_OK ? GD_ERR_INVALID_ARGUMENT : st,
                                    "amp scale fill invalid shape");
    }
    st = gd_amp_scaler_begin_step(ctx, scaler);
    if (st != GD_OK) {
        return st;
    }
    dst_buffer = (gd_backend_buffer *)dst->storage.buffer;
    scale_buffer = (gd_backend_buffer *)scaler->scale_state.storage.buffer;
    st = gd_backend_amp_fill_scale(gd_context_backend(ctx),
                                   dst_buffer,
                                   gd_tensor_storage_offset(dst),
                                   (size_t)numel,
                                   (uint32_t)dst->dtype,
                                   scale_buffer,
                                   gd_tensor_storage_offset(&scaler->scale_state));
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend amp scale fill failed");
    }
    dst->version += 1U;
    if (dst->version == 0U) {
        dst->version = 1U;
    }
    return GD_OK;
}

gd_status gd_amp_scaler_scale_tensor(gd_context *ctx,
                                     gd_amp_scaler *scaler,
                                     const gd_tensor *src,
                                     gd_tensor *dst)
{
    gd_backend_buffer *dst_buffer;
    gd_backend_buffer *src_buffer;
    gd_backend_buffer *scale_buffer;
    int64_t numel = 0;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || scaler == NULL || src == NULL || dst == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, src);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, dst);
    if (st != GD_OK) {
        return st;
    }
    if (src->dtype != dst->dtype || src->rank != dst->rank || !gd_tensor_is_contiguous(src) ||
        !gd_tensor_is_contiguous(dst)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                    "amp tensor scale requires matching contiguous tensors");
    }
    for (i = 0U; i < src->rank; ++i) {
        if (src->shape[i] != dst->shape[i]) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT,
                                        "amp tensor scale shape mismatch");
        }
    }
    st = gd_tensor_numel(src, &numel);
    if (st != GD_OK || numel <= 0 || (uint64_t)numel > (uint64_t)SIZE_MAX) {
        return gd_context_set_error(ctx, st == GD_OK ? GD_ERR_INVALID_ARGUMENT : st,
                                    "amp tensor scale invalid shape");
    }
    st = gd_amp_scaler_begin_step(ctx, scaler);
    if (st != GD_OK) {
        return st;
    }
    dst_buffer = (gd_backend_buffer *)dst->storage.buffer;
    src_buffer = (gd_backend_buffer *)src->storage.buffer;
    scale_buffer = (gd_backend_buffer *)scaler->scale_state.storage.buffer;
    st = gd_backend_amp_scale(gd_context_backend(ctx),
                              dst_buffer,
                              gd_tensor_storage_offset(dst),
                              src_buffer,
                              gd_tensor_storage_offset(src),
                              (size_t)numel,
                              (uint32_t)dst->dtype,
                              scale_buffer,
                              gd_tensor_storage_offset(&scaler->scale_state));
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend amp tensor scale failed");
    }
    dst->version += 1U;
    if (dst->version == 0U) {
        dst->version = 1U;
    }
    return GD_OK;
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
        const int64_t unit_shape[1] = {1};
        const float one = 1.0f;
        size_t state_f32_count = (size_t)optimizer->param_count * (size_t)GD_OPTIM_STATE_F32_FIELDS;
        size_t state_i32_count = (size_t)optimizer->param_count + 1U;
        int64_t state_f32_shape[1];
        int64_t state_i32_shape[1];
        float *state_f32_values;
        uint32_t *state_i32_values;
        if (state_f32_count == 0U) {
            state_f32_count = 1U;
        }
        if (state_f32_count > (size_t)INT64_MAX || state_i32_count > (size_t)INT64_MAX) {
            gd_optimizer_destroy(optimizer);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "adamw scalar state shape overflow");
        }
        state_f32_shape[0] = (int64_t)state_f32_count;
        state_i32_shape[0] = (int64_t)state_i32_count;
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, unit_shape),
                             256U,
                             &optimizer->unit_scale);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        st = gd_tensor_write(ctx, &optimizer->unit_scale, &one, sizeof(one));
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_F32,
                             gd_shape_make(1U, state_f32_shape),
                             256U,
                             &optimizer->device_state_f32);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        st = gd_tensor_empty(ctx,
                             GD_ARENA_STATE,
                             GD_DTYPE_I32,
                             gd_shape_make(1U, state_i32_shape),
                             256U,
                             &optimizer->device_state_i32);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        state_f32_values = (float *)calloc(state_f32_count, sizeof(state_f32_values[0]));
        state_i32_values = (uint32_t *)calloc(state_i32_count, sizeof(state_i32_values[0]));
        if (state_f32_values == NULL || state_i32_values == NULL) {
            free(state_f32_values);
            free(state_i32_values);
            gd_optimizer_destroy(optimizer);
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                        "adamw scalar state initialization failed");
        }
        for (i = 0U; i < optimizer->param_count; ++i) {
            state_f32_values[(size_t)GD_OPTIM_BETA1_POWER_FIELD * optimizer->param_count + i] =
                optimizer->params[i].beta1_power;
            state_f32_values[(size_t)GD_OPTIM_BETA2_POWER_FIELD * optimizer->param_count + i] =
                optimizer->params[i].beta2_power;
        }
        st = gd_tensor_write(ctx,
                             &optimizer->device_state_f32,
                             state_f32_values,
                             state_f32_count * sizeof(state_f32_values[0]));
        if (st == GD_OK) {
            st = gd_tensor_write(ctx,
                                 &optimizer->device_state_i32,
                                 state_i32_values,
                                 state_i32_count * sizeof(state_i32_values[0]));
        }
        free(state_f32_values);
        free(state_i32_values);
        if (st != GD_OK) {
            gd_optimizer_destroy(optimizer);
            return st;
        }
        optimizer->unit_scale.requires_grad = false;
        optimizer->unit_scale.is_leaf = false;
        optimizer->device_state_f32.requires_grad = false;
        optimizer->device_state_f32.is_leaf = false;
        optimizer->device_state_i32.requires_grad = false;
        optimizer->device_state_i32.is_leaf = false;
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
                                                  const gd_optimizer_param *slot,
                                                  const gd_tensor *grad,
                                                  const gd_amp_scaler *scaler,
                                                  gd_backend_amp_unscale_desc *out_desc)
{
    gd_status st;
    if (ctx == NULL || slot == NULL || grad == NULL || scaler == NULL || out_desc == NULL) {
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
    out_desc->inv_scale_buffer = (gd_backend_buffer *)scaler->scale_state.storage.buffer;
    out_desc->inv_scale_offset = gd_tensor_storage_offset(&scaler->scale_state) +
                                 (size_t)GD_AMP_INV_SCALE_INDEX * sizeof(float);
    out_desc->found_inf_buffer = (gd_backend_buffer *)scaler->flag_state.storage.buffer;
    out_desc->found_inf_offset = gd_tensor_storage_offset(&scaler->flag_state) +
                                 (size_t)GD_AMP_FOUND_INF_INDEX * sizeof(int32_t);
    return GD_OK;
}

static gd_status gd_optimizer_build_grad_norm_desc(gd_context *ctx,
                                                   const gd_optimizer_param *slot,
                                                   const gd_tensor *grad,
                                                   const gd_tensor *grad_scale,
                                                   size_t grad_scale_extra_offset,
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
    if (ctx == NULL || slot == NULL || grad == NULL || grad_scale == NULL ||
        partial_buffer == NULL || out_desc == NULL || out_groups == NULL) {
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
    out_desc->grad_scale_buffer = (gd_backend_buffer *)grad_scale->storage.buffer;
    out_desc->grad_scale_offset = gd_tensor_storage_offset(grad_scale) + grad_scale_extra_offset;
    if (found_inf != NULL) {
        out_desc->found_inf_buffer = (gd_backend_buffer *)found_inf->storage.buffer;
        out_desc->found_inf_offset = gd_tensor_storage_offset(found_inf) +
                                     (size_t)GD_AMP_FOUND_INF_INDEX * sizeof(int32_t);
        out_desc->has_found_inf = 1U;
    }
    *out_groups = groups;
    return GD_OK;
}

static gd_status gd_optimizer_apply_grad_clip_scaled(gd_context *ctx,
                                                     gd_optimizer *optimizer,
                                                     float max_norm,
                                                     const gd_tensor *grad_scale,
                                                     size_t grad_scale_extra_offset,
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
    if (grad_scale == NULL) {
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
                                               grad_scale_extra_offset,
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
                                           optimizer,
                                           i,
                                           slot,
                                           &grad,
                                           NULL,
                                           0U,
                                           NULL,
                                           &optimizer->config,
                                           lr,
                                           false,
                                           false,
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
    st = gd_optimizer_apply_grad_clip_scaled(ctx,
                                             optimizer,
                                             max_grad_norm,
                                             &optimizer->unit_scale,
                                             0U,
                                             NULL);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           optimizer,
                                           i,
                                           slot,
                                           &optimizer->step_grads[i],
                                           &optimizer->grad_clip_scale,
                                           0U,
                                           NULL,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           false,
                                           false,
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
    st = gd_amp_scaler_begin_step(ctx, scaler);
    if (st != GD_OK) {
        return st;
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
                                             slot,
                                             &grad,
                                             scaler,
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
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           optimizer,
                                           i,
                                           slot,
                                           &optimizer->step_grads[i],
                                           NULL,
                                           0U,
                                           &scaler->flag_state,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           true,
                                           adamw_desc_count == 0U,
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
    if (adamw_desc_count != 0U) {
        st = gd_amp_scaler_finish_step(ctx, scaler);
        if (st != GD_OK) {
            return st;
        }
        optimizer->host_state_dirty = true;
    } else {
        scaler->step_begun = false;
    }
    return GD_OK;
}

gd_status gd_optimizer_step_amp_clip_lr(gd_context *ctx,
                                        gd_optimizer *optimizer,
                                        gd_amp_scaler *scaler,
                                        float lr,
                                        float max_grad_norm)
{
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
    st = gd_amp_scaler_begin_step(ctx, scaler);
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
    st = gd_optimizer_apply_grad_clip_scaled(ctx,
                                             optimizer,
                                             max_grad_norm,
                                             &scaler->scale_state,
                                             (size_t)GD_AMP_INV_SCALE_INDEX * sizeof(float),
                                             &scaler->flag_state);
    if (st != GD_OK) {
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        if (!optimizer->step_has_grad[i]) {
            continue;
        }
        st = gd_optimizer_build_adamw_desc(ctx,
                                           optimizer,
                                           i,
                                           slot,
                                           &optimizer->step_grads[i],
                                           &optimizer->grad_clip_scale,
                                           0U,
                                           &scaler->flag_state,
                                           &optimizer->config,
                                           lr,
                                           true,
                                           true,
                                           adamw_desc_count == 0U,
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
    if (adamw_desc_count != 0U) {
        st = gd_amp_scaler_finish_step(ctx, scaler);
        if (st != GD_OK) {
            return st;
        }
        optimizer->host_state_dirty = true;
    } else {
        scaler->step_begun = false;
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

gd_status gd_optimizer_sync_state(gd_context *ctx, gd_optimizer *optimizer)
{
    size_t state_f32_count;
    size_t state_i32_count;
    float *state_f32_values;
    uint32_t *state_i32_values;
    uint32_t i;
    gd_status st;
    if (ctx == NULL || optimizer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!optimizer->host_state_dirty) {
        return GD_OK;
    }
    state_f32_count = (size_t)optimizer->param_count * (size_t)GD_OPTIM_STATE_F32_FIELDS;
    if (state_f32_count == 0U) {
        state_f32_count = 1U;
    }
    state_i32_count = (size_t)optimizer->param_count + 1U;
    state_f32_values = (float *)calloc(state_f32_count, sizeof(state_f32_values[0]));
    state_i32_values = (uint32_t *)calloc(state_i32_count, sizeof(state_i32_values[0]));
    if (state_f32_values == NULL || state_i32_values == NULL) {
        free(state_f32_values);
        free(state_i32_values);
        return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY,
                                    "optimizer scalar state sync allocation failed");
    }
    st = gd_tensor_read_f32(ctx, &optimizer->device_state_f32, state_f32_values, state_f32_count);
    if (st == GD_OK) {
        st = gd_tensor_read(ctx,
                            &optimizer->device_state_i32,
                            state_i32_values,
                            state_i32_count * sizeof(state_i32_values[0]));
    }
    if (st != GD_OK) {
        free(state_f32_values);
        free(state_i32_values);
        return st;
    }
    for (i = 0U; i < optimizer->param_count; ++i) {
        gd_optimizer_param *slot = &optimizer->params[i];
        slot->beta1_power = state_f32_values[(size_t)GD_OPTIM_BETA1_POWER_FIELD * optimizer->param_count + i];
        slot->beta2_power = state_f32_values[(size_t)GD_OPTIM_BETA2_POWER_FIELD * optimizer->param_count + i];
        slot->step = (uint64_t)state_i32_values[i];
    }
    optimizer->step = (uint64_t)state_i32_values[optimizer->param_count];
    optimizer->host_state_dirty = false;
    free(state_f32_values);
    free(state_i32_values);
    return GD_OK;
}

uint64_t gd_optimizer_step_count(const gd_optimizer *optimizer)
{
    return optimizer != NULL ? optimizer->step : 0U;
}

gd_status gd_optimizer_save_state(gd_context *ctx,
                                  const gd_optimizer *optimizer,
                                  const char *path)
{
    gd_module state_module;
    gd_module_save_options options;
    char *metadata = NULL;
    size_t metadata_len = 0U;
    gd_status st;
    if (ctx == NULL || optimizer == NULL || path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_optimizer_sync_state(ctx, (gd_optimizer *)optimizer);
    if (st != GD_OK) {
        return st;
    }
    memset(&state_module, 0, sizeof(state_module));
    st = gd_optimizer_build_state_metadata(optimizer, &metadata, &metadata_len);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "optimizer checkpoint metadata allocation failed");
    }
    st = gd_optimizer_build_state_module(ctx, (gd_optimizer *)optimizer, &state_module);
    if (st == GD_OK) {
        options = gd_module_save_options_default();
        options.metadata = metadata;
        options.metadata_len = metadata_len;
        options.include_buffers = true;
        st = gd_module_save_state(ctx, &state_module, path, &options);
    }
    gd_module_deinit(&state_module);
    free(metadata);
    return st == GD_OK ? GD_OK : gd_context_set_error(ctx, st, "optimizer checkpoint save failed");
}

gd_status gd_optimizer_load_state(gd_context *ctx,
                                  gd_optimizer *optimizer,
                                  const char *path,
                                  bool strict)
{
    gd_module state_module;
    gd_module_load_options options;
    gd_optimizer_saved_state saved;
    char *metadata = NULL;
    size_t metadata_len = 0U;
    gd_status st;
    if (ctx == NULL || optimizer == NULL || path == NULL || path[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&state_module, 0, sizeof(state_module));
    memset(&saved, 0, sizeof(saved));
    st = gd_checkpoint_read_metadata(path, &metadata, &metadata_len);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "optimizer checkpoint metadata read failed");
    }
    (void)metadata_len;
    st = gd_optimizer_parse_state_metadata(ctx, optimizer, metadata, &saved);
    free(metadata);
    metadata = NULL;
    if (st != GD_OK) {
        gd_optimizer_saved_state_free(&saved);
        return st;
    }
    st = gd_optimizer_build_state_module(ctx, optimizer, &state_module);
    if (st == GD_OK) {
        options = gd_module_load_options_default();
        options.strict = strict;
        options.load_buffers = true;
        st = gd_module_load_state(ctx, &state_module, path, &options);
    }
    gd_module_deinit(&state_module);
    if (st == GD_OK) {
        gd_optimizer_apply_saved_state(optimizer, &saved);
        st = gd_optimizer_write_device_scalar_state(ctx, optimizer);
    }
    gd_optimizer_saved_state_free(&saved);
    return st == GD_OK ? GD_OK : gd_context_set_error(ctx, st, "optimizer checkpoint load failed");
}
