#include <gradients/module.h>
#include <gradients/ops.h>

#include "../core/memory_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static gd_status gd_module_set_error(gd_context *ctx,
                                     gd_status status,
                                     const char *message)
{
    return ctx != NULL ? gd_context_set_error(ctx, status, message) : status;
}

static gd_status gd_module_copy_name(char dst[GD_MODULE_NAME_MAX], const char *name)
{
    size_t i;
    if (dst == NULL || name == NULL || name[0] == '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i + 1U < GD_MODULE_NAME_MAX && name[i] != '\0'; ++i) {
        dst[i] = name[i];
    }
    if (name[i] != '\0') {
        return GD_ERR_INVALID_ARGUMENT;
    }
    dst[i] = '\0';
    return GD_OK;
}

static bool gd_module_names_equal(const char *a, const char *b)
{
    return strncmp(a, b, GD_MODULE_NAME_MAX) == 0;
}

static bool gd_module_has_entry_name(const gd_module *module, const char *name)
{
    uint32_t i;
    if (module == NULL || name == NULL) {
        return false;
    }
    for (i = 0U; i < module->param_count; ++i) {
        if (gd_module_names_equal(module->params[i].name, name)) {
            return true;
        }
    }
    for (i = 0U; i < module->buffer_count; ++i) {
        if (gd_module_names_equal(module->buffers[i].name, name)) {
            return true;
        }
    }
    for (i = 0U; i < module->child_count; ++i) {
        if (gd_module_names_equal(module->children[i].name, name)) {
            return true;
        }
    }
    return false;
}

static gd_status gd_module_grow_array(void **items,
                                      uint32_t *capacity,
                                      uint32_t count,
                                      size_t elem_size)
{
    void *grown;
    uint32_t new_capacity;
    if (items == NULL || capacity == NULL || elem_size == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (count < *capacity) {
        return GD_OK;
    }
    new_capacity = *capacity == 0U ? 4U : *capacity * 2U;
    if (new_capacity <= *capacity || (size_t)new_capacity > SIZE_MAX / elem_size) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    grown = realloc(*items, (size_t)new_capacity * elem_size);
    if (grown == NULL) {
        return GD_ERR_OUT_OF_MEMORY;
    }
    memset((unsigned char *)grown + (size_t)(*capacity) * elem_size,
           0,
           (size_t)(new_capacity - *capacity) * elem_size);
    *items = grown;
    *capacity = new_capacity;
    return GD_OK;
}

static gd_status gd_module_reserve_param(gd_module *module)
{
    return gd_module_grow_array((void **)&module->params,
                                &module->param_capacity,
                                module->param_count,
                                sizeof(module->params[0]));
}

static gd_status gd_module_reserve_buffer(gd_module *module)
{
    return gd_module_grow_array((void **)&module->buffers,
                                &module->buffer_capacity,
                                module->buffer_count,
                                sizeof(module->buffers[0]));
}

static gd_status gd_module_reserve_child(gd_module *module)
{
    return gd_module_grow_array((void **)&module->children,
                                &module->child_capacity,
                                module->child_count,
                                sizeof(module->children[0]));
}

static bool gd_module_wildcard_match(const char *pattern, const char *text)
{
    const char *star = NULL;
    const char *retry = NULL;
    if (pattern == NULL || text == NULL) {
        return false;
    }
    while (*text != '\0') {
        if (*pattern == '*') {
            star = pattern++;
            retry = text;
        } else if (*pattern == *text) {
            pattern++;
            text++;
        } else if (star != NULL) {
            pattern = star + 1;
            retry++;
            text = retry;
        } else {
            return false;
        }
    }
    while (*pattern == '*') {
        pattern++;
    }
    return *pattern == '\0';
}

static uint64_t gd_module_hash_string(const char *text)
{
    uint64_t hash = 1469598103934665603ULL;
    if (text == NULL) {
        return hash;
    }
    while (*text != '\0') {
        hash ^= (uint64_t)(unsigned char)*text;
        hash *= 1099511628211ULL;
        text++;
    }
    return hash;
}

static uint64_t gd_module_mix_seed(uint64_t seed, const char *path)
{
    uint64_t hash = gd_module_hash_string(path);
    return seed ^ (hash + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

static gd_status gd_module_join_path(const char *prefix,
                                     const char *name,
                                     char out[GD_MODULE_PATH_MAX])
{
    int n;
    if (name == NULL || name[0] == '\0' || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (prefix == NULL || prefix[0] == '\0') {
        n = snprintf(out, GD_MODULE_PATH_MAX, "%s", name);
    } else {
        n = snprintf(out, GD_MODULE_PATH_MAX, "%s.%s", prefix, name);
    }
    if (n < 0 || (size_t)n >= GD_MODULE_PATH_MAX) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static bool gd_tensor_same_param_identity(const gd_tensor *a, const gd_tensor *b)
{
    uint32_t i;
    if (a == NULL || b == NULL) {
        return false;
    }
    if (a == b) {
        return true;
    }
    if (a->dtype != b->dtype || a->rank != b->rank || a->storage.buffer != b->storage.buffer ||
        a->storage.offset != b->storage.offset || a->storage.nbytes != b->storage.nbytes ||
        a->view_offset != b->view_offset) {
        return false;
    }
    for (i = 0U; i < a->rank; ++i) {
        if (a->shape[i] != b->shape[i] || a->strides[i] != b->strides[i]) {
            return false;
        }
    }
    return true;
}

static gd_status gd_param_set_append(gd_param_set *set,
                                     const char path[GD_MODULE_PATH_MAX],
                                     gd_tensor *tensor,
                                     int32_t group_index,
                                     float lr_mult,
                                     float weight_decay,
                                     bool trainable)
{
    gd_status st;
    gd_param_ref *item;
    if (set == NULL || path == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_grow_array((void **)&set->items,
                              &set->capacity,
                              set->count,
                              sizeof(set->items[0]));
    if (st != GD_OK) {
        return st;
    }
    item = &set->items[set->count];
    memset(item, 0, sizeof(*item));
    (void)snprintf(item->path, sizeof(item->path), "%s", path);
    item->tensor = tensor;
    item->group_index = group_index;
    item->lr_mult = lr_mult;
    item->weight_decay = weight_decay;
    item->trainable = trainable;
    set->count += 1U;
    return GD_OK;
}

static bool gd_param_set_contains_tensor(const gd_param_set *set, const gd_tensor *tensor)
{
    uint32_t i;
    if (set == NULL || tensor == NULL) {
        return false;
    }
    for (i = 0U; i < set->count; ++i) {
        if (gd_tensor_same_param_identity(set->items[i].tensor, tensor)) {
            return true;
        }
    }
    return false;
}

static int32_t gd_module_select_group(const char *path,
                                      const gd_param_group *groups,
                                      uint32_t group_count)
{
    uint32_t i;
    if (path == NULL || groups == NULL) {
        return -1;
    }
    for (i = 0U; i < group_count; ++i) {
        if (groups[i].match != NULL && gd_module_wildcard_match(groups[i].match, path)) {
            return (int32_t)i;
        }
    }
    return -1;
}

static gd_status gd_module_collect_params_impl(const gd_module *module,
                                               const char *prefix,
                                               const gd_param_group *groups,
                                               uint32_t group_count,
                                               gd_param_set *out)
{
    uint32_t i;
    gd_status st;
    char child_path[GD_MODULE_PATH_MAX];
    if (module == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < module->param_count; ++i) {
        char param_path[GD_MODULE_PATH_MAX];
        int32_t group_index;
        float lr_mult = 1.0f;
        float weight_decay = 0.0f;
        bool trainable;
        gd_module_param_entry *entry = &module->params[i];
        if (entry->tensor == NULL || gd_param_set_contains_tensor(out, entry->tensor)) {
            continue;
        }
        st = gd_module_join_path(prefix, entry->name, param_path);
        if (st != GD_OK) {
            return st;
        }
        group_index = gd_module_select_group(param_path, groups, group_count);
        trainable = entry->trainable && entry->tensor->requires_grad;
        if (group_index >= 0) {
            lr_mult = groups[(uint32_t)group_index].lr_mult;
            weight_decay = groups[(uint32_t)group_index].weight_decay;
            if (!groups[(uint32_t)group_index].trainable) {
                trainable = false;
            }
        }
        st = gd_param_set_append(out,
                                 param_path,
                                 entry->tensor,
                                 group_index,
                                 lr_mult,
                                 weight_decay,
                                 trainable);
        if (st != GD_OK) {
            return st;
        }
    }
    for (i = 0U; i < module->child_count; ++i) {
        const gd_module_child_entry *child = &module->children[i];
        if (child->module == NULL) {
            continue;
        }
        st = gd_module_join_path(prefix, child->name, child_path);
        if (st != GD_OK) {
            return st;
        }
        st = gd_module_collect_params_impl(child->module,
                                           child_path,
                                           groups,
                                           group_count,
                                           out);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

static gd_status gd_module_apply_trainable(gd_module *module,
                                           const char *prefix,
                                           const char *pattern,
                                           bool trainable)
{
    uint32_t i;
    gd_status st;
    if (module == NULL || pattern == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < module->param_count; ++i) {
        char param_path[GD_MODULE_PATH_MAX];
        gd_module_param_entry *entry = &module->params[i];
        st = gd_module_join_path(prefix, entry->name, param_path);
        if (st != GD_OK) {
            return st;
        }
        if (gd_module_wildcard_match(pattern, param_path)) {
            entry->trainable = trainable;
            if (entry->tensor != NULL) {
                entry->tensor->requires_grad = trainable;
            }
        }
    }
    for (i = 0U; i < module->child_count; ++i) {
        char child_path[GD_MODULE_PATH_MAX];
        gd_module_child_entry *child = &module->children[i];
        if (child->module == NULL) {
            continue;
        }
        st = gd_module_join_path(prefix, child->name, child_path);
        if (st != GD_OK) {
            return st;
        }
        st = gd_module_apply_trainable(child->module, child_path, pattern, trainable);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

static gd_status gd_module_init_tensor(gd_context *ctx,
                                       gd_arena_kind arena,
                                       const gd_tensor_spec *spec,
                                       const gd_init_spec *init,
                                       gd_tensor *out)
{
    gd_init_spec actual_init;
    if (ctx == NULL || spec == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    actual_init = init != NULL ? *init : gd_init_empty();
    switch (actual_init.kind) {
    case GD_INIT_EMPTY:
        return gd_tensor_empty(ctx,
                               arena,
                               spec->dtype,
                               spec->rank,
                               spec->shape,
                               spec->alignment,
                               out);
    case GD_INIT_ZERO:
        return gd_tensor_zeros(ctx,
                               arena,
                               spec->dtype,
                               spec->rank,
                               spec->shape,
                               spec->alignment,
                               out);
    case GD_INIT_ONE:
        return gd_tensor_ones(ctx,
                              arena,
                              spec->dtype,
                              spec->rank,
                              spec->shape,
                              spec->alignment,
                              out);
    case GD_INIT_RAND_UNIFORM:
        return gd_tensor_rand_uniform(ctx,
                                      arena,
                                      spec->dtype,
                                      spec->rank,
                                      spec->shape,
                                      spec->alignment,
                                      actual_init.seed,
                                      actual_init.low,
                                      actual_init.high,
                                      out);
    default:
        return gd_module_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid tensor init kind");
    }
}

gd_tensor_spec gd_tensor_spec_make(gd_dtype dtype,
                                   uint32_t rank,
                                   const int64_t *shape,
                                   size_t alignment)
{
    gd_tensor_spec spec;
    uint32_t i;
    memset(&spec, 0, sizeof(spec));
    spec.dtype = dtype;
    spec.rank = rank;
    spec.alignment = alignment;
    if (shape != NULL && rank <= GD_MAX_DIMS) {
        for (i = 0U; i < rank; ++i) {
            spec.shape[i] = shape[i];
        }
    }
    return spec;
}

gd_linear_layer_config gd_linear_layer_config_make(int64_t in_features,
                                                   int64_t out_features,
                                                   gd_dtype dtype,
                                                   uint64_t seed)
{
    gd_linear_layer_config config;
    memset(&config, 0, sizeof(config));
    config.in_features = in_features;
    config.out_features = out_features;
    config.dtype = dtype;
    config.use_bias = true;
    config.alignment = 256U;
    config.seed = seed;
    config.weight_low = -0.02f;
    config.weight_high = 0.02f;
    return config;
}

gd_linear_layer_config gd_linear_layer_config_build(int64_t in_features,
                                                    int64_t out_features,
                                                    gd_dtype dtype)
{
    return gd_linear_layer_config_make(in_features, out_features, dtype, 0U);
}

gd_status gd_module_init(gd_context *ctx, gd_module *module, const char *name)
{
    gd_status st;
    if (module == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(module, 0, sizeof(*module));
    st = gd_module_copy_name(module->name, name);
    if (st != GD_OK) {
        memset(module, 0, sizeof(*module));
        return gd_module_set_error(ctx, st, "invalid module name");
    }
    module->training = true;
    return GD_OK;
}

gd_status gd_module_init_child(gd_context *ctx,
                               gd_module *parent,
                               const char *name,
                               gd_module *child)
{
    gd_status st;
    if (parent == NULL || child == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_init(ctx, child, name);
    if (st != GD_OK) {
        return st;
    }
    st = gd_module_add_child(parent, name, child);
    if (st != GD_OK) {
        gd_module_deinit(child);
        return gd_module_set_error(ctx, st, "failed to add child module");
    }
    return GD_OK;
}

void gd_module_deinit(gd_module *module)
{
    uint32_t i;
    if (module == NULL) {
        return;
    }
    for (i = 0U; i < module->child_count; ++i) {
        if (module->children[i].module != NULL &&
            module->children[i].module->parent == module) {
            module->children[i].module->parent = NULL;
        }
    }
    free(module->params);
    free(module->buffers);
    free(module->children);
    memset(module, 0, sizeof(*module));
}

gd_status gd_module_add_param(gd_module *module, const char *name, gd_tensor *tensor)
{
    gd_status st;
    char copied[GD_MODULE_NAME_MAX];
    gd_module_param_entry *entry;
    if (module == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_copy_name(copied, name);
    if (st != GD_OK) {
        return st;
    }
    if (gd_module_has_entry_name(module, copied)) {
        return GD_ERR_BAD_STATE;
    }
    st = gd_module_reserve_param(module);
    if (st != GD_OK) {
        return st;
    }
    entry = &module->params[module->param_count];
    memset(entry, 0, sizeof(*entry));
    (void)snprintf(entry->name, sizeof(entry->name), "%s", copied);
    entry->tensor = tensor;
    entry->trainable = true;
    tensor->requires_grad = true;
    module->param_count += 1U;
    return GD_OK;
}

gd_status gd_module_add_buffer(gd_module *module, const char *name, gd_tensor *tensor)
{
    gd_status st;
    char copied[GD_MODULE_NAME_MAX];
    gd_module_buffer_entry *entry;
    if (module == NULL || tensor == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_copy_name(copied, name);
    if (st != GD_OK) {
        return st;
    }
    if (gd_module_has_entry_name(module, copied)) {
        return GD_ERR_BAD_STATE;
    }
    st = gd_module_reserve_buffer(module);
    if (st != GD_OK) {
        return st;
    }
    entry = &module->buffers[module->buffer_count];
    memset(entry, 0, sizeof(*entry));
    (void)snprintf(entry->name, sizeof(entry->name), "%s", copied);
    entry->tensor = tensor;
    tensor->requires_grad = false;
    module->buffer_count += 1U;
    return GD_OK;
}

gd_status gd_module_add_child(gd_module *parent, const char *name, gd_module *child)
{
    gd_status st;
    char copied[GD_MODULE_NAME_MAX];
    gd_module_child_entry *entry;
    if (parent == NULL || child == NULL || parent == child) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_copy_name(copied, name);
    if (st != GD_OK) {
        return st;
    }
    if (gd_module_has_entry_name(parent, copied) || child->parent != NULL) {
        return GD_ERR_BAD_STATE;
    }
    st = gd_module_reserve_child(parent);
    if (st != GD_OK) {
        return st;
    }
    entry = &parent->children[parent->child_count];
    memset(entry, 0, sizeof(*entry));
    (void)snprintf(entry->name, sizeof(entry->name), "%s", copied);
    entry->module = child;
    child->parent = parent;
    child->training = parent->training;
    parent->child_count += 1U;
    return GD_OK;
}

gd_status gd_module_param(gd_context *ctx,
                          gd_module *module,
                          const char *name,
                          const gd_tensor_spec *spec,
                          const gd_init_spec *init,
                          gd_tensor *out)
{
    gd_status st;
    if (ctx == NULL || module == NULL || spec == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_init_tensor(ctx, GD_ARENA_PARAMS, spec, init, out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_module_add_param(module, name, out);
    if (st != GD_OK) {
        return gd_module_set_error(ctx, st, "failed to register module parameter");
    }
    return GD_OK;
}

gd_status gd_module_buffer(gd_context *ctx,
                           gd_module *module,
                           const char *name,
                           const gd_tensor_spec *spec,
                           const gd_init_spec *init,
                           gd_tensor *out)
{
    gd_status st;
    if (ctx == NULL || module == NULL || spec == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_init_tensor(ctx, GD_ARENA_PARAMS, spec, init, out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_module_add_buffer(module, name, out);
    if (st != GD_OK) {
        return gd_module_set_error(ctx, st, "failed to register module buffer");
    }
    return GD_OK;
}

void gd_module_set_training(gd_module *module, bool training)
{
    uint32_t i;
    if (module == NULL) {
        return;
    }
    module->training = training;
    for (i = 0U; i < module->child_count; ++i) {
        gd_module_set_training(module->children[i].module, training);
    }
}

gd_status gd_module_freeze(gd_module *module, const char *pattern)
{
    if (module == NULL || pattern == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_module_apply_trainable(module, module->name, pattern, false);
}

gd_status gd_module_unfreeze(gd_module *module, const char *pattern)
{
    if (module == NULL || pattern == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_module_apply_trainable(module, module->name, pattern, true);
}

static gd_status gd_module_init_param_tensor(gd_context *ctx,
                                             gd_tensor *tensor,
                                             const char *path,
                                             const gd_init_spec *init)
{
    if (ctx == NULL || tensor == NULL || init == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    switch (init->kind) {
    case GD_INIT_EMPTY:
        return GD_OK;
    case GD_INIT_ZERO:
        return gd_tensor_zero_(ctx, tensor);
    case GD_INIT_ONE:
        return gd_tensor_one_(ctx, tensor);
    case GD_INIT_RAND_UNIFORM:
        return gd_tensor_rand_uniform_(ctx,
                                       tensor,
                                       gd_module_mix_seed(init->seed, path),
                                       init->low,
                                       init->high);
    default:
        return gd_module_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid module param init kind");
    }
}

static gd_status gd_module_init_params_impl(gd_context *ctx,
                                            gd_module *module,
                                            const char *prefix,
                                            const char *pattern,
                                            const gd_init_spec *init)
{
    uint32_t i;
    gd_status st;
    if (ctx == NULL || module == NULL || init == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    for (i = 0U; i < module->param_count; ++i) {
        char param_path[GD_MODULE_PATH_MAX];
        gd_module_param_entry *entry = &module->params[i];
        if (entry->tensor == NULL) {
            continue;
        }
        st = gd_module_join_path(prefix, entry->name, param_path);
        if (st != GD_OK) {
            return gd_module_set_error(ctx, st, "module param init path too long");
        }
        if (pattern == NULL || gd_module_wildcard_match(pattern, param_path)) {
            st = gd_module_init_param_tensor(ctx, entry->tensor, param_path, init);
            if (st != GD_OK) {
                return st;
            }
        }
    }
    for (i = 0U; i < module->child_count; ++i) {
        char child_path[GD_MODULE_PATH_MAX];
        gd_module_child_entry *child = &module->children[i];
        if (child->module == NULL) {
            continue;
        }
        st = gd_module_join_path(prefix, child->name, child_path);
        if (st != GD_OK) {
            return gd_module_set_error(ctx, st, "module child init path too long");
        }
        st = gd_module_init_params_impl(ctx, child->module, child_path, pattern, init);
        if (st != GD_OK) {
            return st;
        }
    }
    return GD_OK;
}

gd_status gd_module_init_params_uniform(gd_context *ctx,
                                        gd_module *module,
                                        const char *pattern,
                                        float low,
                                        float high,
                                        uint64_t seed)
{
    gd_init_spec init;
    if (ctx == NULL || module == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!(low <= high)) {
        return gd_module_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid module uniform init range");
    }
    init = gd_init_rand_uniform(seed, low, high);
    return gd_module_init_params_impl(ctx, module, module->name, pattern, &init);
}

gd_status gd_module_init_params_zero(gd_context *ctx,
                                     gd_module *module,
                                     const char *pattern)
{
    gd_init_spec init;
    if (ctx == NULL || module == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    init = gd_init_zero();
    return gd_module_init_params_impl(ctx, module, module->name, pattern, &init);
}

gd_param_group gd_param_group_build(const char *name,
                                    const char *match,
                                    float lr_mult,
                                    float weight_decay,
                                    bool trainable)
{
    gd_param_group group;
    group.name = name;
    group.match = match;
    group.lr_mult = lr_mult;
    group.weight_decay = weight_decay;
    group.trainable = trainable;
    return group;
}

void gd_param_set_init(gd_param_set *set)
{
    if (set != NULL) {
        memset(set, 0, sizeof(*set));
    }
}

void gd_param_set_free(gd_param_set *set)
{
    if (set == NULL) {
        return;
    }
    free(set->items);
    memset(set, 0, sizeof(*set));
}

gd_status gd_module_parameters(gd_context *ctx,
                               const gd_module *module,
                               gd_param_set *out)
{
    return gd_module_collect_params(ctx, module, NULL, 0U, out);
}

gd_status gd_module_collect_params(gd_context *ctx,
                                   const gd_module *module,
                                   const gd_param_group *groups,
                                   uint32_t group_count,
                                   gd_param_set *out)
{
    gd_status st;
    if (module == NULL || out == NULL || (group_count > 0U && groups == NULL)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    gd_param_set_init(out);
    st = gd_module_collect_params_impl(module, module->name, groups, group_count, out);
    if (st != GD_OK) {
        gd_param_set_free(out);
        return gd_module_set_error(ctx, st, "failed to collect module parameters");
    }
    return GD_OK;
}

gd_status gd_module_list_init_child(gd_context *ctx,
                                    gd_module *parent,
                                    const char *name,
                                    gd_module_list *list,
                                    uint32_t count)
{
    gd_status st;
    if (parent == NULL || list == NULL || count == 0U) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(list, 0, sizeof(*list));
    list->items = (gd_module **)calloc(count, sizeof(list->items[0]));
    if (list->items == NULL) {
        return gd_module_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "module list allocation failed");
    }
    list->count = count;
    st = gd_module_init_child(ctx, parent, name, &list->mod);
    if (st != GD_OK) {
        free(list->items);
        memset(list, 0, sizeof(*list));
        return st;
    }
    return GD_OK;
}

gd_status gd_module_list_set(gd_module_list *list, uint32_t index, gd_module *child)
{
    gd_status st;
    char name[GD_MODULE_NAME_MAX];
    int n;
    if (list == NULL || child == NULL || index >= list->count) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (list->items[index] != NULL && list->items[index] != child) {
        return GD_ERR_BAD_STATE;
    }
    if (list->items[index] == child) {
        return GD_OK;
    }
    n = snprintf(name, sizeof(name), "%u", index);
    if (n < 0 || (size_t)n >= sizeof(name)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_module_add_child(&list->mod, name, child);
    if (st != GD_OK) {
        return st;
    }
    list->items[index] = child;
    return GD_OK;
}

void gd_module_list_deinit(gd_module_list *list)
{
    if (list == NULL) {
        return;
    }
    gd_module_deinit(&list->mod);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

gd_status gd_module_dict_init_child(gd_context *ctx,
                                    gd_module *parent,
                                    const char *name,
                                    gd_module_dict *dict)
{
    if (parent == NULL || dict == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(dict, 0, sizeof(*dict));
    return gd_module_init_child(ctx, parent, name, &dict->mod);
}

gd_status gd_module_dict_set(gd_module_dict *dict, const char *name, gd_module *child)
{
    if (dict == NULL || child == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_module_add_child(&dict->mod, name, child);
}

void gd_module_dict_deinit(gd_module_dict *dict)
{
    if (dict == NULL) {
        return;
    }
    gd_module_deinit(&dict->mod);
    memset(dict, 0, sizeof(*dict));
}

gd_status gd_linear_layer_init(gd_context *ctx,
                               gd_linear_layer *layer,
                               const char *name,
                               const gd_linear_layer_config *config)
{
    gd_status st;
    gd_tensor_spec weight_spec;
    gd_tensor_spec bias_spec;
    gd_init_spec weight_init;
    gd_init_spec bias_init;
    int64_t weight_shape[2];
    int64_t bias_shape[1];
    if (ctx == NULL || layer == NULL || config == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (config->in_features <= 0 || config->out_features <= 0 ||
        (config->dtype != GD_DTYPE_F16 && config->dtype != GD_DTYPE_BF16 &&
         config->dtype != GD_DTYPE_F32)) {
        return gd_module_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "invalid linear layer config");
    }
    memset(layer, 0, sizeof(*layer));
    st = gd_module_init(ctx, &layer->mod, name);
    if (st != GD_OK) {
        return st;
    }
    layer->in_features = config->in_features;
    layer->out_features = config->out_features;
    layer->has_bias = config->use_bias;
    weight_shape[0] = config->in_features;
    weight_shape[1] = config->out_features;
    weight_spec = gd_tensor_spec_make(config->dtype, 2U, weight_shape, config->alignment);
    weight_init = gd_init_rand_uniform(config->seed, config->weight_low, config->weight_high);
    st = gd_module_param(ctx, &layer->mod, "weight", &weight_spec, &weight_init, &layer->weight);
    if (st != GD_OK) {
        gd_module_deinit(&layer->mod);
        return st;
    }
    if (config->use_bias) {
        bias_shape[0] = config->out_features;
        bias_spec = gd_tensor_spec_make(config->dtype, 1U, bias_shape, config->alignment);
        bias_init = gd_init_zero();
        st = gd_module_param(ctx, &layer->mod, "bias", &bias_spec, &bias_init, &layer->bias);
        if (st != GD_OK) {
            gd_module_deinit(&layer->mod);
            return st;
        }
    }
    return GD_OK;
}

gd_status gd_linear_layer_init_child(gd_context *ctx,
                                     gd_module *parent,
                                     const char *name,
                                     gd_linear_layer *layer,
                                     const gd_linear_layer_config *config)
{
    gd_status st;
    if (parent == NULL || layer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_linear_layer_init(ctx, layer, name, config);
    if (st != GD_OK) {
        return st;
    }
    st = gd_module_add_child(parent, name, &layer->mod);
    if (st != GD_OK) {
        gd_linear_layer_deinit(layer);
        return gd_module_set_error(ctx, st, "failed to add linear layer child");
    }
    return GD_OK;
}

void gd_linear_layer_deinit(gd_linear_layer *layer)
{
    if (layer == NULL) {
        return;
    }
    gd_module_deinit(&layer->mod);
    memset(layer, 0, sizeof(*layer));
}

gd_status gd_linear_layer_forward(gd_context *ctx,
                                  gd_linear_layer *layer,
                                  const gd_tensor *x,
                                  gd_tensor *out)
{
    if (layer == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return gd_linear(ctx, x, &layer->weight, layer->has_bias ? &layer->bias : NULL, out);
}
