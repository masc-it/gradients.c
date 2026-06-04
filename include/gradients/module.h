#ifndef GRADIENTS_MODULE_H
#define GRADIENTS_MODULE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <gradients/status.h>
#include <gradients/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GD_MODULE_NAME_MAX 64U
#define GD_MODULE_PATH_MAX 256U

typedef enum gd_module_init_kind {
    GD_INIT_EMPTY = 0,
    GD_INIT_ZERO = 1,
    GD_INIT_ONE = 2,
    GD_INIT_RAND_UNIFORM = 3,
} gd_module_init_kind;

typedef struct gd_init_spec {
    gd_module_init_kind kind;
    uint64_t seed;
    float low;
    float high;
} gd_init_spec;

typedef struct gd_tensor_spec {
    gd_dtype dtype;
    uint32_t rank;
    int64_t shape[GD_MAX_DIMS];
    size_t alignment;
} gd_tensor_spec;

typedef struct gd_param_group {
    const char *name;
    const char *match;
    float lr_mult;
    float weight_decay;
    bool trainable;
} gd_param_group;

typedef struct gd_module_param_entry {
    char name[GD_MODULE_NAME_MAX];
    gd_tensor *tensor;
    bool trainable;
} gd_module_param_entry;

typedef struct gd_module_buffer_entry {
    char name[GD_MODULE_NAME_MAX];
    gd_tensor *tensor;
} gd_module_buffer_entry;

typedef struct gd_module_child_entry {
    char name[GD_MODULE_NAME_MAX];
    struct gd_module *module;
} gd_module_child_entry;

typedef struct gd_module {
    char name[GD_MODULE_NAME_MAX];
    struct gd_module *parent;
    bool training;
    gd_module_param_entry *params;
    uint32_t param_count;
    uint32_t param_capacity;
    gd_module_buffer_entry *buffers;
    uint32_t buffer_count;
    uint32_t buffer_capacity;
    gd_module_child_entry *children;
    uint32_t child_count;
    uint32_t child_capacity;
} gd_module;

typedef struct gd_param_ref {
    char path[GD_MODULE_PATH_MAX];
    gd_tensor *tensor;
    int32_t group_index;
    float lr_mult;
    float weight_decay;
    bool trainable;
} gd_param_ref;

typedef struct gd_param_set {
    gd_param_ref *items;
    uint32_t count;
    uint32_t capacity;
} gd_param_set;

typedef struct gd_module_list {
    gd_module mod;
    gd_module **items;
    uint32_t count;
} gd_module_list;

typedef struct gd_module_dict {
    gd_module mod;
} gd_module_dict;

typedef struct gd_linear_layer_config {
    int64_t in_features;
    int64_t out_features;
    gd_dtype dtype;
    bool use_bias;
    size_t alignment;
    uint64_t seed;
    float weight_low;
    float weight_high;
} gd_linear_layer_config;

typedef struct gd_linear_layer {
    gd_module mod;
    gd_tensor weight;
    gd_tensor bias;
    bool has_bias;
    int64_t in_features;
    int64_t out_features;
} gd_linear_layer;

static inline gd_init_spec gd_init_empty(void)
{
    gd_init_spec init;
    init.kind = GD_INIT_EMPTY;
    init.seed = 0U;
    init.low = 0.0f;
    init.high = 0.0f;
    return init;
}

static inline gd_init_spec gd_init_zero(void)
{
    gd_init_spec init = gd_init_empty();
    init.kind = GD_INIT_ZERO;
    return init;
}

static inline gd_init_spec gd_init_one(void)
{
    gd_init_spec init = gd_init_empty();
    init.kind = GD_INIT_ONE;
    return init;
}

static inline gd_init_spec gd_init_rand_uniform(uint64_t seed, float low, float high)
{
    gd_init_spec init;
    init.kind = GD_INIT_RAND_UNIFORM;
    init.seed = seed;
    init.low = low;
    init.high = high;
    return init;
}

gd_tensor_spec gd_tensor_spec_make(gd_dtype dtype,
                                   uint32_t rank,
                                   const int64_t *shape,
                                   size_t alignment);

gd_linear_layer_config gd_linear_layer_config_make(int64_t in_features,
                                                   int64_t out_features,
                                                   gd_dtype dtype,
                                                   uint64_t seed);

gd_status gd_module_init(gd_context *ctx, gd_module *module, const char *name);
gd_status gd_module_init_child(gd_context *ctx,
                               gd_module *parent,
                               const char *name,
                               gd_module *child);
void gd_module_deinit(gd_module *module);

gd_status gd_module_add_param(gd_module *module, const char *name, gd_tensor *tensor);
gd_status gd_module_add_buffer(gd_module *module, const char *name, gd_tensor *tensor);
gd_status gd_module_add_child(gd_module *parent, const char *name, gd_module *child);

gd_status gd_module_param(gd_context *ctx,
                          gd_module *module,
                          const char *name,
                          const gd_tensor_spec *spec,
                          const gd_init_spec *init,
                          gd_tensor *out);
gd_status gd_module_buffer(gd_context *ctx,
                           gd_module *module,
                           const char *name,
                           const gd_tensor_spec *spec,
                           const gd_init_spec *init,
                           gd_tensor *out);

void gd_module_set_training(gd_module *module, bool training);
gd_status gd_module_freeze(gd_module *module, const char *pattern);
gd_status gd_module_unfreeze(gd_module *module, const char *pattern);

void gd_param_set_init(gd_param_set *set);
void gd_param_set_free(gd_param_set *set);
gd_status gd_module_collect_params(gd_context *ctx,
                                   const gd_module *module,
                                   const gd_param_group *groups,
                                   uint32_t group_count,
                                   gd_param_set *out);

gd_status gd_module_list_init_child(gd_context *ctx,
                                    gd_module *parent,
                                    const char *name,
                                    gd_module_list *list,
                                    uint32_t count);
gd_status gd_module_list_set(gd_module_list *list, uint32_t index, gd_module *child);
void gd_module_list_deinit(gd_module_list *list);

gd_status gd_module_dict_init_child(gd_context *ctx,
                                    gd_module *parent,
                                    const char *name,
                                    gd_module_dict *dict);
gd_status gd_module_dict_set(gd_module_dict *dict, const char *name, gd_module *child);
void gd_module_dict_deinit(gd_module_dict *dict);

gd_status gd_linear_layer_init(gd_context *ctx,
                               gd_linear_layer *layer,
                               const char *name,
                               const gd_linear_layer_config *config);
gd_status gd_linear_layer_init_child(gd_context *ctx,
                                     gd_module *parent,
                                     const char *name,
                                     gd_linear_layer *layer,
                                     const gd_linear_layer_config *config);
void gd_linear_layer_deinit(gd_linear_layer *layer);
gd_status gd_linear_layer_forward(gd_context *ctx,
                                  gd_linear_layer *layer,
                                  const gd_tensor *x,
                                  gd_tensor *out);

#ifdef __cplusplus
}
#endif

#endif /* GRADIENTS_MODULE_H */
