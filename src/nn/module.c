#include "gradients/module.h"

#include <stdlib.h>
#include <string.h>

#include "gradients/ops.h"

#include "../core/internal.h"
#include "../core/tensor_internal.h"

typedef struct module_param {
    char *name;
    gd_tensor *tensor;
} module_param;

typedef struct module_child {
    char *name;
    gd_module *module;
} module_child;

struct gd_module {
    char *type_name;
    module_param *params;
    int n_params;
    int param_cap;
    module_child *children;
    int n_children;
    int child_cap;
    gd_tensor **flat;   /* cached deduped parameter list */
    int n_flat;
};

static char *dup_string(const char *s)
{
    size_t len = 0U;
    char *copy = NULL;

    if (s == NULL) {
        s = "";
    }
    len = strlen(s);
    copy = malloc(len + 1U);
    if (copy != NULL) {
        memcpy(copy, s, len + 1U);
    }
    return copy;
}

gd_status gd_module_create(gd_context *ctx, const char *type_name, gd_module **out)
{
    gd_module *module = NULL;

    if (ctx == NULL || type_name == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_create argument is NULL");
    }
    *out = NULL;

    module = calloc(1U, sizeof(*module));
    if (module == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate module");
    }
    module->type_name = dup_string(type_name);
    if (module->type_name == NULL) {
        free(module);
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate module type name");
    }

    *out = module;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void gd_module_destroy(gd_module *module)
{
    int i = 0;

    if (module == NULL) {
        return;
    }
    for (i = 0; i < module->n_params; ++i) {
        gd_tensor_release(module->params[i].tensor);
        free(module->params[i].name);
    }
    for (i = 0; i < module->n_children; ++i) {
        gd_module_destroy(module->children[i].module);
        free(module->children[i].name);
    }
    free(module->params);
    free(module->children);
    free(module->flat);
    free(module->type_name);
    free(module);
    _gd_set_last_error(GD_OK, NULL);
}

gd_status gd_module_param(gd_module *module, const char *name, gd_tensor *param)
{
    gd_status status = GD_OK;
    char *name_copy = NULL;

    if (module == NULL || name == NULL || param == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_param argument is NULL");
    }
    if (gd_tensor_storage(param) == NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "module parameters must be materialized");
    }

    if (module->n_params == module->param_cap) {
        int new_cap = module->param_cap == 0 ? 8 : module->param_cap * 2;
        module_param *grown = realloc(module->params, (size_t)new_cap * sizeof(*grown));
        if (grown == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow module params");
        }
        module->params = grown;
        module->param_cap = new_cap;
    }

    name_copy = dup_string(name);
    if (name_copy == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to copy param name");
    }
    status = gd_tensor_retain(param);
    if (status != GD_OK) {
        free(name_copy);
        return status;
    }

    module->params[module->n_params].name = name_copy;
    module->params[module->n_params].tensor = param;
    module->n_params += 1;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_module_child(gd_module *module, const char *name, gd_module *child)
{
    char *name_copy = NULL;

    if (module == NULL || name == NULL || child == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_child argument is NULL");
    }
    if (child == module) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "module cannot be its own child");
    }

    if (module->n_children == module->child_cap) {
        int new_cap = module->child_cap == 0 ? 4 : module->child_cap * 2;
        module_child *grown = realloc(module->children, (size_t)new_cap * sizeof(*grown));
        if (grown == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow module children");
        }
        module->children = grown;
        module->child_cap = new_cap;
    }

    name_copy = dup_string(name);
    if (name_copy == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to copy child name");
    }

    /* Ownership of the child transfers to the parent. */
    module->children[module->n_children].name = name_copy;
    module->children[module->n_children].module = child;
    module->n_children += 1;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

static gd_status param_extent(gd_tensor *t, size_t *offset_out, size_t *nbytes_out)
{
    const gd_tensor_desc *desc = _gd_tensor_desc_ptr(t);

    if (desc == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "parameter has no descriptor");
    }
    *offset_out = (size_t)desc->storage_offset_bytes;
    return gd_tensor_desc_nbytes(desc, nbytes_out, NULL);
}

static int same_parameter(gd_tensor *a, gd_tensor *b)
{
    size_t a_off = 0U;
    size_t b_off = 0U;
    size_t a_nb = 0U;
    size_t b_nb = 0U;

    if (gd_tensor_storage(a) != gd_tensor_storage(b)) {
        return 0;
    }
    if (param_extent(a, &a_off, &a_nb) != GD_OK || param_extent(b, &b_off, &b_nb) != GD_OK) {
        return 0;
    }
    return a_off == b_off && a_nb == b_nb;
}

static gd_status collect(gd_module *module,
                         gd_tensor ***arr,
                         int *count,
                         int *cap)
{
    int i = 0;

    for (i = 0; i < module->n_params; ++i) {
        gd_tensor *t = module->params[i].tensor;
        int j = 0;
        int duplicate = 0;

        for (j = 0; j < *count; ++j) {
            if (same_parameter((*arr)[j], t)) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        if (*count == *cap) {
            int new_cap = *cap == 0 ? 8 : *cap * 2;
            gd_tensor **grown = realloc(*arr, (size_t)new_cap * sizeof(*grown));
            if (grown == NULL) {
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow parameter list");
            }
            *arr = grown;
            *cap = new_cap;
        }
        (*arr)[*count] = t;
        *count += 1;
    }

    for (i = 0; i < module->n_children; ++i) {
        gd_status status = collect(module->children[i].module, arr, count, cap);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

gd_status gd_module_parameters(gd_module *module, gd_tensor ***params_out, int *n_out)
{
    gd_status status = GD_OK;
    gd_tensor **arr = NULL;
    int count = 0;
    int cap = 0;

    if (module == NULL || params_out == NULL || n_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_parameters argument is NULL");
    }
    *params_out = NULL;
    *n_out = 0;

    status = collect(module, &arr, &count, &cap);
    if (status != GD_OK) {
        free(arr);
        return status;
    }

    free(module->flat);
    module->flat = arr;
    module->n_flat = count;

    *params_out = module->flat;
    *n_out = module->n_flat;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_module_zero_grad(gd_context *ctx, gd_module *module)
{
    gd_status status = GD_OK;
    gd_tensor **params = NULL;
    int n = 0;

    if (ctx == NULL || module == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_zero_grad argument is NULL");
    }
    status = gd_module_parameters(module, &params, &n);
    if (status != GD_OK) {
        return status;
    }
    return gd_zero_grad(ctx, params, n);
}

gd_status gd_module_save(gd_module *module, const char *path)
{
    if (module == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_save argument is NULL");
    }
    return _gd_error(GD_ERR_UNSUPPORTED, "module save is not implemented yet");
}

gd_status gd_module_load(gd_module *module, const char *path, bool strict)
{
    if (module == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_load argument is NULL");
    }
    (void)strict;
    return _gd_error(GD_ERR_UNSUPPORTED, "module load is not implemented yet");
}
