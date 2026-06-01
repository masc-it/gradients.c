#include "gradients/module.h"

#include <stdint.h>
#include <stdio.h>
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
    gd_context *ctx;
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
    module->ctx = ctx;
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

typedef struct named_param {
    char *name;
    gd_tensor *tensor;
    int seen;
} named_param;

static const unsigned char module_magic[8] = {'G', 'D', 'M', 'O', 'D', '1', 0, 0};

static gd_status write_all(FILE *f, const void *data, size_t nbytes)
{
    if (nbytes == 0U) {
        return GD_OK;
    }
    if (fwrite(data, 1U, nbytes, f) != nbytes) {
        return _gd_error(GD_ERR_IO, "failed to write module checkpoint");
    }
    return GD_OK;
}

static gd_status read_all(FILE *f, void *data, size_t nbytes)
{
    if (nbytes == 0U) {
        return GD_OK;
    }
    if (fread(data, 1U, nbytes, f) != nbytes) {
        return _gd_error(GD_ERR_IO, "failed to read module checkpoint");
    }
    return GD_OK;
}

static char *join_name(const char *prefix, const char *name)
{
    size_t prefix_len = prefix == NULL ? 0U : strlen(prefix);
    size_t name_len = strlen(name);
    size_t total = prefix_len + (prefix_len > 0U ? 1U : 0U) + name_len;
    char *out = malloc(total + 1U);

    if (out == NULL) {
        return NULL;
    }
    if (prefix_len > 0U) {
        memcpy(out, prefix, prefix_len);
        out[prefix_len] = '.';
        memcpy(out + prefix_len + 1U, name, name_len + 1U);
    } else {
        memcpy(out, name, name_len + 1U);
    }
    return out;
}

static gd_status named_push(named_param **arr,
                            int *count,
                            int *cap,
                            char *name,
                            gd_tensor *tensor)
{
    if (*count == *cap) {
        int new_cap = *cap == 0 ? 16 : *cap * 2;
        named_param *grown = realloc(*arr, (size_t)new_cap * sizeof(*grown));
        if (grown == NULL) {
            free(name);
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow named parameter list");
        }
        *arr = grown;
        *cap = new_cap;
    }
    (*arr)[*count].name = name;
    (*arr)[*count].tensor = tensor;
    (*arr)[*count].seen = 0;
    *count += 1;
    return GD_OK;
}

static gd_status collect_named(gd_module *module,
                               const char *prefix,
                               named_param **arr,
                               int *count,
                               int *cap)
{
    int i = 0;

    for (i = 0; i < module->n_params; ++i) {
        char *full = join_name(prefix, module->params[i].name);
        gd_status status;
        if (full == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate parameter name");
        }
        status = named_push(arr, count, cap, full, module->params[i].tensor);
        if (status != GD_OK) {
            return status;
        }
    }
    for (i = 0; i < module->n_children; ++i) {
        char *child_prefix = join_name(prefix, module->children[i].name);
        gd_status status;
        if (child_prefix == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate child prefix");
        }
        status = collect_named(module->children[i].module, child_prefix, arr, count, cap);
        free(child_prefix);
        if (status != GD_OK) {
            return status;
        }
    }
    return GD_OK;
}

static void named_free(named_param *arr, int count)
{
    int i = 0;
    for (i = 0; i < count; ++i) {
        free(arr[i].name);
    }
    free(arr);
}

static named_param *find_named(named_param *arr, int count, const char *name)
{
    int i = 0;
    for (i = 0; i < count; ++i) {
        if (strcmp(arr[i].name, name) == 0) {
            return &arr[i];
        }
    }
    return NULL;
}

gd_status gd_module_save(gd_module *module, const char *path)
{
    gd_status status = GD_OK;
    named_param *params = NULL;
    int n_params = 0;
    int cap = 0;
    FILE *f = NULL;
    uint32_t version = 1U;
    uint32_t count32 = 0U;
    int i = 0;

    if (module == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_save argument is NULL");
    }
    status = collect_named(module, NULL, &params, &n_params, &cap);
    if (status != GD_OK) {
        named_free(params, n_params);
        return status;
    }
    if (n_params < 0) {
        named_free(params, n_params);
        return _gd_error(GD_ERR_INVALID_STATE, "invalid module parameter count");
    }
    count32 = (uint32_t)n_params;
    f = fopen(path, "wb");
    if (f == NULL) {
        named_free(params, n_params);
        return _gd_error(GD_ERR_IO, "failed to open module checkpoint for write");
    }
    status = write_all(f, module_magic, sizeof(module_magic));
    if (status == GD_OK) {
        status = write_all(f, &version, sizeof(version));
    }
    if (status == GD_OK) {
        status = write_all(f, &count32, sizeof(count32));
    }

    for (i = 0; i < n_params && status == GD_OK; ++i) {
        const gd_tensor_desc *desc = _gd_tensor_desc_ptr(params[i].tensor);
        size_t nbytes = 0U;
        uint64_t nbytes64 = 0U;
        uint32_t name_len = (uint32_t)strlen(params[i].name);
        uint32_t dtype = (uint32_t)desc->dtype;
        int32_t ndim = (int32_t)desc->ndim;
        void *buf = NULL;

        status = gd_tensor_desc_nbytes(desc, &nbytes, NULL);
        if (status != GD_OK) {
            break;
        }
        nbytes64 = (uint64_t)nbytes;
        buf = malloc(nbytes);
        if (buf == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate module save buffer");
            break;
        }
        status = gd_tensor_copy_to_cpu(module->ctx, params[i].tensor, buf, nbytes);
        if (status == GD_OK) { status = write_all(f, &name_len, sizeof(name_len)); }
        if (status == GD_OK) { status = write_all(f, params[i].name, name_len); }
        if (status == GD_OK) { status = write_all(f, &dtype, sizeof(dtype)); }
        if (status == GD_OK) { status = write_all(f, &ndim, sizeof(ndim)); }
        if (status == GD_OK) { status = write_all(f, desc->sizes, sizeof(desc->sizes)); }
        if (status == GD_OK) { status = write_all(f, &nbytes64, sizeof(nbytes64)); }
        if (status == GD_OK) { status = write_all(f, buf, nbytes); }
        free(buf);
    }

    if (fclose(f) != 0 && status == GD_OK) {
        status = _gd_error(GD_ERR_IO, "failed to close module checkpoint");
    }
    named_free(params, n_params);
    return status;
}

gd_status gd_module_load(gd_module *module, const char *path, bool strict)
{
    gd_status status = GD_OK;
    named_param *params = NULL;
    int n_params = 0;
    int cap = 0;
    FILE *f = NULL;
    unsigned char magic[8];
    uint32_t version = 0U;
    uint32_t file_count = 0U;
    uint32_t entry = 0U;

    if (module == NULL || path == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_module_load argument is NULL");
    }
    status = collect_named(module, NULL, &params, &n_params, &cap);
    if (status != GD_OK) {
        named_free(params, n_params);
        return status;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        named_free(params, n_params);
        return _gd_error(GD_ERR_IO, "failed to open module checkpoint for read");
    }
    status = read_all(f, magic, sizeof(magic));
    if (status == GD_OK && memcmp(magic, module_magic, sizeof(module_magic)) != 0) {
        status = _gd_error(GD_ERR_INVALID_ARGUMENT, "invalid module checkpoint magic");
    }
    if (status == GD_OK) { status = read_all(f, &version, sizeof(version)); }
    if (status == GD_OK && version != 1U) {
        status = _gd_error(GD_ERR_UNSUPPORTED, "unsupported module checkpoint version");
    }
    if (status == GD_OK) { status = read_all(f, &file_count, sizeof(file_count)); }

    for (entry = 0U; entry < file_count && status == GD_OK; ++entry) {
        uint32_t name_len = 0U;
        char *name = NULL;
        uint32_t dtype = 0U;
        int32_t ndim = 0;
        int64_t sizes[GD_MAX_DIMS];
        uint64_t nbytes64 = 0U;
        size_t nbytes = 0U;
        void *buf = NULL;
        named_param *target = NULL;

        memset(sizes, 0, sizeof(sizes));
        status = read_all(f, &name_len, sizeof(name_len));
        if (status != GD_OK) { break; }
        name = malloc((size_t)name_len + 1U);
        if (name == NULL) {
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate module entry name");
            break;
        }
        status = read_all(f, name, name_len);
        name[name_len] = '\0';
        if (status == GD_OK) { status = read_all(f, &dtype, sizeof(dtype)); }
        if (status == GD_OK) { status = read_all(f, &ndim, sizeof(ndim)); }
        if (status == GD_OK) { status = read_all(f, sizes, sizeof(sizes)); }
        if (status == GD_OK) { status = read_all(f, &nbytes64, sizeof(nbytes64)); }
        if (status != GD_OK) {
            free(name);
            break;
        }
        if (nbytes64 > (uint64_t)SIZE_MAX) {
            free(name);
            status = _gd_error(GD_ERR_INVALID_ARGUMENT, "module tensor is too large");
            break;
        }
        nbytes = (size_t)nbytes64;
        buf = malloc(nbytes);
        if (buf == NULL) {
            free(name);
            status = _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate module load buffer");
            break;
        }
        status = read_all(f, buf, nbytes);
        if (status == GD_OK) {
            target = find_named(params, n_params, name);
            if (target == NULL) {
                if (strict) {
                    status = _gd_error(GD_ERR_INVALID_ARGUMENT,
                                       "module checkpoint contains unknown parameter");
                }
            } else {
                const gd_tensor_desc *desc = _gd_tensor_desc_ptr(target->tensor);
                size_t expected_nbytes = 0U;
                int dim = 0;
                status = gd_tensor_desc_nbytes(desc, &expected_nbytes, NULL);
                if (status == GD_OK && ((uint32_t)desc->dtype != dtype ||
                                        (int32_t)desc->ndim != ndim ||
                                        expected_nbytes != nbytes)) {
                    status = _gd_error(GD_ERR_SHAPE, "module checkpoint tensor metadata mismatch");
                }
                for (dim = 0; status == GD_OK && dim < desc->ndim; ++dim) {
                    if (desc->sizes[dim] != sizes[dim]) {
                        status = _gd_error(GD_ERR_SHAPE,
                                           "module checkpoint tensor shape mismatch");
                    }
                }
                if (status == GD_OK) {
                    status = gd_tensor_copy_from_cpu(module->ctx, target->tensor, buf, nbytes);
                    target->seen = 1;
                }
            }
        }
        free(buf);
        free(name);
    }
    if (status == GD_OK && strict) {
        int i = 0;
        for (i = 0; i < n_params; ++i) {
            if (!params[i].seen) {
                status = _gd_error(GD_ERR_INVALID_ARGUMENT,
                                   "module checkpoint missing parameter");
                break;
            }
        }
    }
    if (fclose(f) != 0 && status == GD_OK) {
        status = _gd_error(GD_ERR_IO, "failed to close module checkpoint");
    }
    named_free(params, n_params);
    return status;
}
