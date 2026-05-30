#include "gradients/context.h"

#include <stdlib.h>
#include <string.h>

#include "internal.h"
#include "../backends/backend.h"

#define GD_DEVICE_TYPE_COUNT 4

struct gd_context {
    gd_device default_device;
    gd_fallback_policy fallback_policy;
    gd_compute_policy compute_policy;
    gd_graph *active_graph;
    char *scope;          /* current joined scope path, e.g. "block0/attn" */
    size_t scope_len;
    size_t *scope_stack;  /* saved lengths for each push */
    int scope_depth;
    int scope_cap;
    _gd_backend *backends[GD_DEVICE_TYPE_COUNT];
};

gd_status gd_context_create(gd_context **out)
{
    gd_context *ctx = NULL;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_context_create out is NULL");
    }

    ctx = calloc(1U, sizeof(*ctx));
    if (ctx == NULL) {
        *out = NULL;
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate gd_context");
    }

    ctx->default_device = (gd_device){GD_DEVICE_CPU, 0};
    ctx->fallback_policy = GD_FALLBACK_NONE;
    ctx->compute_policy = gd_compute_policy_default();

    {
        gd_status status = _gd_cpu_backend_register(ctx);
        if (status != GD_OK) {
            gd_context_destroy(ctx);
            *out = NULL;
            return status;
        }
    }

#if defined(GD_ENABLE_METAL)
    /* Best-effort: register Metal when a GPU + shader library are present.
     * Failure leaves the context CPU-only; Metal is just another backend. */
    (void)_gd_metal_backend_register(ctx);
    _gd_set_last_error(GD_OK, NULL);
#endif

    *out = ctx;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

void gd_context_destroy(gd_context *ctx)
{
    if (ctx != NULL) {
        int i = 0;
        for (i = 0; i < GD_DEVICE_TYPE_COUNT; ++i) {
            _gd_backend *backend = ctx->backends[i];
            if (backend != NULL) {
                if (backend->vt != NULL && backend->vt->shutdown != NULL) {
                    backend->vt->shutdown(backend);
                }
                free(backend);
            }
        }
        free(ctx->scope);
        free(ctx->scope_stack);
    }
    free(ctx);
    _gd_set_last_error(GD_OK, NULL);
}

gd_status _gd_context_register_backend(gd_context *ctx, const _gd_backend_vtable *vt)
{
    gd_status status = GD_OK;
    _gd_backend *backend = NULL;

    if (ctx == NULL || vt == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "register_backend argument is NULL");
    }
    if ((int)vt->type < 0 || (int)vt->type >= GD_DEVICE_TYPE_COUNT) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "backend has unknown device type");
    }
    if (ctx->backends[vt->type] != NULL) {
        return _gd_error(GD_ERR_INVALID_STATE, "backend already registered for device type");
    }

    backend = calloc(1U, sizeof(*backend));
    if (backend == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate backend");
    }
    backend->vt = vt;
    backend->ctx = ctx;
    backend->device_index = 0;
    if (vt->init != NULL) {
        status = vt->init(backend, ctx, 0);
        if (status != GD_OK) {
            free(backend);
            return status;
        }
    }
    ctx->backends[vt->type] = backend;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

_gd_backend *_gd_context_backend(const gd_context *ctx, gd_device device)
{
    if (ctx == NULL) {
        return NULL;
    }
    if ((int)device.type < 0 || (int)device.type >= GD_DEVICE_TYPE_COUNT) {
        return NULL;
    }
    if (device.index != 0) {
        return NULL; /* single device per type in v1 */
    }
    return ctx->backends[device.type];
}

gd_status _gd_device_validate_available(const gd_context *ctx, gd_device device)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "context is NULL");
    }
    if (_gd_context_backend(ctx, device) == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for this device");
    }
    return GD_OK;
}

const char *_gd_context_scope(const gd_context *ctx)
{
    if (ctx == NULL || ctx->scope == NULL) {
        return "";
    }
    return ctx->scope;
}

gd_status _gd_context_scope_push(gd_context *ctx, const char *name)
{
    size_t name_len = 0U;
    size_t new_len = 0U;
    char *grown = NULL;

    if (ctx == NULL || name == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "scope_push argument is NULL");
    }
    if (strchr(name, '/') != NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "scope name must not contain '/'");
    }

    if (ctx->scope_depth == ctx->scope_cap) {
        int new_cap = ctx->scope_cap == 0 ? 8 : ctx->scope_cap * 2;
        size_t *stack = realloc(ctx->scope_stack, (size_t)new_cap * sizeof(*stack));
        if (stack == NULL) {
            return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow scope stack");
        }
        ctx->scope_stack = stack;
        ctx->scope_cap = new_cap;
    }

    name_len = strlen(name);
    new_len = ctx->scope_len + (ctx->scope_len > 0U ? 1U : 0U) + name_len;
    grown = realloc(ctx->scope, new_len + 1U);
    if (grown == NULL) {
        return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to grow scope path");
    }
    ctx->scope = grown;
    ctx->scope_stack[ctx->scope_depth] = ctx->scope_len;
    ctx->scope_depth += 1;
    if (ctx->scope_len > 0U) {
        ctx->scope[ctx->scope_len] = '/';
        memcpy(ctx->scope + ctx->scope_len + 1U, name, name_len + 1U);
    } else {
        memcpy(ctx->scope, name, name_len + 1U);
    }
    ctx->scope_len = new_len;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status _gd_context_scope_pop(gd_context *ctx)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "scope_pop ctx is NULL");
    }
    if (ctx->scope_depth == 0) {
        return _gd_error(GD_ERR_INVALID_STATE, "scope stack is empty");
    }
    ctx->scope_depth -= 1;
    ctx->scope_len = ctx->scope_stack[ctx->scope_depth];
    if (ctx->scope != NULL) {
        ctx->scope[ctx->scope_len] = '\0';
    }
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_status gd_context_set_default_device(gd_context *ctx, gd_device device)
{
    gd_status status = GD_OK;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_context_set_default_device ctx is NULL");
    }

    status = _gd_device_validate_available(ctx, device);
    if (status != GD_OK) {
        return status;
    }

    ctx->default_device = device;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_device gd_context_default_device(const gd_context *ctx)
{
    if (ctx == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_context_default_device ctx is NULL");
        return (gd_device){GD_DEVICE_CPU, 0};
    }

    _gd_set_last_error(GD_OK, NULL);
    return ctx->default_device;
}

gd_status gd_context_set_fallback_policy(gd_context *ctx, gd_fallback_policy policy)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_context_set_fallback_policy ctx is NULL");
    }
    if (policy != GD_FALLBACK_NONE && policy != GD_FALLBACK_CPU_REF) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unknown fallback policy");
    }

    ctx->fallback_policy = policy;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_fallback_policy gd_context_fallback_policy(const gd_context *ctx)
{
    if (ctx == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_context_fallback_policy ctx is NULL");
        return GD_FALLBACK_NONE;
    }

    _gd_set_last_error(GD_OK, NULL);
    return ctx->fallback_policy;
}

static int valid_dtype_or_default(gd_dtype dtype)
{
    return dtype == GD_DTYPE_INVALID || dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16 ||
           dtype == GD_DTYPE_BF16 || dtype == GD_DTYPE_FP8_E4M3 ||
           dtype == GD_DTYPE_FP8_E5M2;
}

gd_status gd_context_set_compute_policy(gd_context *ctx, gd_compute_policy policy)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_context_set_compute_policy ctx is NULL");
    }
    if (!valid_dtype_or_default(policy.compute_dtype) ||
        !valid_dtype_or_default(policy.accum_dtype)) {
        return _gd_error(GD_ERR_DTYPE, "compute policy dtype must be floating or invalid/default");
    }

    if (policy.compute_dtype == GD_DTYPE_INVALID) {
        policy.compute_dtype = GD_DTYPE_F32;
    }
    if (policy.accum_dtype == GD_DTYPE_INVALID) {
        policy.accum_dtype = GD_DTYPE_F32;
    }

    ctx->compute_policy = policy;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

gd_compute_policy gd_context_compute_policy(const gd_context *ctx)
{
    if (ctx == NULL) {
        _gd_set_last_error(GD_ERR_INVALID_ARGUMENT, "gd_context_compute_policy ctx is NULL");
        return gd_compute_policy_default();
    }

    _gd_set_last_error(GD_OK, NULL);
    return ctx->compute_policy;
}

gd_status gd_synchronize(gd_context *ctx, gd_device device)
{
    _gd_backend *backend = NULL;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_synchronize ctx is NULL");
    }

    backend = _gd_context_backend(ctx, device);
    if (backend == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for this device");
    }
    if (backend->vt->synchronize == NULL) {
        _gd_set_last_error(GD_OK, NULL);
        return GD_OK; /* a backend with no async work needs no sync */
    }
    return backend->vt->synchronize(backend);
}

gd_graph *_gd_context_active_graph(const gd_context *ctx)
{
    if (ctx == NULL) {
        return NULL;
    }
    return ctx->active_graph;
}

gd_status _gd_context_set_active_graph(gd_context *ctx, gd_graph *graph)
{
    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "context is NULL");
    }
    ctx->active_graph = graph;
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}
