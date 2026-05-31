#include "gradients/context.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "internal.h"
#include "../backends/backend.h"
#include "../graph/graph_internal.h"

#define GD_DEVICE_TYPE_COUNT 4
#define GD_PROFILE_OP_COUNT 64

typedef enum gd_profile_mode {
    GD_PROFILE_OFF = 0,
    GD_PROFILE_SUMMARY,
    GD_PROFILE_JSON,
    GD_PROFILE_TRACE
} gd_profile_mode;

typedef struct gd_profile_op_stats {
    uint64_t count;
    uint64_t ns;
} gd_profile_op_stats;

typedef struct gd_profile_event_stats {
    uint64_t count;
    uint64_t ns;
    uint64_t bytes;
    uint64_t items;
} gd_profile_event_stats;

typedef struct gd_profile_backend_stats {
    bool seen;
    const char *name;
    gd_device_type type;
    int index;
    uint64_t compiles;
    uint64_t compile_ns;
    uint64_t runs;
    uint64_t run_ns;
    uint64_t nodes_executed;
    uint64_t syncs;
    uint64_t explicit_syncs;
    uint64_t sync_ns;
    uint64_t uploads;
    uint64_t upload_ns;
    uint64_t upload_bytes;
    uint64_t downloads;
    uint64_t blocking_downloads;
    uint64_t download_ns;
    uint64_t download_bytes;
    uint64_t allocs;
    uint64_t frees;
    uint64_t allocated_bytes;
    uint64_t freed_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    gd_profile_op_stats ops[GD_PROFILE_OP_COUNT];
    gd_profile_event_stats events[_GD_PROFILE_EVENT_COUNT];
} gd_profile_backend_stats;

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
    gd_profile_mode profile_mode;
    char *profile_path;
    char *profile_backend_filter;
    gd_profile_backend_stats profile[GD_DEVICE_TYPE_COUNT];
};

static char *dup_cstr(const char *s)
{
    size_t len = 0U;
    char *out = NULL;

    if (s == NULL) {
        return NULL;
    }
    len = strlen(s);
    out = malloc(len + 1U);
    if (out == NULL) {
        return NULL;
    }
    memcpy(out, s, len + 1U);
    return out;
}

static gd_profile_mode parse_profile_mode(void)
{
    const char *env = getenv("GD_PROFILE");

    if (env == NULL || env[0] == '\0' || strcmp(env, "0") == 0 ||
        strcmp(env, "off") == 0 || strcmp(env, "false") == 0) {
        return GD_PROFILE_OFF;
    }
    if (strcmp(env, "json") == 0) {
        return GD_PROFILE_JSON;
    }
    if (strcmp(env, "trace") == 0) {
        return GD_PROFILE_TRACE;
    }
    return GD_PROFILE_SUMMARY;
}

static bool profile_backend_allowed(const gd_context *ctx, const _gd_backend *backend)
{
    const char *filter = NULL;
    const char *name = NULL;
    const char *device_name = NULL;
    const char *p = NULL;

    if (ctx == NULL || backend == NULL || backend->vt == NULL) {
        return false;
    }
    filter = ctx->profile_backend_filter;
    if (filter == NULL || filter[0] == '\0') {
        return true;
    }
    name = backend->vt->name != NULL ? backend->vt->name : "";
    device_name = gd_device_type_name(backend->vt->type);
    p = filter;
    while (*p != '\0') {
        const char *end = strchr(p, ',');
        size_t len = end != NULL ? (size_t)(end - p) : strlen(p);
        while (len > 0U && (*p == ' ' || *p == '\t')) {
            p += 1;
            len -= 1U;
        }
        while (len > 0U && (p[len - 1U] == ' ' || p[len - 1U] == '\t')) {
            len -= 1U;
        }
        if ((strlen(name) == len && strncmp(p, name, len) == 0) ||
            (strlen(device_name) == len && strncmp(p, device_name, len) == 0)) {
            return true;
        }
        if (end == NULL) {
            break;
        }
        p = end + 1;
    }
    return false;
}

static gd_profile_backend_stats *profile_stats(gd_context *ctx, const _gd_backend *backend)
{
    gd_profile_backend_stats *stats = NULL;
    int type = 0;

    if (ctx == NULL || backend == NULL || backend->vt == NULL ||
        !profile_backend_allowed(ctx, backend)) {
        return NULL;
    }
    type = (int)backend->vt->type;
    if (type < 0 || type >= GD_DEVICE_TYPE_COUNT) {
        return NULL;
    }
    stats = &ctx->profile[type];
    stats->seen = true;
    stats->name = backend->vt->name;
    stats->type = backend->vt->type;
    stats->index = backend->device_index;
    return stats;
}

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
    ctx->profile_mode = parse_profile_mode();
    if (ctx->profile_mode != GD_PROFILE_OFF) {
        const char *path = getenv("GD_PROFILE_PATH");
        if (path != NULL && path[0] != '\0') {
            ctx->profile_path = dup_cstr(path);
            if (ctx->profile_path == NULL) {
                gd_context_destroy(ctx);
                *out = NULL;
                return _gd_error(GD_ERR_OUT_OF_MEMORY, "failed to allocate profile path");
            }
        }
        {
            const char *filter = getenv("GD_PROFILE_BACKEND");
            if (filter != NULL && filter[0] != '\0') {
                ctx->profile_backend_filter = dup_cstr(filter);
                if (ctx->profile_backend_filter == NULL) {
                    gd_context_destroy(ctx);
                    *out = NULL;
                    return _gd_error(GD_ERR_OUT_OF_MEMORY,
                                     "failed to allocate profile backend filter");
                }
            }
        }
    }

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

static double ns_to_ms(uint64_t ns)
{
    return (double)ns / 1000000.0;
}

static const char *profile_event_name(_gd_profile_event event)
{
    switch (event) {
    case _GD_PROFILE_EVENT_STAGE_LEAVES:
        return "stage_leaves";
    case _GD_PROFILE_EVENT_ENCODE:
        return "metal_encode";
    case _GD_PROFILE_EVENT_WAIT:
        return "metal_wait";
    case _GD_PROFILE_EVENT_WRITEBACK:
        return "writeback_externals";
    case _GD_PROFILE_EVENT_COPY_ALIAS:
        return "copy_alias_skip";
    case _GD_PROFILE_EVENT_COUNT:
        break;
    }
    return "unknown";
}

static FILE *profile_open_output(const gd_context *ctx, bool *must_close)
{
    FILE *file = stderr;

    *must_close = false;
    if (ctx->profile_path != NULL) {
        file = fopen(ctx->profile_path, "w");
        if (file != NULL) {
            *must_close = true;
        } else {
            file = stderr;
        }
    }
    return file;
}

static void profile_print_summary_backend(FILE *file, const gd_profile_backend_stats *s)
{
    int i = 0;

    if (!s->seen) {
        return;
    }
    (void)fprintf(file, "[gd_profile] backend=%s device=%s:%d\n",
                  s->name != NULL ? s->name : "unknown",
                  gd_device_type_name(s->type), s->index);
    (void)fprintf(file,
                  "[gd_profile]   compiles=%" PRIu64 " compile_ms=%.3f runs=%" PRIu64
                  " run_ms=%.3f nodes=%" PRIu64 "\n",
                  s->compiles, ns_to_ms(s->compile_ns), s->runs, ns_to_ms(s->run_ns),
                  s->nodes_executed);
    (void)fprintf(file,
                  "[gd_profile]   syncs=%" PRIu64 " explicit=%" PRIu64 " sync_ms=%.3f\n",
                  s->syncs, s->explicit_syncs, ns_to_ms(s->sync_ns));
    (void)fprintf(file,
                  "[gd_profile]   uploads=%" PRIu64 " upload_bytes=%" PRIu64
                  " upload_ms=%.3f downloads=%" PRIu64 " blocking_downloads=%" PRIu64
                  " download_bytes=%" PRIu64 " download_ms=%.3f\n",
                  s->uploads, s->upload_bytes, ns_to_ms(s->upload_ns), s->downloads,
                  s->blocking_downloads, s->download_bytes, ns_to_ms(s->download_ns));
    (void)fprintf(file,
                  "[gd_profile]   allocs=%" PRIu64 " frees=%" PRIu64
                  " allocated_bytes=%" PRIu64 " freed_bytes=%" PRIu64
                  " peak_live_bytes=%" PRIu64 " live_bytes=%" PRIu64 "\n",
                  s->allocs, s->frees, s->allocated_bytes, s->freed_bytes,
                  s->peak_live_bytes, s->live_bytes);
    for (i = 0; i < (int)_GD_PROFILE_EVENT_COUNT; ++i) {
        const gd_profile_event_stats *e = &s->events[i];
        if (e->count == 0U) {
            continue;
        }
        (void)fprintf(file,
                      "[gd_profile]   event=%s count=%" PRIu64 " items=%" PRIu64
                      " bytes=%" PRIu64 " ms=%.3f\n",
                      profile_event_name((_gd_profile_event)i), e->count, e->items,
                      e->bytes, ns_to_ms(e->ns));
    }
    for (i = 0; i < GD_PROFILE_OP_COUNT; ++i) {
        const gd_profile_op_stats *op = &s->ops[i];
        if (op->count == 0U) {
            continue;
        }
        (void)fprintf(file, "[gd_profile]   op=%s count=%" PRIu64 "\n",
                      _gd_op_kind_name((_gd_op_kind)i), op->count);
    }
}

static void profile_print_json_backend(FILE *file, const gd_profile_backend_stats *s,
                                       bool *first_backend)
{
    int i = 0;
    bool first = true;

    if (!s->seen) {
        return;
    }
    if (!*first_backend) {
        (void)fprintf(file, ",\n");
    }
    *first_backend = false;
    (void)fprintf(file,
                  "    {\"backend\":\"%s\",\"device\":\"%s:%d\","
                  "\"compiles\":%" PRIu64 ",\"compile_ns\":%" PRIu64
                  ",\"runs\":%" PRIu64 ",\"run_ns\":%" PRIu64
                  ",\"nodes\":%" PRIu64 ",\"syncs\":%" PRIu64
                  ",\"explicit_syncs\":%" PRIu64 ",\"sync_ns\":%" PRIu64
                  ",\"uploads\":%" PRIu64 ",\"upload_bytes\":%" PRIu64
                  ",\"upload_ns\":%" PRIu64 ",\"downloads\":%" PRIu64
                  ",\"blocking_downloads\":%" PRIu64
                  ",\"download_bytes\":%" PRIu64 ",\"download_ns\":%" PRIu64
                  ",\"allocs\":%" PRIu64 ",\"frees\":%" PRIu64
                  ",\"allocated_bytes\":%" PRIu64 ",\"freed_bytes\":%" PRIu64
                  ",\"peak_live_bytes\":%" PRIu64 ",\"live_bytes\":%" PRIu64,
                  s->name != NULL ? s->name : "unknown", gd_device_type_name(s->type),
                  s->index, s->compiles, s->compile_ns, s->runs, s->run_ns,
                  s->nodes_executed, s->syncs, s->explicit_syncs, s->sync_ns,
                  s->uploads, s->upload_bytes, s->upload_ns, s->downloads,
                  s->blocking_downloads, s->download_bytes, s->download_ns,
                  s->allocs, s->frees, s->allocated_bytes, s->freed_bytes,
                  s->peak_live_bytes, s->live_bytes);
    (void)fprintf(file, ",\"events\":{");
    first = true;
    for (i = 0; i < (int)_GD_PROFILE_EVENT_COUNT; ++i) {
        const gd_profile_event_stats *e = &s->events[i];
        if (e->count == 0U) {
            continue;
        }
        if (!first) {
            (void)fprintf(file, ",");
        }
        first = false;
        (void)fprintf(file,
                      "\"%s\":{\"count\":%" PRIu64 ",\"items\":%" PRIu64
                      ",\"bytes\":%" PRIu64 ",\"ns\":%" PRIu64 "}",
                      profile_event_name((_gd_profile_event)i), e->count, e->items,
                      e->bytes, e->ns);
    }
    (void)fprintf(file, "},\"ops\":{");
    first = true;
    for (i = 0; i < GD_PROFILE_OP_COUNT; ++i) {
        const gd_profile_op_stats *op = &s->ops[i];
        if (op->count == 0U) {
            continue;
        }
        if (!first) {
            (void)fprintf(file, ",");
        }
        first = false;
        (void)fprintf(file, "\"%s\":%" PRIu64, _gd_op_kind_name((_gd_op_kind)i),
                      op->count);
    }
    (void)fprintf(file, "}}");
}

static void profile_print(gd_context *ctx)
{
    FILE *file = NULL;
    bool must_close = false;
    int i = 0;

    if (ctx == NULL || ctx->profile_mode == GD_PROFILE_OFF) {
        return;
    }
    file = profile_open_output(ctx, &must_close);
    if (ctx->profile_mode == GD_PROFILE_JSON) {
        bool first_backend = true;
        (void)fprintf(file, "{\n  \"backends\": [\n");
        for (i = 0; i < GD_DEVICE_TYPE_COUNT; ++i) {
            profile_print_json_backend(file, &ctx->profile[i], &first_backend);
        }
        (void)fprintf(file, "\n  ]\n}\n");
    } else {
        for (i = 0; i < GD_DEVICE_TYPE_COUNT; ++i) {
            profile_print_summary_backend(file, &ctx->profile[i]);
        }
    }
    if (must_close) {
        (void)fclose(file);
    }
}

void gd_context_destroy(gd_context *ctx)
{
    if (ctx != NULL) {
        int i = 0;
        profile_print(ctx);
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
        free(ctx->profile_path);
        free(ctx->profile_backend_filter);
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
    uint64_t start = 0U;
    gd_status status = GD_OK;

    if (ctx == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "gd_synchronize ctx is NULL");
    }

    backend = _gd_context_backend(ctx, device);
    if (backend == NULL) {
        return _gd_error(GD_ERR_UNSUPPORTED, "no backend registered for this device");
    }
    if (backend->vt->synchronize == NULL) {
        _gd_profile_record_sync(ctx, backend, 0U, true);
        _gd_set_last_error(GD_OK, NULL);
        return GD_OK; /* a backend with no async work needs no sync */
    }
    start = _gd_profile_enabled(ctx) ? _gd_profile_now_ns() : 0U;
    status = backend->vt->synchronize(backend);
    if (start != 0U) {
        _gd_profile_record_sync(ctx, backend, _gd_profile_now_ns() - start, true);
    }
    return status;
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

bool _gd_profile_enabled(const gd_context *ctx)
{
    return ctx != NULL && ctx->profile_mode != GD_PROFILE_OFF;
}

uint64_t _gd_profile_now_ns(void)
{
    struct timeval tv;

    (void)gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000000ULL + (uint64_t)tv.tv_usec * 1000ULL;
}

static void profile_count_ops(gd_profile_backend_stats *stats, const _gd_node *nodes,
                              int n_nodes)
{
    int i = 0;

    if (stats == NULL || nodes == NULL || n_nodes <= 0) {
        return;
    }
    for (i = 0; i < n_nodes; ++i) {
        int op = (int)nodes[i].op;
        if (op >= 0 && op < GD_PROFILE_OP_COUNT) {
            stats->ops[op].count += 1U;
        }
    }
}

void _gd_profile_record_compile(gd_context *ctx, const _gd_backend *backend,
                                uint64_t elapsed_ns, const _gd_node *nodes, int n_nodes)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    (void)nodes;
    stats->compiles += 1U;
    stats->compile_ns += elapsed_ns;
    if (n_nodes > 0) {
        /* Compile sees the graph shape but op execution counts belong to runs. */
    }
}

void _gd_profile_record_run(gd_context *ctx, const _gd_backend *backend,
                            uint64_t elapsed_ns, const _gd_node *nodes, int n_nodes)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->runs += 1U;
    stats->run_ns += elapsed_ns;
    if (n_nodes > 0) {
        stats->nodes_executed += (uint64_t)n_nodes;
    }
    profile_count_ops(stats, nodes, n_nodes);
}

void _gd_profile_record_sync(gd_context *ctx, const _gd_backend *backend,
                             uint64_t elapsed_ns, bool explicit_call)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->syncs += 1U;
    stats->sync_ns += elapsed_ns;
    if (explicit_call) {
        stats->explicit_syncs += 1U;
    }
}

void _gd_profile_record_upload(gd_context *ctx, const _gd_backend *backend,
                               uint64_t elapsed_ns, size_t nbytes)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->uploads += 1U;
    stats->upload_ns += elapsed_ns;
    stats->upload_bytes += (uint64_t)nbytes;
}

void _gd_profile_record_download(gd_context *ctx, const _gd_backend *backend,
                                 uint64_t elapsed_ns, size_t nbytes, bool blocking_read)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->downloads += 1U;
    stats->download_ns += elapsed_ns;
    stats->download_bytes += (uint64_t)nbytes;
    if (blocking_read) {
        stats->blocking_downloads += 1U;
    }
}

void _gd_profile_record_alloc(gd_context *ctx, const _gd_backend *backend, size_t nbytes)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->allocs += 1U;
    stats->allocated_bytes += (uint64_t)nbytes;
    stats->live_bytes += (uint64_t)nbytes;
    if (stats->live_bytes > stats->peak_live_bytes) {
        stats->peak_live_bytes = stats->live_bytes;
    }
}

void _gd_profile_record_free(gd_context *ctx, const _gd_backend *backend, size_t nbytes)
{
    gd_profile_backend_stats *stats = NULL;

    if (!_gd_profile_enabled(ctx)) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    stats->frees += 1U;
    stats->freed_bytes += (uint64_t)nbytes;
    if (stats->live_bytes >= (uint64_t)nbytes) {
        stats->live_bytes -= (uint64_t)nbytes;
    } else {
        stats->live_bytes = 0U;
    }
}

void _gd_profile_record_event(gd_context *ctx, const _gd_backend *backend,
                              _gd_profile_event event, uint64_t elapsed_ns,
                              size_t nbytes, uint64_t count)
{
    gd_profile_backend_stats *stats = NULL;
    gd_profile_event_stats *e = NULL;

    if (!_gd_profile_enabled(ctx) || (int)event < 0 || event >= _GD_PROFILE_EVENT_COUNT) {
        return;
    }
    stats = profile_stats(ctx, backend);
    if (stats == NULL) {
        return;
    }
    e = &stats->events[event];
    e->count += 1U;
    e->ns += elapsed_ns;
    e->bytes += (uint64_t)nbytes;
    e->items += count;
}
