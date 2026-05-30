#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

/* Internal seams: the perturbing test backend delegates to the registered CPU
 * reference backend and corrupts one output element to force a parity miss. */
#include "../src/backends/backend.h"
#include "../src/core/internal.h"

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());           \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                                \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes, const float *data,
                          gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, ndim, sizes, &desc);
    int64_t numel = 1;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        numel *= sizes[i];
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)numel * sizeof(float));
}

/* Builds a small forward graph: scale(relu(a @ b + a)). */
static gd_status build_graph(gd_context *ctx, gd_tensor *a, gd_tensor *b, gd_graph *g)
{
    gd_tensor *mm = NULL;
    gd_tensor *sum = NULL;
    gd_tensor *relu = NULL;
    gd_tensor *out = NULL;

    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_matmul(ctx, a, b, &mm));    /* [2,2] */
    CHECK_OK(gd_add(ctx, mm, a, &sum));
    CHECK_OK(gd_relu(ctx, sum, &relu));
    CHECK_OK(gd_scale(ctx, relu, 0.5F, &out));
    CHECK_OK(gd_graph_end(ctx));
    /* Drop virtual handles; the parity harness reads values by id, and the
     * graph cannot be destroyed while virtual tensors are live. */
    gd_tensor_release(mm);
    gd_tensor_release(sum);
    gd_tensor_release(relu);
    gd_tensor_release(out);
    return GD_OK;
}

/* -------------------------------------------------------------------------- */
/* Perturbing test backend                                                    */
/*                                                                            */
/* It owns no compute: storage, transfers, compilation and value access all   */
/* delegate to the CPU reference backend (resolved at init). Its only deviation*/
/* is in execute(): after running CPU_REF it bumps the last produced value's   */
/* first element so the parity harness must flag a mismatch.                   */
/* -------------------------------------------------------------------------- */

static _gd_backend *delegate(_gd_backend *self)
{
    return (_gd_backend *)self->impl;
}

static gd_status perturb_init(_gd_backend *self, gd_context *ctx, int device_index)
{
    _gd_backend *cpu = _gd_context_backend(ctx, (gd_device){GD_DEVICE_CPU, 0});

    (void)device_index;
    if (cpu == NULL) {
        return _gd_error(GD_ERR_INTERNAL, "perturb backend needs CPU backend");
    }
    self->impl = cpu;
    self->caps.host_visible = true;
    self->caps.supports_cpu_ref = false;
    self->caps.default_memory = GD_MEM_HOST;
    return GD_OK;
}

static gd_status perturb_storage_alloc(_gd_backend *self, const gd_storage_desc *desc,
                                       void **handle_out)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->storage_alloc(cpu, desc, handle_out);
}

static void perturb_storage_free(_gd_backend *self, void *handle)
{
    _gd_backend *cpu = delegate(self);
    cpu->vt->storage_free(cpu, handle);
}

static gd_status perturb_storage_host_ptr(_gd_backend *self, void *handle, void **ptr_out)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->storage_host_ptr(cpu, handle, ptr_out);
}

static gd_status perturb_upload(_gd_backend *self, void *dst, size_t off,
                                const void *src, size_t n)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->upload(cpu, dst, off, src, n);
}

static gd_status perturb_download(_gd_backend *self, void *src, size_t off, void *dst, size_t n)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->download(cpu, src, off, dst, n);
}

static gd_status perturb_compile(_gd_backend *self, gd_graph *graph, _gd_executable **out)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->compile(cpu, graph, out);
}

static gd_status perturb_value_storage(_gd_backend *self, _gd_executable *exe, int value_id,
                                       gd_storage **storage_out, size_t *offset_out)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->value_storage(cpu, exe, value_id, storage_out, offset_out);
}

static gd_status perturb_execute(_gd_backend *self, _gd_executable *exe)
{
    _gd_backend *cpu = delegate(self);
    gd_status status = cpu->vt->execute(cpu, exe);
    gd_storage *storage = NULL;
    size_t offset = 0U;
    void *host = NULL;
    int last = -1;
    int id = 0;

    if (status != GD_OK) {
        return status;
    }
    /* Probe for the highest valid value id; the last value is the graph's final
     * (produced) output, which the harness always compares. */
    while (cpu->vt->value_storage(cpu, exe, id, &storage, &offset) == GD_OK) {
        last = id;
        id += 1;
    }
    if (last < 0) {
        _gd_set_last_error(GD_OK, NULL);
        return GD_OK;
    }
    status = cpu->vt->value_storage(cpu, exe, last, &storage, &offset);
    if (status != GD_OK) {
        return status;
    }
    status = gd_storage_data_cpu(storage, &host);
    if (status != GD_OK) {
        return status;
    }
    ((float *)((char *)host + offset))[0] += 1.0F; /* the injected perturbation */
    _gd_set_last_error(GD_OK, NULL);
    return GD_OK;
}

static void perturb_executable_free(_gd_backend *self, _gd_executable *exe)
{
    _gd_backend *cpu = delegate(self);
    cpu->vt->executable_free(cpu, exe);
}

static gd_status perturb_synchronize(_gd_backend *self)
{
    _gd_backend *cpu = delegate(self);
    return cpu->vt->synchronize != NULL ? cpu->vt->synchronize(cpu) : GD_OK;
}

static const _gd_backend_vtable g_perturb_vtable = {
    .type = GD_DEVICE_CUDA, /* unused slot; METAL is owned by the real backend */
    .name = "perturb_test",
    .init = perturb_init,
    .shutdown = NULL,
    .storage_alloc = perturb_storage_alloc,
    .storage_free = perturb_storage_free,
    .storage_host_ptr = perturb_storage_host_ptr,
    .upload = perturb_upload,
    .download = perturb_download,
    .compile = perturb_compile,
    .execute = perturb_execute,
    .execute_until = NULL,
    .executable_free = perturb_executable_free,
    .value_storage = perturb_value_storage,
    .supports_node = NULL, /* supports all ops -> no fallback */
    .synchronize = perturb_synchronize,
};

static int test_identical_compares_equal(gd_context *ctx)
{
    int64_t s22[2] = {2, 2};
    float a[4] = {1, -2, 3, 4};
    float b[4] = {0.5F, 1, -1, 2};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_compare_options opts = {1e-5, 1e-5, false};

    CHECK_OK(make_f32(ctx, 2, s22, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s22, b, &tb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(build_graph(ctx, ta, tb, g));

    /* CPU vs CPU: no mismatch, even at zero tolerance (deterministic). */
    CHECK_OK(gd_graph_compare(g, cpu, cpu, &opts));
    CHECK_OK(gd_graph_compare(g, cpu, cpu, NULL));
    {
        gd_compare_options tight = {0.0, 0.0, true};
        CHECK_OK(gd_graph_compare(g, cpu, cpu, &tight));
    }

    /* Graph is left compiled on the target; it can still run and read. */
    CHECK_OK(gd_graph_run(g));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

static int test_detects_injected_mismatch(gd_context *ctx)
{
    int64_t s22[2] = {2, 2};
    float a[4] = {1, -2, 3, 4};
    float b[4] = {0.5F, 1, -1, 2};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device perturb = {GD_DEVICE_CUDA, 0};
    gd_compare_options opts = {1e-5, 1e-5, false};
    gd_status status = GD_OK;
    const char *err = NULL;

    CHECK_OK(_gd_context_register_backend(ctx, &g_perturb_vtable));

    CHECK_OK(make_f32(ctx, 2, s22, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s22, b, &tb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(build_graph(ctx, ta, tb, g));

    status = gd_graph_compare(g, cpu, perturb, &opts);
    err = gd_last_error();
    CHECK_TRUE(status == GD_ERR_BACKEND);
    CHECK_TRUE(strstr(err, "parity mismatch") != NULL);
    CHECK_TRUE(strstr(err, "node") != NULL);
    CHECK_TRUE(strstr(err, "value") != NULL);
    fprintf(stderr, "[compare] detected: %s\n", err);

    /* Swap reference/target: still detected regardless of ordering. */
    status = gd_graph_compare(g, perturb, cpu, &opts);
    CHECK_TRUE(status == GD_ERR_BACKEND);

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    int rc = 0;

    if (gd_context_create(&ctx) != GD_OK) {
        fprintf(stderr, "context create failed: %s\n", gd_last_error());
        return 1;
    }

    rc |= test_identical_compares_equal(ctx);
    rc |= test_detects_injected_mismatch(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_compare: ok\n");
    }
    return rc;
}
