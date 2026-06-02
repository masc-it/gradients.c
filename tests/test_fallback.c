/* P5 fallback-policy test: a stub backend registered on the Metal device slot
 * that supports no nodes. Under GD_FALLBACK_NONE compile must fail loud; under
 * GD_FALLBACK_CPU_REF the whole graph must run on CPU_REF and produce correct
 * results. Exercises the dispatch seam without any real GPU. */

#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

#include "../src/core/internal.h"
#include "../src/backends/backend.h"

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != GD_OK) {                                                   \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_STATUS(expr, expected)                                               \
    do {                                                                          \
        gd_status status_ = (expr);                                               \
        if (status_ != (expected)) {                                               \
            fprintf(stderr, "%s got %s expected %s; last_error=%s\n",             \
                    #expr, gd_status_name(status_), gd_status_name(expected),     \
                    gd_last_error());                                             \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static gd_status stub_init(_gd_backend *self, gd_context *ctx, int device_index)
{
    (void)ctx;
    (void)device_index;
    self->caps.host_visible = false;
    self->caps.supports_cpu_ref = false;
    return GD_OK;
}

static void stub_shutdown(_gd_backend *self) { (void)self; }

static gd_status stub_check_node(_gd_backend *self,
                                 const gd_graph *graph,
                                 const _gd_node *node)
{
    (void)self;
    (void)graph;
    (void)node;
    return _gd_error(GD_ERR_UNSUPPORTED,
                     "stub backend intentionally supports no graph nodes");
}

static gd_status stub_synchronize(_gd_backend *self)
{
    (void)self;
    return GD_OK;
}

static const _gd_backend_vtable stub_vtable = {
    /* Vulkan slot: Metal is owned by the real backend when auto-registered. */
    .type = GD_DEVICE_VULKAN,
    .name = "stub",
    .init = stub_init,
    .shutdown = stub_shutdown,
    .storage_alloc = NULL,
    .storage_free = NULL,
    .storage_host_ptr = NULL,
    .upload = NULL,
    .download = NULL,
    .compile = NULL,
    .execute = NULL,
    .execute_until = NULL,
    .executable_free = NULL,
    .value_storage = NULL,
    .check_node = stub_check_node,
    .synchronize = stub_synchronize,
};

static gd_status make_f32(gd_context *ctx, int64_t n, const float *data, gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 1, &n, &desc);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float));
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_device stub_dev = {GD_DEVICE_VULKAN, 0};
    int64_t n = 3;
    float a[3] = {1, 2, 3};
    float out[3];
    gd_tensor *ta = NULL;
    gd_tensor *scaled = NULL;
    gd_graph *g = NULL;

    CHECK_OK(gd_context_create(&ctx));
    CHECK_OK(_gd_context_register_backend(ctx, &stub_vtable));

    CHECK_OK(make_f32(ctx, n, a, &ta));

    /* Build once; compile is what applies fallback policy. */
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_scale(ctx, ta, 2.0F, &scaled));
    CHECK_OK(gd_graph_end(ctx));

    /* Policy NONE: target backend supports nothing -> loud failure. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    CHECK_STATUS(gd_graph_compile(g, stub_dev), GD_ERR_UNSUPPORTED);
    CHECK_TRUE(strstr(gd_last_error(), "stub") != NULL);
    CHECK_TRUE(strstr(gd_last_error(), "scale") != NULL);
    CHECK_TRUE(strstr(gd_last_error(), "node 0") != NULL);
    CHECK_TRUE(strstr(gd_last_error(), "intentionally supports no graph nodes") != NULL);

    /* Policy CPU_REF: whole-graph fallback to CPU, correct results. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF));
    CHECK_OK(gd_graph_compile(g, stub_dev));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scaled, out, sizeof(out)));
    CHECK_TRUE(out[0] == 2.0F && out[1] == 4.0F && out[2] == 6.0F);

    gd_tensor_release(scaled);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_context_destroy(ctx);
    printf("fallback ok\n");
    return 0;
}
