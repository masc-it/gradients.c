#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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

static const gd_device CPU = {GD_DEVICE_CPU, 0};
static const gd_device METAL = {GD_DEVICE_METAL, 0};

static int close_to(float a, float b)
{
    float diff = a < b ? b - a : a - b;
    return diff <= 1e-4F * (1.0F + (b < 0.0F ? -b : b));
}

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes, const float *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, CPU, ndim, sizes, &desc);
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

/* Direct Metal run: c = a + b, read back, verify exact values. */
static int test_metal_add_direct(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {10, 20, 30, 40, 50, 60};
    float out[6];
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tc = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &tc));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tc, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] + b[i]));
    }

    gd_tensor_release(tc);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

/* CPU<->Metal parity via the P8 harness, plain + broadcast adds. */
static int test_metal_add_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    float a[6] = {1, -2, 3, -4, 5, -6};
    float b[6] = {0.5F, 1.5F, -2.5F, 3.5F, -4.5F, 5.5F};
    float r[3] = {100, 200, 300};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tr = NULL;
    gd_tensor *ab = NULL;
    gd_tensor *bc = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(make_f32(ctx, 1, s3, r, &tr));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &ab));        /* elementwise */
    CHECK_OK(gd_add(ctx, ab, tr, &bc));        /* broadcast [2,3] + [3] */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ab);
    gd_tensor_release(bc);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tr);
    return 0;
}

/* Fallback policy at dispatch: relu has no Metal kernel in M0. */
static int test_metal_fallback(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float a[4] = {-1, 2, -3, 4};
    float out[4];
    gd_tensor *ta = NULL;
    gd_tensor *tr = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 1, s4, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_relu(ctx, ta, &tr));
    CHECK_OK(gd_graph_end(ctx));

    /* NONE: unsupported op on Metal must fail loud and name the op. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    CHECK_TRUE(gd_graph_compile(g, METAL) == GD_ERR_UNSUPPORTED);
    CHECK_TRUE(strstr(gd_last_error(), "relu") != NULL);

    /* CPU_REF: whole-graph fallback runs correctly on CPU. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tr, out, sizeof(out)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] > 0.0F ? a[i] : 0.0F));
    }
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));

    gd_tensor_release(tr);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
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

    /* Skip gracefully where no Metal backend is registered (no GPU / no lib). */
    if (gd_synchronize(ctx, METAL) != GD_OK) {
        printf("test_metal: skipped (no Metal backend)\n");
        gd_context_destroy(ctx);
        return 0;
    }

    rc |= test_metal_add_direct(ctx);
    rc |= test_metal_add_parity(ctx);
    rc |= test_metal_fallback(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal: ok\n");
    }
    return rc;
}
