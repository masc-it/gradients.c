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
            fprintf(stderr, "%s failed\n", #expr);                               \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static const gd_device CPU = {GD_DEVICE_CPU, 0};
#if defined(GD_ENABLE_METAL) && GD_ENABLE_METAL
static const gd_device METAL = {GD_DEVICE_METAL, 0};
#endif

static int close_f(float a, float b, float tol)
{
    return fabsf(a - b) <= tol;
}

static gd_status make_f32(gd_context *ctx, int64_t n, const float *data, gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, CPU, 1, &n, &desc);

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float));
}

static int test_cpu_dropout_forward_backward_and_run_id(gd_context *ctx)
{
    enum { N = 64 };
    float data[N];
    float y0[N];
    float y1[N];
    float gx[N];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *grad = NULL;
    gd_graph *g = NULL;
    int i = 0;
    int kept = 0;
    int dropped = 0;
    int changed = 0;

    for (i = 0; i < N; ++i) {
        data[i] = 1.0F;
    }
    CHECK_OK(make_f32(ctx, N, data, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_dropout(ctx, x, 0.5F, UINT64_C(0x123456789abcdef0), true, &y));
    CHECK_OK(gd_sum(ctx, y, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));

    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, CPU));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, y0, sizeof(y0)));
    CHECK_OK(gd_tensor_grad(x, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, gx, sizeof(gx)));
    for (i = 0; i < N; ++i) {
        CHECK_TRUE(close_f(y0[i], gx[i], 1e-6F));
        if (close_f(y0[i], 0.0F, 1e-6F)) {
            dropped += 1;
        } else {
            CHECK_TRUE(close_f(y0[i], 2.0F, 1e-6F));
            kept += 1;
        }
    }
    CHECK_TRUE(kept > 0 && dropped > 0);

    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, CPU));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, y1, sizeof(y1)));
    for (i = 0; i < N; ++i) {
        if (!close_f(y0[i], y1[i], 1e-6F)) {
            changed += 1;
        }
    }
    CHECK_TRUE(changed > 0);

    (void)grad;
    gd_tensor_release(loss);
    gd_tensor_release(y);
    gd_tensor_release(x);
    CHECK_OK(gd_graph_destroy(g));
    return 0;
}

static int test_cpu_dropout_f16_forward(gd_context *ctx)
{
    enum { N = 64 };
    float data[N];
    float out[N];
    gd_tensor *x = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *yh = NULL;
    gd_tensor *yf = NULL;
    gd_graph *g = NULL;
    int i = 0;
    int kept = 0;
    int dropped = 0;

    for (i = 0; i < N; ++i) {
        data[i] = 1.0F;
    }
    CHECK_OK(make_f32(ctx, N, data, &x));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_dropout(ctx, xh, 0.25F, UINT64_C(0xf00dd00d), true, &yh));
    CHECK_OK(gd_cast(ctx, yh, GD_DTYPE_F32, &yf));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, CPU));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, CPU));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, yf, out, sizeof(out)));
    for (i = 0; i < N; ++i) {
        if (out[i] < 0.1F) {
            CHECK_TRUE(close_f(out[i], 0.0F, 1e-6F));
            dropped += 1;
        } else {
            CHECK_TRUE(close_f(out[i], 1.333F, 2e-3F));
            kept += 1;
        }
    }
    CHECK_TRUE(kept > 0 && dropped > 0);

    gd_tensor_release(yf);
    gd_tensor_release(yh);
    gd_tensor_release(xh);
    gd_tensor_release(x);
    CHECK_OK(gd_graph_destroy(g));
    return 0;
}

#if defined(GD_ENABLE_METAL) && GD_ENABLE_METAL
static int test_metal_dropout_f32_f16_backward_parity(gd_context *ctx)
{
    enum { N = 64 };
    float data[N];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *yh = NULL;
    gd_tensor *yf = NULL;
    gd_tensor *z = NULL;
    gd_tensor *loss = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    int i = 0;

    for (i = 0; i < N; ++i) {
        data[i] = 0.25F + 0.01F * (float)i;
    }
    CHECK_OK(make_f32(ctx, N, data, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_dropout(ctx, x, 0.375F, UINT64_C(0xfeedface1234), true, &y));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_dropout(ctx, xh, 0.25F, UINT64_C(0xdeadbeef5678), true, &yh));
    CHECK_OK(gd_cast(ctx, yh, GD_DTYPE_F32, &yf));
    CHECK_OK(gd_add(ctx, y, yf, &z));
    CHECK_OK(gd_sum(ctx, z, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(loss);
    gd_tensor_release(z);
    gd_tensor_release(yf);
    gd_tensor_release(yh);
    gd_tensor_release(xh);
    gd_tensor_release(y);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    gd_tensor_release(x);
    CHECK_OK(gd_graph_destroy(g));
    return 0;
}
#endif

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    CHECK_OK(gd_context_set_default_device(ctx, CPU));
    if (test_cpu_dropout_forward_backward_and_run_id(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_cpu_dropout_f16_forward(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
#if defined(GD_ENABLE_METAL) && GD_ENABLE_METAL
    if (gd_synchronize(ctx, METAL) == GD_OK) {
        if (test_metal_dropout_f32_f16_backward_parity(ctx) != 0) {
            gd_context_destroy(ctx);
            return 1;
        }
    } else {
        (void)fprintf(stderr, "metal unavailable; skipping dropout metal parity\n");
    }
#endif
    gd_context_destroy(ctx);
    return 0;
}
