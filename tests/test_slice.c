#include "gradients/gradients.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
                    #expr, gd_status_name(status_), gd_status_name(expected),      \
                    gd_last_error());                                             \
            return 1;                                                             \
        }                                                                         \
    } while (0)

#define CHECK_TRUE(expr)                                                           \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed at %s:%d\n", #expr, __FILE__, __LINE__);  \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static gd_status make_tensor(gd_context *ctx,
                             gd_dtype dtype,
                             int ndim,
                             const int64_t *sizes,
                             gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(dtype, cpu, ndim, sizes, &desc);

    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_empty(ctx, &desc, out);
}

static int check_shape(gd_tensor *t, int ndim, const int64_t *sizes)
{
    int i = 0;

    CHECK_TRUE(gd_tensor_ndim(t) == ndim);
    for (i = 0; i < ndim; ++i) {
        CHECK_TRUE(gd_tensor_size(t, i) == sizes[i]);
    }
    return 0;
}

static int close_to(float a, float b)
{
    return fabsf(a - b) <= 1e-6F;
}

static int test_slice_forward_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t x_shape[3] = {2, 4, 3};
    int64_t y_shape[3] = {2, 2, 3};
    int64_t z_shape[3] = {2, 1, 3};
    float x_data[24];
    float y_expected[12] = {3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20};
    float z_expected[6] = {0, 1, 2, 12, 13, 14};
    float y_got[12];
    float z_got[6];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *z = NULL;
    gd_tensor *bad = NULL;
    gd_graph *g = NULL;
    int i = 0;

    for (i = 0; i < 24; ++i) {
        x_data[i] = (float)i;
    }
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, x_shape, &x));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, x_data, sizeof(x_data)));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_slice(ctx, x, 1, 1, 2, &y));
    CHECK_TRUE(check_shape(y, 3, y_shape) == 0);
    CHECK_OK(gd_slice(ctx, x, -2, 0, 1, &z));
    CHECK_TRUE(check_shape(z, 3, z_shape) == 0);
    CHECK_STATUS(gd_slice(ctx, x, 1, 3, 2, &bad), GD_ERR_SHAPE);
    CHECK_TRUE(bad == NULL);
    CHECK_STATUS(gd_slice(ctx, x, 4, 0, 1, &bad), GD_ERR_SHAPE);
    CHECK_TRUE(bad == NULL);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, y_got, sizeof(y_got)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, z, z_got, sizeof(z_got)));
    CHECK_TRUE(memcmp(y_got, y_expected, sizeof(y_expected)) == 0);
    CHECK_TRUE(memcmp(z_got, z_expected, sizeof(z_expected)) == 0);

    gd_tensor_release(y);
    gd_tensor_release(z);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

static int test_slice_i32_forward_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t x_shape[2] = {3, 2};
    int64_t y_shape[2] = {2, 2};
    int32_t x_data[6] = {10, 11, 20, 21, 30, 31};
    int32_t expected[4] = {20, 21, 30, 31};
    int32_t got[4];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, x_shape, &x));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, x_data, sizeof(x_data)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_slice(ctx, x, 0, 1, 2, &y));
    CHECK_TRUE(check_shape(y, 2, y_shape) == 0);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, got, sizeof(got)));
    CHECK_TRUE(memcmp(got, expected, sizeof(expected)) == 0);

    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

static int test_slice_backward_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t x_shape[3] = {2, 4, 3};
    float x_data[24];
    float grad[24];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *s0 = NULL;
    gd_tensor *s1 = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *x_grad = NULL;
    gd_graph *g = NULL;
    int b = 0;
    int t = 0;
    int d = 0;

    for (d = 0; d < 24; ++d) {
        x_data[d] = (float)d * 0.01F;
    }
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, x_shape, &x));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_slice(ctx, x, 1, 1, 2, &y));
    CHECK_OK(gd_sum(ctx, y, 2, false, &s0));
    CHECK_OK(gd_sum(ctx, s0, 1, false, &s1));
    CHECK_OK(gd_sum(ctx, s1, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_grad(x, &x_grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, x_grad, grad, sizeof(grad)));

    for (b = 0; b < 2; ++b) {
        for (t = 0; t < 4; ++t) {
            for (d = 0; d < 3; ++d) {
                int idx = (b * 4 + t) * 3 + d;
                float want = (t == 1 || t == 2) ? 1.0F : 0.0F;
                CHECK_TRUE(close_to(grad[idx], want));
            }
        }
    }

    gd_tensor_release(loss);
    gd_tensor_release(s1);
    gd_tensor_release(s0);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

static int test_slice_forward_backward_metal(gd_context *ctx)
{
    gd_device metal = {GD_DEVICE_METAL, 0};
    int64_t x_shape[3] = {2, 4, 3};
    int64_t y_shape[3] = {2, 2, 3};
    float x_data[24];
    float y_expected[12] = {3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20};
    float y_got[12];
    float grad[24];
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *s0 = NULL;
    gd_tensor *s1 = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *x_grad = NULL;
    gd_graph *g = NULL;
    int i = 0;

    if (gd_synchronize(ctx, metal) != GD_OK) {
        gd_last_error();
        printf("test_slice: metal skipped\n");
        return 0;
    }
    for (i = 0; i < 24; ++i) {
        x_data[i] = (float)i;
    }
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, x_shape, &x));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, x, x_data, sizeof(x_data)));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_slice(ctx, x, 1, 1, 2, &y));
    CHECK_TRUE(check_shape(y, 3, y_shape) == 0);
    CHECK_OK(gd_sum(ctx, y, 2, false, &s0));
    CHECK_OK(gd_sum(ctx, s0, 1, false, &s1));
    CHECK_OK(gd_sum(ctx, s1, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, metal));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, y_got, sizeof(y_got)));
    CHECK_TRUE(memcmp(y_got, y_expected, sizeof(y_expected)) == 0);
    CHECK_OK(gd_tensor_grad(x, &x_grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, x_grad, grad, sizeof(grad)));
    for (i = 0; i < 24; ++i) {
        int t = (i / 3) % 4;
        float want = (t == 1 || t == 2) ? 1.0F : 0.0F;
        CHECK_TRUE(close_to(grad[i], want));
    }

    gd_tensor_release(loss);
    gd_tensor_release(s1);
    gd_tensor_release(s0);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_slice_forward_cpu(ctx) != 0 || test_slice_i32_forward_cpu(ctx) != 0 ||
        test_slice_backward_cpu(ctx) != 0 || test_slice_forward_backward_metal(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
