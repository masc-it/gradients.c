#include "gradients/gradients.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK_OK(expr)                                                           \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());       \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_STATUS(expr, expected)                                             \
    do {                                                                         \
        gd_status status_ = (expr);                                              \
        if (status_ != (expected)) {                                             \
            fprintf(stderr, "%s got %s expected %s; last_error=%s\n",         \
                    #expr, gd_status_name(status_), gd_status_name(expected),    \
                    gd_last_error());                                            \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_TRUE(expr)                                                         \
    do {                                                                         \
        if (!(expr)) {                                                           \
            fprintf(stderr, "%s failed at %s:%d\n", #expr, __FILE__, __LINE__); \
            return 1;                                                            \
        }                                                                        \
    } while (0)

static gd_status make_f32(gd_context *ctx,
                          gd_device device,
                          const int64_t *shape,
                          int ndim,
                          const float *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, device, ndim, shape, &desc);
    int64_t n = 1;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        n *= shape[i];
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(float));
}

static gd_status make_i32(gd_context *ctx,
                          gd_device device,
                          const int64_t *shape,
                          int ndim,
                          const int32_t *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_I32, device, ndim, shape, &desc);
    int64_t n = 1;
    int i = 0;

    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_empty(ctx, &desc, out);
    if (status != GD_OK) {
        return status;
    }
    for (i = 0; i < ndim; ++i) {
        n *= shape[i];
    }
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)n * sizeof(int32_t));
}

static int close_to(float a, float b)
{
    return fabsf(a - b) <= 1e-5F * (1.0F + fabsf(b));
}

static int test_runner_rebinds_input(gd_context *ctx, gd_device device)
{
    int64_t shape[1] = {2};
    int64_t bad_shape[1] = {3};
    float a_data[2] = {1.0F, 2.0F};
    float b_data[2] = {-1.0F, 3.0F};
    float bad_data[3] = {0.0F, 0.0F, 0.0F};
    int32_t i32_data[2] = {1, 2};
    float out[2] = {0.0F, 0.0F};
    gd_tensor_desc desc;
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *bad = NULL;
    gd_tensor *i32 = NULL;
    gd_graph_input *x_in = NULL;
    gd_graph *g = NULL;
    gd_graph_runner *runner = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 1, shape, &desc));
    CHECK_OK(make_f32(ctx, device, shape, 1, a_data, &a));
    CHECK_OK(make_f32(ctx, device, shape, 1, b_data, &b));
    CHECK_OK(make_f32(ctx, device, bad_shape, 1, bad_data, &bad));
    CHECK_OK(make_i32(ctx, device, shape, 1, i32_data, &i32));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_graph_add_input(ctx, g, "x", &desc, &x, &x_in));
    CHECK_OK(gd_scale(ctx, x, 2.0F, &y));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(x);
    CHECK_OK(gd_graph_compile(g, device));
    CHECK_STATUS(gd_graph_run(g), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_runner_create(g, &runner));
    CHECK_STATUS(gd_graph_runner_run(runner), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_runner_bind(runner, x_in, i32), GD_ERR_DTYPE);
    CHECK_STATUS(gd_graph_runner_bind(runner, x_in, bad), GD_ERR_SHAPE);

    CHECK_OK(gd_graph_runner_bind(runner, x_in, a));
    CHECK_OK(gd_graph_runner_run(runner));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 2.0F));
    CHECK_TRUE(close_to(out[1], 4.0F));

    CHECK_OK(gd_graph_runner_bind(runner, x_in, b));
    CHECK_OK(gd_graph_runner_run(runner));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], -2.0F));
    CHECK_TRUE(close_to(out[1], 6.0F));

    gd_graph_runner_destroy(runner);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(a);
    gd_tensor_release(b);
    gd_tensor_release(bad);
    gd_tensor_release(i32);
    return 0;
}

static int test_runner_two_inputs(gd_context *ctx, gd_device device)
{
    int64_t shape[1] = {2};
    float a_data[2] = {1.0F, 2.0F};
    float b_data[2] = {10.0F, 20.0F};
    float c_data[2] = {-2.0F, 3.0F};
    float out[2] = {0.0F, 0.0F};
    gd_tensor_desc desc;
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *z = NULL;
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *c = NULL;
    gd_graph_input *x_in = NULL;
    gd_graph_input *y_in = NULL;
    gd_graph *g = NULL;
    gd_graph_runner *runner = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, device, 1, shape, &desc));
    CHECK_OK(make_f32(ctx, device, shape, 1, a_data, &a));
    CHECK_OK(make_f32(ctx, device, shape, 1, b_data, &b));
    CHECK_OK(make_f32(ctx, device, shape, 1, c_data, &c));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_graph_add_input(ctx, g, "x", &desc, &x, &x_in));
    CHECK_OK(gd_graph_add_input(ctx, g, "y", &desc, &y, &y_in));
    CHECK_OK(gd_add(ctx, x, y, &z));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(x);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_compile(g, device));
    CHECK_OK(gd_graph_runner_create(g, &runner));

    CHECK_OK(gd_graph_runner_bind(runner, x_in, a));
    CHECK_STATUS(gd_graph_runner_run(runner), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_runner_bind(runner, y_in, b));
    CHECK_OK(gd_graph_runner_run(runner));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, z, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 11.0F));
    CHECK_TRUE(close_to(out[1], 22.0F));

    CHECK_OK(gd_graph_runner_bind(runner, x_in, c));
    CHECK_OK(gd_graph_runner_run(runner));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, z, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 8.0F));
    CHECK_TRUE(close_to(out[1], 23.0F));

    gd_graph_runner_destroy(runner);
    gd_tensor_release(z);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(a);
    gd_tensor_release(b);
    gd_tensor_release(c);
    return 0;
}

static int test_runner_rejects_wrong_device(gd_context *ctx, gd_device compile_device,
                                            gd_device wrong_device)
{
    int64_t shape[1] = {2};
    float data[2] = {1.0F, 2.0F};
    gd_tensor_desc desc;
    gd_tensor *x = NULL;
    gd_tensor *y = NULL;
    gd_tensor *wrong = NULL;
    gd_graph_input *x_in = NULL;
    gd_graph *g = NULL;
    gd_graph_runner *runner = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, compile_device, 1, shape, &desc));
    CHECK_OK(make_f32(ctx, wrong_device, shape, 1, data, &wrong));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_graph_add_input(ctx, g, "x", &desc, &x, &x_in));
    CHECK_OK(gd_scale(ctx, x, 2.0F, &y));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(x);
    CHECK_OK(gd_graph_compile(g, compile_device));
    CHECK_OK(gd_graph_runner_create(g, &runner));
    CHECK_STATUS(gd_graph_runner_bind(runner, x_in, wrong), GD_ERR_DEVICE);

    gd_graph_runner_destroy(runner);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(wrong);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device metal = {GD_DEVICE_METAL, 0};

    CHECK_OK(gd_context_create(&ctx));
    if (test_runner_rebinds_input(ctx, cpu) != 0 ||
        test_runner_two_inputs(ctx, cpu) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (gd_synchronize(ctx, metal) == GD_OK) {
        if (test_runner_rebinds_input(ctx, metal) != 0 ||
            test_runner_two_inputs(ctx, metal) != 0 ||
            test_runner_rejects_wrong_device(ctx, cpu, metal) != 0) {
            gd_context_destroy(ctx);
            return 1;
        }
    }
    gd_context_destroy(ctx);
    printf("graph runner ok\n");
    return 0;
}
