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
    return fabsf(a - b) <= 1e-5F;
}

static int arrays_close(const float *got, const float *want, int n)
{
    int i = 0;

    for (i = 0; i < n; ++i) {
        if (!close_to(got[i], want[i])) {
            fprintf(stderr, "mismatch at %d: got=%g want=%g\n",
                    i, (double)got[i], (double)want[i]);
            return 0;
        }
    }
    return 1;
}

static int test_concat_forward_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t a_shape[3] = {2, 2, 3};
    int64_t b_shape[3] = {2, 1, 3};
    int64_t c_shape[3] = {2, 3, 3};
    int64_t y_shape[3] = {2, 6, 3};
    float a_data[12];
    float b_data[6];
    float c_data[18];
    float got[36];
    float want[36];
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *c = NULL;
    gd_tensor *y = NULL;
    gd_tensor *inputs[3];
    gd_graph *g = NULL;
    int i = 0;
    int bi = 0;

    for (i = 0; i < 12; ++i) {
        a_data[i] = (float)i;
    }
    for (i = 0; i < 6; ++i) {
        b_data[i] = (float)(100 + i);
    }
    for (i = 0; i < 18; ++i) {
        c_data[i] = (float)(200 + i);
    }
    for (bi = 0; bi < 2; ++bi) {
        memcpy(&want[(bi * 6 + 0) * 3], &a_data[(bi * 2) * 3], 6U * sizeof(float));
        memcpy(&want[(bi * 6 + 2) * 3], &b_data[bi * 3], 3U * sizeof(float));
        memcpy(&want[(bi * 6 + 3) * 3], &c_data[(bi * 3) * 3], 9U * sizeof(float));
    }

    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, a_shape, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, b_shape, &b));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, c_shape, &c));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, a, a_data, sizeof(a_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, b, b_data, sizeof(b_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, c, c_data, sizeof(c_data)));
    inputs[0] = a;
    inputs[1] = b;
    inputs[2] = c;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_concat(ctx, inputs, 3, 1, &y));
    CHECK_TRUE(check_shape(y, 3, y_shape) == 0);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, got, sizeof(got)));
    CHECK_TRUE(arrays_close(got, want, 36));

    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(c);
    gd_tensor_release(b);
    gd_tensor_release(a);
    return 0;
}

static int test_concat_i32_and_invalid_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t a_shape[2] = {1, 2};
    int64_t b_shape[2] = {2, 2};
    int64_t bad_shape[2] = {2, 3};
    int64_t y_shape[2] = {3, 2};
    int32_t a_data[2] = {1, 2};
    int32_t b_data[4] = {3, 4, 5, 6};
    int32_t got[6];
    int32_t want[6] = {1, 2, 3, 4, 5, 6};
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *bad_shape_t = NULL;
    gd_tensor *bad = NULL;
    gd_tensor *y = NULL;
    gd_tensor *inputs[2];
    gd_graph *g = NULL;

    CHECK_STATUS(gd_concat(ctx, inputs, 0, 0, &bad), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(bad == NULL);

    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, a_shape, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, b_shape, &b));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_I32, 2, bad_shape, &bad_shape_t));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, a, a_data, sizeof(a_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, b, b_data, sizeof(b_data)));
    inputs[0] = a;
    inputs[1] = b;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_concat(ctx, inputs, 2, 0, &y));
    CHECK_TRUE(check_shape(y, 2, y_shape) == 0);
    inputs[1] = bad_shape_t;
    CHECK_STATUS(gd_concat(ctx, inputs, 2, 0, &bad), GD_ERR_SHAPE);
    CHECK_TRUE(bad == NULL);
    inputs[1] = b;
    CHECK_STATUS(gd_concat(ctx, inputs, 2, 2, &bad), GD_ERR_SHAPE);
    CHECK_TRUE(bad == NULL);
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, y, got, sizeof(got)));
    CHECK_TRUE(memcmp(got, want, sizeof(want)) == 0);

    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(bad_shape_t);
    gd_tensor_release(b);
    gd_tensor_release(a);
    return 0;
}

static int test_concat_backward_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t a_shape[3] = {2, 2, 3};
    int64_t b_shape[3] = {2, 1, 3};
    int64_t c_shape[3] = {2, 3, 3};
    float a_data[12];
    float b_data[6];
    float c_data[18];
    float ga[12];
    float gb[6];
    float gc[18];
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *c = NULL;
    gd_tensor *y = NULL;
    gd_tensor *s0 = NULL;
    gd_tensor *s1 = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *a_grad = NULL;
    gd_tensor *b_grad = NULL;
    gd_tensor *c_grad = NULL;
    gd_tensor *inputs[3];
    gd_graph *g = NULL;
    int i = 0;

    for (i = 0; i < 12; ++i) {
        a_data[i] = (float)i * 0.01F;
    }
    for (i = 0; i < 6; ++i) {
        b_data[i] = (float)i * 0.02F;
    }
    for (i = 0; i < 18; ++i) {
        c_data[i] = (float)i * 0.03F;
    }
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, a_shape, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, b_shape, &b));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, c_shape, &c));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, a, a_data, sizeof(a_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, b, b_data, sizeof(b_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, c, c_data, sizeof(c_data)));
    CHECK_OK(gd_tensor_set_requires_grad(a, true));
    CHECK_OK(gd_tensor_set_requires_grad(b, true));
    CHECK_OK(gd_tensor_set_requires_grad(c, true));
    inputs[0] = a;
    inputs[1] = b;
    inputs[2] = c;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_concat(ctx, inputs, 3, -2, &y));
    CHECK_OK(gd_sum(ctx, y, 2, false, &s0));
    CHECK_OK(gd_sum(ctx, s0, 1, false, &s1));
    CHECK_OK(gd_sum(ctx, s1, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_grad(a, &a_grad));
    CHECK_OK(gd_tensor_grad(b, &b_grad));
    CHECK_OK(gd_tensor_grad(c, &c_grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, a_grad, ga, sizeof(ga)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, b_grad, gb, sizeof(gb)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, c_grad, gc, sizeof(gc)));
    for (i = 0; i < 12; ++i) {
        CHECK_TRUE(close_to(ga[i], 1.0F));
    }
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(gb[i], 1.0F));
    }
    for (i = 0; i < 18; ++i) {
        CHECK_TRUE(close_to(gc[i], 1.0F));
    }

    gd_tensor_release(loss);
    gd_tensor_release(s1);
    gd_tensor_release(s0);
    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(c);
    gd_tensor_release(b);
    gd_tensor_release(a);
    return 0;
}

static int test_concat_slice_f16_metal(gd_context *ctx)
{
    gd_device metal = {GD_DEVICE_METAL, 0};
    int64_t a_shape[3] = {2, 2, 3};
    int64_t b_shape[3] = {2, 1, 3};
    int64_t cat_shape[3] = {2, 3, 3};
    int64_t slice_shape[3] = {2, 2, 3};
    float a_data[12];
    float b_data[6];
    float got[12];
    float want[12] = {3, 4, 5, 20, 21, 22, 9, 10, 11, 23, 24, 25};
    float ga[12];
    float gb[6];
    gd_tensor *a = NULL;
    gd_tensor *b = NULL;
    gd_tensor *ah = NULL;
    gd_tensor *bh = NULL;
    gd_tensor *cat = NULL;
    gd_tensor *slice = NULL;
    gd_tensor *slice_f = NULL;
    gd_tensor *s0 = NULL;
    gd_tensor *s1 = NULL;
    gd_tensor *loss = NULL;
    gd_tensor *a_grad = NULL;
    gd_tensor *b_grad = NULL;
    gd_tensor *inputs[2];
    gd_graph *g = NULL;
    int i = 0;

    if (gd_synchronize(ctx, metal) != GD_OK) {
        gd_last_error();
        printf("test_concat: metal skipped\n");
        return 0;
    }
    for (i = 0; i < 12; ++i) {
        a_data[i] = (float)i;
    }
    for (i = 0; i < 6; ++i) {
        b_data[i] = (float)(20 + i);
    }
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, a_shape, &a));
    CHECK_OK(make_tensor(ctx, GD_DTYPE_F32, 3, b_shape, &b));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, a, a_data, sizeof(a_data)));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, b, b_data, sizeof(b_data)));
    CHECK_OK(gd_tensor_set_requires_grad(a, true));
    CHECK_OK(gd_tensor_set_requires_grad(b, true));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, a, GD_DTYPE_F16, &ah));
    CHECK_OK(gd_cast(ctx, b, GD_DTYPE_F16, &bh));
    inputs[0] = ah;
    inputs[1] = bh;
    CHECK_OK(gd_concat(ctx, inputs, 2, 1, &cat));
    CHECK_TRUE(gd_tensor_dtype(cat) == GD_DTYPE_F16);
    CHECK_TRUE(check_shape(cat, 3, cat_shape) == 0);
    CHECK_OK(gd_slice(ctx, cat, 1, 1, 2, &slice));
    CHECK_TRUE(gd_tensor_dtype(slice) == GD_DTYPE_F16);
    CHECK_TRUE(check_shape(slice, 3, slice_shape) == 0);
    CHECK_OK(gd_cast(ctx, slice, GD_DTYPE_F32, &slice_f));
    CHECK_OK(gd_sum(ctx, slice_f, 2, false, &s0));
    CHECK_OK(gd_sum(ctx, s0, 1, false, &s1));
    CHECK_OK(gd_sum(ctx, s1, 0, false, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, metal));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, slice_f, got, sizeof(got)));
    CHECK_TRUE(arrays_close(got, want, 12));
    CHECK_OK(gd_tensor_grad(a, &a_grad));
    CHECK_OK(gd_tensor_grad(b, &b_grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, a_grad, ga, sizeof(ga)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, b_grad, gb, sizeof(gb)));
    for (i = 0; i < 12; ++i) {
        int t = (i / 3) % 2;
        float want_grad = t == 1 ? 1.0F : 0.0F;
        CHECK_TRUE(close_to(ga[i], want_grad));
    }
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(gb[i], 1.0F));
    }

    gd_tensor_release(loss);
    gd_tensor_release(s1);
    gd_tensor_release(s0);
    gd_tensor_release(slice_f);
    gd_tensor_release(slice);
    gd_tensor_release(cat);
    gd_tensor_release(bh);
    gd_tensor_release(ah);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(b);
    gd_tensor_release(a);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_concat_forward_cpu(ctx) != 0 || test_concat_i32_and_invalid_cpu(ctx) != 0 ||
        test_concat_backward_cpu(ctx) != 0 || test_concat_slice_f16_metal(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
