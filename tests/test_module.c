#include "gradients/gradients.h"

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

#define CHECK_TRUE(expr)                                                          \
    do {                                                                          \
        if (!(expr)) {                                                            \
            fprintf(stderr, "%s failed\n", #expr);                              \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static gd_status make_param(gd_context *ctx, int64_t n, gd_tensor **out)
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
    return gd_tensor_set_requires_grad(*out, true);
}

static int test_collection_and_recursion(gd_context *ctx)
{
    gd_module *root = NULL;
    gd_module *child = NULL;
    gd_tensor *w0 = NULL;
    gd_tensor *w1 = NULL;
    gd_tensor *w2 = NULL;
    gd_tensor **params = NULL;
    int n = 0;

    CHECK_OK(make_param(ctx, 2, &w0));
    CHECK_OK(make_param(ctx, 3, &w1));
    CHECK_OK(make_param(ctx, 4, &w2));

    CHECK_OK(gd_module_create(ctx, "root", &root));
    CHECK_OK(gd_module_create(ctx, "child", &child));
    CHECK_OK(gd_module_param(root, "w0", w0));
    CHECK_OK(gd_module_child(root, "blk", child));
    CHECK_OK(gd_module_param(child, "w1", w1));
    CHECK_OK(gd_module_param(child, "w2", w2));

    CHECK_OK(gd_module_parameters(root, &params, &n));
    CHECK_TRUE(n == 3);
    CHECK_TRUE(params[0] == w0); /* parent params first */
    CHECK_TRUE(params[1] == w1); /* then child params in order */
    CHECK_TRUE(params[2] == w2);

    /* module retains params; releasing local refs keeps them alive */
    gd_tensor_release(w0);
    gd_tensor_release(w1);
    gd_tensor_release(w2);
    CHECK_OK(gd_module_parameters(root, &params, &n));
    CHECK_TRUE(n == 3);
    CHECK_TRUE(gd_tensor_size(params[2], 0) == 4);

    gd_module_destroy(root); /* destroys child recursively */
    return 0;
}

static int test_tied_dedup(gd_context *ctx)
{
    gd_module *root = NULL;
    gd_module *head = NULL;
    gd_tensor *shared = NULL;
    gd_tensor **params = NULL;
    int n = 0;

    CHECK_OK(make_param(ctx, 5, &shared));

    CHECK_OK(gd_module_create(ctx, "root", &root));
    CHECK_OK(gd_module_create(ctx, "head", &head));
    CHECK_OK(gd_module_param(root, "embed", shared));
    CHECK_OK(gd_module_child(root, "head", head));
    CHECK_OK(gd_module_param(head, "lm_head", shared)); /* weight tying */

    CHECK_OK(gd_module_parameters(root, &params, &n));
    CHECK_TRUE(n == 1); /* deduped by (storage, offset, extent) */
    CHECK_TRUE(params[0] == shared);

    gd_tensor_release(shared);
    gd_module_destroy(root);
    return 0;
}

static int test_zero_grad(gd_context *ctx)
{
    gd_module *m = NULL;
    gd_tensor *w = NULL;
    gd_tensor *grad = NULL;
    float ones[3] = {1.0F, 1.0F, 1.0F};
    float out[3];
    int i = 0;

    CHECK_OK(make_param(ctx, 3, &w));
    CHECK_OK(gd_module_create(ctx, "m", &m));
    CHECK_OK(gd_module_param(m, "w", w));

    CHECK_OK(gd_module_zero_grad(ctx, m));
    CHECK_OK(gd_tensor_grad(w, &grad));
    CHECK_TRUE(grad != NULL);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, out, sizeof(out)));
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(out[i] == 0.0F);
    }

    /* dirty the grad, then zero again */
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, ones, sizeof(ones)));
    CHECK_OK(gd_module_zero_grad(ctx, m));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, out, sizeof(out)));
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(out[i] == 0.0F);
    }

    gd_tensor_release(w);
    gd_module_destroy(m);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_collection_and_recursion(ctx) != 0 || test_tied_dedup(ctx) != 0 ||
        test_zero_grad(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
