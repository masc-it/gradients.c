#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

#include "../src/core/tensor_internal.h"

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
                    #expr,                                                        \
                    gd_status_name(status_),                                      \
                    gd_status_name(expected),                                     \
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

static gd_status empty_build(gd_context *ctx, void *user)
{
    (void)ctx;
    (void)user;
    return GD_OK;
}

static gd_status failing_build(gd_context *ctx, void *user)
{
    (void)ctx;
    (void)user;
    return GD_ERR_INTERNAL;
}

static int test_lifecycle(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_device metal = {GD_DEVICE_METAL, 0};
    gd_graph *g = NULL;
    gd_graph *g2 = NULL;

    CHECK_STATUS(gd_graph_create(NULL, &g), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_graph_create(ctx, NULL), GD_ERR_INVALID_ARGUMENT);
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_TRUE(g != NULL);

    CHECK_STATUS(gd_graph_run(g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_compile(g, cpu), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_end(ctx), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_validate(g));

    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_STATUS(gd_graph_begin(ctx, g), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_create(ctx, &g2));
    CHECK_STATUS(gd_graph_begin(ctx, g2), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_validate(g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_destroy(g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_reset(g), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_end(ctx));

    CHECK_STATUS(gd_graph_begin(ctx, g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_run(g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_compile(g, metal), GD_ERR_UNSUPPORTED);
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_STATUS(gd_graph_compile(g, cpu), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_run(g));
    CHECK_STATUS(gd_graph_run_until(g, 0), GD_ERR_INVALID_ARGUMENT);

    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_destroy(g));
    CHECK_OK(gd_graph_destroy(g2));
    CHECK_OK(gd_graph_destroy(NULL));
    return 0;
}

static int test_dump_and_immediate(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_graph *g = NULL;
    FILE *file = NULL;
    char buf[256];
    int found_state = 0;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_dump(g, GD_DUMP_TEXT, "build/test_graph_dump.txt"));
    CHECK_STATUS(gd_graph_dump(g, GD_DUMP_JSON, "build/test_graph_dump.json"),
                 GD_ERR_UNSUPPORTED);

    file = fopen("build/test_graph_dump.txt", "r");
    CHECK_TRUE(file != NULL);
    while (fgets(buf, sizeof(buf), file) != NULL) {
        if (strstr(buf, "graph state=finalized") != NULL &&
            strstr(buf, "nodes=0") != NULL) {
            found_state = 1;
        }
    }
    CHECK_TRUE(fclose(file) == 0);
    CHECK_TRUE(found_state != 0);
    CHECK_OK(gd_graph_destroy(g));

    CHECK_OK(gd_graph_run_immediate(ctx, cpu, empty_build, NULL));
    CHECK_STATUS(gd_graph_run_immediate(ctx, cpu, NULL, NULL), GD_ERR_INVALID_ARGUMENT);
    CHECK_STATUS(gd_graph_run_immediate(ctx, cpu, failing_build, NULL), GD_ERR_INTERNAL);
    return 0;
}

static int test_virtual_tensor_lifetime(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_graph *g = NULL;
    gd_tensor_desc desc;
    gd_tensor *virtual_tensor = NULL;
    int64_t sizes[2] = {2, 2};

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 2, sizes, &desc));
    CHECK_OK(_gd_tensor_create_virtual(g, 0, &desc, &virtual_tensor));
    CHECK_TRUE(virtual_tensor != NULL);
    CHECK_TRUE(_gd_tensor_is_virtual(virtual_tensor));
    CHECK_TRUE(_gd_tensor_value_id(virtual_tensor) == 0);
    CHECK_TRUE(_gd_tensor_graph(virtual_tensor) == g);
    CHECK_TRUE(gd_tensor_storage(virtual_tensor) == NULL);
    CHECK_STATUS(gd_graph_reset(g), GD_ERR_INVALID_STATE);
    CHECK_STATUS(gd_graph_destroy(g), GD_ERR_INVALID_STATE);
    gd_tensor_release(virtual_tensor);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    return 0;
}

static int test_op_requires_active_graph(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t sizes[1] = {1};
    gd_tensor_desc desc;
    gd_tensor *x = NULL;
    gd_tensor *out = NULL;
    gd_graph *g = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 1, sizes, &desc));
    CHECK_OK(gd_tensor_empty(ctx, &desc, &x));
    CHECK_STATUS(gd_relu(ctx, x, &out), GD_ERR_INVALID_STATE);
    CHECK_TRUE(out == NULL);

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_relu(ctx, x, &out));
    CHECK_TRUE(out != NULL);
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(out);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_lifecycle(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_dump_and_immediate(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_virtual_tensor_lifetime(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    if (test_op_requires_active_graph(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
