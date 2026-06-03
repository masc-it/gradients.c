#include "gradients/gradients.h"

#include <stdio.h>
#include <string.h>

#include "../src/backends/backend.h"
#include "../src/core/internal.h"
#include "../src/graph/graph_internal.h"
#include "../src/ops/op_impl.h"

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

static int test_check_node_guards(void)
{
    gd_context *ctx = NULL;
    gd_device cpu_dev = {GD_DEVICE_CPU, 0};
    _gd_backend *cpu = NULL;
    _gd_node node = {0};

    CHECK_OK(gd_context_create(&ctx));
    cpu = _gd_context_backend(ctx, cpu_dev);
    CHECK_TRUE(cpu != NULL);

    node.op = _GD_OP_INVALID;
    node.n_inputs = 0;
    node.n_outputs = 0;
    CHECK_STATUS(_gd_backend_check_node(cpu, NULL, &node), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(strstr(gd_last_error(), "unknown op kind") != NULL);

    node.op = _GD_OP_SCALE;
    node.n_inputs = 0;
    node.n_outputs = 1;
    CHECK_STATUS(_gd_backend_check_node(cpu, NULL, &node), GD_ERR_INVALID_ARGUMENT);
    CHECK_TRUE(strstr(gd_last_error(), "arity") != NULL);

    node.op = _GD_OP_BACKWARD;
    node.n_inputs = 1;
    node.n_outputs = 0;
    CHECK_STATUS(_gd_backend_check_node(cpu, NULL, &node), GD_ERR_UNSUPPORTED);
    CHECK_TRUE(strstr(gd_last_error(), "pseudo op") != NULL);

    node.op = _GD_OP_SCALE;
    node.n_inputs = 1;
    node.n_outputs = 1;
    CHECK_OK(_gd_backend_check_node(cpu, NULL, &node));

    gd_context_destroy(ctx);
    return 0;
}

static int test_check_graph_reports_bad_node(void)
{
    gd_context *ctx = NULL;
    gd_device cpu_dev = {GD_DEVICE_CPU, 0};
    _gd_backend *cpu = NULL;
    gd_tensor *input = NULL;
    gd_tensor *output = NULL;
    gd_graph *graph = NULL;
    float values[2] = {1.0F, 2.0F};
    int bad_node = -1;

    CHECK_OK(gd_context_create(&ctx));
    cpu = _gd_context_backend(ctx, cpu_dev);
    CHECK_TRUE(cpu != NULL);

    CHECK_OK(make_f32(ctx, 2, values, &input));
    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_scale(ctx, input, 2.0F, &output));
    CHECK_OK(gd_graph_end(ctx));

    CHECK_OK(_gd_backend_check_graph(cpu, graph, &bad_node));
    CHECK_TRUE(bad_node == -1);

    graph->nodes[0].op = _GD_OP_BACKWARD;
    graph->nodes[0].n_inputs = 1;
    graph->nodes[0].n_outputs = 0;
    CHECK_STATUS(_gd_backend_check_graph(cpu, graph, &bad_node), GD_ERR_UNSUPPORTED);
    CHECK_TRUE(bad_node == 0);
    CHECK_TRUE(strstr(gd_last_error(), "pseudo op") != NULL);

    gd_tensor_release(output);
    CHECK_OK(gd_graph_destroy(graph));
    gd_tensor_release(input);
    gd_context_destroy(ctx);
    return 0;
}

int main(void)
{
    if (test_check_node_guards() != 0) {
        return 1;
    }
    if (test_check_graph_reports_bad_node() != 0) {
        return 1;
    }
    printf("backend contract ok\n");
    return 0;
}
