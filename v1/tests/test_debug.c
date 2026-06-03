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

static int file_contains(const char *path, const char *needle)
{
    FILE *f = fopen(path, "r");
    char buf[512];
    int found = 0;

    if (f == NULL) {
        return 0;
    }
    while (fgets(buf, sizeof(buf), f) != NULL) {
        if (strstr(buf, needle) != NULL) {
            found = 1;
        }
    }
    (void)fclose(f);
    return found;
}

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes, const float *data,
                          gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    int64_t numel = 1;
    int i = 0;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, ndim, sizes, &desc);
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

static int test_scopes_and_dump(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *y = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_f32(ctx, 1, s4, a, &ta));
    CHECK_OK(make_f32(ctx, 1, s4, b, &tb));
    CHECK_OK(gd_tensor_set_requires_grad(ta, true));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_scope_push(ctx, "block0"));
    CHECK_OK(gd_scope_push(ctx, "ffn"));
    CHECK_OK(gd_add(ctx, ta, tb, &y));
    CHECK_OK(gd_tensor_set_name(y, "ffn_out"));
    CHECK_OK(gd_scope_pop(ctx));
    CHECK_OK(gd_scope_pop(ctx));
    CHECK_STATUS(gd_scope_pop(ctx), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_end(ctx));

    CHECK_OK(gd_graph_dump(g, GD_DUMP_TEXT, "build/test_debug_dump.txt"));
    CHECK_TRUE(file_contains("build/test_debug_dump.txt", "scope=block0/ffn"));
    CHECK_TRUE(file_contains("build/test_debug_dump.txt", "name=ffn_out"));
    CHECK_TRUE(file_contains("build/test_debug_dump.txt", "requires_grad=1"));

    gd_tensor_release(y);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

static int test_assert_and_run_until(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t s3[1] = {3};
    float good[3] = {1, 2, 3};
    float bad[3] = {1, 0, 3}; /* 1/0 -> inf via scale of reciprocal not available; use direct */
    float same[3] = {1, 2, 3};
    gd_tensor *t = NULL;
    gd_tensor *t2 = NULL;
    gd_tensor *scaled = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_f32(ctx, 1, s3, good, &t));
    CHECK_OK(make_f32(ctx, 1, s3, same, &t2));

    /* assert_close on equal tensors passes; run_until runs only first node. */
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_scale(ctx, t, 2.0F, &scaled));   /* node 0 */
    CHECK_OK(gd_assert_close(ctx, t, t2, 1e-6F, 0.0F)); /* node 1 */
    CHECK_OK(gd_assert_finite(ctx, scaled));     /* node 2 */
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    {
        float out[3];
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, scaled, out, sizeof(out)));
        CHECK_TRUE(out[0] == 2.0F && out[2] == 6.0F);
    }
    CHECK_OK(gd_debug_print_tensor(ctx, scaled, 3)); /* virtual tensor, post-run */
    CHECK_OK(gd_debug_print_tensor(ctx, t, 2));      /* materialized, truncated */
    /* run only node 0 */
    CHECK_OK(gd_graph_run_until(g, 0));
    gd_tensor_release(scaled);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));

    /* assert_close failing tensors -> run returns error */
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, t2, bad, sizeof(bad)));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_assert_close(ctx, t, t2, 1e-6F, 0.0F));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_STATUS(gd_graph_run(g), GD_ERR_INVALID_STATE);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));

    gd_tensor_release(t);
    gd_tensor_release(t2);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_scopes_and_dump(ctx) != 0 || test_assert_and_run_until(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    printf("debug ok\n");
    return 0;
}
