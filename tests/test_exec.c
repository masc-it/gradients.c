#include "gradients/gradients.h"

#include <math.h>
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

static int close_to(float a, float b)
{
    float diff = a - b;
    if (diff < 0.0F) {
        diff = -diff;
    }
    return diff <= 1e-5F * (1.0F + (b < 0.0F ? -b : b));
}

static gd_status make_f32(gd_context *ctx, int ndim, const int64_t *sizes, const float *data,
                          gd_tensor **out)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, ndim, sizes, &desc);
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

static int test_elementwise_and_unary(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[3] = {10, 20, 30};
    float out[6];
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *add = NULL;
    gd_tensor *relu = NULL;
    gd_tensor *scaled = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int i = 0;
    float neg[6] = {-1, 2, -3, 4, -5, 6};
    gd_tensor *tn = NULL;

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 1, s3, b, &tb));
    CHECK_OK(make_f32(ctx, 2, s23, neg, &tn));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &add));      /* broadcast */
    CHECK_OK(gd_scale(ctx, ta, 2.0F, &scaled));
    CHECK_OK(gd_relu(ctx, tn, &relu));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, add, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] + b[i % 3]));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scaled, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] * 2.0F));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, relu, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], neg[i] > 0 ? neg[i] : 0.0F));
    }

    gd_tensor_release(add);
    gd_tensor_release(scaled);
    gd_tensor_release(relu);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tn);
    return 0;
}

static int test_matmul_linear(gd_context *ctx)
{
    int64_t a2[2] = {2, 3};
    int64_t b2[2] = {3, 2};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {1, 0, 0, 1, 1, 1};
    float out[4];
    float expect[4] = {4, 5, 10, 11};
    int64_t bs[1] = {2};
    float bias[2] = {100, 200};
    float lexpect[4] = {104, 205, 110, 211};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tbias = NULL;
    gd_tensor *mm = NULL;
    gd_tensor *lin = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, a2, a, &ta));
    CHECK_OK(make_f32(ctx, 2, b2, b, &tb));
    CHECK_OK(make_f32(ctx, 1, bs, bias, &tbias));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_matmul(ctx, ta, tb, &mm));
    CHECK_OK(gd_linear(ctx, ta, tb, tbias, &lin));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, mm, out, sizeof(out)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(close_to(out[i], expect[i]));
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, lin, out, sizeof(out)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(close_to(out[i], lexpect[i]));
    }

    gd_tensor_release(mm);
    gd_tensor_release(lin);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tbias);
    return 0;
}

static int test_softmax_sum_ce(gd_context *ctx)
{
    int64_t s22[2] = {2, 2};
    float logits[4] = {1, 1, 0, 2};
    float out[4];
    int32_t targets[2] = {0, 1};
    gd_tensor *tl = NULL;
    gd_tensor *tt = NULL;
    gd_tensor *sm = NULL;
    gd_tensor *sums = NULL;
    gd_tensor *loss = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc tdesc;
    float loss_val = 0.0F;
    float expected_loss = 0.0F;
    float s2[2];
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, s22, logits, &tl));
    CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, (int64_t[]){2}, &tdesc));
    CHECK_OK(gd_tensor_empty(ctx, &tdesc, &tt));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, tt, targets, sizeof(targets)));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_softmax(ctx, tl, 1, &sm));
    CHECK_OK(gd_sum(ctx, tl, 1, false, &sums));
    CHECK_OK(gd_cross_entropy(ctx, tl, tt, 1, &loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, sm, out, sizeof(out)));
    /* row0 equal logits -> 0.5,0.5 */
    CHECK_TRUE(close_to(out[0], 0.5F) && close_to(out[1], 0.5F));
    CHECK_TRUE(close_to(out[2] + out[3], 1.0F));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, sums, s2, sizeof(s2)));
    CHECK_TRUE(close_to(s2[0], 2.0F) && close_to(s2[1], 2.0F));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &loss_val, sizeof(loss_val)));
    {
        /* row0 target0 p=0.5 -> -log0.5; row1 target1 logits 0,2 */
        double row1 = -(2.0 - log(exp(0.0) + exp(2.0)));
        expected_loss = (float)((-log(0.5) + row1) / 2.0);
    }
    CHECK_TRUE(close_to(loss_val, expected_loss));
    (void)i;

    gd_tensor_release(sm);
    gd_tensor_release(sums);
    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tl);
    gd_tensor_release(tt);
    return 0;
}

static int test_sdpa_prefix_cpu(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t xs[4] = {1, 5, 1, 1};
    int64_t bs[4] = {1, 1, 5, 5};
    float qd[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    float kd[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    float vd[5] = {10.0F, 20.0F, 100.0F, 200.0F, 400.0F};
    float bd[25];
    float native[5];
    float dense[5];
    float expect[5] = {15.0F, 15.0F, 130.0F / 3.0F, 82.5F, 146.0F};
    gd_sdpa_config prefix = {0};
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *bias = NULL;
    gd_tensor *yn = NULL;
    gd_tensor *yd = NULL;
    gd_graph *g = NULL;
    int i = 0;
    int j = 0;

    prefix.causal = true;
    prefix.prefix_len = 2;
    for (i = 0; i < 5; ++i) {
        for (j = 0; j < 5; ++j) {
            int allowed = (i < 2) ? (j < 2) : (j <= i);
            bd[i * 5 + j] = allowed ? 0.0F : -1.0e9F;
        }
    }
    CHECK_OK(make_f32(ctx, 4, xs, qd, &q));
    CHECK_OK(make_f32(ctx, 4, xs, kd, &k));
    CHECK_OK(make_f32(ctx, 4, xs, vd, &v));
    CHECK_OK(make_f32(ctx, 4, bs, bd, &bias));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa(ctx, q, k, v, NULL, &prefix, &yn));
    CHECK_OK(gd_sdpa(ctx, q, k, v, bias, NULL, &yd));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, yn, native, sizeof(native)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, yd, dense, sizeof(dense)));
    for (i = 0; i < 5; ++i) {
        CHECK_TRUE(close_to(native[i], expect[i]));
        CHECK_TRUE(close_to(native[i], dense[i]));
    }

    gd_tensor_release(yn);
    gd_tensor_release(yd);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(q);
    gd_tensor_release(k);
    gd_tensor_release(v);
    gd_tensor_release(bias);
    return 0;
}

static int test_reuse_and_materialize(gd_context *ctx)
{
    int64_t s2[1] = {2};
    float a[2] = {1, 2};
    float b[2] = {3, 4};
    float out[2];
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *sum = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};

    CHECK_OK(make_f32(ctx, 1, s2, a, &ta));
    CHECK_OK(make_f32(ctx, 1, s2, b, &tb));

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_add(ctx, ta, tb, &sum));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, sum, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 4.0F) && close_to(out[1], 6.0F));

    /* reuse graph with new input bytes */
    a[0] = 10.0F;
    a[1] = 20.0F;
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, ta, a, sizeof(a)));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, sum, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 13.0F) && close_to(out[1], 24.0F));

    /* materialize detaches from graph, survives destroy */
    CHECK_OK(gd_tensor_materialize(ctx, sum));
    CHECK_TRUE(gd_tensor_storage(sum) != NULL);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, sum, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 13.0F) && close_to(out[1], 24.0F));

    gd_tensor_release(sum);
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    return 0;
}

static int test_virtual_reshape(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t s23[2] = {2, 3};
    int64_t r6[1] = {6};
    int64_t r32[2] = {3, 2};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float out[6];
    gd_tensor *ta = NULL;
    gd_tensor *scaled = NULL;
    gd_tensor *flat = NULL;
    gd_tensor *resh = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_scale(ctx, ta, 2.0F, &scaled));         /* produced [2,3] */
    CHECK_OK(gd_tensor_reshape(scaled, 1, r6, &flat));  /* virtual view -> [6] */
    CHECK_OK(gd_tensor_reshape(scaled, 2, r32, &resh)); /* virtual view -> [3,2] */
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));

    CHECK_TRUE(gd_tensor_ndim(flat) == 1 && gd_tensor_size(flat, 0) == 6);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, flat, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] * 2.0F));
    }
    CHECK_TRUE(gd_tensor_ndim(resh) == 2 && gd_tensor_size(resh, 0) == 3);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, resh, out, sizeof(out)));
    for (i = 0; i < 6; ++i) {
        CHECK_TRUE(close_to(out[i], a[i] * 2.0F));
    }

    gd_tensor_release(flat);
    gd_tensor_release(resh);
    gd_tensor_release(scaled);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

static int test_synchronize_contract(gd_context *ctx)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    /* Vulkan is never registered; Metal may be auto-registered on macOS. */
    gd_device unregistered = {GD_DEVICE_VULKAN, 0};
    int64_t s2[1] = {2};
    float a[2] = {3, 4};
    float out[2];
    gd_tensor *ta = NULL;
    gd_tensor *scaled = NULL;
    gd_graph *g = NULL;

    CHECK_OK(make_f32(ctx, 1, s2, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_scale(ctx, ta, 3.0F, &scaled));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));

    /* run -> explicit synchronize -> read */
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, cpu));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scaled, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 9.0F) && close_to(out[1], 12.0F));

    /* run -> blocking read (download is blocking by contract) */
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scaled, out, sizeof(out)));
    CHECK_TRUE(close_to(out[0], 9.0F) && close_to(out[1], 12.0F));

    /* synchronize on an unregistered backend fails loud */
    {
        gd_status st = gd_synchronize(ctx, unregistered);
        CHECK_TRUE(st == GD_ERR_UNSUPPORTED);
    }

    gd_tensor_release(scaled);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;

    CHECK_OK(gd_context_create(&ctx));
    if (test_elementwise_and_unary(ctx) != 0 || test_matmul_linear(ctx) != 0 ||
        test_softmax_sum_ce(ctx) != 0 || test_sdpa_prefix_cpu(ctx) != 0 ||
        test_reuse_and_materialize(ctx) != 0 || test_virtual_reshape(ctx) != 0 ||
        test_synchronize_contract(ctx) != 0) {
        gd_context_destroy(ctx);
        return 1;
    }
    gd_context_destroy(ctx);
    return 0;
}
