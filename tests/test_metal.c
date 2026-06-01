#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

static int env_flag_enabled(const char *name)
{
    const char *v = getenv(name);
    return v != NULL && v[0] != '\0' && strcmp(v, "0") != 0 &&
           strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0;
}

static const gd_device CPU = {GD_DEVICE_CPU, 0};
static const gd_device METAL = {GD_DEVICE_METAL, 0};

static int close_to(float a, float b)
{
    float diff = a < b ? b - a : a - b;
    return diff <= 1e-4F * (1.0F + (b < 0.0F ? -b : b));
}

static gd_status make_i32(gd_context *ctx, int ndim, const int64_t *sizes, const int32_t *data,
                          gd_tensor **out);

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

    /* Reusing a compiled Metal graph must restage CPU-backed leaves when their
     * host storage changes. This catches stale shadow-buffer bugs in dirty
     * staging. */
    for (i = 0; i < 6; ++i) {
        a[i] = (float)(100 + i);
    }
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, ta, a, sizeof(a)));
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

/* Elementwise + unary parity: mul (plain + broadcast), scale, relu, silu. */
static int test_metal_unary_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s3[1] = {3};
    float a[6] = {1, -2, 3, -4, 5, -6};
    float b[6] = {0.5F, 1.5F, -2.5F, 3.5F, -4.5F, 5.5F};
    float r[3] = {2, -3, 4};
    gd_tensor *ta = NULL;
    gd_tensor *tb = NULL;
    gd_tensor *tr = NULL;
    gd_tensor *prod = NULL;
    gd_tensor *bcast = NULL;
    gd_tensor *scaled = NULL;
    gd_tensor *act = NULL;
    gd_tensor *gate = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s23, b, &tb));
    CHECK_OK(make_f32(ctx, 1, s3, r, &tr));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_mul(ctx, ta, tb, &prod));      /* elementwise */
    CHECK_OK(gd_mul(ctx, prod, tr, &bcast));   /* broadcast [2,3]*[3] */
    CHECK_OK(gd_scale(ctx, bcast, 0.25F, &scaled));
    CHECK_OK(gd_relu(ctx, scaled, &act));
    CHECK_OK(gd_silu(ctx, ta, &gate));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(prod);
    gd_tensor_release(bcast);
    gd_tensor_release(scaled);
    gd_tensor_release(act);
    gd_tensor_release(gate);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tr);
    return 0;
}

/* Cast parity: F32->I32, F32 identity, F32->F16, and F16->F32. */
static int test_metal_cast_parity(gd_context *ctx)
{
    int64_t s6[1] = {6};
    float a[6] = {1.9F, -2.4F, 3.5F, -4.6F, 0.0F, 1.0F / 3.0F};
    gd_tensor *ta = NULL;
    gd_tensor *ci = NULL;
    gd_tensor *cf = NULL;
    gd_tensor *ch = NULL;
    gd_tensor *chf = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {0.0, 0.0, false};

    CHECK_OK(make_f32(ctx, 1, s6, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, ta, GD_DTYPE_I32, &ci));   /* truncation toward zero */
    CHECK_OK(gd_cast(ctx, ta, GD_DTYPE_F32, &cf));   /* identity */
    CHECK_OK(gd_cast(ctx, ta, GD_DTYPE_F16, &ch));
    CHECK_OK(gd_cast(ctx, ch, GD_DTYPE_F32, &chf));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ci);
    gd_tensor_release(cf);
    gd_tensor_release(ch);
    gd_tensor_release(chf);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

static int test_metal_f16_unsupported_reject(gd_context *ctx)
{
    int64_t s22[2] = {2, 2};
    float data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    gd_tensor *x = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *sum = NULL;
    gd_graph *g = NULL;
    gd_status status = GD_OK;

    CHECK_OK(make_f32(ctx, 2, s22, data, &x));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_sum(ctx, xh, -1, false, &sum));
    CHECK_OK(gd_graph_end(ctx));
    status = gd_graph_compile(g, METAL);
    CHECK_TRUE(status == GD_ERR_UNSUPPORTED);
    gd_tensor_release(sum);
    gd_tensor_release(xh);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(x);
    return 0;
}

static float metal_test_sigmoid(float x)
{
    if (x >= 0.0F) {
        float e = expf(-x);
        return 1.0F / (1.0F + e);
    }
    {
        float e = expf(x);
        return e / (1.0F + e);
    }
}

static float metal_test_powlu(float x1, float x2)
{
    float s = metal_test_sigmoid(x2);
    if (x2 <= 0.0F) {
        return x1 * x2 * s;
    }
    return x1 * powf(x2, 3.0F / (sqrtf(x2) + 1.0F)) * s;
}

static float metal_test_gelu_tanh(float x)
{
    const float c = 0.7978845608028654F;
    float u = c * (x + 0.044715F * x * x * x);
    return 0.5F * x * (1.0F + tanhf(u));
}

static int test_metal_f16_elementwise(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float xdata[4] = {-2.0F, -0.5F, 0.5F, 2.0F};
    float ydata[4] = {0.25F, -1.0F, 3.0F, -0.5F};
    float got_add[4] = {0};
    float got_mul[4] = {0};
    float got_scale[4] = {0};
    float got_relu[4] = {0};
    float got_silu[4] = {0};
    float got_gelu[4] = {0};
    float got_powlu[4] = {0};
    float got_softmax[4] = {0};
    gd_tensor *x = NULL, *y = NULL, *xh = NULL, *yh = NULL;
    gd_tensor *addh = NULL, *mulh = NULL, *scaleh = NULL, *reluh = NULL;
    gd_tensor *siluh = NULL, *geluh = NULL, *powluh = NULL, *softmaxh = NULL;
    gd_tensor *add = NULL, *mul = NULL, *scale = NULL, *relu = NULL;
    gd_tensor *silu = NULL, *gelu = NULL, *powlu = NULL, *softmax = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 1, s4, xdata, &x));
    CHECK_OK(make_f32(ctx, 1, s4, ydata, &y));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cast(ctx, y, GD_DTYPE_F16, &yh));
    CHECK_OK(gd_add(ctx, xh, yh, &addh));
    CHECK_OK(gd_mul(ctx, xh, yh, &mulh));
    CHECK_OK(gd_scale(ctx, xh, 0.5F, &scaleh));
    CHECK_OK(gd_relu(ctx, xh, &reluh));
    CHECK_OK(gd_silu(ctx, xh, &siluh));
    CHECK_OK(gd_gelu(ctx, xh, true, &geluh));
    CHECK_OK(gd_powlu(ctx, xh, yh, 3.0F, &powluh));
    CHECK_OK(gd_softmax(ctx, xh, 0, &softmaxh));
    CHECK_OK(gd_cast(ctx, addh, GD_DTYPE_F32, &add));
    CHECK_OK(gd_cast(ctx, mulh, GD_DTYPE_F32, &mul));
    CHECK_OK(gd_cast(ctx, scaleh, GD_DTYPE_F32, &scale));
    CHECK_OK(gd_cast(ctx, reluh, GD_DTYPE_F32, &relu));
    CHECK_OK(gd_cast(ctx, siluh, GD_DTYPE_F32, &silu));
    CHECK_OK(gd_cast(ctx, geluh, GD_DTYPE_F32, &gelu));
    CHECK_OK(gd_cast(ctx, powluh, GD_DTYPE_F32, &powlu));
    CHECK_OK(gd_cast(ctx, softmaxh, GD_DTYPE_F32, &softmax));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, add, got_add, sizeof(got_add)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, mul, got_mul, sizeof(got_mul)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, scale, got_scale, sizeof(got_scale)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, relu, got_relu, sizeof(got_relu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, silu, got_silu, sizeof(got_silu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, gelu, got_gelu, sizeof(got_gelu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, powlu, got_powlu, sizeof(got_powlu)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, softmax, got_softmax, sizeof(got_softmax)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got_add[i] - (xdata[i] + ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_mul[i] - (xdata[i] * ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_scale[i] - (0.5F * xdata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_relu[i] - (xdata[i] > 0.0F ? xdata[i] : 0.0F)) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_silu[i] - (xdata[i] * metal_test_sigmoid(xdata[i]))) <= 2.0e-2F);
        float sm_sum = expf(xdata[0]) + expf(xdata[1]) + expf(xdata[2]) + expf(xdata[3]);
        CHECK_TRUE(fabsf(got_gelu[i] - metal_test_gelu_tanh(xdata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_powlu[i] - metal_test_powlu(xdata[i], ydata[i])) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_softmax[i] - expf(xdata[i]) / sm_sum) <= 2.0e-2F);
    }
    gd_tensor_release(softmax); gd_tensor_release(powlu); gd_tensor_release(gelu); gd_tensor_release(silu);
    gd_tensor_release(relu); gd_tensor_release(scale); gd_tensor_release(mul); gd_tensor_release(add);
    gd_tensor_release(softmaxh); gd_tensor_release(powluh); gd_tensor_release(geluh); gd_tensor_release(siluh);
    gd_tensor_release(reluh); gd_tensor_release(scaleh); gd_tensor_release(mulh); gd_tensor_release(addh);
    gd_tensor_release(yh); gd_tensor_release(xh); CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(y); gd_tensor_release(x);
    return 0;
}

static int test_metal_f16_embedding_transpose(gd_context *ctx)
{
    int64_t table_shape[2] = {3, 2};
    int64_t ids_shape[1] = {3};
    int64_t matrix_shape[2] = {2, 3};
    float table_data[6] = {1.0F, 1.5F, -2.0F, 0.25F, 3.0F, -4.0F};
    int32_t ids_data[3] = {2, 0, 1};
    float matrix_data[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float weight_data[3] = {1.0F, 0.5F, -1.0F};
    float got_emb[6] = {0};
    float got_tr[6] = {0};
    float got_rms[6] = {0};
    float expect_emb[6] = {3.0F, -4.0F, 1.0F, 1.5F, -2.0F, 0.25F};
    float expect_tr[6] = {1.0F, 4.0F, 2.0F, 5.0F, 3.0F, 6.0F};
    gd_tensor *table = NULL, *ids = NULL, *matrix = NULL, *weight = NULL;
    gd_tensor *table_h = NULL, *matrix_h = NULL, *weight_h = NULL;
    gd_tensor *emb_h = NULL, *tr_h = NULL, *rms_h = NULL;
    gd_tensor *emb = NULL, *tr = NULL, *rms = NULL;
    gd_graph *g = NULL;
    int perm[2] = {1, 0};
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, table_shape, table_data, &table));
    CHECK_OK(make_i32(ctx, 1, ids_shape, ids_data, &ids));
    CHECK_OK(make_f32(ctx, 2, matrix_shape, matrix_data, &matrix));
    CHECK_OK(make_f32(ctx, 1, ids_shape, weight_data, &weight));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, table, GD_DTYPE_F16, &table_h));
    CHECK_OK(gd_cast(ctx, matrix, GD_DTYPE_F16, &matrix_h));
    CHECK_OK(gd_cast(ctx, weight, GD_DTYPE_F16, &weight_h));
    CHECK_OK(gd_embedding(ctx, table_h, ids, &emb_h));
    CHECK_OK(gd_transpose(ctx, matrix_h, perm, 2, &tr_h));
    CHECK_OK(gd_rms_norm(ctx, matrix_h, weight_h, 1.0e-5F, &rms_h));
    CHECK_OK(gd_cast(ctx, emb_h, GD_DTYPE_F32, &emb));
    CHECK_OK(gd_cast(ctx, tr_h, GD_DTYPE_F32, &tr));
    CHECK_OK(gd_cast(ctx, rms_h, GD_DTYPE_F32, &rms));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, emb, got_emb, sizeof(got_emb)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, tr, got_tr, sizeof(got_tr)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, rms, got_rms, sizeof(got_rms)));
    for (i = 0; i < 6; ++i) {
        int row = i / 3;
        int col = i % 3;
        float a = matrix_data[row * 3 + 0];
        float b = matrix_data[row * 3 + 1];
        float c = matrix_data[row * 3 + 2];
        float inv = 1.0F / sqrtf((a * a + b * b + c * c) / 3.0F + 1.0e-5F);
        float expect_rms = matrix_data[i] * inv * weight_data[col];
        CHECK_TRUE(fabsf(got_emb[i] - expect_emb[i]) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_tr[i] - expect_tr[i]) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_rms[i] - expect_rms) <= 2.0e-2F);
    }
    gd_tensor_release(rms); gd_tensor_release(tr); gd_tensor_release(emb);
    gd_tensor_release(rms_h); gd_tensor_release(tr_h); gd_tensor_release(emb_h);
    gd_tensor_release(weight_h); gd_tensor_release(matrix_h); gd_tensor_release(table_h);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(weight); gd_tensor_release(matrix); gd_tensor_release(ids); gd_tensor_release(table);
    return 0;
}

static int test_metal_f16_cross_entropy(gd_context *ctx)
{
    int64_t xshape[2] = {2, 3};
    int64_t tshape[1] = {2};
    float logits_data[6] = {1.0F, 2.0F, 0.0F, -1.0F, 0.5F, 3.0F};
    int32_t targets_data[2] = {1, 2};
    float got = 0.0F;
    gd_tensor *x = NULL, *targets = NULL, *xh = NULL, *loss = NULL;
    gd_graph *g = NULL;
    double expect = 0.0;
    int r = 0;

    CHECK_OK(make_f32(ctx, 2, xshape, logits_data, &x));
    CHECK_OK(make_i32(ctx, 1, tshape, targets_data, &targets));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cross_entropy(ctx, xh, targets, 1, &loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &got, sizeof(got)));
    for (r = 0; r < 2; ++r) {
        float *row = logits_data + r * 3;
        float maxv = fmaxf(row[0], fmaxf(row[1], row[2]));
        double sum = exp((double)row[0] - (double)maxv) +
                     exp((double)row[1] - (double)maxv) +
                     exp((double)row[2] - (double)maxv);
        expect += -((double)row[targets_data[r]] - (double)maxv - log(sum));
    }
    expect *= 0.5;
    CHECK_TRUE(fabsf(got - (float)expect) <= 2.0e-2F);
    gd_tensor_release(loss); gd_tensor_release(xh);
    CHECK_OK(gd_graph_destroy(g)); gd_tensor_release(targets); gd_tensor_release(x);
    return 0;
}

static int test_metal_f16_lm_cross_entropy(gd_context *ctx)
{
    int64_t hshape[2] = {2, 3};
    int64_t wshape[2] = {4, 3};
    int64_t tshape[1] = {2};
    float hdata[6] = {0.5F, -1.0F, 2.0F, -0.25F, 0.75F, 1.5F};
    float wdata[12] = {0.1F, -0.2F, 0.3F, 0.4F, 0.5F, -0.6F,
                       -0.7F, 0.8F, 0.9F, -1.0F, 1.1F, 0.2F};
    int32_t targets_data[2] = {0, 3};
    float got = 0.0F;
    gd_tensor *h = NULL, *w = NULL, *targets = NULL, *hh = NULL, *wh = NULL, *loss = NULL;
    gd_graph *g = NULL;
    double expect = 0.0;
    int n = 0;

    CHECK_OK(make_f32(ctx, 2, hshape, hdata, &h));
    CHECK_OK(make_f32(ctx, 2, wshape, wdata, &w));
    CHECK_OK(make_i32(ctx, 1, tshape, targets_data, &targets));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, h, GD_DTYPE_F16, &hh));
    CHECK_OK(gd_cast(ctx, w, GD_DTYPE_F16, &wh));
    CHECK_OK(gd_lm_cross_entropy(ctx, hh, wh, targets, &loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_TRUE(gd_tensor_dtype(loss) == GD_DTYPE_F32);
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &got, sizeof(got)));
    for (n = 0; n < 2; ++n) {
        double logits[4];
        double maxv = -1.0e30;
        double sum = 0.0;
        int v = 0;
        for (v = 0; v < 4; ++v) {
            logits[v] = (double)hdata[n * 3 + 0] * (double)wdata[v * 3 + 0] +
                        (double)hdata[n * 3 + 1] * (double)wdata[v * 3 + 1] +
                        (double)hdata[n * 3 + 2] * (double)wdata[v * 3 + 2];
            if (logits[v] > maxv) {
                maxv = logits[v];
            }
        }
        for (v = 0; v < 4; ++v) {
            sum += exp(logits[v] - maxv);
        }
        expect += -(logits[targets_data[n]] - maxv - log(sum));
    }
    expect *= 0.5;
    CHECK_TRUE(fabsf(got - (float)expect) <= 3.0e-2F);
    gd_tensor_release(loss); gd_tensor_release(wh); gd_tensor_release(hh);
    CHECK_OK(gd_graph_destroy(g)); gd_tensor_release(targets); gd_tensor_release(w); gd_tensor_release(h);
    return 0;
}

static int test_metal_f16_rope(gd_context *ctx)
{
    int64_t xshape[4] = {1, 2, 1, 4};
    int64_t pshape[2] = {1, 2};
    float xdata[8] = {1.0F, 2.0F, 3.0F, 4.0F, 1.0F, 0.0F, 0.0F, 1.0F};
    int32_t pos_data[2] = {0, 1};
    float got[8] = {0};
    gd_rope_config cfg = {10000.0F, 4, false};
    gd_tensor *x = NULL, *pos = NULL, *xh = NULL, *rh = NULL, *r = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 4, xshape, xdata, &x));
    CHECK_OK(make_i32(ctx, 2, pshape, pos_data, &pos));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_rope(ctx, xh, pos, &cfg, &rh));
    CHECK_OK(gd_cast(ctx, rh, GD_DTYPE_F32, &r));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, r, got, sizeof(got)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got[i] - xdata[i]) <= 2.0e-2F);
    }
    {
        float c0 = cosf(1.0F);
        float s0 = sinf(1.0F);
        float c1 = cosf(0.01F);
        float s1 = sinf(0.01F);
        float expect[4] = {c0, -s1, s0, c1};
        for (i = 0; i < 4; ++i) {
            CHECK_TRUE(fabsf(got[4 + i] - expect[i]) <= 2.0e-2F);
        }
    }
    gd_tensor_release(r); gd_tensor_release(rh); gd_tensor_release(xh);
    CHECK_OK(gd_graph_destroy(g)); gd_tensor_release(pos); gd_tensor_release(x);
    return 0;
}

static int test_metal_f16_mps_gemm(gd_context *ctx)
{
    int64_t xshape[2] = {2, 3};
    int64_t wshape[2] = {3, 2};
    float xdata[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F};
    float wdata[6] = {0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F};
    float expect[4] = {3.0F, 4.0F, -1.5F, 7.125F};
    float got_mm[4] = {0};
    float got_lin[4] = {0};
    gd_tensor *x = NULL;
    gd_tensor *w = NULL;
    gd_tensor *xh = NULL;
    gd_tensor *wh = NULL;
    gd_tensor *mmh = NULL;
    gd_tensor *linh = NULL;
    gd_tensor *mm = NULL;
    gd_tensor *lin = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 2, xshape, xdata, &x));
    CHECK_OK(make_f32(ctx, 2, wshape, wdata, &w));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cast(ctx, x, GD_DTYPE_F16, &xh));
    CHECK_OK(gd_cast(ctx, w, GD_DTYPE_F16, &wh));
    CHECK_OK(gd_matmul(ctx, xh, wh, &mmh));
    CHECK_OK(gd_cast(ctx, mmh, GD_DTYPE_F32, &mm));
    CHECK_OK(gd_linear(ctx, xh, wh, NULL, &linh));
    CHECK_OK(gd_cast(ctx, linh, GD_DTYPE_F32, &lin));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_synchronize(ctx, METAL));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, mm, got_mm, sizeof(got_mm)));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, lin, got_lin, sizeof(got_lin)));
    for (i = 0; i < 4; ++i) {
        CHECK_TRUE(fabsf(got_mm[i] - expect[i]) <= 2.0e-2F);
        CHECK_TRUE(fabsf(got_lin[i] - expect[i]) <= 2.0e-2F);
    }
    gd_tensor_release(lin);
    gd_tensor_release(mm);
    gd_tensor_release(linh);
    gd_tensor_release(mmh);
    gd_tensor_release(wh);
    gd_tensor_release(xh);
    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(w);
    gd_tensor_release(x);
    return 0;
}

static gd_status make_i32(gd_context *ctx, int ndim, const int64_t *sizes, const int32_t *data,
                          gd_tensor **out)
{
    gd_tensor_desc desc;
    gd_status status = gd_tensor_desc_contiguous(GD_DTYPE_I32, CPU, ndim, sizes, &desc);
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
    return gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)numel * sizeof(int32_t));
}

/* matmul parity: plain 2D, trans_b, batched, and batched trans_a. */
static int test_metal_matmul_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s34[2] = {3, 4};
    int64_t s43[2] = {4, 3};
    int64_t a3[3] = {2, 2, 3};
    int64_t at3[3] = {2, 3, 2};
    int64_t b3[3] = {2, 3, 4};
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float bt[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    float ab[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float atb[12] = {1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12};
    float bb[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                    12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    gd_tensor *ta = NULL, *tb = NULL, *tbt = NULL, *tab = NULL, *tatb = NULL;
    gd_tensor *tbb = NULL;
    gd_tensor *mm = NULL, *mmt = NULL, *mmb = NULL, *mmbt = NULL;
    gd_graph *g = NULL;
    gd_matmul_desc td = {false, true, {GD_DTYPE_F32, GD_DTYPE_F32}};
    gd_matmul_desc tad = {true, false, {GD_DTYPE_F32, GD_DTYPE_F32}};
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, a, &ta));
    CHECK_OK(make_f32(ctx, 2, s34, b, &tb));
    CHECK_OK(make_f32(ctx, 2, s43, bt, &tbt));
    CHECK_OK(make_f32(ctx, 3, a3, ab, &tab));
    CHECK_OK(make_f32(ctx, 3, at3, atb, &tatb));
    CHECK_OK(make_f32(ctx, 3, b3, bb, &tbb));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_matmul(ctx, ta, tb, &mm));           /* [2,3]@[3,4] */
    CHECK_OK(gd_matmul_ex(ctx, &td, ta, tbt, &mmt)); /* [2,3]@[4,3]^T */
    CHECK_OK(gd_matmul(ctx, tab, tbb, &mmb));        /* [2,2,3]@[2,3,4] */
    CHECK_OK(gd_matmul_ex(ctx, &tad, tatb, tbb, &mmbt)); /* [2,3,2]^T@[2,3,4] */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(mm);
    gd_tensor_release(mmt);
    gd_tensor_release(mmb);
    gd_tensor_release(mmbt);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    gd_tensor_release(tb);
    gd_tensor_release(tbt);
    gd_tensor_release(tab);
    gd_tensor_release(tatb);
    gd_tensor_release(tbb);
    return 0;
}

/* linear parity: with and without bias. */
static int test_metal_linear_parity(gd_context *ctx)
{
    int64_t s23[2] = {2, 3};
    int64_t s34[2] = {3, 4};
    int64_t s4[1] = {4};
    float x[6] = {1, -2, 3, 4, -5, 6};
    float w[12] = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F,
                   0.7F, 0.8F, 0.9F, 1.0F, 1.1F, 1.2F};
    float bias[4] = {0.5F, -0.5F, 1.0F, -1.0F};
    gd_tensor *tx = NULL, *tw = NULL, *tbias = NULL;
    gd_tensor *ln = NULL, *lnb = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s23, x, &tx));
    CHECK_OK(make_f32(ctx, 2, s34, w, &tw));
    CHECK_OK(make_f32(ctx, 1, s4, bias, &tbias));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_linear(ctx, tx, tw, tbias, &lnb));  /* with bias */
    CHECK_OK(gd_linear(ctx, tx, tw, NULL, &ln));    /* no bias */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ln);
    gd_tensor_release(lnb);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tx);
    gd_tensor_release(tw);
    gd_tensor_release(tbias);
    return 0;
}

/* reductions + softmax + rms_norm parity. */
static int test_metal_reduce_parity(gd_context *ctx)
{
    int64_t s234[3] = {2, 3, 4};
    int64_t s24[2] = {2, 4};
    int64_t s4[1] = {4};
    float x[24];
    float wt[4] = {1.0F, 0.5F, 2.0F, 1.5F};
    float x2[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    gd_tensor *tx = NULL, *tx2 = NULL, *tw = NULL;
    gd_tensor *sm = NULL, *mn = NULL, *soft = NULL, *rms = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    int i = 0;

    for (i = 0; i < 24; ++i) {
        x[i] = (float)(i - 12) * 0.3F;
    }
    CHECK_OK(make_f32(ctx, 3, s234, x, &tx));
    CHECK_OK(make_f32(ctx, 2, s24, x2, &tx2));
    CHECK_OK(make_f32(ctx, 1, s4, wt, &tw));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sum(ctx, tx, 1, false, &sm));     /* reduce middle dim */
    CHECK_OK(gd_mean(ctx, tx, 2, true, &mn));     /* reduce last, keepdim */
    CHECK_OK(gd_softmax(ctx, tx, 2, &soft));      /* softmax last dim */
    CHECK_OK(gd_rms_norm(ctx, tx2, tw, 1e-5F, &rms));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(sm);
    gd_tensor_release(mn);
    gd_tensor_release(soft);
    gd_tensor_release(rms);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tx);
    gd_tensor_release(tx2);
    gd_tensor_release(tw);
    return 0;
}

/* PowLU primitive forward parity across branch points. */
static int test_metal_powlu_parity(gd_context *ctx)
{
    int64_t s8[1] = {8};
    float x1[8] = {1.0F, -0.5F, 0.25F, 2.0F, -1.5F, 0.75F, -2.0F, 1.25F};
    float x2[8] = {-20.0F, -1.0e-4F, 0.0F, 1.0e-8F, 1.0e-4F, 0.5F, 3.0F, 20.0F};
    gd_tensor *tx1 = NULL, *tx2 = NULL, *y = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {2e-4, 2e-4, false};

    CHECK_OK(make_f32(ctx, 1, s8, x1, &tx1));
    CHECK_OK(make_f32(ctx, 1, s8, x2, &tx2));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_powlu(ctx, tx1, tx2, 3.0F, &y));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(y);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tx1);
    gd_tensor_release(tx2);
    return 0;
}

/* cross_entropy parity: scalar mean loss, I32 targets. */
static int test_metal_cross_entropy_parity(gd_context *ctx)
{
    int64_t s35[2] = {3, 5};
    int64_t s3[1] = {3};
    float logits[15] = {2.0F, 1.0F, 0.1F, -1.0F, 0.5F,
                        0.2F, 3.0F, -0.5F, 1.5F, 0.0F,
                        -1.0F, 0.3F, 2.2F, 0.7F, 1.1F};
    int32_t targets[3] = {0, 1, 2};
    gd_tensor *tl = NULL, *tt = NULL, *ce = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};

    CHECK_OK(make_f32(ctx, 2, s35, logits, &tl));
    CHECK_OK(make_i32(ctx, 1, s3, targets, &tt));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_cross_entropy(ctx, tl, tt, 1, &ce));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(ce);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(tl);
    gd_tensor_release(tt);
    return 0;
}

/* fused LM-head cross_entropy parity: loss only. */
static int test_metal_lm_cross_entropy_parity(gd_context *ctx)
{
    int64_t sh[2] = {3, 4};
    int64_t sw[2] = {5, 4};
    int64_t st[1] = {3};
    float hd[12] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F,
                    0.7F, -0.8F, 0.9F, -1.0F, 1.1F, -1.2F};
    float wd[20];
    int32_t td[3] = {0, 3, 4};
    gd_tensor *h = NULL, *w = NULL, *targets = NULL, *loss = NULL;
    gd_graph *g = NULL;
    gd_compare_options opts = {1e-4, 1e-4, false};
    int i = 0;

    for (i = 0; i < 20; ++i) {
        wd[i] = 0.03F * (float)(i - 7);
    }
    CHECK_OK(make_f32(ctx, 2, sh, hd, &h));
    CHECK_OK(make_f32(ctx, 2, sw, wd, &w));
    CHECK_OK(make_i32(ctx, 1, st, td, &targets));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_lm_cross_entropy(ctx, h, w, targets, &loss));
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(loss);

    CHECK_OK(gd_graph_compare(g, CPU, METAL, &opts));

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(h);
    gd_tensor_release(w);
    gd_tensor_release(targets);
    return 0;
}

static int64_t tensor_numel(gd_tensor *t)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < gd_tensor_ndim(t); ++i) {
        n *= gd_tensor_size(t, i);
    }
    return n;
}

/* A backward-graph builder: creates its own (deterministic) input tensors,
 * records the forward ops, and returns the scalar loss. Inputs that require
 * grad are returned in `inputs_out`. */
typedef int (*build_loss_fn)(gd_context *ctx, gd_tensor **inputs_out, int *n_out,
                            gd_tensor **loss_out);

/* Runs build->backward on `dev`, capturing each input's gradient into gbuf. */
static int run_grads(gd_context *ctx, gd_device dev, build_loss_fn build,
                     float gbuf[][64], int64_t numel[], int *n_inputs)
{
    gd_tensor *inputs[4] = {0};
    gd_tensor *loss = NULL;
    gd_graph *g = NULL;
    int n = 0;
    int i = 0;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    if (build(ctx, inputs, &n, &loss) != 0) {
        return 1;
    }
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, dev));
    CHECK_OK(gd_graph_run(g));

    for (i = 0; i < n; ++i) {
        gd_tensor *grad = NULL;
        numel[i] = tensor_numel(inputs[i]);
        CHECK_OK(gd_tensor_grad(inputs[i], &grad));
        CHECK_TRUE(grad != NULL);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, gbuf[i], (size_t)numel[i] * sizeof(float)));
    }
    *n_inputs = n;

    gd_tensor_release(loss);
    CHECK_OK(gd_graph_destroy(g));
    for (i = 0; i < n; ++i) {
        gd_tensor_release(inputs[i]);
    }
    return 0;
}

/* Compares CPU vs Metal gradients for a backward graph builder. */
static int backward_parity(gd_context *ctx, build_loss_fn build, const char *name)
{
    float gcpu[4][64];
    float gmtl[4][64];
    int64_t ncpu[4];
    int64_t nmtl[4];
    int n_cpu = 0;
    int n_mtl = 0;
    int i = 0;
    int64_t j = 0;

    if (run_grads(ctx, CPU, build, gcpu, ncpu, &n_cpu) != 0) {
        fprintf(stderr, "%s: CPU run failed\n", name);
        return 1;
    }
    if (run_grads(ctx, METAL, build, gmtl, nmtl, &n_mtl) != 0) {
        fprintf(stderr, "%s: METAL run failed\n", name);
        return 1;
    }
    CHECK_TRUE(n_cpu == n_mtl);
    for (i = 0; i < n_cpu; ++i) {
        CHECK_TRUE(ncpu[i] == nmtl[i]);
        for (j = 0; j < ncpu[i]; ++j) {
            if (!close_to(gcpu[i][j], gmtl[i][j])) {
                fprintf(stderr, "%s: grad[%d][%lld] cpu=%g metal=%g\n",
                        name, i, (long long)j, (double)gcpu[i][j], (double)gmtl[i][j]);
                return 1;
            }
        }
    }
    return 0;
}

/* matmul -> relu -> sum -> mean: relu_bwd, sum_bwd, mean_bwd, matmul backward. */
static int build_mlp(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[2] = {2, 3};
    int64_t sw[2] = {3, 4};
    float xd[6] = {1, -2, 3, -4, 5, -6};
    float wd[12] = {0.1F, -0.2F, 0.3F, -0.4F, 0.5F, -0.6F,
                    0.7F, -0.8F, 0.9F, -1.0F, 1.1F, -1.2F};
    gd_tensor *x = NULL, *w = NULL, *h = NULL, *a = NULL, *s = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sx, xd, &x));
    CHECK_OK(make_f32(ctx, 2, sw, wd, &w));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_tensor_set_requires_grad(w, true));
    CHECK_OK(gd_matmul(ctx, x, w, &h));
    CHECK_OK(gd_relu(ctx, h, &a));
    CHECK_OK(gd_sum(ctx, a, 1, false, &s));
    CHECK_OK(gd_mean(ctx, s, 0, false, &loss));
    gd_tensor_release(h);
    gd_tensor_release(a);
    gd_tensor_release(s);
    inputs[0] = x;
    inputs[1] = w;
    *n = 2;
    *loss_out = loss;
    return 0;
}

/* softmax -> mul(broadcast) -> sum -> mean: softmax_bwd, reduce_to, sum/mean_bwd. */
static int build_softmax(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[2] = {2, 4};
    int64_t sw[1] = {4};
    float xd[8] = {0.5F, -1.0F, 2.0F, 0.3F, -0.7F, 1.2F, 0.0F, -0.4F};
    float wd[4] = {1.0F, -2.0F, 0.5F, 3.0F};
    gd_tensor *x = NULL, *wt = NULL, *y = NULL, *m = NULL, *s = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sx, xd, &x));
    CHECK_OK(make_f32(ctx, 1, sw, wd, &wt));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_tensor_set_requires_grad(wt, true));
    CHECK_OK(gd_softmax(ctx, x, 1, &y));
    CHECK_OK(gd_mul(ctx, y, wt, &m));
    CHECK_OK(gd_sum(ctx, m, 1, false, &s));
    CHECK_OK(gd_mean(ctx, s, 0, false, &loss));
    gd_tensor_release(y);
    gd_tensor_release(m);
    gd_tensor_release(s);
    inputs[0] = x;
    inputs[1] = wt;
    *n = 2;
    *loss_out = loss;
    return 0;
}

/* silu -> mean: silu_bwd, mean_bwd. */
static int build_silu(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[1] = {6};
    float xd[6] = {1.0F, -2.0F, 0.5F, -0.3F, 2.0F, -1.5F};
    gd_tensor *x = NULL, *a = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 1, sx, xd, &x));
    CHECK_OK(gd_tensor_set_requires_grad(x, true));
    CHECK_OK(gd_silu(ctx, x, &a));
    CHECK_OK(gd_mean(ctx, a, 0, false, &loss));
    gd_tensor_release(a);
    inputs[0] = x;
    *n = 1;
    *loss_out = loss;
    return 0;
}

/* powlu -> mean: powlu_bwd multi-output dx1/dx2. */
static int build_powlu(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sx[1] = {6};
    float x1d[6] = {1.0F, -0.5F, 0.25F, 2.0F, -1.5F, 0.75F};
    float x2d[6] = {-2.0F, -0.25F, 0.3F, 0.7F, 2.0F, 5.0F};
    gd_tensor *x1 = NULL, *x2 = NULL, *y = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 1, sx, x1d, &x1));
    CHECK_OK(make_f32(ctx, 1, sx, x2d, &x2));
    CHECK_OK(gd_tensor_set_requires_grad(x1, true));
    CHECK_OK(gd_tensor_set_requires_grad(x2, true));
    CHECK_OK(gd_powlu(ctx, x1, x2, 3.0F, &y));
    CHECK_OK(gd_mean(ctx, y, 0, false, &loss));
    gd_tensor_release(y);
    inputs[0] = x1;
    inputs[1] = x2;
    *n = 2;
    *loss_out = loss;
    return 0;
}

/* cross_entropy -> backward: cross_entropy_bwd. */
static int build_ce(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sl[2] = {3, 5};
    int64_t st[1] = {3};
    float ld[15] = {2.0F, 1.0F, 0.1F, -1.0F, 0.5F,
                    0.2F, 3.0F, -0.5F, 1.5F, 0.0F,
                    -1.0F, 0.3F, 2.2F, 0.7F, 1.1F};
    int32_t td[3] = {0, 1, 2};
    gd_tensor *logits = NULL, *targets = NULL, *loss = NULL;

    CHECK_OK(make_f32(ctx, 2, sl, ld, &logits));
    CHECK_OK(make_i32(ctx, 1, st, td, &targets));
    CHECK_OK(gd_tensor_set_requires_grad(logits, true));
    CHECK_OK(gd_cross_entropy(ctx, logits, targets, 1, &loss));
    gd_tensor_release(targets); /* graph retains it */
    inputs[0] = logits;
    *n = 1;
    *loss_out = loss;
    return 0;
}

/* lm_cross_entropy -> backward: fused d_hidden,d_weight. */
static int build_lmce(gd_context *ctx, gd_tensor **inputs, int *n, gd_tensor **loss_out)
{
    int64_t sh[2] = {3, 4};
    int64_t sw[2] = {5, 4};
    int64_t st[1] = {3};
    float hd[12] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F,
                    0.7F, -0.8F, 0.9F, -1.0F, 1.1F, -1.2F};
    float wd[20];
    int32_t td[3] = {0, 3, 4};
    gd_tensor *h = NULL, *w = NULL, *targets = NULL, *loss = NULL;
    int i = 0;

    for (i = 0; i < 20; ++i) {
        wd[i] = 0.03F * (float)(i - 7);
    }
    CHECK_OK(make_f32(ctx, 2, sh, hd, &h));
    CHECK_OK(make_f32(ctx, 2, sw, wd, &w));
    CHECK_OK(make_i32(ctx, 1, st, td, &targets));
    CHECK_OK(gd_tensor_set_requires_grad(h, true));
    CHECK_OK(gd_tensor_set_requires_grad(w, true));
    CHECK_OK(gd_lm_cross_entropy(ctx, h, w, targets, &loss));
    gd_tensor_release(targets); /* graph retains it */
    inputs[0] = h;
    inputs[1] = w;
    *n = 2;
    *loss_out = loss;
    return 0;
}

static int test_metal_backward_parity(gd_context *ctx, int mps_enabled)
{
    if (backward_parity(ctx, build_mlp, "mlp") != 0) return 1;
    if (backward_parity(ctx, build_softmax, "softmax") != 0) return 1;
    if (backward_parity(ctx, build_silu, "silu") != 0) return 1;
    if (backward_parity(ctx, build_powlu, "powlu") != 0) return 1;
    if (backward_parity(ctx, build_ce, "cross_entropy") != 0) return 1;
    if (mps_enabled && backward_parity(ctx, build_lmce, "lm_cross_entropy") != 0) return 1;
    return 0;
}

static int test_metal_clip_grad_norm(gd_context *ctx)
{
    float p1_init[2] = {1.0F, 2.0F};
    float p2_init[3] = {3.0F, 4.0F, 5.0F};
    float g1[2] = {3.0F, 4.0F};
    float g2[3] = {0.0F, 12.0F, -5.0F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *p1 = NULL;
    gd_tensor *p2 = NULL;
    gd_tensor *params[2];
    gd_tensor *grad = NULL;
    gd_tensor *norm = NULL;
    gd_graph *graph = NULL;
    float got1[2];
    float got2[3];
    float got_norm = 0.0F;
    float expect_norm = sqrtf(194.0F);
    float scale = 7.0F / (expect_norm + 1e-6F);
    int64_t s2[1] = {2};
    int64_t s3[1] = {3};
    int i = 0;

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;

    CHECK_OK(make_f32(ctx, 1, s2, p1_init, &p1));
    CHECK_OK(make_f32(ctx, 1, s3, p2_init, &p2));
    CHECK_OK(gd_tensor_set_requires_grad(p1, true));
    CHECK_OK(gd_tensor_set_requires_grad(p2, true));
    params[0] = p1;
    params[1] = p2;
    CHECK_OK(gd_adamw_create(ctx, params, 2, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g1, sizeof(g1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g2, sizeof(g2)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_clip_grad_norm(ctx, params, 2, 7.0F, &norm));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, METAL));
    CHECK_OK(gd_graph_run(graph));

    CHECK_OK(gd_tensor_copy_to_cpu(ctx, norm, &got_norm, sizeof(got_norm)));
    CHECK_TRUE(close_to(got_norm, expect_norm));
    CHECK_OK(gd_tensor_grad(p1, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got1, sizeof(got1)));
    CHECK_OK(gd_tensor_grad(p2, &grad));
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, got2, sizeof(got2)));
    for (i = 0; i < 2; ++i) {
        CHECK_TRUE(close_to(got1[i], g1[i] * scale));
    }
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(close_to(got2[i], g2[i] * scale));
    }

    gd_tensor_release(norm);
    norm = NULL;
    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(p2);
    gd_tensor_release(p1);
    return 0;
}

/* AdamW reference (double precision), matching the CPU_REF kernel. */
static void ref_adamw(float *p, float *m, float *v, const float *g, int n, int t,
                      float lr, float b1, float b2, float eps, float wd)
{
    double bc1 = 1.0 - pow((double)b1, (double)t);
    double bc2 = 1.0 - pow((double)b2, (double)t);
    int i = 0;
    for (i = 0; i < n; ++i) {
        double mi = (double)b1 * (double)m[i] + (1.0 - (double)b1) * (double)g[i];
        double vi = (double)b2 * (double)v[i] + (1.0 - (double)b2) * (double)g[i] * (double)g[i];
        double mhat = mi / bc1;
        double vhat = vi / bc2;
        double pp = (double)p[i];
        m[i] = (float)mi;
        v[i] = (float)vi;
        pp -= (double)lr * (double)wd * pp;
        pp -= (double)lr * mhat / (sqrt(vhat) + (double)eps);
        p[i] = (float)pp;
    }
}

/* AdamW on Metal: in-place param/m/v/step updates + write-back over 3 steps. */
static int test_metal_adamw(gd_context *ctx)
{
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.5F, -1.0F, 0.25F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float ref_p[3], ref_m[3] = {0, 0, 0}, ref_v[3] = {0, 0, 0};
    float got[3];
    int step = 0, i = 0;
    int64_t s3[1] = {3};

    cfg.lr = 0.1F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.01F;
    memcpy(ref_p, pinit, sizeof(pinit));

    CHECK_OK(make_f32(ctx, 1, s3, pinit, &param));
    CHECK_OK(gd_tensor_set_requires_grad(param, true));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, METAL));

    for (step = 1; step <= 3; ++step) {
        CHECK_OK(gd_graph_run(graph));
        ref_adamw(ref_p, ref_m, ref_v, g, 3, step, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps,
                  cfg.weight_decay);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
        for (i = 0; i < 3; ++i) {
            CHECK_TRUE(close_to(got[i], ref_p[i]));
        }
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

static int test_metal_adamw_lr_tensor(gd_context *ctx)
{
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.5F, -1.0F, 0.25F};
    float lrs[2] = {0.1F, 0.01F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_tensor *lr = NULL;
    gd_graph *graph = NULL;
    float ref_p[3], ref_m[3] = {0, 0, 0}, ref_v[3] = {0, 0, 0};
    float got[3];
    int step = 0, i = 0;
    int64_t s3[1] = {3};

    cfg.lr = 0.0F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.01F;
    memcpy(ref_p, pinit, sizeof(pinit));

    CHECK_OK(make_f32(ctx, 1, s3, pinit, &param));
    CHECK_OK(make_f32(ctx, 0, NULL, &lrs[0], &lr));
    CHECK_OK(gd_tensor_set_requires_grad(param, true));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step_lr(ctx, opt, lr));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, METAL));

    for (step = 1; step <= 2; ++step) {
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, lr, &lrs[step - 1], sizeof(float)));
        CHECK_OK(gd_graph_run(graph));
        ref_adamw(ref_p, ref_m, ref_v, g, 3, step, lrs[step - 1], cfg.beta1,
                  cfg.beta2, cfg.eps, cfg.weight_decay);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
        for (i = 0; i < 3; ++i) {
            CHECK_TRUE(close_to(got[i], ref_p[i]));
        }
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(lr);
    gd_tensor_release(param);
    return 0;
}

/* P4 lazy writeback: a CPU-backed param updated in place on Metal across many
 * runs with NO intermediate reads must equal the reference, and the single
 * read-back at the end must observe all updates (deferred writeback flushed on
 * read). Pre-P4 this path waited + wrote back every step. */
static int test_metal_lazy_writeback(gd_context *ctx)
{
    float pinit[3] = {1.0F, -2.0F, 0.5F};
    float g[3] = {0.25F, 0.5F, -0.75F};
    gd_adamw_config cfg = {0};
    gd_optimizer *opt = NULL;
    gd_tensor *param = NULL;
    gd_tensor *grad = NULL;
    gd_graph *graph = NULL;
    float ref_p[3], ref_m[3] = {0, 0, 0}, ref_v[3] = {0, 0, 0};
    float got[3];
    const int steps = 25;
    int step = 0, i = 0;
    int64_t s3[1] = {3};

    cfg.lr = 0.05F;
    cfg.beta1 = 0.9F;
    cfg.beta2 = 0.999F;
    cfg.eps = 1e-8F;
    cfg.weight_decay = 0.0F;
    memcpy(ref_p, pinit, sizeof(pinit));

    CHECK_OK(make_f32(ctx, 1, s3, pinit, &param));
    CHECK_OK(gd_tensor_set_requires_grad(param, true));
    CHECK_OK(gd_adamw_create(ctx, &param, 1, &cfg, &opt));
    CHECK_OK(gd_optimizer_zero_grad(ctx, opt));
    CHECK_OK(gd_tensor_grad(param, &grad));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, grad, g, sizeof(g)));

    CHECK_OK(gd_graph_create(ctx, &graph));
    CHECK_OK(gd_graph_begin(ctx, graph));
    CHECK_OK(gd_optimizer_step(ctx, opt));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(graph, METAL));

    /* No reads between steps: writeback must be deferred yet not lost. */
    for (step = 1; step <= steps; ++step) {
        CHECK_OK(gd_graph_run(graph));
        ref_adamw(ref_p, ref_m, ref_v, g, 3, step, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps,
                  cfg.weight_decay);
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, param, got, sizeof(got)));
    for (i = 0; i < 3; ++i) {
        CHECK_TRUE(close_to(got[i], ref_p[i]));
    }

    CHECK_OK(gd_graph_reset(graph));
    CHECK_OK(gd_graph_destroy(graph));
    gd_optimizer_destroy(opt);
    gd_tensor_release(param);
    return 0;
}

/* Fallback policy at dispatch: assert_finite has no Metal kernel (debug op). */
static int test_metal_fallback(gd_context *ctx)
{
    int64_t s4[1] = {4};
    float a[4] = {-1, 2, -3, 4};
    float out[4];
    gd_tensor *ta = NULL;
    gd_tensor *act = NULL;
    gd_graph *g = NULL;
    int i = 0;

    CHECK_OK(make_f32(ctx, 1, s4, a, &ta));
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_relu(ctx, ta, &act));
    CHECK_OK(gd_assert_finite(ctx, act)); /* no Metal kernel -> unsupported */
    CHECK_OK(gd_graph_end(ctx));
    gd_tensor_release(act);

    /* NONE: unsupported op on Metal must fail loud and name the op. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    CHECK_TRUE(gd_graph_compile(g, METAL) == GD_ERR_UNSUPPORTED);
    CHECK_TRUE(strstr(gd_last_error(), "assert_finite") != NULL);

    /* CPU_REF: whole-graph fallback runs correctly on CPU. */
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_CPU_REF));
    CHECK_OK(gd_graph_compile(g, METAL));
    CHECK_OK(gd_graph_run(g));
    CHECK_OK(gd_context_set_fallback_policy(ctx, GD_FALLBACK_NONE));
    (void)out;
    (void)i;

    CHECK_OK(gd_graph_destroy(g));
    gd_tensor_release(ta);
    return 0;
}

int main(void)
{
    gd_context *ctx = NULL;
    int rc = 0;
    int mps_enabled = env_flag_enabled("GD_METAL_MPS");

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
    rc |= test_metal_unary_parity(ctx);
    rc |= test_metal_cast_parity(ctx);
    rc |= test_metal_f16_unsupported_reject(ctx);
    rc |= test_metal_f16_elementwise(ctx);
    rc |= test_metal_f16_embedding_transpose(ctx);
    rc |= test_metal_f16_cross_entropy(ctx);
    rc |= test_metal_f16_rope(ctx);
    rc |= test_metal_matmul_parity(ctx);
    rc |= test_metal_linear_parity(ctx);
    if (mps_enabled) {
        rc |= test_metal_f16_mps_gemm(ctx);
        rc |= test_metal_f16_lm_cross_entropy(ctx);
    }
    rc |= test_metal_reduce_parity(ctx);
    rc |= test_metal_powlu_parity(ctx);
    rc |= test_metal_cross_entropy_parity(ctx);
    if (mps_enabled) {
        rc |= test_metal_lm_cross_entropy_parity(ctx);
    }
    rc |= test_metal_backward_parity(ctx, mps_enabled);
    rc |= test_metal_clip_grad_norm(ctx);
    rc |= test_metal_adamw(ctx);
    rc |= test_metal_adamw_lr_tensor(ctx);
    rc |= test_metal_lazy_writeback(ctx);
    rc |= test_metal_fallback(ctx);

    gd_context_destroy(ctx);
    if (rc == 0) {
        printf("test_metal: ok\n");
    }
    return rc;
}
