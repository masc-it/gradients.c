#include "gradients/gradients.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAX_ELEMS 256

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

typedef gd_status (*build_fn)(gd_context *ctx, void *user, gd_tensor **loss_out);

static int64_t tensor_numel(gd_tensor *t)
{
    int64_t n = 1;
    int i = 0;
    for (i = 0; i < gd_tensor_ndim(t); ++i) {
        n *= gd_tensor_size(t, i);
    }
    return n;
}

/* Finite-difference gradcheck for a scalar loss built by `build`. */
static int gradcheck(gd_context *ctx,
                     build_fn build,
                     void *user,
                     gd_tensor **inputs,
                     int n_inputs,
                     const char *name)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_graph *g = NULL;
    gd_tensor *loss = NULL;
    const float h = 1e-3F;
    int i = 0;

    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(build(ctx, user, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));

    for (i = 0; i < n_inputs; ++i) {
        gd_tensor *grad = NULL;
        float analytic[MAX_ELEMS];
        float orig[MAX_ELEMS];
        int64_t numel = tensor_numel(inputs[i]);
        int64_t j = 0;

        CHECK_TRUE(numel <= MAX_ELEMS);
        CHECK_OK(gd_tensor_grad(inputs[i], &grad));
        CHECK_TRUE(grad != NULL);
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, grad, analytic, (size_t)numel * sizeof(float)));
        CHECK_OK(gd_tensor_copy_to_cpu(ctx, inputs[i], orig, (size_t)numel * sizeof(float)));

        for (j = 0; j < numel; ++j) {
            float lossp = 0.0F;
            float lossm = 0.0F;
            float numeric = 0.0F;
            float diff = 0.0F;
            float tol = 0.0F;
            float saved = orig[j];

            orig[j] = saved + h;
            CHECK_OK(gd_tensor_copy_from_cpu(ctx, inputs[i], orig, (size_t)numel * sizeof(float)));
            CHECK_OK(gd_graph_run(g));
            CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &lossp, sizeof(lossp)));

            orig[j] = saved - h;
            CHECK_OK(gd_tensor_copy_from_cpu(ctx, inputs[i], orig, (size_t)numel * sizeof(float)));
            CHECK_OK(gd_graph_run(g));
            CHECK_OK(gd_tensor_copy_to_cpu(ctx, loss, &lossm, sizeof(lossm)));

            orig[j] = saved;
            CHECK_OK(gd_tensor_copy_from_cpu(ctx, inputs[i], orig, (size_t)numel * sizeof(float)));

            numeric = (lossp - lossm) / (2.0F * h);
            diff = fabsf(numeric - analytic[j]);
            tol = 2e-2F * (1.0F + fabsf(numeric));
            if (diff > tol) {
                fprintf(stderr, "%s input %d elem %lld: analytic=%g numeric=%g\n",
                        name, i, (long long)j, (double)analytic[j], (double)numeric);
                gd_tensor_release(loss);
                (void)gd_graph_reset(g);
                (void)gd_graph_destroy(g);
                return 1;
            }
        }
    }

    gd_tensor_release(loss);
    CHECK_OK(gd_graph_reset(g));
    CHECK_OK(gd_graph_destroy(g));
    return 0;
}

typedef struct two_in {
    gd_tensor *a;
    gd_tensor *b;
} two_in;

static gd_status build_add_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_add(ctx, t->a, t->b, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_add_bcast_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user; /* a:[2,3], b:[3] -> broadcast add -> sum all */
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_add(ctx, t->a, t->b, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, &r); /* [2,3]->[3] */
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out); /* ->scalar */
    gd_tensor_release(r);
    return status;
}

static gd_status build_mul_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_mul(ctx, t->a, t->b, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_square_mean(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user; /* uses only a; y = a*a; loss = mean(y) -> grad 2a/n */
    gd_tensor *y = NULL;
    gd_status status = gd_mul(ctx, t->a, t->a, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_mean(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_scale_relu_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *s = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_scale(ctx, t->a, 1.5F, &s);
    if (status != GD_OK) {
        return status;
    }
    status = gd_relu(ctx, s, &r);
    gd_tensor_release(s);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out);
    gd_tensor_release(r);
    return status;
}

static gd_status build_reshape_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user; /* a:[4]; y=a*a; r=reshape([2,2]); loss=sum sum -> scalar */
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_tensor *s = NULL;
    int64_t shape[2] = {2, 2};
    gd_status status = gd_mul(ctx, t->a, t->a, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_reshape(y, 2, shape, &r);
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, &s); /* [2,2]->[2] */
    gd_tensor_release(r);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, s, 0, false, loss_out);
    gd_tensor_release(s);
    return status;
}

static gd_status build_silu_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_silu(ctx, t->a, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_powlu_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_powlu(ctx, t->a, t->b, 3.0F, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_powlu_m2_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_powlu(ctx, t->a, t->b, 2.0F, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_gelu_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_gelu(ctx, t->a, false, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_gelu_tanh_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_status status = gd_gelu(ctx, t->a, true, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, loss_out);
    gd_tensor_release(y);
    return status;
}

static gd_status build_transpose_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user; /* a:[2,3] -> transpose [3,2] -> gelu -> sum -> scalar */
    gd_tensor *tr = NULL;
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    int perm[2] = {1, 0};
    gd_status status = gd_transpose(ctx, t->a, perm, 2, &tr);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gelu(ctx, tr, false, &y); /* nonlinearity: grad not constant */
    gd_tensor_release(tr);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 1, false, &r); /* [3,2]->[3] */
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out);
    gd_tensor_release(r);
    return status;
}

typedef struct emb_in {
    gd_tensor *table;
    gd_tensor *ids;
} emb_in;

typedef struct rope_in {
    gd_tensor *x;
    gd_tensor *pos;
} rope_in;

typedef struct sdpa_in {
    gd_tensor *q;
    gd_tensor *k;
    gd_tensor *v;
    gd_tensor *bias; /* optional additive bias (constant; tests q/k/v grads) */
    bool causal;
    int prefix_len;
} sdpa_in;

static gd_status build_sdpa_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    sdpa_in *s = user; /* q[1,3,2,4], k/v[1,3,1,4] (GQA) -> sdpa -> reshape -> mean */
    gd_sdpa_config cfg = {0};
    gd_tensor *y = NULL;
    gd_tensor *fl = NULL;
    int64_t flat[1] = {24};
    gd_status status = GD_OK;

    cfg.causal = s->causal;
    cfg.prefix_len = s->prefix_len;
    status = gd_sdpa(ctx, s->q, s->k, s->v, s->bias, &cfg, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_reshape(y, 1, flat, &fl);
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_mean(ctx, fl, 0, false, loss_out);
    gd_tensor_release(fl);
    return status;
}

static gd_status build_rope_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    rope_in *r = user; /* x:[1,2,2,4] (grad) -> rope -> gelu -> reshape[16] -> mean */
    gd_tensor *y = NULL;
    gd_tensor *gel = NULL;
    gd_tensor *fl = NULL;
    int64_t flat[1] = {16};
    gd_status status = gd_rope(ctx, r->x, r->pos, NULL, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_gelu(ctx, y, false, &gel);
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_tensor_reshape(gel, 1, flat, &fl);
    gd_tensor_release(gel);
    if (status != GD_OK) {
        return status;
    }
    status = gd_mean(ctx, fl, 0, false, loss_out);
    gd_tensor_release(fl);
    return status;
}

static gd_status build_embedding_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    emb_in *e = user; /* table:[V,d] (grad), ids -> emb[N,d] -> sum -> scalar */
    gd_tensor *emb = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_embedding(ctx, e->table, e->ids, &emb);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, emb, 1, false, &r); /* [N,d]->[N] */
    gd_tensor_release(emb);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out);
    gd_tensor_release(r);
    return status;
}

static gd_status build_rms_norm_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user; /* a:[2,3] (x, grad), b:[3] (weight, grad) -> rms_norm -> mean */
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_rms_norm(ctx, t->a, t->b, 1e-5F, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 1, false, &r); /* [2,3]->[2] */
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out);
    gd_tensor_release(r);
    return status;
}

static gd_status build_matmul_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    two_in *t = user;
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_matmul(ctx, t->a, t->b, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, &r); /* [m,n]->[n] */
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, r, 0, false, loss_out); /* ->scalar */
    gd_tensor_release(r);
    return status;
}

typedef struct mm_cfg {
    gd_tensor *a;
    gd_tensor *b;
    bool trans_a;
    bool trans_b;
} mm_cfg;

static gd_status build_matmul_ex_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    mm_cfg *c = user;
    gd_matmul_desc desc = {c->trans_a, c->trans_b, {GD_DTYPE_F32, GD_DTYPE_F32}};
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_matmul_ex(ctx, &desc, c->a, c->b, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, &r);
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    /* reduce remaining dims to scalar */
    while (gd_tensor_ndim(r) > 0) {
        gd_tensor *next = NULL;
        status = gd_sum(ctx, r, 0, false, &next);
        gd_tensor_release(r);
        if (status != GD_OK) {
            return status;
        }
        r = next;
    }
    *loss_out = r;
    return GD_OK;
}

typedef struct lin_cfg {
    gd_tensor *x;
    gd_tensor *w;
    gd_tensor *bias;
    bool trans_w;
} lin_cfg;

static gd_status build_linear_sum(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    lin_cfg *c = user;
    gd_linear_desc desc = {c->trans_w, {GD_DTYPE_F32, GD_DTYPE_F32}};
    gd_tensor *y = NULL;
    gd_tensor *r = NULL;
    gd_status status = gd_linear_ex(ctx, &desc, c->x, c->w, c->bias, &y);
    if (status != GD_OK) {
        return status;
    }
    status = gd_sum(ctx, y, 0, false, &r);
    gd_tensor_release(y);
    if (status != GD_OK) {
        return status;
    }
    while (gd_tensor_ndim(r) > 0) {
        gd_tensor *next = NULL;
        status = gd_sum(ctx, r, 0, false, &next);
        gd_tensor_release(r);
        if (status != GD_OK) {
            return status;
        }
        r = next;
    }
    *loss_out = r;
    return GD_OK;
}

typedef struct ce_in {
    gd_tensor *logits;
    gd_tensor *targets;
} ce_in;

static gd_status build_ce(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    ce_in *t = user;
    return gd_cross_entropy(ctx, t->logits, t->targets, 1, loss_out);
}

typedef struct lmce_in {
    gd_tensor *hidden;
    gd_tensor *weight;
    gd_tensor *targets;
} lmce_in;

static gd_status build_lmce(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    lmce_in *t = user;
    return gd_lm_cross_entropy(ctx, t->hidden, t->weight, t->targets, loss_out);
}

typedef struct mlp_in {
    gd_tensor *x;
    gd_tensor *w1;
    gd_tensor *w2;
    gd_tensor *targets;
} mlp_in;

static gd_status build_mlp(gd_context *ctx, void *user, gd_tensor **loss_out)
{
    mlp_in *m = user;
    gd_tensor *h = NULL;
    gd_tensor *a = NULL;
    gd_tensor *logits = NULL;
    gd_status status = gd_linear(ctx, m->x, m->w1, NULL, &h);
    if (status != GD_OK) {
        return status;
    }
    status = gd_relu(ctx, h, &a);
    gd_tensor_release(h);
    if (status != GD_OK) {
        return status;
    }
    status = gd_linear(ctx, a, m->w2, NULL, &logits);
    gd_tensor_release(a);
    if (status != GD_OK) {
        return status;
    }
    status = gd_cross_entropy(ctx, logits, m->targets, 1, loss_out);
    gd_tensor_release(logits);
    return status;
}

static gd_status make_grad_input(gd_context *ctx, int ndim, const int64_t *sizes,
                                 const float *data, gd_tensor **out)
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
    status = gd_tensor_copy_from_cpu(ctx, *out, data, (size_t)numel * sizeof(float));
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_set_requires_grad(*out, true);
}

int main(void)
{
    gd_context *ctx = NULL;
    int64_t s4[1] = {4};
    int64_t a2[2] = {2, 3};
    int64_t b2[2] = {3, 2};
    int64_t l2[2] = {2, 3};
    float da[4] = {-1.0F, 0.5F, 2.0F, -3.0F};
    float db[4] = {0.7F, -1.2F, 0.3F, 1.1F};
    float ma[6] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F};
    float mb[6] = {1.0F, -1.0F, 0.5F, 0.25F, -0.75F, 0.2F};
    float logits[6] = {0.2F, -0.5F, 1.0F, -1.0F, 0.3F, 0.8F};

    CHECK_OK(gd_context_create(&ctx));

    {
        two_in t = {0};
        gd_tensor *in[2];
        CHECK_OK(make_grad_input(ctx, 1, s4, da, &t.a));
        CHECK_OK(make_grad_input(ctx, 1, s4, db, &t.b));
        in[0] = t.a;
        in[1] = t.b;
        CHECK_TRUE(gradcheck(ctx, build_add_sum, &t, in, 2, "add") == 0);
        CHECK_TRUE(gradcheck(ctx, build_mul_sum, &t, in, 2, "mul") == 0);
        CHECK_TRUE(gradcheck(ctx, build_square_mean, &t, in, 1, "square_mean") == 0);
        CHECK_TRUE(gradcheck(ctx, build_scale_relu_sum, &t, in, 1, "scale_relu") == 0);
        CHECK_TRUE(gradcheck(ctx, build_silu_sum, &t, in, 1, "silu") == 0);
        CHECK_TRUE(gradcheck(ctx, build_powlu_sum, &t, in, 2, "powlu") == 0);
        CHECK_TRUE(gradcheck(ctx, build_powlu_m2_sum, &t, in, 2, "powlu_m2") == 0);
        CHECK_TRUE(gradcheck(ctx, build_gelu_sum, &t, in, 1, "gelu") == 0);
        CHECK_TRUE(gradcheck(ctx, build_gelu_tanh_sum, &t, in, 1, "gelu_tanh") == 0);
        CHECK_TRUE(gradcheck(ctx, build_reshape_sum, &t, in, 1, "reshape") == 0);
        gd_tensor_release(t.a);
        gd_tensor_release(t.b);
    }

    {
        /* transpose: a[2,3] -> [3,2] */
        two_in t = {0};
        gd_tensor *in[1];
        int64_t s23[2] = {2, 3};
        float adata[6] = {0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F};
        CHECK_OK(make_grad_input(ctx, 2, s23, adata, &t.a));
        in[0] = t.a;
        CHECK_TRUE(gradcheck(ctx, build_transpose_sum, &t, in, 1, "transpose") == 0);
        gd_tensor_release(t.a);
    }

    {
        /* rms_norm: x[2,3] (grad) + weight[3] (grad). */
        two_in t = {0};
        gd_tensor *in[2];
        int64_t s23[2] = {2, 3};
        int64_t s3[1] = {3};
        float xd[6] = {0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F};
        float wd[3] = {1.0F, 0.5F, -0.8F};
        CHECK_OK(make_grad_input(ctx, 2, s23, xd, &t.a));
        CHECK_OK(make_grad_input(ctx, 1, s3, wd, &t.b));
        in[0] = t.a;
        in[1] = t.b;
        CHECK_TRUE(gradcheck(ctx, build_rms_norm_sum, &t, in, 2, "rms_norm") == 0);
        gd_tensor_release(t.a);
        gd_tensor_release(t.b);
    }

    {
        /* embedding: table[5,4] (grad) gathered by ids[6] with repeats. */
        emb_in e = {0};
        gd_tensor *in[1];
        gd_device cpu = {GD_DEVICE_CPU, 0};
        gd_tensor_desc idesc;
        int64_t vs[2] = {5, 4};
        int64_t is[1] = {6};
        float tdata[20];
        int32_t ids[6] = {0, 2, 2, 4, 1, 2};
        int k = 0;
        for (k = 0; k < 20; ++k) {
            tdata[k] = 0.1F * (float)(k - 10);
        }
        CHECK_OK(make_grad_input(ctx, 2, vs, tdata, &e.table));
        CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, is, &idesc));
        CHECK_OK(gd_tensor_empty(ctx, &idesc, &e.ids));
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, e.ids, ids, sizeof(ids)));
        in[0] = e.table;
        CHECK_TRUE(gradcheck(ctx, build_embedding_sum, &e, in, 1, "embedding") == 0);
        gd_tensor_release(e.table);
        gd_tensor_release(e.ids);
    }

    {
        /* rope: x[1,2,2,4] (grad) with positions {0,1}. */
        rope_in r = {0};
        gd_tensor *in[1];
        gd_device cpu = {GD_DEVICE_CPU, 0};
        gd_tensor_desc pdesc;
        int64_t xs[4] = {1, 2, 2, 4};
        int64_t ps[2] = {1, 2};
        float xd[16];
        int32_t pid[2] = {0, 1};
        int k = 0;
        for (k = 0; k < 16; ++k) {
            xd[k] = 0.2F * (float)(k - 8);
        }
        CHECK_OK(make_grad_input(ctx, 4, xs, xd, &r.x));
        CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 2, ps, &pdesc));
        CHECK_OK(gd_tensor_empty(ctx, &pdesc, &r.pos));
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, r.pos, pid, sizeof(pid)));
        in[0] = r.x;
        CHECK_TRUE(gradcheck(ctx, build_rope_sum, &r, in, 1, "rope") == 0);
        gd_tensor_release(r.x);
        gd_tensor_release(r.pos);
    }

    {
        /* sdpa with GQA: q[1,3,2,4], k/v[1,3,1,4]; dense and causal. */
        sdpa_in s = {0};
        gd_tensor *in[3];
        int64_t qs[4] = {1, 3, 2, 4};
        int64_t ks[4] = {1, 3, 1, 4};
        float qd[24];
        float kd[12];
        float vd[12];
        int j = 0;
        for (j = 0; j < 24; ++j) {
            qd[j] = 0.15F * (float)((j % 7) - 3);
        }
        for (j = 0; j < 12; ++j) {
            kd[j] = 0.2F * (float)((j % 5) - 2);
            vd[j] = 0.1F * (float)((j % 4) - 1);
        }
        CHECK_OK(make_grad_input(ctx, 4, qs, qd, &s.q));
        CHECK_OK(make_grad_input(ctx, 4, ks, kd, &s.k));
        CHECK_OK(make_grad_input(ctx, 4, ks, vd, &s.v));
        in[0] = s.q;
        in[1] = s.k;
        in[2] = s.v;
        s.causal = false;
        CHECK_TRUE(gradcheck(ctx, build_sdpa_sum, &s, in, 3, "sdpa") == 0);
        s.causal = true;
        CHECK_TRUE(gradcheck(ctx, build_sdpa_sum, &s, in, 3, "sdpa_causal") == 0);
        s.prefix_len = 2;
        CHECK_TRUE(gradcheck(ctx, build_sdpa_sum, &s, in, 3, "sdpa_prefix") == 0);
        s.prefix_len = 0;
        {
            /* additive bias broadcast over [B,Hq,Tq,Tk] = [1,1,3,3]; verifies
             * q/k/v grads are correct in the presence of a bias term. */
            gd_device cpu = {GD_DEVICE_CPU, 0};
            gd_tensor_desc bdesc;
            int64_t bs[4] = {1, 1, 3, 3};
            float bd[9] = {0.0F, -1e9F, -1e9F, 0.0F, 0.0F, -1e9F, 0.2F, -0.1F, 0.3F};
            CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_F32, cpu, 4, bs, &bdesc));
            CHECK_OK(gd_tensor_empty(ctx, &bdesc, &s.bias));
            CHECK_OK(gd_tensor_copy_from_cpu(ctx, s.bias, bd, sizeof(bd)));
            s.causal = false;
            CHECK_TRUE(gradcheck(ctx, build_sdpa_sum, &s, in, 3, "sdpa_bias") == 0);
            gd_tensor_release(s.bias);
            s.bias = NULL;
        }
        gd_tensor_release(s.q);
        gd_tensor_release(s.k);
        gd_tensor_release(s.v);
    }

    {
        two_in t = {0};
        gd_tensor *in[2];
        CHECK_OK(make_grad_input(ctx, 2, a2, ma, &t.a));
        CHECK_OK(make_grad_input(ctx, 2, b2, mb, &t.b));
        in[0] = t.a;
        in[1] = t.b;
        CHECK_TRUE(gradcheck(ctx, build_matmul_sum, &t, in, 2, "matmul") == 0);
        gd_tensor_release(t.a);
        gd_tensor_release(t.b);
    }

    {
        two_in t = {0};
        gd_tensor *in[2];
        int64_t s23[2] = {2, 3};
        int64_t s3[1] = {3};
        float adata[6] = {0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F};
        float bdata[3] = {1.0F, -2.0F, 0.5F};
        CHECK_OK(make_grad_input(ctx, 2, s23, adata, &t.a));
        CHECK_OK(make_grad_input(ctx, 1, s3, bdata, &t.b));
        in[0] = t.a;
        in[1] = t.b;
        CHECK_TRUE(gradcheck(ctx, build_add_bcast_sum, &t, in, 2, "add_broadcast") == 0);
        gd_tensor_release(t.a);
        gd_tensor_release(t.b);
    }

    {
        /* transposed matmul: a[3,2]^T @ b[3,4] -> [2,4] */
        mm_cfg c = {0};
        gd_tensor *in[2];
        int64_t at[2] = {3, 2};
        int64_t bt[2] = {3, 4};
        float ad[6] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F};
        float bd[12];
        int k = 0;
        for (k = 0; k < 12; ++k) {
            bd[k] = 0.1F * (float)(k - 6);
        }
        CHECK_OK(make_grad_input(ctx, 2, at, ad, &c.a));
        CHECK_OK(make_grad_input(ctx, 2, bt, bd, &c.b));
        c.trans_a = true;
        c.trans_b = false;
        in[0] = c.a;
        in[1] = c.b;
        CHECK_TRUE(gradcheck(ctx, build_matmul_ex_sum, &c, in, 2, "matmul_trans_a") == 0);
        gd_tensor_release(c.a);
        gd_tensor_release(c.b);
    }

    {
        /* batched matmul with broadcast: a[2,2,3] @ b[3,4] -> [2,2,4] */
        mm_cfg c = {0};
        gd_tensor *in[2];
        int64_t as[3] = {2, 2, 3};
        int64_t bs[2] = {3, 4};
        float ad[12];
        float bd[12];
        int k = 0;
        for (k = 0; k < 12; ++k) {
            ad[k] = 0.05F * (float)(k - 6);
            bd[k] = 0.1F * (float)(6 - k);
        }
        CHECK_OK(make_grad_input(ctx, 3, as, ad, &c.a));
        CHECK_OK(make_grad_input(ctx, 2, bs, bd, &c.b));
        c.trans_a = false;
        c.trans_b = false;
        in[0] = c.a;
        in[1] = c.b;
        CHECK_TRUE(gradcheck(ctx, build_matmul_ex_sum, &c, in, 2, "matmul_batched_bcast") == 0);
        gd_tensor_release(c.a);
        gd_tensor_release(c.b);
    }

    {
        /* linear with trans_w (tied-LM-head style): x[2,3], w[4,3], bias[4] */
        lin_cfg c = {0};
        gd_tensor *in[3];
        int64_t xs[2] = {2, 3};
        int64_t ws[2] = {4, 3};
        int64_t bs[1] = {4};
        float xd[6] = {0.2F, -0.1F, 0.4F, -0.3F, 0.5F, 0.1F};
        float wd[12];
        float bd[4] = {0.1F, -0.2F, 0.3F, -0.4F};
        int k = 0;
        for (k = 0; k < 12; ++k) {
            wd[k] = 0.08F * (float)(k - 6);
        }
        CHECK_OK(make_grad_input(ctx, 2, xs, xd, &c.x));
        CHECK_OK(make_grad_input(ctx, 2, ws, wd, &c.w));
        CHECK_OK(make_grad_input(ctx, 1, bs, bd, &c.bias));
        c.trans_w = true;
        in[0] = c.x;
        in[1] = c.w;
        in[2] = c.bias;
        CHECK_TRUE(gradcheck(ctx, build_linear_sum, &c, in, 3, "linear_trans_w") == 0);
        gd_tensor_release(c.x);
        gd_tensor_release(c.w);
        gd_tensor_release(c.bias);
    }

    {
        ce_in t = {0};
        gd_tensor *in[1];
        gd_device cpu = {GD_DEVICE_CPU, 0};
        gd_tensor_desc tdesc;
        int32_t targets[2] = {2, 0};
        CHECK_OK(make_grad_input(ctx, 2, l2, logits, &t.logits));
        CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, (int64_t[]){2}, &tdesc));
        CHECK_OK(gd_tensor_empty(ctx, &tdesc, &t.targets));
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, t.targets, targets, sizeof(targets)));
        in[0] = t.logits;
        CHECK_TRUE(gradcheck(ctx, build_ce, &t, in, 1, "cross_entropy") == 0);
        gd_tensor_release(t.logits);
        gd_tensor_release(t.targets);
    }

    {
        lmce_in t = {0};
        gd_tensor *in[2];
        gd_device cpu = {GD_DEVICE_CPU, 0};
        gd_tensor_desc tdesc;
        int64_t hs[2] = {2, 3};
        int64_t ws[2] = {4, 3};
        int32_t targets[2] = {2, 0};
        float hd[6] = {0.1F, -0.2F, 0.3F, 0.4F, -0.5F, 0.6F};
        float wd[12] = {0.05F, -0.03F, 0.07F, 0.11F, -0.13F, 0.17F,
                        -0.19F, 0.23F, -0.29F, 0.31F, -0.37F, 0.41F};
        CHECK_OK(make_grad_input(ctx, 2, hs, hd, &t.hidden));
        CHECK_OK(make_grad_input(ctx, 2, ws, wd, &t.weight));
        CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, (int64_t[]){2}, &tdesc));
        CHECK_OK(gd_tensor_empty(ctx, &tdesc, &t.targets));
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, t.targets, targets, sizeof(targets)));
        in[0] = t.hidden;
        in[1] = t.weight;
        CHECK_TRUE(gradcheck(ctx, build_lmce, &t, in, 2, "lm_cross_entropy") == 0);
        gd_tensor_release(t.hidden);
        gd_tensor_release(t.weight);
        gd_tensor_release(t.targets);
    }

    {
        mlp_in m = {0};
        gd_tensor *in[3];
        gd_device cpu = {GD_DEVICE_CPU, 0};
        gd_tensor_desc tdesc;
        int32_t targets[2] = {1, 0};
        int64_t xs[2] = {2, 3};
        int64_t w1s[2] = {3, 4};
        int64_t w2s[2] = {4, 2};
        float xd[6] = {0.1F, 0.2F, -0.3F, 0.4F, -0.1F, 0.5F};
        float w1[12];
        float w2[8];
        int k = 0;
        for (k = 0; k < 12; ++k) {
            w1[k] = 0.1F * (float)(k - 6);
        }
        for (k = 0; k < 8; ++k) {
            w2[k] = 0.05F * (float)(k - 4);
        }
        CHECK_OK(make_grad_input(ctx, 2, xs, xd, &m.x));
        CHECK_OK(make_grad_input(ctx, 2, w1s, w1, &m.w1));
        CHECK_OK(make_grad_input(ctx, 2, w2s, w2, &m.w2));
        CHECK_OK(gd_tensor_desc_contiguous(GD_DTYPE_I32, cpu, 1, (int64_t[]){2}, &tdesc));
        CHECK_OK(gd_tensor_empty(ctx, &tdesc, &m.targets));
        CHECK_OK(gd_tensor_copy_from_cpu(ctx, m.targets, targets, sizeof(targets)));
        in[0] = m.x;
        in[1] = m.w1;
        in[2] = m.w2;
        CHECK_TRUE(gradcheck(ctx, build_mlp, &m, in, 3, "mlp") == 0);
        gd_tensor_release(m.x);
        gd_tensor_release(m.w1);
        gd_tensor_release(m.w2);
        gd_tensor_release(m.targets);
    }

    gd_context_destroy(ctx);
    printf("autograd gradcheck ok\n");
    return 0;
}
