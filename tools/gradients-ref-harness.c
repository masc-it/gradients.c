#include "gradients/gradients.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_OK(expr)                                                            \
    do {                                                                          \
        gd_status status__ = (expr);                                              \
        if (status__ != GD_OK) {                                                  \
            fprintf(stderr, "%s failed: %s\n", #expr, gd_last_error());          \
            return 1;                                                             \
        }                                                                         \
    } while (0)

static int parse_i64(const char *s, int64_t *out_value)
{
    char *end = NULL;
    long long v = 0;

    if (s == NULL || out_value == NULL) {
        return 1;
    }
    errno = 0;
    v = strtoll(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 1;
    }
    *out_value = (int64_t)v;
    return 0;
}

static int parse_int_arg(const char *s, int *out_value)
{
    int64_t v = 0;

    if (parse_i64(s, &v) != 0 || v < (int64_t)INT32_MIN || v > (int64_t)INT32_MAX) {
        return 1;
    }
    *out_value = (int)v;
    return 0;
}

static int parse_float_arg(const char *s, float *out_value)
{
    char *end = NULL;
    float v = 0.0F;

    if (s == NULL || out_value == NULL) {
        return 1;
    }
    errno = 0;
    v = strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) {
        return 1;
    }
    *out_value = v;
    return 0;
}

static int checked_numel(int ndim, const int64_t *sizes, int64_t *out_numel)
{
    int64_t n = 1;
    int i = 0;

    if (ndim < 0 || sizes == NULL || out_numel == NULL) {
        return 1;
    }
    for (i = 0; i < ndim; ++i) {
        if (sizes[i] < 0 || (sizes[i] != 0 && n > INT64_MAX / sizes[i])) {
            return 1;
        }
        n *= sizes[i];
    }
    *out_numel = n;
    return 0;
}

static int checked_nbytes(int ndim, const int64_t *sizes, size_t elem, size_t *out_nbytes)
{
    int64_t n = 0;

    if (out_nbytes == NULL || checked_numel(ndim, sizes, &n) != 0 || n < 0) {
        return 1;
    }
    if ((uint64_t)n > (uint64_t)(SIZE_MAX / elem)) {
        return 1;
    }
    *out_nbytes = (size_t)n * elem;
    return 0;
}

static int join_path(const char *dir, const char *name, char *out_path, size_t cap)
{
    int n = 0;

    if (dir == NULL || name == NULL || out_path == NULL || cap == 0U) {
        return 1;
    }
    n = snprintf(out_path, cap, "%s/%s", dir, name);
    if (n < 0 || (size_t)n >= cap) {
        return 1;
    }
    return 0;
}

static int read_exact(const char *dir, const char *name, void *dst, size_t nbytes)
{
    char path[4096];
    FILE *f = NULL;
    size_t got = 0U;

    if (join_path(dir, name, path, sizeof(path)) != 0) {
        fprintf(stderr, "path too long for %s\n", name);
        return 1;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "open read %s failed\n", path);
        return 1;
    }
    got = fread(dst, 1U, nbytes, f);
    if (got != nbytes || fgetc(f) != EOF) {
        fclose(f);
        fprintf(stderr, "read size mismatch for %s\n", path);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "close read %s failed\n", path);
        return 1;
    }
    return 0;
}

static int write_exact(const char *dir, const char *name, const void *src, size_t nbytes)
{
    char path[4096];
    FILE *f = NULL;
    size_t wrote = 0U;

    if (join_path(dir, name, path, sizeof(path)) != 0) {
        fprintf(stderr, "path too long for %s\n", name);
        return 1;
    }
    f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "open write %s failed\n", path);
        return 1;
    }
    wrote = fwrite(src, 1U, nbytes, f);
    if (wrote != nbytes) {
        fclose(f);
        fprintf(stderr, "write size mismatch for %s\n", path);
        return 1;
    }
    if (fclose(f) != 0) {
        fprintf(stderr, "close write %s failed\n", path);
        return 1;
    }
    return 0;
}

static int make_tensor(gd_context *ctx,
                       gd_dtype dtype,
                       int ndim,
                       const int64_t *sizes,
                       const char *dir,
                       const char *name,
                       int requires_grad,
                       gd_tensor **out_tensor)
{
    gd_device cpu = {GD_DEVICE_CPU, 0};
    gd_tensor_desc desc;
    gd_tensor *t = NULL;
    size_t nbytes = 0U;
    void *buf = NULL;

    CHECK_OK(gd_tensor_desc_contiguous(dtype, cpu, ndim, sizes, &desc));
    CHECK_OK(gd_tensor_desc_nbytes(&desc, &nbytes, NULL));
    buf = malloc(nbytes == 0U ? 1U : nbytes);
    if (buf == NULL) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    if (read_exact(dir, name, buf, nbytes) != 0) {
        free(buf);
        return 1;
    }
    CHECK_OK(gd_tensor_empty(ctx, &desc, &t));
    CHECK_OK(gd_tensor_copy_from_cpu(ctx, t, buf, nbytes));
    free(buf);
    if (requires_grad != 0) {
        CHECK_OK(gd_tensor_set_requires_grad(t, true));
    }
    *out_tensor = t;
    return 0;
}

static int copy_tensor_out(gd_context *ctx,
                           const char *dir,
                           const char *name,
                           gd_tensor *t)
{
    int ndim = 0;
    int64_t sizes[GD_MAX_DIMS];
    size_t nbytes = 0U;
    void *buf = NULL;
    int i = 0;

    ndim = gd_tensor_ndim(t);
    for (i = 0; i < ndim; ++i) {
        sizes[i] = gd_tensor_size(t, i);
    }
    if (checked_nbytes(ndim, sizes, gd_dtype_sizeof(gd_tensor_dtype(t)), &nbytes) != 0) {
        fprintf(stderr, "bad tensor size for %s\n", name);
        return 1;
    }
    buf = malloc(nbytes == 0U ? 1U : nbytes);
    if (buf == NULL) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    CHECK_OK(gd_tensor_copy_to_cpu(ctx, t, buf, nbytes));
    if (write_exact(dir, name, buf, nbytes) != 0) {
        free(buf);
        return 1;
    }
    free(buf);
    return 0;
}

static int copy_grad_out(gd_context *ctx, const char *dir, const char *name, gd_tensor *leaf)
{
    gd_tensor *grad = NULL;

    CHECK_OK(gd_tensor_grad(leaf, &grad));
    if (grad == NULL) {
        fprintf(stderr, "missing grad for %s\n", name);
        return 1;
    }
    return copy_tensor_out(ctx, dir, name, grad);
}

static int dot_sum_loss(gd_context *ctx,
                        gd_tensor *value,
                        gd_tensor *grad_weight,
                        int64_t numel,
                        gd_tensor **loss_out)
{
    int64_t flat[1];
    gd_tensor *vf = NULL;
    gd_tensor *gf = NULL;
    gd_tensor *prod = NULL;
    gd_tensor *flat_prod = NULL;
    gd_status status = GD_OK;

    flat[0] = numel;
    status = gd_cast(ctx, value, GD_DTYPE_F32, &vf);
    if (status == GD_OK) { status = gd_cast(ctx, grad_weight, GD_DTYPE_F32, &gf); }
    if (status == GD_OK) { status = gd_mul(ctx, vf, gf, &prod); }
    if (status == GD_OK) { status = gd_tensor_reshape(prod, 1, flat, &flat_prod); }
    if (status == GD_OK) { status = gd_sum(ctx, flat_prod, 0, false, loss_out); }
    gd_tensor_release(flat_prod);
    gd_tensor_release(prod);
    gd_tensor_release(gf);
    gd_tensor_release(vf);
    if (status != GD_OK) {
        fprintf(stderr, "dot_sum_loss failed: %s\n", gd_last_error());
        return 1;
    }
    return 0;
}

static int add_loss(gd_context *ctx, gd_tensor **accum, gd_tensor *term)
{
    gd_tensor *sum = NULL;

    if (*accum == NULL) {
        CHECK_OK(gd_tensor_retain(term));
        *accum = term;
        return 0;
    }
    CHECK_OK(gd_add(ctx, *accum, term, &sum));
    gd_tensor_release(*accum);
    *accum = sum;
    return 0;
}

static int run_powlu(int argc, char **argv)
{
    gd_context *ctx = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t n = 0;
    int64_t shape[1];
    float m = 0.0F;
    gd_tensor *x1 = NULL;
    gd_tensor *x2 = NULL;
    gd_tensor *go = NULL;
    gd_tensor *y = NULL;
    gd_tensor *loss = NULL;

    if (argc != 5 || parse_i64(argv[3], &n) != 0 || parse_float_arg(argv[4], &m) != 0 || n <= 0) {
        fprintf(stderr, "usage: %s powlu DIR N M\n", argv[0]);
        return 1;
    }
    shape[0] = n;
    CHECK_OK(gd_context_create(&ctx));
    if (make_tensor(ctx, GD_DTYPE_F16, 1, shape, argv[2], "x1.bin", 1, &x1) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 1, shape, argv[2], "x2.bin", 1, &x2) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 1, shape, argv[2], "go.bin", 0, &go) != 0) { return 1; }
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_powlu(ctx, x1, x2, m, &y));
    if (dot_sum_loss(ctx, y, go, n, &loss) != 0) { return 1; }
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    if (copy_tensor_out(ctx, argv[2], "out.bin", y) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_x1.bin", x1) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_x2.bin", x2) != 0) { return 1; }
    gd_tensor_release(loss);
    gd_tensor_release(y);
    gd_graph_destroy(g);
    gd_tensor_release(go);
    gd_tensor_release(x2);
    gd_tensor_release(x1);
    gd_context_destroy(ctx);
    return 0;
}

static int run_lmce(int argc, char **argv)
{
    gd_context *ctx = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t rows = 0;
    int64_t d = 0;
    int64_t vocab = 0;
    int ignore_index = 0;
    int has_ignore = 0;
    int64_t hs[2];
    int64_t ws[2];
    int64_t ts[1];
    gd_lm_cross_entropy_desc desc;
    gd_tensor *hidden = NULL;
    gd_tensor *weight = NULL;
    gd_tensor *targets = NULL;
    gd_tensor *loss = NULL;

    if (argc != 8 || parse_i64(argv[3], &rows) != 0 || parse_i64(argv[4], &d) != 0 ||
        parse_i64(argv[5], &vocab) != 0 || parse_int_arg(argv[6], &ignore_index) != 0 ||
        parse_int_arg(argv[7], &has_ignore) != 0 || rows <= 0 || d <= 0 || vocab <= 0) {
        fprintf(stderr, "usage: %s lmce DIR ROWS D VOCAB IGNORE_INDEX HAS_IGNORE\n", argv[0]);
        return 1;
    }
    hs[0] = rows; hs[1] = d;
    ws[0] = vocab; ws[1] = d;
    ts[0] = rows;
    desc.has_ignore_index = has_ignore != 0;
    desc.ignore_index = ignore_index;
    CHECK_OK(gd_context_create(&ctx));
    if (make_tensor(ctx, GD_DTYPE_F16, 2, hs, argv[2], "hidden.bin", 1, &hidden) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 2, ws, argv[2], "weight.bin", 1, &weight) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_I32, 1, ts, argv[2], "targets.bin", 0, &targets) != 0) { return 1; }
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_lm_cross_entropy_ex(ctx, &desc, hidden, weight, targets, &loss));
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    if (copy_tensor_out(ctx, argv[2], "loss.bin", loss) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_hidden.bin", hidden) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_weight.bin", weight) != 0) { return 1; }
    gd_tensor_release(loss);
    gd_graph_destroy(g);
    gd_tensor_release(targets);
    gd_tensor_release(weight);
    gd_tensor_release(hidden);
    gd_context_destroy(ctx);
    return 0;
}

static int run_rms_norm_qkv(int argc, char **argv)
{
    gd_context *ctx = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t rows = 0;
    int64_t d = 0;
    int64_t qn = 0;
    int64_t kn = 0;
    int64_t vn = 0;
    int64_t xs[2];
    int64_t gs[1];
    int64_t qs[2];
    int64_t ks[2];
    int64_t vs[2];
    float eps = 0.0F;
    gd_tensor *x = NULL;
    gd_tensor *gamma = NULL;
    gd_tensor *wq = NULL;
    gd_tensor *wk = NULL;
    gd_tensor *wv = NULL;
    gd_tensor *go_norm = NULL;
    gd_tensor *go_q = NULL;
    gd_tensor *go_k = NULL;
    gd_tensor *go_v = NULL;
    gd_tensor *norm = NULL;
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *term = NULL;
    gd_tensor *loss = NULL;

    if (argc != 9 || parse_i64(argv[3], &rows) != 0 || parse_i64(argv[4], &d) != 0 ||
        parse_i64(argv[5], &qn) != 0 || parse_i64(argv[6], &kn) != 0 ||
        parse_i64(argv[7], &vn) != 0 || parse_float_arg(argv[8], &eps) != 0 ||
        rows <= 0 || d <= 0 || qn <= 0 || kn <= 0 || vn <= 0) {
        fprintf(stderr, "usage: %s rms_norm_qkv DIR ROWS D Q K V EPS\n", argv[0]);
        return 1;
    }
    xs[0] = rows; xs[1] = d;
    gs[0] = d;
    qs[0] = d; qs[1] = qn;
    ks[0] = d; ks[1] = kn;
    vs[0] = d; vs[1] = vn;
    CHECK_OK(gd_context_create(&ctx));
    if (make_tensor(ctx, GD_DTYPE_F16, 2, xs, argv[2], "x.bin", 1, &x) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 1, gs, argv[2], "gamma.bin", 1, &gamma) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 2, qs, argv[2], "wq.bin", 1, &wq) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 2, ks, argv[2], "wk.bin", 1, &wk) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 2, vs, argv[2], "wv.bin", 1, &wv) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 2, xs, argv[2], "go_norm.bin", 0, &go_norm) != 0) { return 1; }
    xs[1] = qn;
    if (make_tensor(ctx, GD_DTYPE_F16, 2, xs, argv[2], "go_q.bin", 0, &go_q) != 0) { return 1; }
    xs[1] = kn;
    if (make_tensor(ctx, GD_DTYPE_F16, 2, xs, argv[2], "go_k.bin", 0, &go_k) != 0) { return 1; }
    xs[1] = vn;
    if (make_tensor(ctx, GD_DTYPE_F16, 2, xs, argv[2], "go_v.bin", 0, &go_v) != 0) { return 1; }
    xs[1] = d;
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_rms_norm_qkv(ctx, x, gamma, wq, wk, wv, eps, &norm, &q, &k, &v));
    if (dot_sum_loss(ctx, norm, go_norm, rows * d, &term) != 0) { return 1; }
    if (add_loss(ctx, &loss, term) != 0) { return 1; }
    gd_tensor_release(term); term = NULL;
    if (dot_sum_loss(ctx, q, go_q, rows * qn, &term) != 0) { return 1; }
    if (add_loss(ctx, &loss, term) != 0) { return 1; }
    gd_tensor_release(term); term = NULL;
    if (dot_sum_loss(ctx, k, go_k, rows * kn, &term) != 0) { return 1; }
    if (add_loss(ctx, &loss, term) != 0) { return 1; }
    gd_tensor_release(term); term = NULL;
    if (dot_sum_loss(ctx, v, go_v, rows * vn, &term) != 0) { return 1; }
    if (add_loss(ctx, &loss, term) != 0) { return 1; }
    gd_tensor_release(term); term = NULL;
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    if (copy_tensor_out(ctx, argv[2], "out_norm.bin", norm) != 0) { return 1; }
    if (copy_tensor_out(ctx, argv[2], "out_q.bin", q) != 0) { return 1; }
    if (copy_tensor_out(ctx, argv[2], "out_k.bin", k) != 0) { return 1; }
    if (copy_tensor_out(ctx, argv[2], "out_v.bin", v) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_x.bin", x) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_gamma.bin", gamma) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_wq.bin", wq) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_wk.bin", wk) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_wv.bin", wv) != 0) { return 1; }
    gd_tensor_release(loss);
    gd_tensor_release(v); gd_tensor_release(k); gd_tensor_release(q); gd_tensor_release(norm);
    gd_graph_destroy(g);
    gd_tensor_release(go_v); gd_tensor_release(go_k); gd_tensor_release(go_q); gd_tensor_release(go_norm);
    gd_tensor_release(wv); gd_tensor_release(wk); gd_tensor_release(wq); gd_tensor_release(gamma); gd_tensor_release(x);
    gd_context_destroy(ctx);
    return 0;
}

static int run_sdpa_varlen(int argc, char **argv)
{
    gd_context *ctx = NULL;
    gd_graph *g = NULL;
    gd_device cpu = {GD_DEVICE_CPU, 0};
    int64_t total = 0;
    int64_t bsz = 0;
    int64_t hq = 0;
    int64_t hkv = 0;
    int64_t dh = 0;
    int causal = 0;
    int window = 0;
    int prefix_len = 0;
    int max_seqlen = 0;
    float scale = 0.0F;
    int64_t qs[3];
    int64_t ks[3];
    int64_t cs[1];
    gd_sdpa_varlen_config cfg;
    gd_tensor *q = NULL;
    gd_tensor *k = NULL;
    gd_tensor *v = NULL;
    gd_tensor *cu = NULL;
    gd_tensor *go = NULL;
    gd_tensor *y = NULL;
    gd_tensor *loss = NULL;

    if (argc != 13 || parse_i64(argv[3], &total) != 0 || parse_i64(argv[4], &bsz) != 0 ||
        parse_i64(argv[5], &hq) != 0 || parse_i64(argv[6], &hkv) != 0 ||
        parse_i64(argv[7], &dh) != 0 || parse_int_arg(argv[8], &causal) != 0 ||
        parse_int_arg(argv[9], &window) != 0 || parse_int_arg(argv[10], &prefix_len) != 0 ||
        parse_int_arg(argv[11], &max_seqlen) != 0 || parse_float_arg(argv[12], &scale) != 0 ||
        total <= 0 || bsz <= 0 || hq <= 0 || hkv <= 0 || dh <= 0) {
        fprintf(stderr, "usage: %s sdpa_varlen DIR TOTAL B HQ HKV DH CAUSAL WINDOW PREFIX MAXSEQ SCALE\n", argv[0]);
        return 1;
    }
    qs[0] = total; qs[1] = hq; qs[2] = dh;
    ks[0] = total; ks[1] = hkv; ks[2] = dh;
    cs[0] = bsz + 1;
    cfg.scale = scale;
    cfg.causal = causal != 0;
    cfg.sliding_window = window;
    cfg.prefix_len = prefix_len;
    cfg.max_seqlen = max_seqlen;
    CHECK_OK(gd_context_create(&ctx));
    if (make_tensor(ctx, GD_DTYPE_F16, 3, qs, argv[2], "q.bin", 1, &q) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 3, ks, argv[2], "k.bin", 1, &k) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 3, ks, argv[2], "v.bin", 1, &v) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_I32, 1, cs, argv[2], "cu.bin", 0, &cu) != 0) { return 1; }
    if (make_tensor(ctx, GD_DTYPE_F16, 3, qs, argv[2], "go.bin", 0, &go) != 0) { return 1; }
    CHECK_OK(gd_graph_create(ctx, &g));
    CHECK_OK(gd_graph_begin(ctx, g));
    CHECK_OK(gd_sdpa_varlen(ctx, q, k, v, cu, &cfg, &y));
    if (dot_sum_loss(ctx, y, go, total * hq * dh, &loss) != 0) { return 1; }
    CHECK_OK(gd_backward(ctx, loss));
    CHECK_OK(gd_graph_end(ctx));
    CHECK_OK(gd_graph_compile(g, cpu));
    CHECK_OK(gd_graph_run(g));
    if (copy_tensor_out(ctx, argv[2], "out.bin", y) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_q.bin", q) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_k.bin", k) != 0) { return 1; }
    if (copy_grad_out(ctx, argv[2], "grad_v.bin", v) != 0) { return 1; }
    gd_tensor_release(loss);
    gd_tensor_release(y);
    gd_graph_destroy(g);
    gd_tensor_release(go); gd_tensor_release(cu); gd_tensor_release(v); gd_tensor_release(k); gd_tensor_release(q);
    gd_context_destroy(ctx);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s OP ...\n", argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "powlu") == 0) {
        return run_powlu(argc, argv);
    }
    if (strcmp(argv[1], "lmce") == 0) {
        return run_lmce(argc, argv);
    }
    if (strcmp(argv[1], "rms_norm_qkv") == 0) {
        return run_rms_norm_qkv(argc, argv);
    }
    if (strcmp(argv[1], "sdpa_varlen") == 0) {
        return run_sdpa_varlen(argc, argv);
    }
    fprintf(stderr, "unknown op %s\n", argv[1]);
    return 1;
}
