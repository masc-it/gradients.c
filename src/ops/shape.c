#include "ops_internal.h"

#include <stdint.h>

#include "../core/internal.h"
#include "../core/tensor_internal.h"

bool _gd_dtype_is_float(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_BF16;
}

bool _gd_dtype_is_integer(gd_dtype dtype)
{
    return dtype == GD_DTYPE_I32 || dtype == GD_DTYPE_I64;
}

static gd_status normalize_dim(int dim, int ndim, int *out)
{
    int normalized = dim;

    if (out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "normalize_dim out is NULL");
    }
    if (ndim <= 0) {
        return _gd_error(GD_ERR_SHAPE, "tensor has no reducible dimensions");
    }
    if (normalized < 0) {
        normalized += ndim;
    }
    if (normalized < 0 || normalized >= ndim) {
        return _gd_error(GD_ERR_SHAPE, "dimension index is out of range");
    }
    *out = normalized;
    return GD_OK;
}

static gd_status broadcast_shapes(const int64_t *a,
                                  int a_ndim,
                                  const int64_t *b,
                                  int b_ndim,
                                  int64_t *out,
                                  int *out_ndim)
{
    int ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    int i = 0;

    if (ndim > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "broadcast result exceeds GD_MAX_DIMS");
    }
    for (i = 0; i < ndim; ++i) {
        int64_t av = i < ndim - a_ndim ? 1 : a[i - (ndim - a_ndim)];
        int64_t bv = i < ndim - b_ndim ? 1 : b[i - (ndim - b_ndim)];

        if (av != bv && av != 1 && bv != 1) {
            return _gd_error(GD_ERR_SHAPE, "shapes are not broadcastable");
        }
        out[i] = av > bv ? av : bv;
    }
    *out_ndim = ndim;
    return GD_OK;
}

static gd_status require_same_device(gd_tensor *a, gd_tensor *b)
{
    if (!gd_device_equal(gd_tensor_device(a), gd_tensor_device(b))) {
        return _gd_error(GD_ERR_DEVICE, "op inputs must share a device");
    }
    return GD_OK;
}

gd_status _gd_infer_elementwise(gd_tensor *a, gd_tensor *b, gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *da = NULL;
    const gd_tensor_desc *db = NULL;
    int64_t sizes[GD_MAX_DIMS];
    int ndim = 0;

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "elementwise argument is NULL");
    }
    if (gd_tensor_dtype(a) != gd_tensor_dtype(b)) {
        return _gd_error(GD_ERR_DTYPE, "elementwise inputs must share dtype");
    }
    status = require_same_device(a, b);
    if (status != GD_OK) {
        return status;
    }

    da = _gd_tensor_desc_ptr(a);
    db = _gd_tensor_desc_ptr(b);
    status = broadcast_shapes(da->sizes, da->ndim, db->sizes, db->ndim, sizes, &ndim);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_desc_contiguous(da->dtype, da->device, ndim, sizes, out);
}

gd_status _gd_infer_unary_float(gd_tensor *x, gd_tensor_desc *out)
{
    const gd_tensor_desc *dx = NULL;

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unary argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(x))) {
        return _gd_error(GD_ERR_DTYPE, "op requires a floating-point input");
    }
    dx = _gd_tensor_desc_ptr(x);
    return gd_tensor_desc_contiguous(dx->dtype, dx->device, dx->ndim, dx->sizes, out);
}

gd_status _gd_infer_matmul(gd_tensor *a,
                           gd_tensor *b,
                           bool trans_a,
                           bool trans_b,
                           gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *da = NULL;
    const gd_tensor_desc *db = NULL;
    int64_t a_mat[2];
    int64_t b_mat[2];
    int64_t batch[GD_MAX_DIMS];
    int64_t sizes[GD_MAX_DIMS];
    int batch_ndim = 0;
    int out_ndim = 0;
    int i = 0;

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "matmul argument is NULL");
    }
    if (gd_tensor_dtype(a) != gd_tensor_dtype(b)) {
        return _gd_error(GD_ERR_DTYPE, "matmul inputs must share dtype");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(a))) {
        return _gd_error(GD_ERR_DTYPE, "matmul requires floating-point inputs");
    }
    status = require_same_device(a, b);
    if (status != GD_OK) {
        return status;
    }

    da = _gd_tensor_desc_ptr(a);
    db = _gd_tensor_desc_ptr(b);
    if (da->ndim < 2 || db->ndim < 2) {
        return _gd_error(GD_ERR_SHAPE, "matmul inputs must be at least 2D");
    }

    a_mat[0] = da->sizes[da->ndim - 2];
    a_mat[1] = da->sizes[da->ndim - 1];
    if (trans_a) {
        int64_t tmp = a_mat[0];
        a_mat[0] = a_mat[1];
        a_mat[1] = tmp;
    }
    b_mat[0] = db->sizes[db->ndim - 2];
    b_mat[1] = db->sizes[db->ndim - 1];
    if (trans_b) {
        int64_t tmp = b_mat[0];
        b_mat[0] = b_mat[1];
        b_mat[1] = tmp;
    }
    if (a_mat[1] != b_mat[0]) {
        return _gd_error(GD_ERR_SHAPE, "matmul inner dimensions do not match");
    }

    status = broadcast_shapes(da->sizes, da->ndim - 2, db->sizes, db->ndim - 2, batch,
                              &batch_ndim);
    if (status != GD_OK) {
        return status;
    }
    if (batch_ndim + 2 > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "matmul result exceeds GD_MAX_DIMS");
    }
    for (i = 0; i < batch_ndim; ++i) {
        sizes[i] = batch[i];
    }
    sizes[batch_ndim] = a_mat[0];
    sizes[batch_ndim + 1] = b_mat[1];
    out_ndim = batch_ndim + 2;

    return gd_tensor_desc_contiguous(da->dtype, da->device, out_ndim, sizes, out);
}

gd_status _gd_infer_linear(gd_tensor *x,
                           gd_tensor *w,
                           gd_tensor *bias,
                           bool trans_w,
                           gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dx = NULL;
    const gd_tensor_desc *dw = NULL;
    int64_t sizes[GD_MAX_DIMS];
    int64_t in_features = 0;
    int64_t out_features = 0;
    int i = 0;

    if (x == NULL || w == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "linear argument is NULL");
    }
    if (gd_tensor_dtype(x) != gd_tensor_dtype(w)) {
        return _gd_error(GD_ERR_DTYPE, "linear input and weight must share dtype");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(x))) {
        return _gd_error(GD_ERR_DTYPE, "linear requires floating-point inputs");
    }
    status = require_same_device(x, w);
    if (status != GD_OK) {
        return status;
    }

    dx = _gd_tensor_desc_ptr(x);
    dw = _gd_tensor_desc_ptr(w);
    if (dx->ndim < 1) {
        return _gd_error(GD_ERR_SHAPE, "linear input must have at least one dimension");
    }
    if (dw->ndim != 2) {
        return _gd_error(GD_ERR_SHAPE, "linear weight must be 2D");
    }

    in_features = dx->sizes[dx->ndim - 1];
    if (trans_w) {
        /* weight stored [out, in]; logical op uses w^T => [in, out] */
        if (dw->sizes[1] != in_features) {
            return _gd_error(GD_ERR_SHAPE, "linear weight inner dimension mismatch");
        }
        out_features = dw->sizes[0];
    } else {
        if (dw->sizes[0] != in_features) {
            return _gd_error(GD_ERR_SHAPE, "linear weight inner dimension mismatch");
        }
        out_features = dw->sizes[1];
    }

    if (bias != NULL) {
        const gd_tensor_desc *db = NULL;

        if (gd_tensor_dtype(bias) != dx->dtype) {
            return _gd_error(GD_ERR_DTYPE, "linear bias must share dtype");
        }
        status = require_same_device(x, bias);
        if (status != GD_OK) {
            return status;
        }
        db = _gd_tensor_desc_ptr(bias);
        if (db->ndim != 1 || db->sizes[0] != out_features) {
            return _gd_error(GD_ERR_SHAPE, "linear bias must be [out_features]");
        }
    }

    for (i = 0; i < dx->ndim - 1; ++i) {
        sizes[i] = dx->sizes[i];
    }
    sizes[dx->ndim - 1] = out_features;

    return gd_tensor_desc_contiguous(dx->dtype, dx->device, dx->ndim, sizes, out);
}

gd_status _gd_infer_reduce(gd_tensor *x,
                           int dim,
                           bool keepdim,
                           int *norm_dim_out,
                           gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dx = NULL;
    int64_t sizes[GD_MAX_DIMS];
    int norm_dim = 0;
    int out_ndim = 0;
    int i = 0;
    int j = 0;

    if (x == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "reduce argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(x))) {
        return _gd_error(GD_ERR_DTYPE, "reduction requires a floating-point input");
    }

    dx = _gd_tensor_desc_ptr(x);
    status = normalize_dim(dim, dx->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }

    for (i = 0; i < dx->ndim; ++i) {
        if (i == norm_dim) {
            if (keepdim) {
                sizes[j++] = 1;
            }
        } else {
            sizes[j++] = dx->sizes[i];
        }
    }
    out_ndim = j;

    *norm_dim_out = norm_dim;
    status = gd_tensor_desc_contiguous(dx->dtype, dx->device, out_ndim, sizes, out);
    return status;
}

gd_status _gd_infer_softmax(gd_tensor *x, int dim, int *norm_dim_out, gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dx = NULL;
    int norm_dim = 0;

    if (x == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "softmax argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(x))) {
        return _gd_error(GD_ERR_DTYPE, "softmax requires a floating-point input");
    }
    dx = _gd_tensor_desc_ptr(x);
    status = normalize_dim(dim, dx->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    *norm_dim_out = norm_dim;
    return gd_tensor_desc_contiguous(dx->dtype, dx->device, dx->ndim, dx->sizes, out);
}

gd_status _gd_infer_cross_entropy(gd_tensor *logits,
                                  gd_tensor *targets,
                                  int class_dim,
                                  int *norm_dim_out,
                                  gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dl = NULL;
    const gd_tensor_desc *dt = NULL;
    int norm_dim = 0;
    int i = 0;
    int j = 0;

    if (logits == NULL || targets == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "cross_entropy argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(logits))) {
        return _gd_error(GD_ERR_DTYPE, "cross_entropy logits must be floating-point");
    }
    if (!_gd_dtype_is_integer(gd_tensor_dtype(targets))) {
        return _gd_error(GD_ERR_DTYPE, "cross_entropy targets must be I32 or I64");
    }
    status = require_same_device(logits, targets);
    if (status != GD_OK) {
        return status;
    }

    dl = _gd_tensor_desc_ptr(logits);
    dt = _gd_tensor_desc_ptr(targets);
    status = normalize_dim(class_dim, dl->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    if (dt->ndim != dl->ndim - 1) {
        return _gd_error(GD_ERR_SHAPE, "cross_entropy targets rank must be logits rank minus one");
    }
    for (i = 0; i < dl->ndim; ++i) {
        if (i == norm_dim) {
            continue;
        }
        if (dt->sizes[j] != dl->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "cross_entropy targets shape mismatch");
        }
        ++j;
    }

    *norm_dim_out = norm_dim;
    return gd_tensor_desc_contiguous(dl->dtype, dl->device, 0, NULL, out);
}

gd_status _gd_infer_cast(gd_tensor *x, gd_dtype dtype, gd_tensor_desc *out)
{
    const gd_tensor_desc *dx = NULL;

    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "cast argument is NULL");
    }
    if (dtype == GD_DTYPE_INVALID || dtype == GD_DTYPE_QUANTIZED) {
        return _gd_error(GD_ERR_DTYPE, "cast target dtype is invalid");
    }
    if (gd_dtype_sizeof(dtype) == 0U) {
        return _gd_error(GD_ERR_DTYPE, "cast target dtype has no fixed size");
    }
    dx = _gd_tensor_desc_ptr(x);
    return gd_tensor_desc_contiguous(dtype, dx->device, dx->ndim, dx->sizes, out);
}

gd_status _gd_infer_transpose(gd_tensor *x, const int *perm, int ndim, gd_tensor_desc *out)
{
    const gd_tensor_desc *dx = NULL;
    int64_t sizes[GD_MAX_DIMS];
    bool seen[GD_MAX_DIMS] = {0};
    int i = 0;

    if (x == NULL || perm == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "transpose argument is NULL");
    }
    dx = _gd_tensor_desc_ptr(x);
    if (ndim != dx->ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose perm rank must match tensor rank");
    }
    for (i = 0; i < ndim; ++i) {
        int axis = perm[i];
        if (axis < 0 || axis >= ndim || seen[axis]) {
            return _gd_error(GD_ERR_SHAPE, "transpose perm must be a permutation");
        }
        seen[axis] = true;
        sizes[i] = dx->sizes[axis];
    }
    return gd_tensor_desc_contiguous(dx->dtype, dx->device, ndim, sizes, out);
}

gd_status _gd_infer_embedding(gd_tensor *table, gd_tensor *ids, gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dt = NULL;
    const gd_tensor_desc *di = NULL;
    int64_t sizes[GD_MAX_DIMS];
    int i = 0;

    if (table == NULL || ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "embedding argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(table))) {
        return _gd_error(GD_ERR_DTYPE, "embedding table must be floating-point");
    }
    if (!_gd_dtype_is_integer(gd_tensor_dtype(ids))) {
        return _gd_error(GD_ERR_DTYPE, "embedding ids must be I32 or I64");
    }
    status = require_same_device(table, ids);
    if (status != GD_OK) {
        return status;
    }
    dt = _gd_tensor_desc_ptr(table);
    di = _gd_tensor_desc_ptr(ids);
    if (dt->ndim != 2) {
        return _gd_error(GD_ERR_SHAPE, "embedding table must be 2D [vocab, dim]");
    }
    if (di->ndim < 1 || di->ndim + 1 > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "embedding ids rank is out of range");
    }
    for (i = 0; i < di->ndim; ++i) {
        sizes[i] = di->sizes[i];
    }
    sizes[di->ndim] = dt->sizes[1];
    return gd_tensor_desc_contiguous(dt->dtype, dt->device, di->ndim + 1, sizes, out);
}

gd_status _gd_infer_rope(gd_tensor *x, gd_tensor *pos_ids, gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dx = NULL;
    const gd_tensor_desc *dp = NULL;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t pos_rows = 1;
    int64_t pos_numel = 1;
    int i = 0;

    if (x == NULL || pos_ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rope argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(x))) {
        return _gd_error(GD_ERR_DTYPE, "rope input must be floating-point");
    }
    if (!_gd_dtype_is_integer(gd_tensor_dtype(pos_ids))) {
        return _gd_error(GD_ERR_DTYPE, "rope position ids must be I32 or I64");
    }
    status = require_same_device(x, pos_ids);
    if (status != GD_OK) {
        return status;
    }
    dx = _gd_tensor_desc_ptr(x);
    dp = _gd_tensor_desc_ptr(pos_ids);
    if (dx->ndim < 2) {
        return _gd_error(GD_ERR_SHAPE, "rope input must be at least [.., heads, head_dim]");
    }
    heads = dx->sizes[dx->ndim - 2];
    head_dim = dx->sizes[dx->ndim - 1];
    for (i = 0; i < dx->ndim - 2; ++i) {
        pos_rows *= dx->sizes[i];
    }
    for (i = 0; i < dp->ndim; ++i) {
        pos_numel *= dp->sizes[i];
    }
    if (pos_numel != pos_rows) {
        return _gd_error(GD_ERR_SHAPE, "rope position count must equal product of leading dims");
    }
    (void)heads;
    (void)head_dim;
    return gd_tensor_desc_contiguous(dx->dtype, dx->device, dx->ndim, dx->sizes, out);
}

gd_status _gd_infer_sdpa(gd_tensor *q, gd_tensor *k, gd_tensor *v, gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    const gd_tensor_desc *dq = NULL;
    const gd_tensor_desc *dk = NULL;
    const gd_tensor_desc *dv = NULL;
    int i = 0;

    if (q == NULL || k == NULL || v == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa argument is NULL");
    }
    if (!_gd_dtype_is_float(gd_tensor_dtype(q))) {
        return _gd_error(GD_ERR_DTYPE, "sdpa requires floating-point inputs");
    }
    if (gd_tensor_dtype(q) != gd_tensor_dtype(k) || gd_tensor_dtype(q) != gd_tensor_dtype(v)) {
        return _gd_error(GD_ERR_DTYPE, "sdpa q/k/v must share dtype");
    }
    status = require_same_device(q, k);
    if (status != GD_OK) {
        return status;
    }
    status = require_same_device(q, v);
    if (status != GD_OK) {
        return status;
    }
    dq = _gd_tensor_desc_ptr(q);
    dk = _gd_tensor_desc_ptr(k);
    dv = _gd_tensor_desc_ptr(v);
    if (dq->ndim != 4 || dk->ndim != 4 || dv->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE, "sdpa q/k/v must be 4D [batch, seq, heads, head_dim]");
    }
    /* k and v share [batch, seq_k, kv_heads, head_dim]. */
    for (i = 0; i < 4; ++i) {
        if (dk->sizes[i] != dv->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "sdpa k and v shapes must match");
        }
    }
    if (dq->sizes[0] != dk->sizes[0]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa batch mismatch");
    }
    if (dq->sizes[3] != dk->sizes[3]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa head_dim mismatch");
    }
    if (dk->sizes[2] == 0 || dq->sizes[2] % dk->sizes[2] != 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa query heads must be a multiple of kv heads");
    }
    /* output o = [batch, seq_q, q_heads, head_dim] (== q shape). */
    return gd_tensor_desc_contiguous(dq->dtype, dq->device, 4, dq->sizes, out);
}
