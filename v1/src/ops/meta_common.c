#include "meta_common.h"

#include "../core/internal.h"

bool _gd_dtype_is_float(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F32 || dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_BF16;
}

bool _gd_dtype_is_integer(gd_dtype dtype)
{
    return dtype == GD_DTYPE_I32 || dtype == GD_DTYPE_I64;
}

gd_status _gd_meta_normalize_dim(int dim, int ndim, int *out)
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

gd_status _gd_meta_broadcast_shapes(const int64_t *a,
                                    int a_ndim,
                                    const int64_t *b,
                                    int b_ndim,
                                    int64_t *out,
                                    int *out_ndim)
{
    int ndim = a_ndim > b_ndim ? a_ndim : b_ndim;
    int i = 0;

    if (a == NULL || b == NULL || out == NULL || out_ndim == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "broadcast_shapes argument is NULL");
    }
    if (a_ndim < 0 || b_ndim < 0) {
        return _gd_error(GD_ERR_SHAPE, "broadcast rank is invalid");
    }
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

gd_status _gd_meta_require_same_device(const gd_tensor_desc *a,
                                       const gd_tensor_desc *b,
                                       const char *message)
{
    if (a == NULL || b == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "device check argument is NULL");
    }
    if (!gd_device_equal(a->device, b->device)) {
        return _gd_error(GD_ERR_DEVICE,
                         message != NULL ? message : "op inputs must share a device");
    }
    return GD_OK;
}

gd_status _gd_meta_set_output_count(int expected, int *n_outputs)
{
    if (n_outputs == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "meta output count is NULL");
    }
    *n_outputs = expected;
    return GD_OK;
}

gd_status _gd_meta_elementwise(const gd_tensor_desc *a,
                               const gd_tensor_desc *b,
                               gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int ndim = 0;

    if (a == NULL || b == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "elementwise argument is NULL");
    }
    if (a->dtype != b->dtype) {
        return _gd_error(GD_ERR_DTYPE, "elementwise inputs must share dtype");
    }
    status = _gd_meta_require_same_device(a, b, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_broadcast_shapes(a->sizes, a->ndim, b->sizes, b->ndim, sizes, &ndim);
    if (status != GD_OK) {
        return status;
    }
    return gd_tensor_desc_contiguous(a->dtype, a->device, ndim, sizes, out);
}

gd_status _gd_meta_unary_float(const gd_tensor_desc *x, gd_tensor_desc *out)
{
    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "unary argument is NULL");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "op requires a floating-point input");
    }
    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, x->sizes, out);
}

gd_status _gd_meta_powlu(const gd_tensor_desc *x1,
                         const gd_tensor_desc *x2,
                         gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int i = 0;

    if (x1 == NULL || x2 == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "powlu argument is NULL");
    }
    if (x1->dtype != x2->dtype) {
        return _gd_error(GD_ERR_DTYPE, "powlu inputs must share dtype");
    }
    if (!_gd_dtype_is_float(x1->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "powlu requires floating-point inputs");
    }
    status = _gd_meta_require_same_device(x1, x2, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (x1->ndim != x2->ndim) {
        return _gd_error(GD_ERR_SHAPE, "powlu inputs must have equal shape");
    }
    for (i = 0; i < x1->ndim; ++i) {
        if (x1->sizes[i] != x2->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "powlu inputs must have equal shape");
        }
    }
    return gd_tensor_desc_contiguous(x1->dtype, x1->device, x1->ndim, x1->sizes, out);
}

gd_status _gd_meta_matmul(const gd_tensor_desc *a,
                          const gd_tensor_desc *b,
                          bool trans_a,
                          bool trans_b,
                          gd_tensor_desc *out)
{
    gd_status status = GD_OK;
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
    if (a->dtype != b->dtype) {
        return _gd_error(GD_ERR_DTYPE, "matmul inputs must share dtype");
    }
    if (!_gd_dtype_is_float(a->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "matmul requires floating-point inputs");
    }
    status = _gd_meta_require_same_device(a, b, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (a->ndim < 2 || b->ndim < 2) {
        return _gd_error(GD_ERR_SHAPE, "matmul inputs must be at least 2D");
    }

    a_mat[0] = a->sizes[a->ndim - 2];
    a_mat[1] = a->sizes[a->ndim - 1];
    if (trans_a) {
        int64_t tmp = a_mat[0];
        a_mat[0] = a_mat[1];
        a_mat[1] = tmp;
    }
    b_mat[0] = b->sizes[b->ndim - 2];
    b_mat[1] = b->sizes[b->ndim - 1];
    if (trans_b) {
        int64_t tmp = b_mat[0];
        b_mat[0] = b_mat[1];
        b_mat[1] = tmp;
    }
    if (a_mat[1] != b_mat[0]) {
        return _gd_error(GD_ERR_SHAPE, "matmul inner dimensions do not match");
    }

    status = _gd_meta_broadcast_shapes(a->sizes, a->ndim - 2, b->sizes, b->ndim - 2,
                                       batch, &batch_ndim);
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

    return gd_tensor_desc_contiguous(a->dtype, a->device, out_ndim, sizes, out);
}

gd_status _gd_meta_linear(const gd_tensor_desc *x,
                          const gd_tensor_desc *w,
                          const gd_tensor_desc *bias,
                          bool trans_w,
                          gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int64_t in_features = 0;
    int64_t out_features = 0;
    int i = 0;

    if (x == NULL || w == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "linear argument is NULL");
    }
    if (x->dtype != w->dtype) {
        return _gd_error(GD_ERR_DTYPE, "linear input and weight must share dtype");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "linear requires floating-point inputs");
    }
    status = _gd_meta_require_same_device(x, w, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (x->ndim < 1) {
        return _gd_error(GD_ERR_SHAPE, "linear input must have at least one dimension");
    }
    if (w->ndim != 2) {
        return _gd_error(GD_ERR_SHAPE, "linear weight must be 2D");
    }

    in_features = x->sizes[x->ndim - 1];
    if (trans_w) {
        if (w->sizes[1] != in_features) {
            return _gd_error(GD_ERR_SHAPE, "linear weight inner dimension mismatch");
        }
        out_features = w->sizes[0];
    } else {
        if (w->sizes[0] != in_features) {
            return _gd_error(GD_ERR_SHAPE, "linear weight inner dimension mismatch");
        }
        out_features = w->sizes[1];
    }

    if (bias != NULL) {
        if (bias->dtype != x->dtype) {
            return _gd_error(GD_ERR_DTYPE, "linear bias must share dtype");
        }
        status = _gd_meta_require_same_device(x, bias, "op inputs must share a device");
        if (status != GD_OK) {
            return status;
        }
        if (bias->ndim != 1 || bias->sizes[0] != out_features) {
            return _gd_error(GD_ERR_SHAPE, "linear bias must be [out_features]");
        }
    }

    for (i = 0; i < x->ndim - 1; ++i) {
        sizes[i] = x->sizes[i];
    }
    sizes[x->ndim - 1] = out_features;

    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, sizes, out);
}

gd_status _gd_meta_reduce(const gd_tensor_desc *x,
                          int dim,
                          bool keepdim,
                          int *norm_dim_out,
                          gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int norm_dim = 0;
    int out_ndim = 0;
    int i = 0;
    int j = 0;

    if (x == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "reduce argument is NULL");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "reduction requires a floating-point input");
    }

    status = _gd_meta_normalize_dim(dim, x->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }

    for (i = 0; i < x->ndim; ++i) {
        if (i == norm_dim) {
            if (keepdim) {
                sizes[j++] = 1;
            }
        } else {
            sizes[j++] = x->sizes[i];
        }
    }
    out_ndim = j;

    *norm_dim_out = norm_dim;
    return gd_tensor_desc_contiguous(x->dtype, x->device, out_ndim, sizes, out);
}

gd_status _gd_meta_reduce_to(const gd_tensor_desc *x,
                             const gd_tensor_desc *target,
                             gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int ndim = 0;
    int i = 0;

    if (x == NULL || target == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "reduce_to argument is NULL");
    }
    if (x->dtype != target->dtype) {
        return _gd_error(GD_ERR_DTYPE, "reduce_to target must share dtype");
    }
    status = _gd_meta_require_same_device(x, target, "reduce_to target must share device");
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_broadcast_shapes(target->sizes, target->ndim, x->sizes, x->ndim,
                                       sizes, &ndim);
    if (status != GD_OK) {
        return status;
    }
    if (ndim != x->ndim) {
        return _gd_error(GD_ERR_SHAPE, "reduce_to target is not broadcastable to input");
    }
    for (i = 0; i < ndim; ++i) {
        if (sizes[i] != x->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "reduce_to target is not broadcastable to input");
        }
    }
    *out = *target;
    return GD_OK;
}

gd_status _gd_meta_softmax(const gd_tensor_desc *x,
                           int dim,
                           int *norm_dim_out,
                           gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int norm_dim = 0;

    if (x == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "softmax argument is NULL");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "softmax requires a floating-point input");
    }
    status = _gd_meta_normalize_dim(dim, x->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    *norm_dim_out = norm_dim;
    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, x->sizes, out);
}

gd_status _gd_meta_cross_entropy(const gd_tensor_desc *logits,
                                 const gd_tensor_desc *targets,
                                 int class_dim,
                                 int *norm_dim_out,
                                 gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int norm_dim = 0;
    int i = 0;
    int j = 0;

    if (logits == NULL || targets == NULL || out == NULL || norm_dim_out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "cross_entropy argument is NULL");
    }
    if (!_gd_dtype_is_float(logits->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "cross_entropy logits must be floating-point");
    }
    if (!_gd_dtype_is_integer(targets->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "cross_entropy targets must be I32 or I64");
    }
    status = _gd_meta_require_same_device(logits, targets, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }

    status = _gd_meta_normalize_dim(class_dim, logits->ndim, &norm_dim);
    if (status != GD_OK) {
        return status;
    }
    if (targets->ndim != logits->ndim - 1) {
        return _gd_error(GD_ERR_SHAPE, "cross_entropy targets rank must be logits rank minus one");
    }
    for (i = 0; i < logits->ndim; ++i) {
        if (i == norm_dim) {
            continue;
        }
        if (targets->sizes[j] != logits->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "cross_entropy targets shape mismatch");
        }
        ++j;
    }

    *norm_dim_out = norm_dim;
    return gd_tensor_desc_contiguous(GD_DTYPE_F32, logits->device, 0, NULL, out);
}

gd_status _gd_meta_lm_cross_entropy(const gd_tensor_desc *hidden,
                                    const gd_tensor_desc *weight,
                                    const gd_tensor_desc *targets,
                                    gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int i = 0;

    if (hidden == NULL || weight == NULL || targets == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "lm_cross_entropy argument is NULL");
    }
    if (hidden->dtype != weight->dtype) {
        return _gd_error(GD_ERR_DTYPE, "lm_cross_entropy hidden and weight must share dtype");
    }
    if (!_gd_dtype_is_float(hidden->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "lm_cross_entropy hidden/weight must be floating-point");
    }
    if (!_gd_dtype_is_integer(targets->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "lm_cross_entropy targets must be I32 or I64");
    }
    status = _gd_meta_require_same_device(hidden, weight, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_require_same_device(hidden, targets, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }

    if (hidden->ndim < 1) {
        return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy hidden must have rank >= 1");
    }
    if (weight->ndim != 2) {
        return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy weight must be [vocab, dim]");
    }
    if (weight->sizes[1] != hidden->sizes[hidden->ndim - 1]) {
        return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy hidden dim must match weight dim");
    }
    if (targets->ndim != hidden->ndim - 1) {
        return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy targets rank mismatch");
    }
    for (i = 0; i < targets->ndim; ++i) {
        if (targets->sizes[i] != hidden->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "lm_cross_entropy targets shape mismatch");
        }
    }
    return gd_tensor_desc_contiguous(GD_DTYPE_F32, hidden->device, 0, NULL, out);
}

gd_status _gd_meta_cast(const gd_tensor_desc *x, gd_dtype dtype, gd_tensor_desc *out)
{
    if (x == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "cast argument is NULL");
    }
    if (dtype == GD_DTYPE_INVALID || dtype == GD_DTYPE_QUANTIZED) {
        return _gd_error(GD_ERR_DTYPE, "cast target dtype is invalid");
    }
    if (gd_dtype_sizeof(dtype) == 0U) {
        return _gd_error(GD_ERR_DTYPE, "cast target dtype has no fixed size");
    }
    return gd_tensor_desc_contiguous(dtype, x->device, x->ndim, x->sizes, out);
}

gd_status _gd_meta_transpose(const gd_tensor_desc *x,
                             const int *perm,
                             int ndim,
                             gd_tensor_desc *out)
{
    int64_t sizes[GD_MAX_DIMS];
    bool seen[GD_MAX_DIMS] = {0};
    int i = 0;

    if (x == NULL || perm == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "transpose argument is NULL");
    }
    if (ndim != x->ndim) {
        return _gd_error(GD_ERR_SHAPE, "transpose perm rank must match tensor rank");
    }
    for (i = 0; i < ndim; ++i) {
        int axis = perm[i];
        if (axis < 0 || axis >= ndim || seen[axis]) {
            return _gd_error(GD_ERR_SHAPE, "transpose perm must be a permutation");
        }
        seen[axis] = true;
        sizes[i] = x->sizes[axis];
    }
    return gd_tensor_desc_contiguous(x->dtype, x->device, ndim, sizes, out);
}

gd_status _gd_meta_embedding(const gd_tensor_desc *table,
                             const gd_tensor_desc *ids,
                             gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t sizes[GD_MAX_DIMS];
    int i = 0;

    if (table == NULL || ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "embedding argument is NULL");
    }
    if (!_gd_dtype_is_float(table->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "embedding table must be floating-point");
    }
    if (!_gd_dtype_is_integer(ids->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "embedding ids must be I32 or I64");
    }
    status = _gd_meta_require_same_device(table, ids, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (table->ndim != 2) {
        return _gd_error(GD_ERR_SHAPE, "embedding table must be 2D [vocab, dim]");
    }
    if (ids->ndim < 1 || ids->ndim + 1 > GD_MAX_DIMS) {
        return _gd_error(GD_ERR_SHAPE, "embedding ids rank is out of range");
    }
    for (i = 0; i < ids->ndim; ++i) {
        sizes[i] = ids->sizes[i];
    }
    sizes[ids->ndim] = table->sizes[1];
    return gd_tensor_desc_contiguous(table->dtype, table->device, ids->ndim + 1, sizes, out);
}

gd_status _gd_meta_rope(const gd_tensor_desc *x,
                        const gd_tensor_desc *pos_ids,
                        gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int64_t heads = 0;
    int64_t head_dim = 0;
    int64_t pos_rows = 1;
    int64_t pos_numel = 1;
    int i = 0;

    if (x == NULL || pos_ids == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "rope argument is NULL");
    }
    if (!_gd_dtype_is_float(x->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "rope input must be floating-point");
    }
    if (!_gd_dtype_is_integer(pos_ids->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "rope position ids must be I32 or I64");
    }
    status = _gd_meta_require_same_device(x, pos_ids, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (x->ndim < 2) {
        return _gd_error(GD_ERR_SHAPE, "rope input must be at least [.., heads, head_dim]");
    }
    heads = x->sizes[x->ndim - 2];
    head_dim = x->sizes[x->ndim - 1];
    for (i = 0; i < x->ndim - 2; ++i) {
        pos_rows *= x->sizes[i];
    }
    for (i = 0; i < pos_ids->ndim; ++i) {
        pos_numel *= pos_ids->sizes[i];
    }
    if (pos_numel != pos_rows) {
        return _gd_error(GD_ERR_SHAPE, "rope position count must equal product of leading dims");
    }
    (void)heads;
    (void)head_dim;
    return gd_tensor_desc_contiguous(x->dtype, x->device, x->ndim, x->sizes, out);
}

gd_status _gd_meta_sdpa(const gd_tensor_desc *q,
                        const gd_tensor_desc *k,
                        const gd_tensor_desc *v,
                        gd_tensor_desc *out)
{
    gd_status status = GD_OK;
    int i = 0;

    if (q == NULL || k == NULL || v == NULL || out == NULL) {
        return _gd_error(GD_ERR_INVALID_ARGUMENT, "sdpa argument is NULL");
    }
    if (!_gd_dtype_is_float(q->dtype)) {
        return _gd_error(GD_ERR_DTYPE, "sdpa requires floating-point inputs");
    }
    if (q->dtype != k->dtype || q->dtype != v->dtype) {
        return _gd_error(GD_ERR_DTYPE, "sdpa q/k/v must share dtype");
    }
    status = _gd_meta_require_same_device(q, k, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    status = _gd_meta_require_same_device(q, v, "op inputs must share a device");
    if (status != GD_OK) {
        return status;
    }
    if (q->ndim != 4 || k->ndim != 4 || v->ndim != 4) {
        return _gd_error(GD_ERR_SHAPE, "sdpa q/k/v must be 4D [batch, seq, heads, head_dim]");
    }
    for (i = 0; i < 4; ++i) {
        if (k->sizes[i] != v->sizes[i]) {
            return _gd_error(GD_ERR_SHAPE, "sdpa k and v shapes must match");
        }
    }
    if (q->sizes[0] != k->sizes[0]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa batch mismatch");
    }
    if (q->sizes[3] != k->sizes[3]) {
        return _gd_error(GD_ERR_SHAPE, "sdpa head_dim mismatch");
    }
    if (k->sizes[2] == 0 || q->sizes[2] % k->sizes[2] != 0) {
        return _gd_error(GD_ERR_SHAPE, "sdpa query heads must be a multiple of kv heads");
    }
    return gd_tensor_desc_contiguous(q->dtype, q->device, 4, q->sizes, out);
}
