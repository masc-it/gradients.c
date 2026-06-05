#include <gradients/ops.h>

#include "../_shared/binary/binary_core.h"

#include <string.h>

static gd_status gd_mul_validate_f16(gd_context *ctx,
                                     const gd_tensor *x,
                                     const gd_tensor *y,
                                     const gd_tensor *grad_out)
{
    if (ctx == NULL || x == NULL || y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (x->dtype != GD_DTYPE_F16 || y->dtype != GD_DTYPE_F16 ||
        (grad_out != NULL && grad_out->dtype != GD_DTYPE_F16)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "mul is optimized for f16 tensors only");
    }
    return GD_OK;
}

static bool gd_mul_suffix_reduce_compatible(const gd_tensor *src, const gd_tensor *dst)
{
    uint32_t dim;
    uint32_t prefix;
    bool suffix_started = false;
    size_t src_count;
    size_t dst_count;
    int64_t numel;
    if (src == NULL || dst == NULL || dst->rank > src->rank || gd_tensor_numel(src, &numel) != GD_OK ||
        numel <= 0) {
        return false;
    }
    src_count = (size_t)numel;
    if (gd_tensor_numel(dst, &numel) != GD_OK || numel <= 0) {
        return false;
    }
    dst_count = (size_t)numel;
    if (src_count <= dst_count || dst_count == 0U || src_count % dst_count != 0U) {
        return false;
    }
    prefix = src->rank - dst->rank;
    for (dim = 0U; dim < src->rank; ++dim) {
        int64_t dst_dim = dim < prefix ? 1 : dst->shape[dim - prefix];
        if (!suffix_started && dst_dim == 1 && src->shape[dim] != 1) {
            continue;
        }
        suffix_started = true;
        if (dst_dim != src->shape[dim]) {
            return false;
        }
    }
    return true;
}

static bool gd_mul_suffix_reduce_should_fuse(const gd_tensor *src, const gd_tensor *dst)
{
    int64_t src_numel;
    int64_t dst_numel;
    if (!gd_mul_suffix_reduce_compatible(src, dst) || gd_tensor_numel(src, &src_numel) != GD_OK ||
        gd_tensor_numel(dst, &dst_numel) != GD_OK || dst_numel <= 0) {
        return false;
    }
    return (src_numel / dst_numel) <= 32;
}

static gd_status gd_mul_empty_like(gd_context *ctx, const gd_tensor *shape_like, gd_tensor *out)
{
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || shape_like == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_empty(ctx, GD_ARENA_SCRATCH, shape_like->dtype, gd_shape_make(shape_like->rank, shape_like->shape), 256U, out);
    if (st != GD_OK) {
        return st;
    }
    out->is_leaf = false;
    return GD_OK;
}

static gd_status gd_mul_backward_direct_fused(gd_context *ctx,
                                              const gd_tensor *x,
                                              const gd_tensor *y,
                                              const gd_tensor *grad_out,
                                              gd_tensor *grad_x,
                                              gd_tensor *grad_y)
{
    gd_tensor dx;
    gd_tensor dy;
    gd_backend_tensor_view xv;
    gd_backend_tensor_view yv;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view dxv;
    gd_backend_tensor_view dyv;
    gd_status st;
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL || grad_x == NULL || grad_y == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_binary_shapes_equal(x, y) || !gd_binary_shapes_equal(x, grad_out)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_mul_empty_like(ctx, x, &dx);
    if (st != GD_OK) {
        return st;
    }
    st = gd_mul_empty_like(ctx, y, &dy);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(x, &xv) || !gd_op_tensor_view_from_tensor(y, &yv) ||
        !gd_op_tensor_view_from_tensor(grad_out, &gv) || !gd_op_tensor_view_from_tensor(&dx, &dxv) ||
        !gd_op_tensor_view_from_tensor(&dy, &dyv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "mul backward direct invalid tensor view");
    }
    st = gd_backend_mul_backward_direct(gd_context_backend(ctx), &xv, &yv, &gv, &dxv, &dyv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend mul backward direct failed");
    }
    *grad_x = dx;
    *grad_y = dy;
    return GD_OK;
}

static gd_status gd_mul_backward_suffix_fused(gd_context *ctx,
                                              const gd_tensor *grad_out,
                                              const gd_tensor *other,
                                              const gd_tensor *shape_like,
                                              gd_tensor *out)
{
    gd_tensor result;
    gd_backend_tensor_view gv;
    gd_backend_tensor_view ov;
    gd_backend_tensor_view rv;
    gd_status st;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || grad_out == NULL || other == NULL || shape_like == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    if (!gd_binary_shapes_equal(grad_out, other) || !gd_mul_suffix_reduce_compatible(grad_out, shape_like)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_mul_empty_like(ctx, shape_like, &result);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_op_tensor_view_from_tensor(grad_out, &gv) || !gd_op_tensor_view_from_tensor(other, &ov) ||
        !gd_op_tensor_view_from_tensor(&result, &rv)) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "mul backward suffix invalid tensor view");
    }
    st = gd_backend_mul_reduce_suffix(gd_context_backend(ctx), &gv, &ov, &rv);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "backend mul backward suffix failed");
    }
    *out = result;
    return GD_OK;
}

gd_status gd_mul(gd_context *ctx,
                 const gd_tensor *x,
                 const gd_tensor *y,
                 gd_tensor *out)
{
    gd_status st = gd_mul_validate_f16(ctx, x, y, NULL);
    if (st != GD_OK) {
        return st;
    }
    return gd_binary_apply_impl(ctx, x, y, out, gd_backend_mul, GD_OP_MUL, "mul", true);
}

gd_status gd_mul_backward(gd_context *ctx,
                          const gd_tensor *x,
                          const gd_tensor *y,
                          const gd_tensor *grad_out,
                          gd_tensor *grad_x,
                          gd_tensor *grad_y)
{
    gd_status st;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (grad_y != NULL) {
        memset(grad_y, 0, sizeof(*grad_y));
    }
    if (ctx == NULL || x == NULL || y == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_mul_validate_f16(ctx, x, y, grad_out);
    if (st != GD_OK) {
        return st;
    }
    st = gd_binary_validate_backward(ctx, x, y, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (grad_x != NULL && grad_y != NULL && gd_binary_shapes_equal(x, y) && gd_binary_shapes_equal(x, grad_out)) {
        return gd_mul_backward_direct_fused(ctx, x, y, grad_out, grad_x, grad_y);
    }
    if (grad_x != NULL) {
        if (gd_binary_shapes_equal(y, grad_out) && gd_mul_suffix_reduce_should_fuse(grad_out, x)) {
            st = gd_mul_backward_suffix_fused(ctx, grad_out, y, x, grad_x);
            if (st != GD_OK) {
                return st;
            }
        } else {
            gd_tensor full_dx;
            st = gd_binary_apply_impl(ctx, grad_out, y, &full_dx, gd_backend_mul, GD_OP_MUL, "mul backward dx", false);
            if (st != GD_OK) {
                return st;
            }
            if (gd_binary_shapes_equal(&full_dx, x)) {
                *grad_x = full_dx;
            } else {
                st = gd_binary_sum_to_shape(ctx, &full_dx, x, 1.0f, grad_x);
                if (st != GD_OK) {
                    return st;
                }
            }
        }
    }
    if (grad_y != NULL) {
        if (gd_binary_shapes_equal(x, grad_out) && gd_mul_suffix_reduce_should_fuse(grad_out, y)) {
            st = gd_mul_backward_suffix_fused(ctx, grad_out, x, y, grad_y);
            if (st != GD_OK) {
                return st;
            }
        } else {
            gd_tensor full_dy;
            st = gd_binary_apply_impl(ctx, grad_out, x, &full_dy, gd_backend_mul, GD_OP_MUL, "mul backward dy", false);
            if (st != GD_OK) {
                return st;
            }
            if (gd_binary_shapes_equal(&full_dy, y)) {
                *grad_y = full_dy;
            } else {
                st = gd_binary_sum_to_shape(ctx, &full_dy, y, 1.0f, grad_y);
                if (st != GD_OK) {
                    return st;
                }
            }
        }
    }
    return GD_OK;
}
