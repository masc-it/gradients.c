#include "reshape_impl.h"

#include <gradients/ops.h>

#include "../autograd_impl.h"
#include "../op_common.h"

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool gd_reshape_autograd_dtype(gd_dtype dtype)
{
    return dtype == GD_DTYPE_F16 || dtype == GD_DTYPE_F32;
}

static bool gd_reshape_i64_mul_overflow(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL || a < 0 || b < 0) {
        return true;
    }
    if (a != 0 && b > INT64_MAX / a) {
        return true;
    }
    *out = a * b;
    return false;
}

static void gd_reshape_set_contiguous_strides(gd_tensor *tensor)
{
    uint32_t i;
    int64_t stride = 1;
    if (tensor == NULL) {
        return;
    }
    for (i = 0U; i < GD_MAX_DIMS; ++i) {
        tensor->strides[i] = 0;
    }
    for (i = tensor->rank; i > 0U; --i) {
        uint32_t dim = i - 1U;
        tensor->strides[dim] = stride;
        stride *= tensor->shape[dim];
    }
}

static gd_status gd_reshape_resolve_shape(gd_context *ctx,
                                          gd_shape requested,
                                          int64_t input_numel,
                                          gd_shape *out_shape)
{
    uint32_t i;
    int32_t infer_index = -1;
    int64_t known_product = 1;
    gd_shape resolved;
    if (out_shape != NULL) {
        memset(out_shape, 0, sizeof(*out_shape));
    }
    if (ctx == NULL || out_shape == NULL || requested.rank > GD_MAX_DIMS || input_numel <= 0) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    memset(&resolved, 0, sizeof(resolved));
    resolved.rank = requested.rank;
    for (i = 0U; i < requested.rank; ++i) {
        int64_t dim = requested.dims[i];
        if (dim == -1) {
            if (infer_index >= 0) {
                return gd_context_set_error(ctx,
                                            GD_ERR_INVALID_ARGUMENT,
                                            "reshape allows at most one inferred dimension");
            }
            infer_index = (int32_t)i;
            resolved.dims[i] = -1;
            continue;
        }
        if (dim <= 0) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "reshape dimensions must be positive, except one -1");
        }
        if (gd_reshape_i64_mul_overflow(known_product, dim, &known_product)) {
            return gd_context_set_error(ctx, GD_ERR_OUT_OF_MEMORY, "reshape dimension product overflow");
        }
        resolved.dims[i] = dim;
    }
    if (infer_index >= 0) {
        if (known_product <= 0 || input_numel % known_product != 0) {
            return gd_context_set_error(ctx,
                                        GD_ERR_INVALID_ARGUMENT,
                                        "reshape inferred dimension is not integral");
        }
        resolved.dims[infer_index] = input_numel / known_product;
        if (resolved.dims[infer_index] <= 0) {
            return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reshape inferred dimension is invalid");
        }
    } else if (known_product != input_numel) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reshape element count mismatch");
    }
    *out_shape = resolved;
    return GD_OK;
}

gd_status gd_reshape_view_impl(gd_context *ctx,
                               const gd_tensor *x,
                               gd_shape shape,
                               bool record_autograd,
                               gd_tensor *out)
{
    gd_status st;
    gd_shape resolved;
    gd_tensor y;
    int64_t input_numel;
    uint32_t i;
    bool needs_grad;
    if (out != NULL) {
        memset(out, 0, sizeof(*out));
    }
    if (ctx == NULL || x == NULL || out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    if (!gd_tensor_is_contiguous(x)) {
        return gd_context_set_error(ctx, GD_ERR_UNSUPPORTED, "reshape requires contiguous input");
    }
    st = gd_tensor_numel(x, &input_numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "reshape invalid input shape");
    }
    st = gd_reshape_resolve_shape(ctx, shape, input_numel, &resolved);
    if (st != GD_OK) {
        return st;
    }
    needs_grad = record_autograd && x->requires_grad;
    if (needs_grad && !gd_reshape_autograd_dtype(x->dtype)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "reshape autograd supports f16/f32 tensors only");
    }

    y = *x;
    st = gd_context_next_tensor_id(ctx, &y.id);
    if (st != GD_OK) {
        return st;
    }
    y.version = 0U;
    y.rank = resolved.rank;
    for (i = 0U; i < GD_MAX_DIMS; ++i) {
        y.shape[i] = 0;
    }
    for (i = 0U; i < resolved.rank; ++i) {
        y.shape[i] = resolved.dims[i];
    }
    gd_reshape_set_contiguous_strides(&y);
    y.is_view = true;
    y.requires_grad = false;
    y.is_leaf = false;
    st = gd_tensor_validate(ctx, &y);
    if (st != GD_OK) {
        return st;
    }
    if (needs_grad) {
        const gd_tensor *inputs[1] = {x};
        gd_tensor *outputs[1] = {&y};
        st = gd_autograd_record(ctx,
                                GD_OP_RESHAPE,
                                inputs,
                                1U,
                                outputs,
                                1U,
                                NULL,
                                0U,
                                NULL,
                                0U);
        if (st != GD_OK) {
            return st;
        }
    }
    *out = y;
    return GD_OK;
}

gd_status gd_reshape(gd_context *ctx,
                     const gd_tensor *x,
                     gd_shape shape,
                     gd_tensor *out)
{
    return gd_reshape_view_impl(ctx, x, shape, true, out);
}

gd_status gd_reshape_backward(gd_context *ctx,
                              const gd_tensor *x,
                              const gd_tensor *grad_out,
                              gd_tensor *grad_x)
{
    gd_status st;
    int64_t x_numel;
    int64_t grad_numel;
    if (grad_x != NULL) {
        memset(grad_x, 0, sizeof(*grad_x));
    }
    if (ctx == NULL || x == NULL || grad_out == NULL) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    st = gd_tensor_validate(ctx, x);
    if (st != GD_OK) {
        return st;
    }
    st = gd_tensor_validate(ctx, grad_out);
    if (st != GD_OK) {
        return st;
    }
    if (x->dtype != grad_out->dtype) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reshape backward dtype mismatch");
    }
    if (!gd_tensor_is_contiguous(x) || !gd_tensor_is_contiguous(grad_out)) {
        return gd_context_set_error(ctx,
                                    GD_ERR_UNSUPPORTED,
                                    "reshape backward requires contiguous tensors");
    }
    st = gd_tensor_numel(x, &x_numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "reshape backward invalid input shape");
    }
    st = gd_tensor_numel(grad_out, &grad_numel);
    if (st != GD_OK) {
        return gd_context_set_error(ctx, st, "reshape backward invalid grad_out shape");
    }
    if (x_numel != grad_numel) {
        return gd_context_set_error(ctx, GD_ERR_INVALID_ARGUMENT, "reshape backward element count mismatch");
    }
    if (grad_x == NULL) {
        return GD_OK;
    }
    return gd_reshape_view_impl(ctx,
                                grad_out,
                                gd_shape_make(x->rank, x->shape),
                                false,
                                grad_x);
}
