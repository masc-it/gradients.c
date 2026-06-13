#include "../../backends/metal/metal_backend_internal.h"
#include "../_shared/loss/metal_pairwise_loss_common.h"
#include "metal_huber_types.h"

typedef struct gd_huber_metal_attrs {
    float delta;
} gd_huber_metal_attrs;

static id<MTLComputePipelineState> gd_huber_forward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_forward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_forward_f32_pso;
    }
    return nil;
}

static id<MTLComputePipelineState> gd_huber_backward_pso(gd_backend *backend, uint32_t dtype)
{
    if (backend == NULL) {
        return nil;
    }
    if (dtype == (uint32_t)GD_DTYPE_F16) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_backward_f16_pso;
    }
    if (dtype == (uint32_t)GD_DTYPE_F32) {
        return (__bridge id<MTLComputePipelineState>)backend->huber_backward_f32_pso;
    }
    return nil;
}

static gd_status gd_huber_validate_metal_attrs(const void *attrs)
{
    const gd_huber_metal_attrs *huber_attrs = (const gd_huber_metal_attrs *)attrs;
    if (huber_attrs == NULL || !(huber_attrs->delta > 0.0f)) {
        return GD_ERR_INVALID_ARGUMENT;
    }
    return GD_OK;
}

static void gd_huber_init_forward_args(void *dst,
                                       const gd_backend_tensor_view *x,
                                       const gd_backend_tensor_view *y,
                                       const gd_backend_tensor_view *out,
                                       uint64_t chunk_size,
                                       float scale,
                                       uint32_t simdgroups,
                                       const void *attrs)
{
    const gd_huber_metal_attrs *huber_attrs = (const gd_huber_metal_attrs *)attrs;
    gd_metal_huber_args *args = (gd_metal_huber_args *)dst;
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->out_offset = (uint64_t)out->offset;
    args->count = (uint64_t)x->count;
    args->chunk_size = chunk_size;
    args->scale = scale;
    args->delta = huber_attrs->delta;
    args->simdgroups = simdgroups;
}

static void gd_huber_init_backward_args(void *dst,
                                        const gd_backend_tensor_view *x,
                                        const gd_backend_tensor_view *y,
                                        const gd_backend_tensor_view *grad_out,
                                        const gd_backend_tensor_view *grad_x,
                                        const gd_backend_tensor_view *grad_y,
                                        float scale,
                                        const void *attrs)
{
    const gd_huber_metal_attrs *huber_attrs = (const gd_huber_metal_attrs *)attrs;
    gd_metal_huber_args *args = (gd_metal_huber_args *)dst;
    args->x_offset = (uint64_t)x->offset;
    args->y_offset = (uint64_t)y->offset;
    args->grad_out_offset = (uint64_t)grad_out->offset;
    args->out_offset = grad_x != NULL ? (uint64_t)grad_x->offset : 0U;
    args->dy_offset = grad_y != NULL ? (uint64_t)grad_y->offset : 0U;
    args->count = (uint64_t)x->count;
    args->scale = scale;
    args->delta = huber_attrs->delta;
    args->write_x = grad_x != NULL ? 1U : 0U;
    args->write_y = grad_y != NULL ? 1U : 0U;
}

static const gd_metal_pairwise_loss_config GD_HUBER_METAL_LOSS_CONFIG = {
    .args_size = sizeof(gd_metal_huber_args),
    .max_simdgroups = GD_METAL_HUBER_MAX_SIMDGROUPS,
    .forward_pso = gd_huber_forward_pso,
    .backward_pso = gd_huber_backward_pso,
    .validate_attrs = gd_huber_validate_metal_attrs,
    .init_forward_args = gd_huber_init_forward_args,
    .init_backward_args = gd_huber_init_backward_args,
};

gd_status gd_backend_huber_forward(gd_backend *backend,
                                   const gd_backend_tensor_view *x,
                                   const gd_backend_tensor_view *y,
                                   const gd_backend_tensor_view *out,
                                   uint64_t chunk_size,
                                   float scale,
                                   float delta)
{
    const gd_huber_metal_attrs attrs = {.delta = delta};
    return gd_metal_pairwise_loss_forward(backend,
                                          x,
                                          y,
                                          out,
                                          chunk_size,
                                          scale,
                                          &attrs,
                                          &GD_HUBER_METAL_LOSS_CONFIG);
}

gd_status gd_backend_huber_backward(gd_backend *backend,
                                    const gd_backend_tensor_view *x,
                                    const gd_backend_tensor_view *y,
                                    const gd_backend_tensor_view *grad_out,
                                    const gd_backend_tensor_view *grad_x,
                                    const gd_backend_tensor_view *grad_y,
                                    float scale,
                                    float delta)
{
    const gd_huber_metal_attrs attrs = {.delta = delta};
    return gd_metal_pairwise_loss_backward(backend,
                                           x,
                                           y,
                                           grad_out,
                                           grad_x,
                                           grad_y,
                                           scale,
                                           &attrs,
                                           &GD_HUBER_METAL_LOSS_CONFIG);
}
